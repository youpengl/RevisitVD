# coding=utf-8

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
os.environ["HF_ENDPOINT"] = "https://huggingface.co"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import gc
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
import json

from tqdm import tqdm
import multiprocessing
import warnings
warnings.filterwarnings('ignore')
cpu_cont = multiprocessing.cpu_count()
import torch.nn.functional as F
from deepspeed.accelerator import get_accelerator
from transformers import (get_linear_schedule_with_warmup)

from transformers import (AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig)

from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    PeftModel,
    PeftModelForSequenceClassification,
    prepare_model_for_kbit_training
)

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed


logger = get_logger(__name__)
                                  
class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_ids,
                 attention_mask,
                 label,
                 index
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.label = label
        self.index = index
        
import re

def code_cleaner_pdbert(func):
    code = re.sub(r' +|\t+', ' ', func)
    return code

def convert_examples_to_features(js, tokenizer, args):
    """convert examples to token ids"""
    source_tokens = tokenizer.encode_plus(
        ' '.join(js['func'].split()),
        # code_cleaner_pdbert(js['func']),
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt"
    )
    input_ids = source_tokens["input_ids"]
    attention_mask = source_tokens["attention_mask"]

    return InputFeatures(input_ids, attention_mask, js['target'], js['idx'])

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        with open(file_path) as f:
            count = 0
            for line in tqdm(f):
                js=json.loads(line.strip())
                self.examples.append(convert_examples_to_features(js, tokenizer, args))
                    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        input_ids = self.examples[i].input_ids
        attention_mask = self.examples[i].attention_mask
        label = self.examples[i].label
        index = self.examples[i].index
        return (input_ids, attention_mask, torch.tensor(label), index)
    

def train(args, accelerator, tokenizer, train_dataset, eval_dataset, model):
    """ Train the model """ 
    train_sampler = RandomSampler(train_dataset)
    
    train_dataloader = DataLoader(train_dataset, sampler = train_sampler, 
                                  batch_size = args.train_batch_size, num_workers = 4, pin_memory=False)
    
    eval_sampler = SequentialSampler(eval_dataset)
    
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4, pin_memory=False)
    
    args.max_steps = args.epoch * len(train_dataloader)
    args.num_train_epochs = args.epoch
    
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    if args.ft_head:
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0}
        ]
    else:
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr = args.learning_rate, eps = args.adam_epsilon)
        
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = args.max_steps * 0.1 if args.warmup_steps == -1 else args.warmup_steps,
                                                num_training_steps = args.max_steps)

    model, train_dataloader, eval_dataloader, optimizer, scheduler = accelerator.prepare(
        model, train_dataloader, eval_dataloader, optimizer, scheduler
    )
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size = %d",
                accelerator.num_processes * args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)
    
    if not args.resume:
        step = 0
        best_bacc = 0.0
        args.start_epoch = 0
        
    else:
        step = 382
        best_bacc = 71.7
        args.start_epoch = 1
        
        for _ in range(args.start_epoch):
            bar = iter(train_dataloader)
        scheduler.step(step)
        current_lr = scheduler.get_last_lr()[0]
        for group in optimizer.param_groups:
            group['lr'] = current_lr
    
    model.zero_grad()
    model.train()

    for idx in range(args.start_epoch, int(args.num_train_epochs)): 
        bar = tqdm(train_dataloader, total = len(train_dataloader), disable=not accelerator.is_local_main_process)
        for batch in bar:
            input_ids = batch[0].squeeze(1)
            attention_mask = batch[1].squeeze(1)
            labels = batch[2]
            with accelerator.accumulate(model):
                output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = output[0]
                logits = output[1]

                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  

            step += 1

        del bar, batch, input_ids, attention_mask, labels, output, loss, logits

        if args.evaluate_during_training:  
            logger.info(f"***** Epoch {idx} finished : step {step} *****")
            results = evaluate(args, accelerator, model, eval_dataset, eval_dataloader)
            for key, value in results.items():
                logger.info("  %s = %s", key, round(value,4))     

        if results['eval_bacc'] > best_bacc:
            best_bacc = results['eval_bacc']
            logger.info("  "+"*"*20)
            logger.info("  Best bacc:%s", round(best_bacc, 4))          
            
            checkpoint_prefix = f'{args.localtime}/{args.project}/checkpoint-best-bacc'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            accelerator.wait_for_everyone()
            accelerator.save_state(output_dir)
            if accelerator.is_main_process:
                model.module.active_peft_config.save_pretrained(output_dir)

            logger.info(f"  Saving best balance accuracy lora adapter checkpoint at epoch {idx} to {output_dir}")
            logger.info("  "+"*"*20)              

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        from zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict
        convert_zero_checkpoint_to_fp32_state_dict(output_dir, output_dir, safe_serialization=True, exclude_frozen_parameters=True)
        os.rename(os.path.join(output_dir, 'model.safetensors'), os.path.join(output_dir, 'adapter_model.safetensors'))

def calculate_metrics(labels, preds):
    acc=accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    TN, FP, FN, TP = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    tnr = float(TN / (TN + FP))
    fpr = float(FP / (FP + TN))
    fnr = float(FN / (TP + FN))
    return round(acc, 4) * 100, round(prec, 4) * 100, \
        round(recall, 4) * 100, round(f1, 4) * 100, round(tnr, 4) * 100, \
            round(fpr, 4) * 100, round(fnr, 4) * 100, round((recall+tnr)/2, 4) *100

def evaluate(args, accelerator, model, eval_dataset, eval_dataloader):
    eval_output_dir = args.output_dir

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir, exist_ok=True)
        
    # Eval!
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    model.eval()
    losses = []
    probs = [] 
    labels = []
    bar = tqdm(eval_dataloader, total = len(eval_dataloader), disable=not accelerator.is_local_main_process)
    for batch in bar:  
        input_ids = batch[0].squeeze(1)
        attention_mask = batch[1].squeeze(1)
        label = batch[2]

        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=label)
            lm_loss = output[0]
            logit = output[1]

        losses.append(accelerator.gather_for_metrics(lm_loss.repeat(args.eval_batch_size)))
        logit, label = accelerator.gather_for_metrics((logit, label))
        prob = torch.nn.functional.softmax(logit, dim=-1)
        probs.append(prob.cpu().numpy())
        labels.append(label.cpu().numpy())
            
    
    losses = torch.cat(losses)
    eval_loss = torch.mean(losses)
    probs = np.concatenate(probs, 0)
    labels = np.concatenate(labels ,0)

    preds = probs[:, 1] > 0.5

    eval_acc, eval_prec, eval_recall, eval_f1, eval_tnr, eval_fpr, eval_fnr, eval_bacc = calculate_metrics(labels, preds)
    perplexity = torch.tensor(eval_loss)

    result = {
        "eval_loss": float(perplexity),
        "eval_acc": eval_acc,
        "eval_prec": eval_prec,
        "eval_recall": eval_recall,
        "eval_f1": eval_f1,
        "eval_tnr": eval_tnr,
        "eval_fpr": eval_fpr,
        "eval_fnr": eval_fnr,
        "eval_bacc": eval_bacc
    }
    
    return result

def test(args, accelerator, model, test_dataset, test_dataloader):
    # import time
    model, test_dataloader = accelerator.prepare(model, test_dataloader)
    model.eval()
    probs=[]   
    labels=[]
    bar = tqdm(test_dataloader, total=len(test_dataloader), disable=not accelerator.is_local_main_process)
    for batch in bar:
        input_ids = batch[0].squeeze(1)
        attention_mask = batch[1].squeeze(1)
        label = batch[2]
                
        with torch.no_grad():
            logit = model(input_ids=input_ids, attention_mask=attention_mask, labels=label)[1]
        logit, label = accelerator.gather_for_metrics((logit, label))
        prob = torch.nn.functional.softmax(logit, dim=-1)
        probs.append(prob.cpu().numpy())
        labels.append(label.cpu().numpy())

    probs = np.concatenate(probs, 0)
    labels = np.concatenate(labels, 0)
    
    preds = probs[:, 1] > 0.5

    test_acc, test_prec, test_recall, test_f1, test_tnr, test_fpr, test_fnr, test_bacc = calculate_metrics(labels, preds)

    result = {
        "test_acc": test_acc,
        "test_prec": test_prec,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "test_tnr": test_tnr,
        "test_fpr": test_fpr,
        "test_fnr": test_fnr,
        "test_bacc": test_bacc
    }

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        test_project = args.test_project if args.test_project else args.project
        name = 'bacc'

        if not os.path.exists(os.path.join(args.output_dir, args.localtime, test_project, name)):
            os.makedirs(os.path.join(args.output_dir, args.localtime, test_project, name))
            
        with open(os.path.join(args.output_dir, args.localtime, test_project, name, "predictions.txt"),'w') as f:
            for example, pred in zip(test_dataset.examples, preds):
                f.write(f"idx: {example.index}, pred: {1 if pred else 0}, target: {example.label}\n")

        save_path = os.path.join(args.output_dir, args.localtime, test_project, name, "result.npz")

        np.savez(save_path, test_result=result)


    return result
           
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument('--project', type=str, required=True, help="using dataset from this project.")
    parser.add_argument('--test_project', type=str, required=False, help="test setup name.")
    parser.add_argument("--train_data_file", default=None, type=str, required=False,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--run_dir', type=str, default="runs", help="parent directory to store run stats.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input valid data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input test data file to evaluate the perplexity on (a text file).")   
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")   
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--test_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_steps', type=int, default=999999999999,
                        help="Log every X updates steps.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epoch', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--max-patience', type=int, default=20, help="Max iterations for model with no improvement.")
    parser.add_argument('--gpu', type=int, default=-1, help="specify gpu id, -1 denotes using all" )

    
    # peft
    parser.add_argument('--ft_head', action='store_true', help="whether to fine-tune the head only")
    parser.add_argument('--lora', action='store_true', help="whether to use lora")
    parser.add_argument('--q', action='store_true', help="whether to use qlora")
    parser.add_argument('--lora_rank', type=int, default=64)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--resume', action='store_true', help="resume training")
    parser.add_argument('--localtime', type=str, default='2026-01-01-00:00:00')
    parser.add_argument('--basetime', type=str, default='2026-01-02-00:00:00')

    args = parser.parse_args()

    if args.do_train:
        accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    else:
        accelerator = Accelerator()
    
            
    if not os.path.exists('logs'):
        os.makedirs('logs', exist_ok=True)

    logging.basicConfig(
        filename = f'logs/{args.localtime}.log',
        format = "%(asctime)s - %(levelname)s - %(message)s",
        level = logging.INFO,
        datefmt = "%Y-%m-%d %H:%M:%S")
    
    logger.info(accelerator.state, main_process_only=False)
    

    # Set seed
    set_seed(args.seed)
    
    if args.lora:
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules="all-linear",
            lora_dropout=args.lora_dropout,
            bias='none',
            inference_mode=False,
            task_type=TaskType.SEQ_CLS
    )

    if args.q:
        q_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if args.q:
            model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, quantization_config=q_config, trust_remote_code=True, use_cache=False)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, trust_remote_code=True, use_cache=False)

        if not model.config.pad_token_id:
            model.config.pad_token_id = model.config.eos_token_id

        if args.ft_head:
            for name, param in model.named_parameters():
                if "score" not in name:
                    param.requires_grad = False 
        
        if args.q:
            model = prepare_model_for_kbit_training(model)

        if args.lora:
            if not args.resume:
                model = get_peft_model(model, lora_config)
            else:
                model = PeftModel.from_pretrained(model, 'xxx', is_trainable=True)   

        train_dataset = TextDataset(tokenizer, args, args.train_data_file)
        eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)
        train(args, accelerator, tokenizer, train_dataset, eval_dataset, model)

    gc.collect()
    torch.cuda.empty_cache()
    
    if args.do_test:
        if args.q:
            model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, quantization_config=q_config, trust_remote_code=True, use_cache=False)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, trust_remote_code=True, use_cache=False)

        if not model.config.pad_token_id:
            model.config.pad_token_id = model.config.eos_token_id

        test_dataset = TextDataset(tokenizer, args, args.test_data_file)
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.test_batch_size, num_workers=4, pin_memory=False)

        checkpoint_prefix = f'{args.localtime}/{args.project}/checkpoint-best-bacc' if args.do_train else f'{args.basetime}/{args.project}/checkpoint-best-bacc'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        if args.q:
            accelerator.wait_for_everyone()
            model = prepare_model_for_kbit_training(model)

        if args.lora:
            accelerator.wait_for_everyone()
            model = PeftModel.from_pretrained(model, output_dir)   

        logger.info(f"***** Doing test *****")
        results = test(args, accelerator, model, test_dataset, test_dataloader)
        for key, value in results.items():
            logger.info("  %s = %s", key, round(value,4)) 

        import torch.distributed as dist

        dist.destroy_process_group()

if __name__ == "__main__":
    main()
