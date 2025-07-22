# coding=utf-8
# Reference: https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection
# Reference: https://github.com/salesforce/CodeT5/blob/main/CodeT5/run_defect.py

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import re
os.environ["HF_ENDPOINT"] = "https://huggingface.co"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import random
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import json
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange
import multiprocessing
from model import Model, DefectModel, Graph_Model
import warnings
warnings.filterwarnings('ignore')
cpu_cont = multiprocessing.cpu_count()
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup, 
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          T5Config, T5ForConditionalGeneration,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)

from datetime import datetime

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
}

from parser import DFG_c
from parser import (remove_comments_and_docstrings,
            tree_to_token_index,
            index_to_code_token)

from tree_sitter import Language, Parser
LANGUAGE = Language('parser/my-languages.so', 'c')
parser = Parser()
parser.set_language(LANGUAGE) 
parser = [parser, DFG_c]  
                                  
def extract_dataflow(code, parser, lang):
    #remove comments
    try:
        code = remove_comments_and_docstrings(code, lang)
    except:
        pass    
    #obtain dataflow
    if lang == "php":
        code = "<?php" + code + "?>"    
    try:
        tree = parser[0].parse(bytes(code,'utf8'))    
        # tree = cpp_parser(code)    
        root_node = tree.root_node    
                
        tokens_index = tree_to_token_index(root_node)     
        code = code.split('\n')
        code_tokens = [index_to_code_token(x,code) for x in tokens_index]  
        index_to_code={}
        for idx,(index,code) in enumerate(zip(tokens_index, code_tokens)):
            index_to_code[index] = (idx,code)  
        try:
            DFG,_ = parser[1](root_node,index_to_code,{}) 
        except:
            DFG = []
        DFG = sorted(DFG, key = lambda x:x[1])
        indexs = set()
        for d in DFG:
            if len(d[-1]) != 0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG = []
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg = new_DFG
        
    except:
        dfg = []

    return code_tokens, dfg

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_ids,
                 idx,
                 label

    ):
        self.input_ids = input_ids
        self.idx = str(idx)
        self.label = label

class Graph_InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_ids,
                 position_idx,
                 dfg_to_code, 
                 dfg_to_dfg,
                 idx,
                 label

    ):
        self.input_ids = input_ids
        self.position_idx = position_idx
        self.dfg_to_code = dfg_to_code
        self.dfg_to_dfg = dfg_to_dfg
        self.idx = str(idx)
        self.label = label
        
def code_cleaner_pdbert(func):
    code = re.sub(r' +|\t+', ' ', func)
    return code

def code_cleaner(func):
    code = ' '.join(func.split())
    return code

def convert_examples_to_features(js,tokenizer,args):
    if "PDBERT" in args.project:
        code = code_cleaner_pdbert(js['func'])
    elif "GraphCodeBERT" in args.project:
        code = js['func']
    else:
        code = code_cleaner(js['func'])

    if "CodeT5" in args.project:
        source_ids = tokenizer.encode(code, max_length=args.block_size, padding='max_length', truncation=True)
        if sum([1 for si in source_ids if si == 2]) > 1:
            # print(code)
            last_index = len(source_ids) - 1 - source_ids[::-1].index(2)
            # print(f"old input: {source_ids}")
            source_ids = [x for i, x in enumerate(source_ids) if x != 2 or i == last_index]
            paddings = [0] * (512 - len(source_ids))
            source_ids.extend(paddings)
            # print(f"new input: {source_ids}")

    elif 'UniXCoder' in args.project:
        code_tokens = tokenizer.tokenize(code)
        code_tokens = code_tokens[:args.block_size-4]
        code_tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + code_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(code_tokens)
        padding_length = args.block_size - len(source_ids)
        source_ids += [tokenizer.pad_token_id]*padding_length
        
    elif 'GraphCodeBERT' in args.project:
        code_tokens, dfg = extract_dataflow(code, parser, 'c')
        code_tokens = [tokenizer.tokenize('@ '+x)[1:] if idx!=0 else tokenizer.tokenize(x) for idx, x in enumerate(code_tokens)]
        ori2cur_pos = {}
        ori2cur_pos[-1] = (0, 0)
        for i in range(len(code_tokens)):
            ori2cur_pos[i] = (ori2cur_pos[i-1][1], ori2cur_pos[i-1][1] + len(code_tokens[i]))    
        code_tokens = [y for x in code_tokens for y in x]  
        
        #truncating
        code_tokens = code_tokens[:args.block_size + args.data_flow_length - 3 - min(len(dfg), args.data_flow_length)][:args.block_size - 3]
        source_tokens =[tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        position_idx = [i + tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]
        dfg = dfg[:args.block_size + args.data_flow_length - len(source_tokens)]
        source_tokens += [x[0] for x in dfg]
        position_idx += [0 for x in dfg]
        source_ids += [tokenizer.unk_token_id for x in dfg]
        padding_length = args.block_size + args.data_flow_length - len(source_ids)
        position_idx += [tokenizer.pad_token_id] * padding_length
        source_ids += [tokenizer.pad_token_id] * padding_length      
        
        #reindex
        reverse_index = {}
        for idx, x in enumerate(dfg):
            reverse_index[x[1]] = idx
        for idx, x in enumerate(dfg):
            dfg[idx] = x[:-1] + ([reverse_index[i] for i in x[-1] if i in reverse_index],)    
        dfg_to_dfg = [x[-1] for x in dfg]
        dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
        length = len([tokenizer.cls_token])
        dfg_to_code = [(x[0] + length,x[1] + length) for x in dfg_to_code]  
                
        return Graph_InputFeatures(source_ids, position_idx, dfg_to_code, dfg_to_dfg, js['idx'], js['target'])
        
    else:
        code_tokens = tokenizer.tokenize(code)
        code_tokens = code_tokens[:args.block_size-2]
        code_tokens =[tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]       
        source_ids =  tokenizer.convert_tokens_to_ids(code_tokens)
        padding_length = args.block_size - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length
        
    return InputFeatures(source_ids, js['idx'], js['target'])

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
        return (torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label))
            
class Graph_TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        self.args = args
        
        with open(file_path) as f:
            count = 0
            for line in tqdm(f):
                js=json.loads(line.strip())
                self.examples.append(convert_examples_to_features(js, tokenizer, args))
                    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        #calculate graph-guided masked function
        attn_mask = np.zeros((self.args.block_size + self.args.data_flow_length,
                        self.args.block_size + self.args.data_flow_length),dtype=bool)
        #calculate begin index of node and max length of input
        node_index = sum([i > 1 for i in self.examples[item].position_idx])
        max_length = sum([i != 1 for i in self.examples[item].position_idx])
        #sequence can attend to sequence
        attn_mask[:node_index,:node_index] = True
        #special tokens attend to all tokens
        for idx, i in enumerate(self.examples[item].input_ids):
            if i in [0,2]:
                attn_mask[idx,:max_length] = True
        #nodes attend to code tokens that are identified from
        for idx, (a,b) in enumerate(self.examples[item].dfg_to_code):
            if a < node_index and b < node_index:
                attn_mask[idx+node_index,a:b] = True
                attn_mask[a:b,idx+node_index] = True
        #nodes attend to adjacent nodes 
        for idx, nodes in enumerate(self.examples[item].dfg_to_dfg):
            for a in nodes:
                if a + node_index < len(self.examples[item].position_idx):
                    attn_mask[idx + node_index,a + node_index] = True                    
                    
        return  (torch.tensor(self.examples[item].input_ids),
                torch.tensor(self.examples[item].position_idx),
                torch.tensor(attn_mask),                
                torch.tensor(self.examples[item].label))
          
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def train(args, train_dataset, eval_dataset, model, tokenizer):
    """ Train the model """ 
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    
    train_dataloader = DataLoader(train_dataset, sampler = train_sampler, 
                                  batch_size = args.train_batch_size, num_workers = 4, pin_memory = True)
    
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    
    eval_sampler = SequentialSampler(eval_dataset)
    
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4, pin_memory=True)
    
    args.max_steps = args.epoch * len(train_dataloader)
    args.num_train_epochs = args.epoch
    model.to(args.device)
    
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr = args.learning_rate, eps = args.adam_epsilon)
        
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = args.max_steps * 0.1 if args.warmup_steps == -1 else args.warmup_steps,
                                                num_training_steps = args.max_steps)


    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)
    
    global_step = args.start_step
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_bacc = 0.0

    tensorboard_logdir = f'{args.run_dir}/{args.localtime}/{args.project}/'
    writer = SummaryWriter(tensorboard_logdir)

    model.zero_grad()
 
    step = 0
    for idx in range(args.start_epoch, int(args.num_train_epochs)): 
        bar = tqdm(train_dataloader, total = len(train_dataloader))
        tr_num = 0
        train_loss = 0
        logits_lst = []
        labels_lst = []
        for local_step, batch in enumerate(bar):
            if "GraphCodeBERT" in args.project:
                (inputs_ids, position_idx, attn_mask, labels) = [x.to(args.device) for x in batch]
            else:   
                (inputs, labels) = [x.to(args.device) for x in batch]
                
            model.train()

            if "GraphCodeBERT" in args.project:
                loss, logits = model(inputs_ids, position_idx, attn_mask, labels)
            else:
                loss, logits = model(inputs, labels)
            
            logits_lst.append(logits.detach().cpu().numpy())
            labels_lst.append(labels.detach().cpu().numpy())

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
                
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
                
            avg_loss = round(train_loss / tr_num, 5)
            bar.set_description("epoch {} loss {}".format(idx,avg_loss)) # sum_loss / num_steps
            
            if (local_step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  
                global_step += 1
                avg_loss = round(np.exp((tr_loss - logging_loss) / (global_step- tr_nb)), 4)
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging_loss = tr_loss
                    tr_nb = global_step
            
            # log after every logging_steps (e.g., 100)
            if (step + 1) % args.logging_steps == 0:
                avg_loss = round(train_loss / tr_num, 5)
                step_logits_lst = np.concatenate(logits_lst, 0)
                step_labels_lst = np.concatenate(labels_lst, 0)
                if args.model_type in set(['codet5', 't5']):
                    step_preds_lst = step_logits_lst[:,1] > 0.5
                else:
                    step_preds_lst = step_logits_lst[:,0] > 0.5
                
                logger.info(f"***** Evaluate on valid set: Epoch {idx}: step {step} *****")

                train_acc, train_prec, train_recall, train_f1, train_tnr, train_fpr, train_fnr, train_bacc = calculate_metrics(step_labels_lst, step_preds_lst)
                if args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                    results = evaluate(args, model, tokenizer, eval_dataset, eval_dataloader, eval_when_training=True)
                    for key, value in results.items():
                        logger.info("  %s = %s", key, round(value, 4))        
                                    
                valid_loss, valid_acc, valid_prec, valid_recall, valid_f1, valid_tnr, valid_fpr, valid_fnr, valid_bacc = results.values()
                writer.add_scalars('Loss', {'train': avg_loss, 'valid': valid_loss}, step)
                writer.add_scalars('Acc', {'train': train_acc, 'valid': valid_acc}, step)
                writer.add_scalars('F1', {'train': train_f1, 'valid': valid_f1}, step)
                writer.add_scalars('Prec', {'train': train_prec, 'valid': valid_prec}, step)
                writer.add_scalars('Recall', {'train': train_recall, 'valid': valid_recall}, step)
                writer.add_scalars('TNR', {'train': train_tnr, 'valid': valid_tnr}, step)
                writer.add_scalars('FPR', {'train': train_fpr, 'valid': valid_fpr}, step)
                writer.add_scalars('FNR', {'train': train_fnr, 'valid': valid_fnr}, step)
                writer.add_scalars('Balanced_Accuracy', {'train': train_bacc, 'valid': valid_bacc}, step)
   
                # Save model checkpoint   
                if results['eval_bacc'] > best_bacc:
                    best_bacc = results['eval_bacc']
                    logger.info("  "+"*"*20)  
                    logger.info("  Best bacc:%s",round(best_bacc, 4))
                    checkpoint_prefix = f'{args.localtime}/{args.project}/checkpoint-best-bacc'
                    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)                        
                    model_to_save = model.module if hasattr(model,'module') else model
                    output_dir = os.path.join(output_dir, f'model.bin') 
                    torch.save(model_to_save.state_dict(), output_dir)
                    logger.info(f"  Saving best balanced accruacy model checkpoint at epoch {idx} step {step} to {output_dir}")
                    logger.info("  "+"*"*20)

                          
            # increment step within the same epoch
            step += 1
            
        # log after every epoch
        avg_loss = round(train_loss/tr_num, 5)
        logits_lst = np.concatenate(logits_lst, 0)
        labels_lst = np.concatenate(labels_lst, 0)

        if args.model_type in set(['codet5', 't5']):
            preds_lst = logits_lst[:,1] > 0.5
        else:
            preds_lst = logits_lst[:,0] > 0.5

        train_acc, train_prec, train_recall, train_f1, train_tnr, train_fpr, train_fnr, train_bacc = calculate_metrics(labels_lst, preds_lst)
        
        if args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
            logger.info(f"***** Epoch {idx} finished : step {step} *****")
            results = evaluate(args, model, tokenizer, eval_dataset, eval_dataloader, eval_when_training=True)
            for key, value in results.items():
                logger.info("  %s = %s", key, round(value,4))                    
        valid_loss, valid_acc, valid_prec, valid_recall, valid_f1, valid_tnr, valid_fpr, valid_fnr, valid_bacc = results.values()
        writer.add_scalars('Loss', {'train': avg_loss, 'valid': valid_loss}, step)
        writer.add_scalars('Acc', {'train': train_acc, 'valid': valid_acc}, step)
        writer.add_scalars('F1', {'train': train_f1, 'valid': valid_f1}, step)
        writer.add_scalars('Prec', {'train': train_prec, 'valid': valid_prec}, step)
        writer.add_scalars('Recall', {'train': train_recall, 'valid': valid_recall}, step)
        writer.add_scalars('TNR', {'train': train_tnr, 'valid': valid_tnr}, step)
        writer.add_scalars('FPR', {'train': train_fpr, 'valid': valid_fpr}, step)
        writer.add_scalars('FNR', {'train': train_fnr, 'valid': valid_fnr}, step)
        writer.add_scalars('Balanced_Accuracy', {'train': train_bacc, 'valid': valid_bacc}, step)
        
        # Save model checkpoint    
        if results['eval_bacc'] > best_bacc:
            best_bacc = results['eval_bacc']
            logger.info("  "+"*"*20)
            logger.info("  Best bacc:%s",round(best_bacc,4))          
            
            checkpoint_prefix = f'{args.localtime}/{args.project}/checkpoint-best-bacc'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)                        
            model_to_save = model.module if hasattr(model,'module') else model
            output_dir = os.path.join(output_dir, f'model.bin') 
            torch.save(model_to_save.state_dict(), output_dir)
            logger.info(f"  Saving best balance accuracy model checkpoint at epoch {idx} to {output_dir}")
            logger.info("  "+"*"*20) 

    writer.close()                

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

def evaluate(args, model, tokenizer, eval_dataset, eval_dataloader, eval_when_training=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    
    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)
        
    # Eval!
    # logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = [] 
    labels = []
    for batch in tqdm(eval_dataloader):
        if "GraphCodeBERT" in args.project:
            (inputs_ids, position_idx, attn_mask, label) = [x.to(args.device) for x in batch]
        else:   
            (inputs, label) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            if "GraphCodeBERT" in args.project:
                lm_loss, logit = model(inputs_ids, position_idx, attn_mask, label)
            else:
                lm_loss, logit = model(inputs, label)
                
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
            
        nb_eval_steps += 1
        
    logits = np.concatenate(logits,0)
    labels = np.concatenate(labels,0)

    if args.model_type == 'codet5':
        preds = logits[:,1] > 0.5
    else:
        preds = logits[:,0] > 0.5

    eval_acc, eval_prec, eval_recall, eval_f1, eval_tnr, eval_fpr, eval_fnr, eval_bacc = calculate_metrics(labels, preds)
    eval_loss = eval_loss / nb_eval_steps
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

def test(args, model, tokenizer, test_dataset, test_dataloader, name):

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    logits=[]   
    labels=[]
    for batch in tqdm(test_dataloader, total=len(test_dataloader)):
        if "GraphCodeBERT" in args.project:
            (inputs_ids, position_idx, attn_mask, label) = [x.to(args.device) for x in batch]
        else:   
            (inputs, label) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            if "GraphCodeBERT" in args.project:
                logit = model(inputs_ids, position_idx, attn_mask)
            else:
                logit = model(inputs)
                
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())

    logits = np.concatenate(logits,0)
    labels = np.concatenate(labels,0)
    
    if args.model_type in set(['codet5', 't5']):
        preds = logits[:,1] > 0.5
        vuln_scores = logits[:,1].tolist()
    else:
        preds = logits[:,0] > 0.5
        vuln_scores = logits[:,0].tolist()
        
    # for the convenience of saving different testing results
    test_project = args.test_project if args.test_project else args.project
    if not os.path.exists(os.path.join(args.output_dir, args.localtime, test_project, name)):
        os.makedirs(os.path.join(args.output_dir, args.localtime, test_project, name))


    with open(os.path.join(args.output_dir, args.localtime, test_project, name, "predictions.txt"),'w') as f:
        for example, pred in zip(test_dataset.examples, preds):
            f.write(f"idx: {example.idx}, pred: {1 if pred else 0}, target: {example.label}\n")

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

    save_path = os.path.join(args.output_dir, args.localtime, test_project, name, "result.npz")

    np.savez(save_path, test_result=result)
    
    return result
                
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument('--project', type=str, required=True, help="using dataset from this project.")
    parser.add_argument('--test_project', type=str, required=False, help="test setup name.")
    parser.add_argument('--model_dir', type=str, required=True, help="directory to store the model weights.")
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--run_dir', type=str, default="runs", help="parent directory to store run stats.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input valid data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input test data file to evaluate the perplexity on (a text file).")    
    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--data_flow_length", default=-1, type=int,
                        help="for GraphCodeBERT")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")   
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
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
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epoch', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gpu', type=int, default=-1, help="specify gpu id, -1 denotes using all" )
    parser.add_argument('--localtime', type=str, default='2026-01-03-00:00:00')
    parser.add_argument('--basetime', type=str, default='2026-01-02-00:00:00')
    
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.gpu == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
        args.n_gpu = 1
        
    args.device = device
    args.per_gpu_train_batch_size = args.train_batch_size // args.n_gpu
    args.per_gpu_eval_batch_size = args.eval_batch_size // args.n_gpu

    def log_time(sec, what):
        t = datetime.now()
        return t.timetuple()
    logging.Formatter.converter = log_time
    
    if not os.path.exists('logs'):
        os.makedirs('logs')
    logging.basicConfig(
        filename = f'logs/{args.localtime}.log',
        format = "%(asctime)s - %(levelname)s - %(message)s",
        level = logging.INFO,
        datefmt = "%Y-%m-%d %H:%M:%S",
        filemode = 'w')

    # Set seed
    set_seed(args.seed)

    args.start_epoch = 0
    args.start_step = 0

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)

    if args.model_type == "codet5":
        config.num_labels = 2
    else:
        config.num_labels = 1
        
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)

    model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)    

    if args.model_type == 'codet5':
        model = DefectModel(model, config, tokenizer, args)
    elif 'GraphCodeBERT' in args.project:
        model = Graph_Model(model, config, tokenizer, args)
    else:
        model = Model(model, config, tokenizer, args)
        
    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if 'GraphCodeBERT' in args.project:
            train_dataset = Graph_TextDataset(tokenizer, args, args.train_data_file)
        else:
            train_dataset = TextDataset(tokenizer, args, args.train_data_file)

        if 'GraphCodeBERT' in args.project:
            eval_dataset = Graph_TextDataset(tokenizer, args, args.eval_data_file)
        else:
            eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)
        
        train(args, train_dataset, eval_dataset, model, tokenizer)

    if args.do_test:
        if 'GraphCodeBERT' in args.project:
            test_dataset = Graph_TextDataset(tokenizer, args, args.test_data_file)
        else:  
            test_dataset = TextDataset(tokenizer, args, args.test_data_file)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size)
            
        checkpoint_prefix = f'{args.localtime}/{args.project}/checkpoint-best-bacc/model.bin' if args.do_train else f'{args.basetime}/{args.project}/checkpoint-best-bacc/model.bin'
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        if not os.path.exists(output_dir):
            print('no best bacc model for test')
            exit()
        model.load_state_dict(torch.load(output_dir))                  
        model.to(args.device)
        result = test(args, model, tokenizer, test_dataset, test_dataloader, 'bacc')
        logger.info("***** Test results of best bacc model *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key], 4)))
        
if __name__ == "__main__":
    main()
