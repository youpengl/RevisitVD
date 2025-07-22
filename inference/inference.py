from __future__ import absolute_import, division, print_function

import os
import json
import random
import torch
import argparse
import numpy as np
from tqdm import tqdm
from transformers import (AutoTokenizer, AutoModelForCausalLM)
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import gc
from parser import DFG_c
import logging
from tree_sitter import Language, Parser
from utils import *
from tokenization import *
from datetime import datetime
import pytz
import time
datetime_NewYork = datetime.now(pytz.timezone('US/central'))
localtime = datetime_NewYork.strftime("%Y-%m-%d-%H:%M:%S")
logger = logging.getLogger(__name__)

LANGUAGE = Language('parser/my-languages.so', 'c')
parser = Parser()
parser.set_language(LANGUAGE) 
parser = [parser, DFG_c]  

import tiktoken
import openai
from openai import OpenAI

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Replace it with your own API_KEY
os.environ["OPENAI_API_KEY"] = "<OPENAI_API_KEY>"

def tokenize(args, prefix, dialogs, tokenizer, model_name_or_path):
    if 'CodeLlama' in model_name_or_path:
        return tokenize_llama(args, prefix, dialogs, tokenizer)
    elif 'deepseek-coder' in model_name_or_path:
        return tokenize_deepseek(args, prefix, dialogs, tokenizer)
    elif 'Mistral' in model_name_or_path:
        return tokenize_mistral(args, prefix, dialogs, tokenizer)
    elif 'starchat-beta' in model_name_or_path:
        return tokenize_starchat(args, prefix, dialogs, tokenizer)
    elif 'WizardCoder' in model_name_or_path:
        return tokenize_wizardcoder(args, prefix, dialogs, tokenizer)
    elif 'gpt' in model_name_or_path:
        return tokenize_gpt(args, dialogs)
    else:
        raise NotImplementedError(model_name_or_path)
    
def zero_shot(code, cpg_prompt="", graph_prompt_type=None):
    prefix = None
    if graph_prompt_type == 'data_flow':
        graph_prompt_type = 'Data flow'
    elif graph_prompt_type == 'api_call':
        graph_prompt_type = 'API call'
    else:
        graph_prompt_type = ''
        
    system_prompt = "You are a code security expert who excels at detecting vulnerabilities. "

    prompt_template = """Is the following function vulnerable? Please answer Yes or No.\n\n```\n{code}\n{cpg_prompt}```"""

    dialogs = [{"role": "system", "content": system_prompt}]
    dialogs = dialogs + [{"role": "user", "content": prompt_template.format(code=code, cpg_prompt=f"\n/* {graph_prompt_type}: "+cpg_prompt+"*/\n" if cpg_prompt!="" else "")}]

    return prefix, dialogs

def few_shots(code, args, incontext_examples): 
    nvul_answer = 'No, the function is not vulnerable.'
    vul_answer = 'Yes, the function is vulnerable.'
    prefix = None
    system_prompt = "You are a code security expert who excels at detecting vulnerabilities. "
    system_dialog = [{"role": "system", "content": system_prompt}]
    query = "Is the following function vulnerable? Please answer Yes or No."
    if args.random_shots:
        ids = np.random.choice(np.arange(0, int(len(incontext_examples)/2)), args.num_pairs, replace=False)
        selected_examples = []
        for idx in ids:
            jdx = idx * 2
            selected_examples.extend([incontext_examples[jdx], incontext_examples[jdx+1]])
    else:
        selected_examples = load_jsonl_file(args.few_shots_samples_path)[:args.num_pairs]
    
    ic_dialogs = []
    content_map = [nvul_answer, vul_answer]
    for ex in selected_examples:
        func = " ".join(ex["func"].split())
        dialog = [
            {"role": "user", "content": f"""{query}\n```\n{func}\n```"""},
            {"role": "assistant", "content": content_map[ex["target"]]},
        ]
        ic_dialogs.extend(dialog)
    
    
    q_dialogs = [{"role": "user", "content": f"{query}\n```\n{code}\n```"}]
    merged_content = ""
    for dialog in ic_dialogs:
        merged_content += dialog["content"] + "\n\n"
        
    q_dialogs[0]["content"] = merged_content + q_dialogs[0]["content"]
    ic_dialogs = []
    dialogs = system_dialog + ic_dialogs + q_dialogs
    
    return prefix, dialogs

def convert_examples_to_features(code, tokenizer, args, incontext_examples=None):
    if args.graph_prompt_type == 'data_flow':
        dfg, variables = extract_dataflow(code, parser, 'c')
        cpg_prompt = dfg2description(dfg, variables)
        
    elif args.graph_prompt_type == 'api_call':
        cpg_prompt = extract_api_call(code, parser, 'c')
    
    elif args.graph_prompt_type == 'flatten_AST':
        code = extract_flatten_AST(code, parser, 'c', tokenizer)
        cpg_prompt = ""
    
    else:
        cpg_prompt = ""
        
    code = " ".join(code.split())
    if args.prompt_type == 'zero_shot':
        prefix, dialogs = zero_shot(code, cpg_prompt=cpg_prompt, graph_prompt_type=args.graph_prompt_type)
    elif args.prompt_type == 'few_shots':
        prefix, dialogs = few_shots(code, args, incontext_examples)
        
    input_ids = tokenize(args, prefix, dialogs, tokenizer, args.model_name_or_path)

    try:
        if input_ids.shape[1] > args.max_length:
            print('The length of input_ids is above the max input size of tokenizer!')
    except:
        pass
    
    return input_ids


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_openai_chat(messages, args):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    try:
        response = client.chat.completions.create(
            model=args.model_name_or_path,
            messages=messages,
            max_tokens=10,
            temperature=args.temperature,
            seed=args.seed,
            n=1,
            )
        response_content = response.choices[0].message.content

        return response_content

    # when encounter RateLimit or Connection Error, sleep for 5 or specified seconds and try again
    except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError) as error:
        print(f"RateLimit or Connection Error: {error}")
        time.sleep(5)
        return None
    # when encounter bad request errors, print the error message and return None
    except openai.BadRequestError as error:
        print(f"Bad Request Error: {error}")
        return None
    except:
        print("Unknown Errors")
        return None
    
def evaluate(test_dataset, model, tokenizer, args):
    """ Evaluate the model """

    # Eval!
    print("***** Running evaluation *****")
    print(f"  Num examples = {len(test_dataset)}")

    model.eval()
    labels = []
    preds = []
    invalid = []
    pred_info = []
    for batch in tqdm(test_dataset, total=len(test_dataset)):
        input_ids = batch[1].to(model.device)
        with torch.no_grad():
            try:
                if args.temperature != 0.:                 
                    output = model.generate(input_ids, 
                                            max_new_tokens=10 if not args.CoT else 200, 
                                            do_sample=True,
                                            top_p=0.9,
                                            temperature=args.temperature,
                                            pad_token_id=tokenizer.pad_token_id
                                            )
                else:        
                    output = model.generate(input_ids, 
                                            max_new_tokens=10 if not args.CoT else 200, 
                                            do_sample=False,
                                            pad_token_id=tokenizer.pad_token_id
                                            )
            except:
                try:
                    max_new_tokens = args.max_length - input_ids.shape[1]
                    if args.temperature != 0.:                 
                        output = model.generate(input_ids, 
                                                max_new_tokens=max_new_tokens if not args.CoT else 200, 
                                                do_sample=True,
                                                top_p=0.9,
                                                temperature=args.temperature,
                                                pad_token_id=tokenizer.pad_token_id
                                                )
                    else:        
                        output = model.generate(input_ids, 
                                                max_new_tokens=max_new_tokens if not args.CoT else 200, 
                                                do_sample=False,
                                                pad_token_id=tokenizer.pad_token_id
                                                )         
                except:
                    print("exception!!!!")
                    gc.collect()
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
            
            pred = tokenizer.decode(output[0][len(input_ids[0]):])

            # pred = extract_binary_response(pred, args)
            
            pred_info.append({'idx': batch[0], 'pred': pred, 'target': batch[2]})

            pred = pred.lower()
            if 'yes' in pred or 'is vulnerable' in pred or 'true' in pred:
                preds.append(1)
                labels.append(batch[2])
            
            elif 'no' in pred or 'is not vulnerable' in pred or 'false' in pred:
                preds.append(0)
                labels.append(batch[2])
            
            else:
                invalid.append(pred)
               
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

    if not os.path.exists(f"results/{args.localtime}"):
        os.makedirs(f"results/{args.localtime}")
    np.savez(f"results/{args.localtime}/{args.model_name_or_path.split('/')[1]}.npz", pred_info=pred_info, invalid=invalid)
    
    invalid_ratio = len(invalid)/len(test_dataset)
    acc = accuracy_score(labels, preds)
    recall = recall_score(labels, preds)
    precision = precision_score(labels, preds)
    f1 = f1_score(labels, preds)
    TN, FP, FN, TP = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    tnr = TN / (TN + FP)
    fpr = FP / (FP + TN)
    fnr = FN / (TP + FN)
    bacc = (recall + tnr) / 2

    results = {
        "f1": float(f1),
        "recall": float(recall),
        "precision": float(precision),
        "acc": float(acc),
        "tnr": float(tnr),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "bacc": float(bacc),
        "invalid ratio": float(invalid_ratio)
    }
                
    return results


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input test data file to evaluate the perplexity on (a jsonl file).")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--max_length", default=-1, type=int,
                        help="Optional input sequence length after tokenization.")
    parser.add_argument('--seed', type=int, default=123456,
                        help="random seed for initialization.")  
    parser.add_argument('--prompt_type', type=str, default=None,
                    help="[zero_shot, few_shots]")   
    parser.add_argument('--num_pairs', type=int, default=None,
                help="the number of pairs used in few-shot incontext learning")   
    parser.add_argument("--few_shots_samples_path", default=None, type=str,
                    help="for loading few-shots samples")
    parser.add_argument("--random_shots", action="store_true",
                help="randomly loading few-shots samples")
    parser.add_argument("--graph_prompt_type", default=None, type=str,
                help="add descriptions of code property graph to prompt")
    parser.add_argument("--CoT", action="store_true",
                help="add CoT prompt")
    parser.add_argument("--temperature", default=0., type=float,
                help="temperature")
    

    def log_time(sec, what):
        t = datetime.now()
        return t.timetuple()
    logging.Formatter.converter = log_time
    
    if not os.path.exists('logs'):
        os.makedirs('logs')
        
    logging.basicConfig(
        filename = f'logs/{localtime}.log',
        format = "%(asctime)s - %(levelname)s - %(message)s",
        level = logging.INFO,
        datefmt = "%Y-%m-%d %H:%M:%S")
         
    # Print arguments
    args = parser.parse_args()
    args.localtime = localtime
    logger.info("parameters %s", args)
    
    # Build model
    if 'gpt-3.5-turbo-0125' == args.model_name_or_path:
        args.max_length = 16385
        tokenizer = None
    elif 'gpt-4-0125-preview' == args.model_name_or_path:
        args.max_length = 128000
        tokenizer = None
    else:
        import dotenv
        dotenv.load_dotenv()
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False if args.model_name_or_path != "WizardLMTeam/WizardCoder-33B-V1.1" else True, trust_remote_code=True)
        
        if tokenizer.pad_token == None: 
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
        args.device = model.device
        
        # codellama, deepseek coder 16384, mistral 32768, starchat 8192, wizardcoder 8192
        args.max_length = model.config.max_position_embeddings
    
    
    if args.prompt_type == 'few_shots':
        incontext_examples = load_jsonl_file(args.few_shots_samples_path)
    else:
        incontext_examples = None
    test_dataset = []
    with open(args.test_data_file) as f:
        count = 0
        for line in f:
            line = line.strip()
            js = json.loads(line)
            test_dataset.append((js["idx"], convert_examples_to_features(js["func"], tokenizer, args, incontext_examples), js["target"]))
    
    if 'gpt' in args.model_name_or_path:
        file_path = f"results/{args.model_name_or_path}-{args.prompt_type}-{args.graph_prompt_type}"
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        file_path = os.path.join(file_path, 'prediction.jsonl')
        if not os.path.exists(file_path):
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write("")
        json_file = load_jsonl_file(file_path)

        json_ids = [jf['idx'] for jf in json_file]
        error = 0
        results = []
        print(f"Requesting {args.model_name_or_path} to respond to {len(test_dataset)} prompts ...")
        if not os.path.exists(f"results/{args.localtime}"):
            os.makedirs(f"results/{args.localtime}")
            
        for batch in tqdm(test_dataset, total=len(test_dataset)):
            if batch[0] in json_ids: 
                continue

            if error == 10:
                break
            result = {}
            p = batch[1]
            result['idx'] = batch[0]
            result['target'] = batch[2]
            response = get_openai_chat(p, args)

            if response is None:
                error += 1
                result['response'] = "ERROR"
            else:
                result["response"] = response
                error = 0
                results.append(result)

                with open(file_path, "a") as file:
                    file.write(json.dumps(result) + "\n")

        save_to_jsonl(results, f"results/{args.localtime}/{args.model_name_or_path}.jsonl")

            
    else:
        result = evaluate(test_dataset, model, tokenizer, args)
        logger.info("***** Test results *****")
        for key in result.keys():
            logger.info("{} = {}".format(key, str(round(result[key] * 100 if "map" in key else result[key], 4))))       


if __name__ == "__main__":
    main()
