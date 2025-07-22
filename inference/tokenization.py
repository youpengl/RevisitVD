import torch
import tiktoken

def bound_check(system_prompt, user_prompt, tokenizer, args):

    len_sys = len(tokenizer.encode(system_prompt, add_special_tokens=False))
    
    len_cot = len(tokenizer.encode("Let's think step by step.", add_special_tokens=False)) if args.CoT else 0
    
    if args.CoT:
        space = len_sys + len_cot + 200 # 200 for output
    else:
        space = len_sys + 0 + 20 # 20 for output
    
    user_input_ids = tokenizer.encode(
        user_prompt,
        add_special_tokens=False,
        truncation=True, 
        return_tensors="pt",
        max_length=args.max_length-space)
    
    user_prompt = tokenizer.decode(user_input_ids[0])

    if len(user_input_ids[0]) == args.max_length - space:
        return user_prompt, True
        
    return user_prompt, False

def bound_check_for_gpt(system_prompt, user_prompt, args):

    encoding = tiktoken.encoding_for_model(args.model_name_or_path)

    len_sys = len(encoding.encode(system_prompt))
    
    len_cot = len(encoding.encode("Let's think step by step.")) if args.CoT else 0
    
    if args.CoT:
        space = len_sys + len_cot + 200 # 200 for output
    else:
        space = len_sys + 0 + 30 # 20 for output
    
    user_input_ids = encoding.encode(user_prompt)

    user_prompt = encoding.decode(user_input_ids[:args.max_length-space])
    
    if len(encoding.encode(user_prompt)) == args.max_length - space:
        return user_prompt, True
        
    return user_prompt, False

def tokenize_gpt(args, dialogs):
    for prompt in dialogs:
        if prompt['role'] == 'system':
            system_prompt = prompt["content"]
        elif prompt['role'] == 'user':
            user_prompt = prompt["content"]

    user_prompt, ifmax = bound_check_for_gpt(system_prompt, user_prompt, args)
 
    if ifmax:
        if args.graph_prompt_type in ['data_flow', 'api_call']:
            user_prompt += f'*/\n\'\'\'\n'
        else:
            user_prompt += f'\n\'\'\'\n'

    if args.CoT:
        user_prompt += f'Let\'s think step by step.\n\n'
        
    for prompt in dialogs:
        if prompt['role'] == 'system':
            prompt["content"] = system_prompt
        elif prompt['role'] == 'user':
            prompt["content"] = user_prompt

    # text += system_prompt + user_prompt   

    return dialogs

def tokenize_wizardcoder(args, prefix, dialogs, tokenizer):
    # https://huggingface.co/WizardLM/WizardCoder-15B-V1.0#inference
    text = ""
    for prompt in dialogs:
        if prompt['role'] == 'system':
            system_prompt = prompt["content"] + "\n\n"
        elif prompt['role'] == 'user':
            user_prompt = "### Instruction:\n" + prompt["content"] + "\n"

    user_prompt, ifmax = bound_check(system_prompt, user_prompt, tokenizer, args)
 
    if ifmax:
        if args.graph_prompt_type in ['data_flow', 'api_call']:
            user_prompt += f'*/\n\'\'\'\n'
        else:
            user_prompt += f'\n\'\'\'\n'

    if args.CoT:
        user_prompt += f'Let\'s think step by step.\n\n'
        
    text += system_prompt + user_prompt   
    text += "### Response:\n"
    if prefix is not None:
        text += prefix
    
    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=args.max_length).to(args.device)
    return input_ids
    
def tokenize_starchat(args, prefix, dialogs, tokenizer):
    # https://huggingface.co/HuggingFaceH4/starchat-beta#intended-uses--limitations
    system_token, user_token = "<|system|>", "<|user|>"
    assistant_token, end_token = "<|assistant|>", "<|end|>"
    text = ""
    for prompt in dialogs:
        if prompt['role'] == 'system':
            system_prompt = system_token + "\n" + prompt['content'] + end_token + "\n"
            
        elif prompt['role'] == 'user':
            user_prompt = user_token + "\n" + prompt["content"] + end_token + "\n"

    user_prompt, ifmax = bound_check(system_prompt, user_prompt, tokenizer, args)
 
    if ifmax:
        if args.graph_prompt_type in ['data_flow', 'api_call']:
            user_prompt += f'*/\n\'\'\'\nLet\'s think step by step.{end_token}\n' if args.CoT else f'\n\'\'\'{end_token}\n'   
        else:
            user_prompt += f'\n\'\'\'\nLet\'s think step by step.{end_token}\n' if args.CoT else f'\n\'\'\'{end_token}\n'   
    else:
        user_prompt = user_token + "\n" + prompt["content"] + (f'\nLet\'s think step by step.' if args.CoT else "") + end_token + "\n"
        
    text += system_prompt + user_prompt
    text += assistant_token
    
    if prefix is not None:
        text += prefix

    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=args.max_length).to(args.device)
    return input_ids

def tokenize_mistral(args, prefix, dialogs, tokenizer):
    
    # Official format:
    # <s>[INST] System Prompt + Instruction [/INST] Model answer</s>[INST] Follow-up instruction [/INST]
    # Reference: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/discussions/73#6555e77abce11b9cde9a79f0
    assert len(dialogs) >= 1, "Invalid dialogs"

    system_prompt = dialogs.pop(0)["content"]
    user_prompt = dialogs[0]["content"]

    user_prompt, ifmax = bound_check(system_prompt, user_prompt, tokenizer, args)

    if ifmax:
        if args.graph_prompt_type in ['data_flow', 'api_call']:
            user_prompt += f'*/\n\'\'\''
        else:
            user_prompt += f'\n\'\'\''
    
    if args.CoT:
        user_prompt += f'\nLet\'s think step by step.'
        
    dialogs = [
        {
            "role": dialogs[0]["role"],
            "content": system_prompt + '\n\n' + user_prompt,
        }
    ]
    input_ids = tokenizer.apply_chat_template(dialogs, 
                                              truncation=True, 
                                              max_length=args.max_length)
    
    if prefix is not None:
        input_ids.extend(tokenizer.encode(prefix, add_special_tokens=False))
    
    input_ids = torch.tensor([input_ids]).to(args.device)
    return input_ids

def tokenize_deepseek(args, prefix, dialogs, tokenizer):
    # https://github.com/deepseek-ai/deepseek-coder
    text = ""
    for prompt in dialogs:
        if prompt['role'] == 'system':
            system_prompt = prompt["content"] + "\n"
        elif prompt['role'] == 'user':
            user_prompt = "### Instruction:\n" + prompt["content"] + "\n"

    user_prompt, ifmax = bound_check(system_prompt, user_prompt, tokenizer, args)
 
    if ifmax:
        if args.graph_prompt_type in ['data_flow', 'api_call']:
            user_prompt += f'*/\n\'\'\'\n'
        else:
            user_prompt += f'\n\'\'\'\n'
    
    if args.CoT:
        user_prompt += f'Let\'s think step by step.\n'
        
    text += system_prompt + user_prompt   
    text += "### Response:\n"
    if prefix is not None:
        text += prefix
    
    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=args.max_length).to(args.device)
    return input_ids

def tokenize_llama(args, prefix, dialogs, tokenizer):
    # https://huggingface.co/blog/llama2#how-to-prompt-llama-2
    assert len(dialogs) >= 1, "Invalid dialogs"
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    
    system_prompt = B_SYS + dialogs[0]["content"] + E_SYS
    
    user_prompt = dialogs[1]["content"]

    user_prompt, ifmax = bound_check(system_prompt, user_prompt, tokenizer, args)

    if ifmax:
        if args.graph_prompt_type in ['data_flow', 'api_call']:
            user_prompt += f'*/\n\'\'\''
        else:
            user_prompt += f'\n\'\'\''

    if args.CoT:
        user_prompt += f'\nLet\'s think step by step.'
        
    dialogs = [
        {
            "role": dialogs[1]["role"],
            "content": system_prompt + user_prompt,
        }
    ]
    dialog_tokens = []

    input_ids = tokenizer.encode(
        f"{B_INST} {(dialogs[-1]['content'])} {E_INST}",
        add_special_tokens=False,
        truncation=True, 
        max_length=args.max_length
    )

    dialog_tokens.extend([tokenizer.bos_token_id] + input_ids)
    
    if prefix is not None:
        dialog_tokens.extend(tokenizer.encode(prefix, add_special_tokens=False))
    
    return torch.tensor([dialog_tokens]).to(args.device)