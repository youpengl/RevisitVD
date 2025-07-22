import json
import re
from collections import defaultdict
from parser import (remove_comments_and_docstrings,
            tree_to_token_index,
            index_to_code_token)

def load_jsonl_file(filename):
    with open(filename, 'r') as f:
        data = []
        for line in f:
            # Convert each line into a JSON object
            data_object = json.loads(line)
            data.append(data_object)
    return data

def save_to_jsonl(data, filename):
  with open(filename, 'w') as f:
    for item in data:
      # Convert dictionary to JSON string
      json_string = json.dumps(item)
      # Write the JSON string with a newline character
      f.write(json_string + '\n')

# below for extracting dataflow
def get_variables(node):
    variables = []
    for child in node.children:
        if child.type == 'identifier':
            variables.append(child.text)
        variables.extend(get_variables(child))

    return list(set(variables))

def get_funcs(node):
    funcs = []
    for child in node.children:
        if child.type == 'call_expression':
            funcs.append(child.text)
        funcs.extend(get_funcs(child))
    
    return list(set(funcs))

def ree(word, func):

    pattern = fr"\b{word}"

    match_found = bool(re.search(pattern, func))
    
    return match_found

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
        variables = get_variables(root_node)
        variables = [v.decode() for v in variables]
        funcs = get_funcs(root_node)
        funcs = [func.decode().split('(')[0] for func in funcs]
        new_variables = []
        for v in variables:
            flag = False
            for func in funcs:
                if ree(v, func) and func.endswith(v) and '->'+v not in func and '.'+v not in func:
                    flag = True
                    break
                if v+'<' in func:
                    flag = True
                    break
            if not flag:
                new_variables.append(v)
                
        tokens_index = tree_to_token_index(root_node)     
        code = code.split('\n')
        code_tokens = [index_to_code_token(x,code) for x in tokens_index]  
        index_to_code={}
        for idx,(index,code) in enumerate(zip(tokens_index, code_tokens)):
            index_to_code[index] = (idx,code)  
        try:
            DFG,_ = parser[1](root_node, index_to_code, {}) 
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

    reverse_index = {}
    for idx, x in enumerate(dfg):
        reverse_index[x[1]] = idx
    for idx, x in enumerate(dfg):
        dfg[idx] = x[:-1] + ([reverse_index[i] for i in x[-1] if i in reverse_index],)   
        
    # dfg = [d for d in dfg if d[0] in new_variables]
    
    return dfg, new_variables

# below for dfg2description
def check(a, b):
    if a == b:
        return True
    else:
        return False

def reorder(sample_data):
    new_sample_data  = []
    for d in sample_data:
        if len(d[3]) > 1 and len(d[4]) > 1:
            d = list(d)
            new_d4 = []
            for idx, dd in enumerate(d[3]):
                jdx = 0
                while not check(dd, sample_data[d[4][jdx]][0]):
                    jdx += 1
                new_d4.append(d[4][jdx])
            d[4] = new_d4
            d = tuple(d)
        new_sample_data.append(d)
    return new_sample_data

def analyze_relationships(data, variables):
    # Create dictionaries to track occurrences of each variable
    var_counts = defaultdict(int)
    var_indices = defaultdict(list)
    
    # First pass: count occurrences and store indices
    for i, (var, line_num, relationship, deps, indices) in enumerate(data):
        var_counts[var] += 1
        var_indices[var].append(i)
    
    # Helper function to get ordinal
    def get_ordinal(n):
        if n == 1:
            return "1st"
        elif n == 2:
            return "2nd"
        elif n == 3:
            return "3rd"
        else:
            return f"{n}th"
            
    # Helper function to find occurrence number
    def get_occurrence_number(var, current_index):
        return var_indices[var].index(current_index) + 1
        
    # Helper function to get dependency descriptions
    def get_dependency_description(deps, indices):
        if not deps or not indices:
            return ""
            
        dep_strings = []
        
        # for comesFrom like ['a'], [1, 3, 4, 6, 8]
        if len(deps) != len(indices):
            deps = deps * len(indices) 
            
        for dep, idx in zip(deps, indices):
            occurrence = get_occurrence_number(dep, idx)
            dep_strings.append(f"the {get_ordinal(occurrence)} {dep}")
            
        if len(dep_strings) == 1:
            return dep_strings[0]
        
        elif len(dep_strings) == 2:
            return f"{dep_strings[0]} and {dep_strings[1]}"
        else:
            return ", ".join(dep_strings[:-1]) + f", and {dep_strings[-1]}"
    
    # Generate sentences
    dfg_prompt = ""
    dfg_list = []
    for i, (var, line_num, relationship, deps, indices) in enumerate(data):
        if deps and indices and var in variables:  # Has dependencies
            var_occurrence = get_occurrence_number(var, i)
            
            if relationship == 'computedFrom':
                dfg_prompt += f"The {get_ordinal(var_occurrence)} {var} is computed from {get_dependency_description(deps, indices)}. "
                dfg_list.append(f"The {get_ordinal(var_occurrence)} {var} is computed from {get_dependency_description(deps, indices)}.")
                
            elif relationship == 'comesFrom':
                dfg_prompt += f"The {get_ordinal(var_occurrence)} {var} comes from {get_dependency_description(deps, indices)}. "
                dfg_list.append(f"The {get_ordinal(var_occurrence)} {var} comes from {get_dependency_description(deps, indices)}.")
                
    return dfg_prompt

def dfg2description(dfg, variables):
    dfg = reorder(dfg)
    cpg_prompt = analyze_relationships(dfg, variables)
    return cpg_prompt

# below for extracting api call
def extract_api_call(code, parser, lang):
    
    def check_tree(node):
        if node.type == 'call_expression':
            calls.append(node.text.decode())
        if node.children:
            for child in node.children:
                check_tree(child)
                
    #remove comments
    try:
        code = remove_comments_and_docstrings(code, lang)
    except:
        pass    
    
    #obtain api_call
    if lang == "php":
        code = "<?php" + code + "?>"  
          
    try:
        tree = parser[0].parse(bytes(code,'utf8')).root_node  
        calls = []            
        check_tree(tree)
        calls = ["".join(call.split('(')[0].split()) for call in calls]
        cfg_prompt = ""
        for idx, call in enumerate(calls):
            if idx == 0:
                cfg_prompt += f"The program first calls {call}, "
            elif idx == len(calls) - 1:
                cfg_prompt += f"and finally calls {call}."
            else:
                cfg_prompt += f"then calls {call}, "
                
    except:
        cfg_prompt = ""
        
    return cfg_prompt

# below for extracting flatten AST
def travel(root_node, index_to_code, tokenizer):
    """Given a AST node, return AST travel sequence using Algo in the paper: https://arxiv.org/pdf/2203.03850.pdf"""
    #if (len(root_node.children) == 0 or root_node.type == 'string' or root_node.type == 'comment' or 'comment' in root_node.type):
    if (len(root_node.children)==0 or root_node.type=='string' or root_node.type=='string_literal') and root_node.type!='comment':
        index = (root_node.start_point,root_node.end_point)
        code = index_to_code[index][1]
        # return tokenizer.tokenize(code)
        return [code]
    else:
        code_tokens = []
        for child in root_node.children:
            code_tokens += travel(child,index_to_code,tokenizer)
        # remove nodes that have only one children for reducing length
        if len(root_node.children) != 1:
            return ["AST#" + root_node.type.replace("#","") + "#Left"] + code_tokens + ["AST#" + root_node.type.replace("#","") + "#Right"] 
        else:
            return code_tokens
        
def extract_flatten_AST(code, parser, lang, tokenizer):
    """Given a code, return its AST flatten sequence"""
    if lang == "php":
        code = "<?php "+code+"?>" 
    # remove comment
    try:
        code = remove_comments_and_docstrings(code, lang)
    except:
        pass
    # parse source code
    tree = parser[0].parse(bytes(code,'utf8'))  
    
    # obtain AST sequence
    root_node = tree.root_node  
    tokens_index = tree_to_token_index(root_node)     
    code = code.split('\n')
    code_tokens = [index_to_code_token(x, code) for x in tokens_index]  
    index_to_code = {}
    for idx,(index,code) in enumerate(zip(tokens_index, code_tokens)):
        index_to_code[index] = (idx,code)  

    code_tokens = travel(root_node, index_to_code, tokenizer)
    
    cfg_prompt = " ".join(code_tokens)
    
    return cfg_prompt
