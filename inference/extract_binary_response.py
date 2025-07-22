import re
from rich import print as rprint
from rich.padding import Padding
from rich.markup import escape

debug = __name__ == "__main__"

# synonyms for "vulnerability" or "vulnerable"
VULN=r"\b("\
r"bug|buggy|bugs|potentially buggy|"\
r"cwe|cwes|"\
r"issue|issues|"\
r"flaw|flaws|"\
r"vulnerabilities|vulnerability|vulnerable|"\
r"buffer-overflow|buffer overflow|memory-leak|memory leak|race-condition|race condition|use-after-free|use after free|segmentation fault|undefined behavior|unexpected behavior|integer-overflow|integer overflow|"\
r"crash"\
r")\b"
VULN_ADJECTIVE=r"\b(buggy|potentially buggy|vulnerable)\b"
NVULN=r"\b(correct|fine|bug-free|well-|correctly|valid|functioning as intended|safe|secure|straightforward)\b"
SAME_SENTENCE = r"[^.;,\n]"
SAME_SENTENCE_PLUS = SAME_SENTENCE+r"+" # zero or more characters in the same sentence
SAME_SENTENCE_STAR = SAME_SENTENCE+r"*" # zero or more characters in the same sentence

NVUL_PATTERNS = [
    SAME_SENTENCE_STAR+r"\bno(, |\.|\n|$)",
    r"\b(appears to be|is) "+NVULN, # NOTE: "is valid" may not always work
    r"\b(does not|do not|don't|doesn't|has no|is not(?! possible to determine)|isn't(?! possible to determine)|there are no|contains none of|contains no|cannot detect)\b"+SAME_SENTENCE_PLUS+VULN,
    r"answer: no",
    ]
VUL_PATTERNS = [
    SAME_SENTENCE_STAR+r"\byes(, |\.|\n|$)",
    SAME_SENTENCE_STAR+r"\b(contains(?! (no|none|the function))|\b(could|can)\b(?! (not))[^.;\n]+(cause|lead to|result in)|will result in|point out|has|does have|which is a(?! (good practice))|(?<!(if) )there is(?! (no|none))|(?<!(if) )there are(?! (no|none)))\b"+SAME_SENTENCE_PLUS+VULN,
    SAME_SENTENCE_STAR+r"\b(appears to be|(function|code|snippet|`) is(?! (not|also not|no|none)|"+NVULN+r"))\b[^.;\n,]+"+VULN_ADJECTIVE,
    r"\bcan be (exploited|fixed)\b",
    r"will not[^.;\n,]+(free)[^.;\n,]+memory",
    r"memory[^.;\n,]+will not[^.;\n,]+(cleaned up)",
    r"(function|code|snippet|`) will fail",
    r"answer: yes",
    ]

def extract_response_from_chunk(text):
    for i, pattern in enumerate(VUL_PATTERNS):
        m = re.finditer(pattern, text)
        for match in m:
            matched = True
            if i == 1:
                does_not_contain = re.compile(r"""
                    \b(
                        ( # "does not contain" and similar
                            (is\ not|does\ not|does\ not\ appear\ to|unlikely\ to|not\ likely\ to)\s+
                            (perform|performing|seem\ to\ be\ performing|involve|contain|affected\ by|check|have)
                        )
                        |no # or "no"
                        |free\ of
                        |or\ perform\ any
                        |not\ prone\ to
                    )\b
                    [^.;\n]+ # allow adjectives like "arithmetic operations"
                    (operation|operations|"""+re.sub(r"(\s)", r"\\\1", VULN)+r")", flags=re.VERBOSE).search(match.group(0))
                not_possible = re.compile(r"""(
                    (cannot|difficult\ to|impossible\ to|not\ possible\ to)
                    [^.;\n]+ # allow adverbs like "confidently"
                    (determine|say|rule\ out)
                    |need\ more\ information)""", flags=re.VERBOSE).search(match.group(0))
                matched = matched \
                    and not does_not_contain \
                    and not not_possible
            if i == 2:
                is_not = re.compile(r"""
                    \b(
                        ( # "does not contain" and similar
                            (is\ not|unlikely\ to|not\ likely\ to)\s+
                            (perform|involve|contain|affected\ by|check)
                        )
                    )\b
                    [^.;\n]+ # allow adjectives like "arithmetic operations"
                    (operation|operations|"""+VULN+r")", flags=re.VERBOSE).search(match.group(0))
                well_written = re.compile(r"appears to be[^.;\n,]+"+NVULN).search(match.group(0))
                not_possible = re.compile(r"""(
                    (cannot|difficult\ to|impossible\ to|not\ possible\ to)
                    [^.;\n]+ # allow adverbs like "confidently"
                    (determine|say|rule\ out)
                    |need\ more\ information)""", flags=re.VERBOSE).search(match.group(0))
                matched = matched \
                    and not well_written \
                    and not is_not \
                    and not not_possible
            if matched:
                if debug: print("VUL:", i, match.group(0))
                return 1
    for i, pattern in enumerate(NVUL_PATTERNS):
        m = re.finditer(pattern, text)
        for match in m:
            if i == 0:
                if "please answer yes or no" in match.group(0):
                    continue
            if debug: print("NVUL:", i, match.group(0))
            return 0
    m = re.finditer(r"^(- |[0-9]+\.)?cwe-", text)
    for match in m:
        if debug: print("VUL CWE:", match.group(0))
        return 1
    m = re.finditer(r"the vulnerability types present are", text)
    for match in m:
        if debug: print("VUL types:", match.group(0))
        return 1
    return 2

def extract_response(text):
    first_chunk = re.sub(r"\n```[a-z]+\n", "\n\n", text.strip()).split("\n\n")[0] # Consider code blocks as line breaks
    if debug:
        print("Chunk:")
        rprint(Padding(escape(first_chunk), (0, 4)))
    response = extract_response_from_chunk(first_chunk)
    if response == 2 or "can be improved" in first_chunk:
        if debug:
            print("Parsing rest:")
            rprint(Padding(escape(text), (0, 4)))
        response = extract_response_from_chunk(text)
    return response

# Models

def trim_openai_response(response):
    return response

def extract_openai_response(response):
    res = response
    res = res.strip().lower()
    return extract_response(res)

def trim_llama_response(response):
    res = response
    res = res.partition("</s>")[0]
    return res

def extract_llama_response(response):
    res = trim_llama_response(response).strip().lower()
    return extract_response(res)

def trim_starchat_response(response):
    res = response
    res = res.partition("<|end|>")[0]
    return res

def extract_starchat_response(response):
    res = trim_starchat_response(response).strip().lower()
    return extract_response(res)

def trim_starcoder_ta_response(response):
    res: str = response
    human_idx = res.find("Human:")
    blankline_idx = res.find("\n\n")
    separator_idx = res.find("-----")
    min_idx = min(len(res) if idx == -1 else idx for idx in [human_idx, blankline_idx, separator_idx])
    res = res[:min_idx]
    return res

def extract_starcoder_ta_response(response):
    res = trim_starcoder_ta_response(response).strip().lower()
    return extract_response(res)

def trim_wizardcoder_response(response):
    res = response
    res = res.partition("<|endoftext|>")[0]
    if "### Instruction:" in res:
        res = res[:res.find("### Instruction:")]
    return res

def extract_wizardcoder_response(response):
    res = trim_wizardcoder_response(response).strip().lower()
    return extract_response(res)

def trim_mistral_response(response):
    res = response
    res = res.partition("</s>")[0]
    return res

def extract_mistral_response(response):
    res = trim_mistral_response(response).strip().lower()
    return extract_response(res)

def pre_trim(prompt_type, response):
    if "zero_shot" == prompt_type:
        if "Question" in response and not response.startswith("Question:"):
            response = response[:response.find("Question")]
    if "\n```" in response:
        first_block = response.find("\n```")
        second_block = response.find("\n```", first_block + len("\n```")) + len("\n```")
        response = response[:first_block] + response[second_block:]
    lines = response.splitlines(keepends=False)
    for line in lines:
        if line.lower().startswith("answer:") and "reasoning:" in response:
            response = line
    return response

# Models
def extract_binary_response(response, args):
    response = pre_trim(args.prompt_type, response)
    if 'openai' in args.model_name_or_path:
        return extract_openai_response(response)
    # elif api_type == "starchat":
    #     return extract_starchat_response(response)
    elif 'starcoder' in args.model_name_or_path:
        return extract_starcoder_ta_response(response)
    elif 'WizardCoder' in args.model_name_or_path:
        return extract_wizardcoder_response(response)
    elif 'CodeLlama' in args.model_name_or_path or 'deepseek-coder' in args.model_name_or_path:
        return extract_llama_response(response)
    elif 'Mistral' in args.model_name_or_path:
        return extract_mistral_response(response)
    else:
        raise NotImplementedError(args.model_name_or_path)
