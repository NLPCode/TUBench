from transformers import AutoModel, AutoTokenizer

import torch
import random
import numpy as np
import argparse
import json
from gpt4 import read_input
import os
def set_seed(seed, n_gpu=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='openai.')
    parser.add_argument('--model', type=str, default="internlm-xcomposer-vl-7b", choices=["internlm-xcomposer-vl-7b", "internlm-xcomposer2-vl-7b"])
    parser.add_argument('--max_tokens', type=int, default=512, 
                        help='The number of max tokens for the generated output.')

    parser.add_argument('--dataset', type=str, default='UCR') 
    parser.add_argument('--restart_idx', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    output_dir = os.path.join(f'../outputs/{args.dataset}')
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{output_dir}/{args.model}.jsonl"
    print(args)
    configs = json.load(open(f'../utils/{args.dataset}/prompt.json', 'r'))
    
    set_seed(args.seed)
    model_path = f"../../HuggingfaceModels/internlm/{args.model}" # your downloaded model path.
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if args.model == 'internlm-xcomposer-vl-7b':
        model.tokenizer = tokenizer
    i = 0
    with open(output_filename,  'a', encoding='utf-8') as fw:
        for question, choice, label, filename, image_filename in read_input(args.dataset):
            i+=1
            if i<=args.restart_idx:
                continue
            text_prompt = configs['user_prompt'].replace('##question##', question).replace('##choices##', choice)
            prompt = '<ImageHere>'+text_prompt
            if args.model == 'internlm-xcomposer-vl-7b':  
                response, _ = model.chat(prompt, image=image_filename, history=[], 
                                        max_new_tokens=args.max_tokens, do_sample=False, num_beams=1)
            else:
                response, _ = model.chat(tokenizer, query=prompt, image=image_filename, history=[], 
                        max_new_tokens=args.max_tokens, do_sample=False, num_beams=1)
            d = {"filename": filename,"gold_label": label, "response": response, "text_prompt": prompt}
            fw.write(json.dumps(d, ensure_ascii=False) +'\n')
            fw.flush()
    