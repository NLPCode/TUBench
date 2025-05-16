from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

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
    parser.add_argument('--model', type=str, default="ChatTruth-7B")
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
    model_path = f"../../HuggingfaceModels/{args.model}" # your downloaded model path.
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # use cuda device
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True).eval()
    model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)
    model.generation_config.do_sample = False
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    model.generation_config.max_new_tokens = args.max_tokens
    
    i = 0
    with open(output_filename,  'a', encoding='utf-8') as fw:
        for question, choice, label, filename, image_filename in read_input(args.dataset):
            i+=1
            if i<=args.restart_idx:
                continue
            text_prompt = configs['user_prompt'].replace('##question##', question).replace('##choices##', choice)
                
            query = tokenizer.from_list_format([
                {'image': image_filename},
                {'text': text_prompt},
            ])
            response, history = model.chat(tokenizer, query=query, history=None)
            d = {"filename": filename,"gold_label": label, "response": response, "text_prompt": text_prompt}
            fw.write(json.dumps(d, ensure_ascii=False) +'\n')
            fw.flush()
    