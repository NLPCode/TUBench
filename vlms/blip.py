from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
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
    parser.add_argument('--model', type=str, default="blip2-opt-2.7b", choices=["blip2-opt-2.7b", "blip2-opt-6.7b", "blip2-flan-t5-xxl", "instructblip-vicuna-7b", "instructblip-vicuna-13b", "instructblip-flan-t5-xxl"])
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
    model_path = f"../../HuggingfaceModels/Salesforce/{args.model}" # your downloaded model path.
    if args.model == "blip2-opt-2.7b" or args.model == "blip2-opt-6.7b" or args.model == "blip2-flan-t5-xxl": 
        model = Blip2ForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16)
        processor = Blip2Processor.from_pretrained(model_path)
    else:
        model = InstructBlipForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16)
        processor = InstructBlipProcessor.from_pretrained(model_path)
    model.to("cuda:0")
    model.eval()
    i = 0
    with open(output_filename,  'a', encoding='utf-8') as fw:
        for question, choice, label, filename, image_filename in read_input(args.dataset):
            i+=1
            if i<=args.restart_idx:
                continue
            
            text_prompt = configs['user_prompt'].replace('##question##', question).replace('##choices##', choice)
            image = Image.open(image_filename)
            inputs = processor(text=text_prompt, images=image, return_tensors="pt").to("cuda:0")

            generate_ids = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False, num_beams=1)
            response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            # print(response)
            # exit()
            
            d = {"filename": filename,"gold_label": label, "response": response, "text_prompt": text_prompt}
            fw.write(json.dumps(d, ensure_ascii=False) +'\n')
            fw.flush()

