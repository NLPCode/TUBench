from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
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
    parser.add_argument('--model', type=str, default="llava-1.5-7b-hf", choices=["llava-1.5-7b-hf", "llava-1.5-13b-hf", "llava-v1.6-mistral-7b-hf", "llava-v1.6-vicuna-7b-hf", "llava-v1.6-vicuna-13b-hf"])
    parser.add_argument('--max_tokens', type=int, default=512, 
                        help='The number of max tokens for the generated output.')
    # parser.add_argument('--temperature', type=float, default=1e-20)   
    parser.add_argument('--wait_time', type=int, default=10, 
                        help='Retry your request after a specified seconds.')  
    parser.add_argument('--retry_times', type=int, default=10, 
                        help='The maximum number of retry times.')  
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
    model_path = f"../../HuggingfaceModels/llava-hf/{args.model}" # your downloaded model path.
    if args.model == "llava-1.5-7b-hf" or args.model == "llava-1.5-13b-hf": 
        model = LlavaForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        processor = AutoProcessor.from_pretrained(model_path)
    else:
        model = LlavaNextForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        processor = LlavaNextProcessor.from_pretrained(model_path)
    model.to("cuda:0")
    model.eval()
    i = 0
    with open(output_filename,  'a', encoding='utf-8') as fw:
        for question, choice, label, filename, image_filename in read_input(args.dataset):
            i+=1
            if i<=args.restart_idx:
                continue
            
            text_prompt = configs['user_prompt'].replace('##question##', question).replace('##choices##', choice)
            if args.model == "llava-1.5-7b-hf" or args.model == "llava-1.5-13b-hf":
                prompt = f"USER: <image>\n{text_prompt} ASSISTANT:"
            elif args.model == "llava-v1.6-vicuna-7b-hf" or args.model == "llava-v1.6-vicuna-13b-hf":
                prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{text_prompt} ASSISTANT:"
            elif args.model == "llava-v1.6-mistral-7b-hf":
                prompt = f"[INST] <image>\n{text_prompt} [/INST]"
            else:
                raise ValueError("")
            image = Image.open(image_filename)

            inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda:0")

            generate_ids = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False, num_beams=1)
            prompt_len = inputs["input_ids"].shape[1]
            generate_ids = generate_ids[:, prompt_len:]
            response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            d = {"filename": filename,"gold_label": label, "response": response, "text_prompt": prompt}
            fw.write(json.dumps(d, ensure_ascii=False) +'\n')
            fw.flush()
            
    