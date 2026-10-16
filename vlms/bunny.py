from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
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
    parser.add_argument('--model', type=str, default="Bunny-v1_0-4B", choices=["Bunny-v1_0-4B", "Bunny-v1_1-4B", "Bunny-Llama-3-8B-V", "Bunny-v1_1-Llama-3-8B-V"])
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
    model_path = f"../../HuggingfaceModels/BAAI/{args.model}" # your downloaded model path.

    # create model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16, # float32 for cpu
        device_map='auto',
        trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True)
    
    model.to("cuda:0")
    model.eval()
    i = 0
    with open(output_filename,  'a', encoding='utf-8') as fw:
        for question, choice, label, filename, image_filename in read_input(args.dataset):
            i+=1
            if i<=args.restart_idx:
                continue
            
            text_prompt = configs['user_prompt'].replace('##question##', question).replace('##choices##', choice)
            
            
            # text prompt
            prompt = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{text_prompt} ASSISTANT:"
            text_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]
            input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1][1:], dtype=torch.long).unsqueeze(0).to("cuda:0")

            # image, sample images can be found in images folder
            image = Image.open(image_filename)
            image_tensor = model.process_images([image], model.config).to(dtype=model.dtype, device="cuda:0")

            # generate
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                max_new_tokens=args.max_tokens, do_sample=False, num_beams=1,
                use_cache=True,
                repetition_penalty=1.0 # increase this to avoid chattering
            )[0]

            response = tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()

            d = {"filename": filename,"gold_label": label, "response": response, "text_prompt": text_prompt}
            fw.write(json.dumps(d, ensure_ascii=False) +'\n')
            fw.flush()

