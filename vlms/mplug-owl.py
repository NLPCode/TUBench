from PIL import Image
import requests
from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl.tokenization_mplug_owl import MplugOwlTokenizer
from mplug_owl.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor

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
    parser.add_argument('--model', type=str, default="mplug-owl-llama-7b", choices=["mplug-owl-llama-7b"])
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
    model_path = f"../../HuggingfaceModels/MAGAer13/{args.model}" # your downloaded model path.
    model = MplugOwlForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )
    image_processor = MplugOwlImageProcessor.from_pretrained(model_path)
    tokenizer = MplugOwlTokenizer.from_pretrained(model_path)
    processor = MplugOwlProcessor(image_processor, tokenizer)
    
    model.to("cuda:0")
    model.eval()
    i = 0
    with open(output_filename,  'a', encoding='utf-8') as fw:
        for question, choice, label, filename, image_filename in read_input(args.dataset):
            i+=1
            if i<=args.restart_idx:
                continue
            
            text_prompt = configs['user_prompt'].replace('##question##', question).replace('##choices##', choice)
            
            
            # We use a human/AI template to organize the context as a multi-turn conversation.
            # <image> denotes an image placehold.
            prompt = f"The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nHuman: <image>\nHuman: {text_prompt}\nAI: "
            
            image = Image.open(image_filename)
            inputs = processor(text=[prompt], images=[image], return_tensors="pt").to("cuda:0")
            inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
            with torch.no_grad():
                generate_ids = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False, num_beams=1)
            response = tokenizer.decode(generate_ids.tolist()[0], skip_special_tokens=True)
            d = {"filename": filename,"gold_label": label, "response": response, "text_prompt": prompt}
            fw.write(json.dumps(d, ensure_ascii=False) +'\n')
            fw.flush()

