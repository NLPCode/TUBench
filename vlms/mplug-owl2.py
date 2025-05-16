import torch
from PIL import Image
from transformers import TextStreamer

import random
import numpy as np
import argparse
import json
from gpt4 import read_input
import os

from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.conversation import conv_templates, SeparatorStyle
from mplug_owl2.model.builder import load_pretrained_model
from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

def set_seed(seed, n_gpu=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='openai.')
    parser.add_argument('--model', type=str, default="mplug-owl2-llama2-7b", choices=["mplug-owl2-llama2-7b", "mplug_owl_2_1"])
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
    if args.model == "mplug-owl2-llama2-7b":
        model_path = '../../HuggingfaceModels/MAGAer13/mplug-owl2-llama2-7b'
    else:
        model_path = '../../HuggingfaceModels/Mizukiluke/mplug_owl_2_1'
    
    
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit=False, load_4bit=False, device="cuda")

    model.to("cuda:0")
    model.eval()
    i = 0
    with open(output_filename,  'a', encoding='utf-8') as fw:
        for question, choice, label, filename, image_filename in read_input(args.dataset):
            i+=1
            if i<=args.restart_idx:
                continue
            
            text_prompt = configs['user_prompt'].replace('##question##', question).replace('##choices##', choice)
            conv = conv_templates["mplug_owl2"].copy()
            roles = conv.roles
            
            
            image = Image.open(image_filename).convert('RGB')
            max_edge = max(image.size) # We recommand you to resize to squared image for BEST performance.
            image = image.resize((max_edge, max_edge))

            image_tensor = process_images([image], image_processor)
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)

            inp = DEFAULT_IMAGE_TOKEN + text_prompt
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
            stop_str = conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    max_new_tokens=args.max_tokens,
                    do_sample=False,
                    num_beams=1,
                    streamer=streamer,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

            response = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            
            d = {"filename": filename,"gold_label": label, "response": response, "text_prompt": prompt}
            fw.write(json.dumps(d, ensure_ascii=False) +'\n')
            fw.flush()

