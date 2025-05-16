import google.generativeai as genai
import PIL.Image
import os
import json
import argparse
import time

from gpt4 import read_input
class Gemini:
    def __init__(self, api_key: str, model: str, max_tokens: int = 256, 
                 temperature: float =  0, wait_time: int  = 10, retry_times: int  = 5):
        
        self.api_key = api_key
        self.model = model

        self.max_tokens = max_tokens
        self.temperature  = temperature
        self.wait_time = wait_time
        self.retry_times = retry_times
        
        genai.configure(api_key=api_key)
        generation_config = { 
            "temperature": temperature, 
            "max_output_tokens": max_tokens
        } 
        self.model = genai.GenerativeModel(args.model, generation_config=generation_config)


    def generate(self, text_prompt: str, image_path: str, text_only: bool)->str:
        img = PIL.Image.open(image_path)
        retry = True
        retry_times = 0
        while retry and retry_times < self.retry_times:
            try:
                if text_only:
                    response = self.model.generate_content(text_prompt)
                else:
                    response = self.model.generate_content([text_prompt, img])
                text = response.text
                return text
            except:
                
                retry_times += 1
                print(f"{retry_times} retry times.")
                time.sleep(self.wait_time)
        exit()
        # return "Failed to connect to Gemini API"
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Google Gemini.')
    parser.add_argument('--api_key', type=str, required=False) 
    parser.add_argument('--model', type=str, default="gemini-pro-vision", 
                        choices=["gemini-pro-vision", "gemini-1.5-flash-latest", "gemini-1.5-pro-latest"]) 
    parser.add_argument('--max_tokens', type=int, default=256, 
                        help='The number of max tokens for the generated output.')   
    parser.add_argument('--temperature', type=float, default=1e-20)   
    parser.add_argument('--wait_time', type=int, default=30, 
                        help='Retry your request after a specified seconds.')  
    parser.add_argument('--retry_times', type=int, default=100, 
                        help='The maximum number of retry times.')  
    parser.add_argument('--dataset', type=str, default='UCR') 
    parser.add_argument('--restart_idx', type=int, default=0) 
    parser.add_argument('--text_only', action='store_true')
    args = parser.parse_args()
    os.environ['http_proxy'] = 'http://127.0.0.1:7890'
    os.environ['https_proxy'] = 'http://127.0.0.1:7890'
    os.environ['all_proxy'] = 'socks5://127.0.0.1:7890'
    
    output_dir = os.path.join(f'../outputs/{args.dataset}')
    # output_dir = os.path.join(f'../outputs/{args.dataset}_comparison')
    os.makedirs(output_dir, exist_ok=True)
    if args.text_only:
        output_filename = f"{output_dir}/{args.model}_text_only.jsonl"
    else:
        output_filename = f"{output_dir}/{args.model}.jsonl"
        output_filename = f"{output_dir}/{args.model}_order3.jsonl"
    if args.model == 'gemini-1.5-flash-latest':
        args.model  = 'gemini-1.5-flash-001'
    elif args.model == 'gemini-1.5-pro-latest':
        args.model = 'gemini-1.5-pro-001'
    print(args)
    configs = json.load(open(f'../utils/{args.dataset}/prompt.json', 'r'))

    vllm = Gemini(  api_key=args.api_key, model=args.model,
                    temperature=args.temperature, max_tokens=args.max_tokens,
                    wait_time=args.wait_time, retry_times=args.retry_times)
    i = 0
    with open(output_filename,  'a', encoding='utf-8') as fw:
        for inputs in read_input(args.dataset):
            question, choice, label, filename, image_filename = list(inputs)[:5]
            i+=1

            if i<=args.restart_idx:
                continue
            if args.text_only:
                text_prompt = configs['llm_prompt'].replace('##question##', question).replace('##choices##', choice)
            else:             
                text_prompt = configs['user_prompt'].replace('##question##', question).replace('##choices##', choice)
                text_prompt = configs['user_prompt_order3'].replace('##question##', question).replace('##choices##', choice)
                
            response = vllm.generate(text_prompt, image_filename, text_only=args.text_only)
            d = {"filename": filename,"gold_label": label, "response": response, "text_prompt": text_prompt}
            fw.write(json.dumps(d, ensure_ascii=False) +'\n')
            # fw.write(json.dumps(d).encode('utf-8').decode('unicode_escape') +'\n')
            fw.flush()
