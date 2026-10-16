import dashscope
import os
import json
import argparse
import time

from gpt4 import read_input
class QWenVL:
    def __init__(self, api_key: str, model: str, max_tokens: int = 256, 
                 temperature: float =  0, wait_time: int  = 10, retry_times: int  = 5):
        
        self.api_key = api_key
        self.model = model

        self.max_tokens = max_tokens
        self.temperature  = temperature
        self.wait_time = wait_time
        self.retry_times = retry_times
        

    def generate(self, text_prompt: str, image_path: str)->str:
        messages = [{
            'role': 'system',
            'content': [{
                'text': 'You are a helpful assistant.'
            }]
            }, 
            {
            'role': 'user',
            'content': [
                {
                    'image': image_path
                },
                {
                    'text': text_prompt
                },
            ]
        }]

        retry = True
        retry_times = 0
        while retry and retry_times < self.retry_times:

            try:
                if self.model=="qwen-vl-chat-v1":
                    # the performance of this model is poor
                    response = dashscope.MultiModalConversation.call(model=dashscope.MultiModalConversation.Models.qwen_vl_chat_v1, messages=messages, 
                                                                    temperature=self.temperature 
                                                )
                else:
                    response = dashscope.MultiModalConversation.call(model=self.model, messages=messages, 
                                                    temperature=self.temperature, max_tokens=self.max_tokens)
                text =  response["output"]["choices"][0]["message"]["content"][0]["text"]
                # text =  response.output.choices[0].message.content[0]["text"]
                return text
            except:
                retry_times += 1
                print(f"{retry_times} retry times.")
                time.sleep(self.wait_time)
        return "Failed to connect to QWenVL API"

      
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='openai.')
    parser.add_argument('--api_key', type=str, required=True) 
    parser.add_argument('--model', type=str, default="qwen-vl-plus", choices=['qwen-vl-max', 'qwen-vl-plus', 'qwen-vl-chat-v1'])  
    parser.add_argument('--max_tokens', type=int, default=256, 
                        help='The number of max tokens for the generated output.')
    parser.add_argument('--temperature', type=float, default=1e-20)   
    parser.add_argument('--wait_time', type=int, default=10, 
                        help='Retry your request after a specified seconds.')  
    parser.add_argument('--retry_times', type=int, default=10, 
                        help='The maximum number of retry times.')  
    parser.add_argument('--dataset', type=str, default='UCR') 
    parser.add_argument('--restart_idx', type=int, default=0) 
    args = parser.parse_args()
    output_dir = os.path.join(f'../outputs/{args.dataset}')
    # output_dir = os.path.join(f'../outputs/{args.dataset}_comparison')
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{output_dir}/{args.model}.jsonl"
    print(args)
    configs = json.load(open(f'../utils/{args.dataset}/prompt.json', 'r'))
    dashscope.api_key = args.api_key
    vllm = QWenVL(api_key=args.api_key, model=args.model,
                  temperature=args.temperature, max_tokens=args.max_tokens,
                  wait_time=args.wait_time, retry_times=args.retry_times)
    i = 0
    with open(output_filename,  'a', encoding='utf-8') as fw:
        for question, choice, label, filename, image_filename in read_input(args.dataset):
            i+=1
            if i<=args.restart_idx:
                continue
            text_prompt = configs['user_prompt'].replace('##question##', question).replace('##choices##', choice)
            response = vllm.generate(text_prompt, image_filename)
            # print(response)
            d = {"filename": filename,"gold_label": label, "response": response, "text_prompt": text_prompt}
            # fw.write(json.dumps(d)+'\n')
            fw.write(json.dumps(d, ensure_ascii=False) +'\n')
            fw.flush()
