# -*- coding: UTF-8 -*-
# run with python version>3.6
# pip install openai


import argparse
import base64
import requests
import os
from tqdm import tqdm
import time
import json


    
class OpenAIGPT4Vision:
    def __init__(self, api_key: str, model: str, max_tokens: int = 256, 
                 temperature: float =  0, wait_time: int  = 10, retry_times: int  = 5):
        
        self.api_key = api_key
        self.model = model

        self.max_tokens = max_tokens
        self.temperature  = temperature
        self.wait_time = wait_time
        self.retry_times = retry_times
        
        self.headers = {"Content-Type": "application/json", 
                        "Authorization": f"Bearer {api_key}"}
    
    def create_few_shot_prompt(self, n_shot: int, question_answer: str, text_prompt: str, image_path: str):
        content = [
            {
            "type": "text",
            "text": text_prompt
            },
        ]
        if n_shot==3:
            demonstration_dir_list = ['../utils/tabmwp/10003']
        else:
            demonstration_dir_list = ['../utils/tabmwp/10003', '../utils/tabmwp/11830']
        for demonstration_dir in demonstration_dir_list:
            qa_dict = json.load(open(f'{demonstration_dir}/qa.json', 'r'))
            _question_answer = f"Question: {qa_dict['question']}\nAnswer Choices: {qa_dict['choices']}\n"
            for i in range(3):
                _image_path  = f"{demonstration_dir}/{i+1}.png"
                base64_image = self.encode_image(_image_path)
                content.append(
                                {
                                "type": "image_url",
                                "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                                }
                                }
                            )
                content.append( 
                                {
                                "type": "text",
                                "text": _question_answer + qa_dict[f'solution{i+1}']
                                }
                            )
        base64_image = self.encode_image(image_path)
        content.append(
                        {
                        "type": "image_url",
                        "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high"
                        }
                        }
                    )
        content.append( 
                        {
                        "type": "text",
                        "text": question_answer
                        }
                    )
        return content
    # Function to encode the image
    @staticmethod
    def encode_image(image_path:str) ->str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    

    def generate(self, n_shot: int, question_answer:str, text_prompt: str, image_path: str)->str:
        
        if n_shot == 0:
            base64_image = self.encode_image(image_path)
            content = [
                        {
                        "type": "text",
                        "text": text_prompt
                        },
                        {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                        }
                        
                        }
                    ]
        else:
            content = self.create_few_shot_prompt(n_shot, question_answer, text_prompt, image_path)
        payload = {
            "model": self.model,
            "messages": [
                {
                "role": "user",
                "content": content
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "n": 1
            
        }
        retry = True
        retry_times = 0
        while retry and retry_times < self.retry_times:
            # response = requests.post("https://api.openai.com/v1/chat/completions", headers=self.headers, json=payload)
            try:
                response = requests.post("https://api.chatanywhere.com.cn/v1/chat/completions", headers=self.headers, json=payload)
                print(response)
                response.encoding = 'utf-8'
                if response.status_code == 200:
                    response_data = response.json()
                    # print(response_data["usage"])
                    text =  response_data["choices"][0]["message"]["content"]
                    # text = text.encode('utf-8').decode('unicode_escape')
                    return text
            except:
                    print(f"Failed to connect to OpenAI API. Retrying...")
                    time.sleep(self.wait_time)
                    retry_times += 1
        return "Failed to connect to OpenAI GPT4V API"

def read_input(dataset, choice=False, negate_flip=False):
    if dataset == 'UCR':
        for i in range(1, 21):
            qa_filenames = [f'{i}/1.txt', f'{i}/1_u1.txt', f'{i}/1_u2.txt', f'{i}/1_u3.txt']
            for qa_filename in qa_filenames:
                if 'u1' in qa_filename:
                    strategy = 1
                elif 'u2' in qa_filename:
                    strategy = 2
                elif 'u3' in qa_filename:
                    strategy = 3
                else:
                    strategy = 0
                
                qa_filename = f'../datasets/{dataset}/{qa_filename}'
                image_filename = qa_filename.replace('txt', 'png')
                if negate_flip:
                    qa_filename = qa_filename.replace('.txt', '_negate_flip.txt')
                with open(qa_filename, 'r') as fr:
                    for line in fr:
                        contents = line.strip().split('\t')
                        assert len(contents) == 2
                        question, label = contents[0], contents[1]
                        yield question, '', label, qa_filename, image_filename, strategy
    elif dataset == "UVQA":
        for i in range(0, 300):
            cur_dir= f'../datasets/{dataset}/{i}'
            # cur_dir= f'../datasets/{dataset}/select_val2014_vqa_comparison/{i}'
            if not os.path.exists(cur_dir):
                continue
            for filename in os.listdir(cur_dir):
                if filename.endswith('jpg'):
                    break
            image_filename = f'{cur_dir}/{filename}'
            qa_filename = f'{cur_dir}/vqa.txt'
            if negate_flip:
                qa_filename = qa_filename.replace('.txt', '_negate_flip.txt')
            with open(qa_filename, 'r') as fr:
                for line in fr:
                    contents = line.strip().split('\t')
                    question, label = contents[1], contents[2]
                    if label == 'Unanswerable':
                        if len(contents) == 4:
                            strategy = int(contents[3])
                        else:
                            strategy = 0
                    else:
                        assert len(contents) == 3
                        strategy = 0
                    yield question, '', label, qa_filename, image_filename, strategy
    elif dataset == 'UGeoQA':
        for i in range(0, 10000):
            cur_dir= f'../datasets/{dataset}/{i}'
            if not os.path.exists(cur_dir):
                continue
            qa_filename = f'{cur_dir}/{i}.json'
            image_filename = f'{cur_dir}/{i}.png'
            data = json.load(open(qa_filename, 'r'))
            choices = data['choices']
            if choice:
                choice_str = choices
            else:
                choice_str = ""
                for j, c in enumerate(choices):
                    choice_str += f"({chr(97+j)}) {c} "
                choice_str = choice_str.strip()
            
            # answerable case
            question = data['subject']
            question = question.replace('()', '?')
            # question = f"{question} Answer Choices: {choice_str}."
            label = data['label']
            label = f"({chr(97+label)})"
            yield question, choice_str, label, qa_filename, image_filename
            
            # unanswerable case
            question = data['unanswerable_subject']
            question = question.replace('()', '?')
            # question = f"{question} Answer Choices: {choice_str}."
            label = "Unanswerable"
            yield question, choice_str, label, qa_filename, image_filename
    elif dataset == 'UTabMWP':
        for i in range(9840):
            cur_dir= f'../datasets/{dataset}/{i}'
            if not os.path.exists(cur_dir):
                continue
            qa_filename = f'{cur_dir}/{i}.json'
            data = json.load(open(qa_filename, 'r'))
            choices = data['choices']
            if choice:
                choice_str = choices
            else:
                choice_str = ""
                for j, c in enumerate(choices):
                    choice_str += f"({chr(97+j)}) {c} "
                choice_str = choice_str.strip()
            question = data['question']
            
            # answerable case
            label = data['choices'].index(data['answer'])
            label = f"({chr(97+label)})"
            image_filename = f'{cur_dir}/{i}.png'
            yield question, choice_str, label, qa_filename, image_filename
            
            # unanswerable case
            label = "Unanswerable"
            image_filename = f'{cur_dir}/{i}_u1.png'
            yield question, choice_str, label, qa_filename, image_filename
            
            
            # # answerable case
            # label = data['choices'].index(data['answer'])
            # label = f"({chr(97+label)})"
            # image_filename = f'{cur_dir}/{i}_1.png'
            # if os.path.exists(image_filename):
            #     yield question, choice_str, label, qa_filename, image_filename
            
    else:
        pass

      
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='openai.')
    parser.add_argument('--api_key', type=str, required=True) 
    parser.add_argument('--model', type=str, default="gpt-4o-mini", choices=['gpt-4-turbo', 'gpt-4o', 'gpt-4o-mini',
                                                                             'gemini-1.5-flash-latest', 'gemini-1.5-pro-latest',])   
    parser.add_argument('--max_tokens', type=int, default=256, 
                        help='The number of max tokens for the generated output.')   
    parser.add_argument('--temperature', type=float, default=1e-20)   
    parser.add_argument('--wait_time', type=int, default=10, 
                        help='Retry your request after a specified seconds.')  
    parser.add_argument('--retry_times', type=int, default=10, 
                        help='The maximum number of retry times.')  
    parser.add_argument('--dataset', type=str, default='UCR') 
    parser.add_argument('--restart_idx', type=int, default=0)
    parser.add_argument('--text_only', action='store_true')
    parser.add_argument('--negate_flip', action='store_true', help='whether use the negated question')
    parser.add_argument('--n_shot', type=int, default=0)
    args = parser.parse_args()
    output_dir = os.path.join(f'../rebuttal_outputs/{args.dataset}')
    # output_dir = os.path.join(f'../outputs/{args.dataset}_comparison')
    os.makedirs(output_dir, exist_ok=True)
    if args.text_only:
        output_filename = f"{output_dir}/{args.model}_text_only.jsonl"
    else:
        if args.n_shot == 0:
            if args.negate_flip:
                output_filename = f"{output_dir}/{args.model}_negate_flip.jsonl"
            else:
                output_filename = f"{output_dir}/{args.model}.jsonl"
        else:
            output_filename = f"{output_dir}/{args.model}_{args.n_shot}shot.jsonl"
        # output_filename = f"{output_dir}/{args.model}_order2.jsonl"
    configs = json.load(open(f'../utils/{args.dataset}/prompt.json', 'r'))
    
    if args.model=='gpt-4o':
        args.model = 'gpt-4o-2024-05-13'
    elif args.model=='gpt-4o-mini':
        args.model = 'gpt-4o-mini-2024-07-18'  
    elif args.model=='gpt-4-turbo':
        args.model = 'gpt-4-turbo-2024-04-09'  
    else:
        pass
        # raise ValueError('')
    print(args)
    
    vllm = OpenAIGPT4Vision(api_key=args.api_key, model=args.model,
                            temperature=args.temperature, max_tokens=args.max_tokens,
                            wait_time=args.wait_time, retry_times=args.retry_times)
    i = 0
    with open(output_filename,  'a', encoding='utf-8') as fw:
        for inputs in read_input(args.dataset, negate_flip=args.negate_flip):
            question, choice, label, filename, image_filename = list(inputs)[:5]
            i+=1

            if i<=args.restart_idx:
                continue   
            if args.text_only:
                text_prompt = configs['llm_prompt'].replace('##question##', question).replace('##choices##', choice)
            else:
                if args.n_shot ==0:             
                    text_prompt = configs['user_prompt'].replace('##question##', question).replace('##choices##', choice)
                    # text_prompt = configs['user_prompt_order2'].replace('##question##', question).replace('##choices##', choice)
                    question_answer = None
                else:
                    text_prompt = configs['few_shot_prompt']
                    question_answer = f"Question: {question}\nAnswer Choices: {choice}\n"
                    
            response = vllm.generate(args.n_shot, question_answer, text_prompt, image_filename)
            # print(response)
            d = {"filename": filename,"gold_label": label, "response": response, "text_prompt": text_prompt}
            # fw.write(json.dumps(d)+'\n')
            fw.write(json.dumps(d, ensure_ascii=False) +'\n')
            fw.flush()
