import os
import json
from dataset import AITZDataset
import random

def save_dataset_jsonl(dataset, save_path):
    # 将数据集存储到指定路径的 JSON 文件中
    with open(save_path, 'w') as f:
        for entry in dataset:
            f.write(json.dumps(entry) + '\n')
            
def save_data_to_json(data, file_path):
    """Save data to a JSON file."""
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def convert_format_to_sft(dataset, mode):
    
    new_data = []
    # for episode in data:
    # breakpoint()
    for data in dataset:
        # breakpoint()
        image_path = data['image_full_path']
        usr_prompt = ''
        # breakpoint()
        for item in data['inputs'][0]['content']:
            # breakpoint()
            if item['type'] == 'text':
                usr_prompt += item['text'] + '\n'
            elif item['type'] == 'image':
                usr_prompt += "<image>"
        if mode == 'think':
            answer = f'''<think>{data['action_think']}</think><action>{data['solution']}</action>'''
        elif mode == 'init':
            answer = f'''<action>{data['solution']}</action>'''
        # elif mode == 'think_in':
        #     usr_prompt+= f'''<think>{data['action_think']}</think>'''
        #     answer = f'''<action>{data['solution']}</action>'''
        elif mode == 'two_mode':
            random_number = random.random()
            if random_number < 0.5:
                answer = f'''<think>{data['action_think']}</think><action>{data['solution']}</action>'''
            else:
                answer = f'''<think>\n\n</think><action>{data['solution']}</action>'''
            
        new_step = {
            "conversations": [
            {
                "from": "human",
                "value": usr_prompt
            },
            {
                "from": "gpt",
                "value": answer
            }
            ],
            "images": [
                image_path
            ]
        }
        new_data.append(new_step)
    return new_data

def convert_format_to_grpo(dataset, mode):
    new_dataset=[]
    print(len(dataset))
    
    for data in dataset:
        image_path = data['image_full_path']
        usr_prompt = ''
        # breakpoint()
        for item in data['inputs'][0]['content']:
            # breakpoint()
            if item['type'] == 'text':
                usr_prompt += item['text'] + '\n'
            elif item['type'] == 'image':
                usr_prompt += "<image>"
        # think_prompt= f'''<think>{data['action_think']}</think>'''
        solution = f'''<action>{data['solution']}</action>'''
        new_dataset.append({
        'image': image_path, # googleapps_1984213201603669913_8.jpg'
        'problem': usr_prompt, # instruct + previous actions
        'solution': solution, # pyautogui.click(x=0.963, y=0.064
        # 'think': think_prompt,
        })

    # print(sampled_data)
    return new_dataset

if __name__ == "__main__":
    mode = 'think'
    domain = 'google_apps'
    AITZDataset = AITZDataset(split="test", mode=mode, domain=domain, data_dir=f"/export3/huangdongchi/hdc_debug/data/train/train_general_aitz_reprocess2.json")
    
    # preprocess the data
    process_path = 'your process path'
    save_dataset_jsonl(AITZDataset, process_path)
    
    
    # sft format
    sft_data = convert_format_to_sft(AITZDataset, mode='two_mode')
    new_sft_data_path = f'train_data/sft/train_two_data_50.json'
    save_data_to_json(sft_data, new_sft_data_path)
    
    ## grpo format
    # grpo_data = convert_format_to_grpo(AITZDataset,mode=mode)
    # new_grpo_data_path = 'train_data/grpo/train_grpo_data_mix.jsonl'
    # save_dataset_jsonl(grpo_data, new_grpo_data_path)
    
    