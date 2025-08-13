import numpy as np
import torch

import pdb
import os
import sys
sys.path.append(".")
# from dset_aitz import AitzDataset
import json
IGNORE_INDEX = -100

from transformers import AutoProcessor
from aitz import aitz_to_openai_qwenvl
from torch.utils.data import DataLoader



class AITZDataset(torch.utils.data.Dataset):
    """
    Creating a custom dataset for reading the processed AITW data
    with GPT-4V labeled detail semantic annotations.
    """
    DATASET_DIR = {
    # 'general': '{}/general',
    'google_apps': '{}/google_apps',
    'install': '{}/install',
    # 'single': '{}/single',
    'web_shopping': '{}/web_shopping',
    }

    def __init__(self, mode='init', split="train", data_dir="dataset/android_in_the_zoo/aitz_reprocess2.json", ratio=1.0, double_sample=False) -> None:
        self.ratio = ratio
        self.double_sample = double_sample
        self.data_dir = data_dir
        # change to your local image directory
        self.image_dir = f'dataset/android_in_the_zoo/{split}'
        self.mode = mode
        self.episode_data = self._load_data_()
        self.data = self._split_to_steps_(self.episode_data, self.mode)
    
    def _load_data_(self): 
        
        episode_data = json.load(open(self.data_dir, "r"))                        
        # ep_data.append(episode_data)      
        return episode_data

    def _split_to_steps_(self, episode_data, mode):
        data = []
        for edx, episode in enumerate(episode_data):
            # history_plain_actions, history_coat_actions = [], []
            for idx, step in enumerate(episode):
                step['subset'] = step['image_path'].split('/')[0]
                step['image_full_path'] = os.path.join(self.image_dir, step['image_path'])
                step['prev_step_id'] = episode[idx-1]['step_id'] if idx > 0 else None
                next_img_path = os.path.join(self.image_dir, episode[idx+1]['image_path']) \
                    if idx + 1 < len(episode) else None
                step['next_image_full_path'] = next_img_path
                
                # inputs format
                if self.mode == 'init':
                    data.append(self.__apply_template__(step))
                elif self.mode == 'think':
                    data.append(self.__apply_template_think__(step))
                elif self.mode == 'try_think':
                    data.append(self.__try_apply_template_think__(step))
        return data

    def __len__(self, ): return len(self.data)

    def __getitem__(self, index): return self.data[index]

    
    def __apply_template__(self, data):
        task = data['instruction']
        action_history = data['action_history']
        image_history = data['image_history']
        full_image_path_history=[]
        if image_history is not None:
            for image_path in image_history:
                full_image_path = os.path.join(self.image_dir, image_path)
                full_image_path_history.append(full_image_path)
        image_path = data['image_full_path']
        inputs = aitz_to_openai_qwenvl(task=task, action_history=action_history, image_history=full_image_path_history, image_path=image_path)
        data['inputs']= inputs
        
        return data
    
    def __apply_template_think__(self,data):
        task = data['instruction']
        action_history = data['action_history']
        image_history = data['image_history']
        full_image_path_history=[]
        if image_history is not None:
            for image_path in image_history:
                full_image_path = os.path.join(self.image_dir, image_path)
                full_image_path_history.append(full_image_path)
        image_path = data['image_full_path']
        inputs = aitz_to_openai_qwenvl(task=task, action_history=action_history, image_history=full_image_path_history, image_path=image_path, mode = 'think')
        data['inputs']= inputs
        
        return data
    
    def __try_apply_template_think__(self,data):
        task = data['instruction']
        action_history = data['action_history']
        image_history = data['image_history']
        action_think = data['action_think']
        full_image_path_history=[]
        if image_history is not None:
            for image_path in image_history:
                full_image_path = os.path.join(self.image_dir, image_path)
                full_image_path_history.append(full_image_path)
        image_path = data['image_full_path']
        inputs = aitz_to_openai_qwenvl(task=task, action_history=action_history, image_history=full_image_path_history, image_path=image_path, think=action_think, mode = 'try_think')
        
        data['inputs']= inputs
        
        return data
        


        
        
    
        
