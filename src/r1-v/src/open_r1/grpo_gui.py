# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import json
import jax.numpy as jnp
import PIL.Image

from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

# from math_verify import parse, verify
from trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer, Qwen2VLGRPOVLLMTrainerModified
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

# from system_prompt.constant import agent_system_message,chat_template, grounding_system_message, until, user_instruction

_TAP_DISTANCE_THRESHOLD = 0.14


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["type", "format", "value"],
        # default_factory=lambda: ["accuracy"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )


def type_reward_json(action1, action2): 
    a1_type = action1.get("action").lower()
    a2_type = action2.get("action").lower()
    reward =0.0
    # if os.getenv("DEBUG_MODE") == "true":
    #         log_path = os.getenv("LOG_PATH")
    #         # local_rank = int(os.getenv("LOCAL_RANK", 0))
    #         with open(log_path, "a") as f:
    #             f.write(f"'a1_type':{a1_type}, 'a2_type':{a2_type}, {a1_type == a2_type}\n")         
    if a1_type == a2_type:
        reward = 1.0
    return reward

def value_reward_json(action1, action2): 
    a1_type = action1.get("action").lower()
    a2_type = action2.get("action").lower()
    reward =0.0

    if a1_type == a2_type == 'click':
        # if a1_type == 'type' or 'input': 
        #     if action2.get("value") in action1.get("value") :
        #         reward= 1.0
        # elif a1_type == 'click': 
        tap_1 = action1.get("position")
        tap_2 = action2.get("position")
        if ( jnp.linalg.norm(jnp.array(tap_1) - jnp.array(tap_2))
                    <= _TAP_DISTANCE_THRESHOLD) :
            reward =1.0
    return reward

def exact_action(data_str, mode):
    # 正则表达式模式
    if mode == 'think':
        pattern = r"<think>.*?</think>\s*<action>(.*?)</action>"
    else:
        pattern = r"<action>(.*?)</action>"
    # 使用 re.search 提取 <action> 标签内的内容
    match = re.search(pattern, data_str, re.DOTALL)
    if match:
        if mode == 'think':
            action_content = match.group(1)
        else:
            action_content = match.group(1)  # 提取第一个括号内的内容
        # print("Extracted action content:", action_content)
    else:
        raise ValueError("No match found in think action.")
    # print( "mode",mode, "raw", data_str, "processed",action_content)
    return action_content

def solution2json(data_str, mode):
    # 将字符串转换为字典
    # value里面也有单引号
    data_str = exact_action(data_str, mode)   
    data_dict = eval(data_str)
    # data_dict = eval(data_str.replace("'", "\""))  # 替换单引号为双引号
    
    if data_dict.get("position") is not None: 
        if isinstance(data_dict.get("position"), list):
            position = data_dict.get("position")  
        else:
            position=json.loads(data_dict.get("position"))
    else:
        position= None
    
    json_data = {
        "action": data_dict.get("action"),
        "value": data_dict.get("value"),
        "position": position  # 将位置字符串转换为列表
    }
    # print('mode',mode, json_data)
    return json_data    

def type_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = 0.0
        try:
            ground_truth = solution2json(sol, mode='init')
            student_answer = solution2json(content, mode='init')
            reward = type_reward_json(student_answer, ground_truth) 
        except Exception:
            pass  # Keep reward as 0.0 if both methods fail
                
        rewards.append(reward)
        # Get the DEBUG_MODE environment variable
        # debug_mode = os.getenv("DEBUG_MODE", "true")
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Type reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards

def value_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = 0.0
        try:
            ground_truth = solution2json(sol, mode='init')
            student_answer = solution2json(content, mode='init')
            reward = value_reward_json(student_answer, ground_truth) 
        except Exception:
            pass  # Keep reward as 0.0 if both methods fail
                
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Value reward: {reward} -------------\n")
        # Get the DEBUG_MODE environment variable
        # debug_mode = os.getenv("DEBUG_MODE", "true")
    return rewards    

def format_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    # pattern = r"<action>.*?</action>"
    # pattern = r"content:\s*(.*?)\s*Thought:\s*(.*?)\s*Action:\s*(.*?)\s*assistantos\s*(.*)"
    pattern = r"<think>.*?</think>\s*<action>.*?</action>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1 if match else 0.0 for match in matches]

reward_funcs_registry = {
    "value": value_reward,
    "type": type_reward,
    "format": format_reward,
}

# SYSTEM_PROMPT = (
#     "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
#     "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
#     "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
#     "<think> reasoning process here </think><answer> answer here </answer>"
# )
# THINKING_PROCESS = '''
# You first thinks about the reasoning process in the mind and then provides the user with the pre-defined function. 
# The reasoning process and functions are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e.,
# <think> reasoning process here </think><answer> pre-defined function, i.e., pyautogui.click(x=0.797, y=0.219), here </answer>
# Do not include the explainations within <answer> </answer> tags, only the following function should be included.
# '''
ANSWER_PROCESS = '''
You need to provide the user with the pre-defined function or the pyautogui actions,
like pyautogui.click(x=0.378, y=0.23), pyautogui.press(keys=['enter']), pyautogui.write(message='macbook pro'). 
The functions is enclosed within <answer> </answer> tags, i.e., <answer> pre-defined function, i.e., pyautogui.click(x=0.797, y=0.219), here </answer>
Do not include the explainations within <answer> </answer> tags, only the function or the pyautogui actions should be included.
'''
# SYSTEM_PROMPT = agent_system_message + ANSWER_PROCESS
# SYSTEM_PROMPT =''



def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load the dataset
    dataset = load_dataset('json', data_files=script_args.dataset_name)

    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                # {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    # QUESTION_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final answer(functions) in <answer> </answer> tags."
    # def load_image(example):
    #     try:
    #         # 注意：这里应该从image_path字段读取，而不是example["image"]
    #         example["image_vllm"] = PIL.Image.open(example["image"]).convert("RGB")
    #     except Exception as e:
    #         print(f"Error loading {example['image']}: {e}")
    #         example["image_vllm"] = None
    #     return example  # 保留所有原始字段
    
    def make_conversation_image(example):
        return  {
                'image_vllm':PIL.Image.open(example['image']),
                'image': example['image'],  # Store path instead of loaded image
                'solution': example['solution'],
                # 'think': example['think'],
                'prompt': [{
                    'role': 'user',
                    'content': [
                        {'type': 'image', 'text': None},
                        {'type': 'text', 'text': example['problem']}
                    ]
                }]
        }


    if "image" in dataset[script_args.dataset_train_split].features:
        print("has image in dataset")
        dataset = dataset.map(make_conversation_image, num_proc=8) 
        # dataset = dataset.map(load_image, num_proc=8)
        # Utilize multiprocessing for faster mapping
        # dataset = dataset.remove_columns(["original_question", "original_answer"])

    else:
        print("no image in dataset")
        dataset = dataset.map(make_conversation)
        dataset = dataset.remove_columns("messages")

    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainerModified
    print("using: ", trainer_cls)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
