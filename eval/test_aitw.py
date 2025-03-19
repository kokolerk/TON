from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from tqdm import tqdm
import re
from math_verify import parse, verify

MODEL_PATH='/export3/huangdongchi/hdc_debug/model_jq/Qwen2-VL-Aguvis-7B-720P'
# MODEL_PATH="/export3/huangdongchi/hdc_debug/R1-V/results" # qwen2vl model or grpoed model on geoqa train
BSZ=50 # 50 reduce it if GPU OOM
OUTPUT_PATH="/export3/huangdongchi/hdc_debug/R1-V/src/eval/eval_results/aguvis-2000.jsonl"
PROMPT_PATH="/export3/huangdongchi/hdc_debug/R1-V/images/aitw/aitw-l1-v1-2000.jsonl"

#We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# default processer
processor = AutoProcessor.from_pretrained(MODEL_PATH)

data = []
with open(PROMPT_PATH, "r") as f:
    for line in f:
        data.append(json.loads(line))


QUESTION_TEMPLATE = "{Question}"
recipient_text = "<|im_start|>assistant<|recipient|>"

messages = []

data = data

for i in data:
    message = [{
        "role": "user",
        "content": [
            {
                "type": "image", 
                "image": f"file://{i['image']}"
            },
            {
                "type": "text",
                "text": QUESTION_TEMPLATE.format(Question=i['problem'])
            }
        ]
    }]
    messages.append(message)




all_outputs = []  # List to store all answers

# Process data in batches
# flag =0
for i in tqdm(range(0, len(messages), BSZ)):
    # flag +=1
    # if flag > 1:
    #     break
    batch_messages = messages[i:i + BSZ]
    
    # Preparation for inference
    text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
    text = [t+recipient_text for t in text]
    image_inputs, video_inputs = process_vision_info(batch_messages)
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        padding_side='left',
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=1024, do_sample=False)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    batch_output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    all_outputs.extend(batch_output_text)
    print(f"Processed batch {i//BSZ + 1}/{(len(messages) + BSZ - 1)//BSZ}")


def action_reward(action1, action2):
    type=0
    para=0
    # breakpoint()
    if not action1 or not action2:
        return type,para
    action1 = action1.strip()
    action2 = action2.strip()
    action1_match = re.search(r'(.+?)\((.+?)\)', action1)
    action2_match = re.search(r'(.+?)\((.+?)\)', action2)
    # breakpoint()
    # 提取类型和参数
    if not action1_match or not action2_match:
        return type,para
    action1_type = action1_match.group(1)  # 括号前部分，如 "pyautogui.click"
    action1_params = action1_match.group(2)  # 括号内部分，如 "x=0.364, y=0.135"
    
    action2_type = action2_match.group(1)
    action2_params = action2_match.group(2)
    
    # print(f"动作1类型: {action1_type}, 参数: {action1_params}")
    # print(f"动作2类型: {action2_type}, 参数: {action2_params}")
    # 比较类型
    if action1_type == action2_type:
        type=1  # 类型相同，获得基础分数
    
    if action1_params == action2_params:
        para=1
        
    return type,para


final_output = []
correct_number_type = 0
correct_number_para = 0
correct_number = 0

for input_example, model_output in zip(data,all_outputs):
    # original_output = model_output
    # breakpoint()
    ground_truth = input_example['solution']
    # modified the answer tags
    ground_truth = re.search(r'<answer>(.*?)</answer>', ground_truth)
    ground_truth = ground_truth.group(1).strip() if ground_truth else ground_truth.strip()
    
    model_answer = re.search(r'assistantos\s*(.*)', model_output, re.DOTALL)
    model_answer = model_answer.group(1).strip() if model_answer else model_output.strip()

    # Count correct answers
    # if model_answer is not None and float(verify(model_answer,parse(ground_truth)))>0:
    if model_answer is not None:
        type, para = action_reward(model_answer, ground_truth)
        correct_number_type += type
        correct_number_para += para
        correct_number += type*para
        if correct_number ==1:
            is_correct = True
        else:
            is_correct = False
    else:
        is_correct = False
    
    try:
        result = {
            'question': input_example,
            'ground_truth': ground_truth,
            'model_output': model_answer,
            'extracted_answer':str(model_answer[0]) if model_answer is not None else None,
            'is_correct':is_correct
        }

    except Exception as e:
        print("no answer parsed",e,model_answer)
        result = {
            'question': input_example,
            'ground_truth': ground_truth,
            'model_output': model_output,
            'extracted_answer':None,
            'is_correct':is_correct
        }



    final_output.append(result)


# Calculate and print accuracy
accuracy = correct_number / len(data) * 100
accuracy_type = correct_number_type / len(data) * 100
accuracy_para = correct_number_para / len(data) * 100
print(f"\nAccuracy type: {accuracy_type:.2f}%")
print(f"\nAccuracy para: {accuracy_para:.2f}%")
print(f"\nAccuracy: {accuracy:.2f}%")

# Save results to a JSON file
output_path = OUTPUT_PATH
with open(output_path, "w") as f:
    json.dump({
        'accuracy': accuracy,
        'accuracy_type': accuracy_type,
        'accuracy_para': accuracy_para,
        'results': final_output
    }, f, indent=2, ensure_ascii=False)

print(f"Results saved to {output_path}")





