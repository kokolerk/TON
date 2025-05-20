import json
import os
from TON.src.eval.aitz_evaluate.process_data import load_json,load_jsonl
import tqdm
from accelerate.utils import gather_object
import numpy as np
from utils_aitz import pred2json, pred2json_post, is_tap_action, _check_tap_actions_match, solution2json,answer2json
# from eval_aitz import calculate_aitz_metrics
import ast
import jax.numpy as jnp
import logging
logging.basicConfig(level=logging.INFO)

llama_TAP_DISTANCE_THRESHOLD = 0.2


def calculate_aitz_metrics(results):
    num = 0
    type_action=0
    match_action=0
    # click
    num_click=0
    type_click=0
    match_click=0
    # text
    num_text=0
    type_text=0
    match_text=0
    # scroll
    num_scroll=0
    type_scroll=0
    match_scroll=0
    # press
    num_press=0
    type_press=0
    match_press=0
    # completed
    num_completed=0
    type_completed=0
    match_completed=0
    
    num_wrong_format=0

    for step in results:
        num += 1
        type_action += step['type_action']
        match_action += step['match_action']

        num_click += step['num_click']
        type_click += step['type_click']
        match_click += step['match_click']

        num_text += step['num_text']
        type_text += step['type_text']
        match_text += step['match_text']

        num_scroll += step['num_scroll']
        type_scroll += step['type_scroll']
        match_scroll += step['match_scroll']

        num_press += step['num_press']
        type_press += step['type_press']
        match_press += step['match_press']

        num_completed += step['num_completed']
        type_completed += step['type_completed']
        match_completed += step['match_completed']

        num_wrong_format += step['num_wrong_format']
    
    logging.info("[Type Avg]: " + str(type_action/num))
    logging.info("[Match Avg]: " + str(match_action/num))
    logging.info("[Type Click]: " + str(type_click/num_click) if num_click > 0 else 0)
    logging.info("[Match Click]: " + str(match_click/num_click) if num_click > 0 else 0)
    logging.info("[Type Text]: " + str(type_text/num_text) if num_text > 0 else 0)
    logging.info("[Match Text]: " + str(match_text/num_text) if num_text > 0 else 0)
    # logging.info("[Type Scroll]: " + str(type_scroll/num_scroll) if num_scroll > 0 else 0)
    logging.info("[Match Scroll]: " + str(match_scroll/num_scroll) if num_scroll > 0 else 0)
    # logging.info("[Type Press]: " + str(type_press/num_press) if num_press > 0 else 0)
    logging.info("[Match Press]: " + str(match_press/num_press) if num_press > 0 else 0)
    # logging.info("[Type Completed]: " + str(type_completed/num_completed) if num_completed > 0 else 0)
    logging.info("[Match Completed]: " + str(match_completed/num_completed) if num_completed > 0 else 0)

    metrics = {
        "Type Avg": type_action / num,
        "Match Avg": match_action / num,
        "Type Click": type_click / num_click if num_click > 0 else 0,
        "Match Click": match_click / num_click if num_click > 0 else 0,
        "Type Text": type_text / num_text if num_text > 0 else 0,
        "Match Text": match_text / num_text if num_text > 0 else 0,
        # "Type Scroll": type_scroll / num_scroll if num_scroll > 0 else 0,
        "Match Scroll": match_scroll / num_scroll if num_scroll > 0 else 0,
        # "Type Press": type_press / num_press if num_press > 0 else 0,
        "Match Press": match_press / num_press if num_press > 0 else 0,
        # "Type Completed": type_completed / num_completed if num_completed > 0 else 0,
        "Match Completed": match_completed / num_completed if num_completed > 0 else 0,
        "Wrong Format": num_wrong_format / num
    }
    return metrics


def  evalute_aitz_llama(answers_unique, outputs_unique, mode):
    results = []
    eval_metric = []
    length = len(answers_unique)
    for i in range(length):
        eval_metric.append(dict(
                type_action=0,  # type
                match_action=0,    # match

                # click
                num_click=0,
                type_click=0,
                match_click=0,

                # type
                num_text=0,
                type_text=0,
                match_text=0,

                # scroll
                num_scroll=0,
                type_scroll=0,
                match_scroll=0,

                # press
                num_press=0,
                type_press=0,
                match_press=0,

                # completed
                num_completed=0,
                type_completed=0,
                match_completed=0,

                num_wrong_format=0,
        ))
    motivation = []    
    for ans_i, predict_i, output_i in zip(answers_unique,outputs_unique,eval_metric):
        # step_result = output_i.copy()
        # episode_id = output_i['ep_id']
        # step_id = output_i['step_id']
        # pred = output_i['sentence'][0]
        # breakpoint()
        try:
            pred_i = answer2json(predict_i, mode)
            action_pred = pred2json_post(pred_i)
            print('pred_i', pred_i)
            # breakpoint()
            action_ref = solution2json(ans_i, mode='think')
            print('action_ref', action_ref)
            # action_ref = answer2json(ans_i, mode)
            action_ref = pred2json_post(action_ref) # 'action_type', 'touch_point', 'lift_point', 'typed_text'
            
            # type accuracy
            if action_pred["action_type"] == action_ref["action_type"]:
                output_i['type_action'] += 1
                motivation.append(1)
            else:
                motivation.append(0)
            # click accuracy
            if action_ref["action_type"] == 4:
                output_i['num_click'] += 1
                if action_pred["action_type"] == 4:
                    output_i['type_click'] += 1
                    # annotation_positions = np.array(ast.literal_eval(output_i["ui_positions"]))
                    tap_1_yx = (float(action_pred["touch_point"][0]), float(action_pred["touch_point"][1]))
                    tap_2_yx = (float(action_ref["touch_point"][0]), float(action_ref["touch_point"][1]))
                    # if _check_tap_actions_match(tap_1_yx=tap_1_yx, \
                    #     tap_2_yx=tap_2_yx, \
                    #     annotation_positions=annotation_positions):
                    if jnp.linalg.norm(jnp.array(tap_1_yx) - jnp.array(tap_2_yx)) <= llama_TAP_DISTANCE_THRESHOLD: 
                        output_i['match_action'] += 1
                        output_i['match_click'] += 1

            # text accuracy
            if action_ref["action_type"] == 3:
                output_i['num_text'] += 1
                if action_pred["action_type"] == 3:
                    output_i['type_text'] += 1
                    if (action_pred["typed_text"] == action_ref["typed_text"]) or (
                            action_pred["typed_text"] in action_ref["typed_text"]) or (
                            action_ref["typed_text"] in action_pred["typed_text"]):
                        output_i['match_text'] += 1
                        output_i['match_action'] += 1

            # scroll accuracy
            if action_ref["action_type"] in [1, 0, 8, 9]:
                output_i['num_scroll'] += 1
                if action_pred["action_type"] in [1, 0, 8, 9]:
                    # output_i['type_scroll'] += 1
                    if action_ref['action_type'] == action_pred['action_type']:
                        output_i['type_scroll'] += 1
                        output_i['match_scroll'] += 1
                        output_i['match_action'] += 1

            # press accuracy
            if action_ref["action_type"] in [5,6,7]:
                output_i['num_press'] += 1
                if action_pred["action_type"] in [5,6,7]:
                    # output_i['type_press'] += 1
                    if action_ref['action_type'] == action_pred['action_type']:
                        output_i['type_press'] += 1
                        output_i['match_press'] += 1
                        output_i['match_action'] += 1

            # completed
            if action_ref["action_type"] == 10:
                output_i['num_completed'] += 1
                if action_ref['action_type'] == action_pred['action_type']:
                    output_i['type_completed'] += 1
                    output_i['match_completed'] += 1
                    output_i['match_action'] += 1

        except Exception as e:
            print(e)
            output_i['num_wrong_format'] += 1
            # print(f"format wrong with {action_pred}; answer is {action_ref}")

        results.append(output_i)

    eval_dict = calculate_aitz_metrics(results)

    
    print(eval_dict)

    return motivation



if __name__=="__main__":
    
    # from llamafactory
    results = '/export3/huangdongchi/hdc_debug/results/aitz_grpo_hybrid/generated_predictions.jsonl'
    results = load_jsonl(results)
    mode = 'think'
    answer = [instance['label'] for instance in results]
    outputs = [instance['predict'] for instance in results]
    answer = gather_object(answer)
    outputs = gather_object(outputs)
    motivation = evalute_aitz_llama(answer, outputs, mode)
    save_path = '/export3/huangdongchi/hdc_debug/empty.json'
    def save_json(save_dict, save_path):
        with open(save_path, 'w') as f:
            json.dump(save_dict, f, indent=4)
    save_json(motivation, save_path)
    # first mode
    # data_dir = '/export3/huangdongchi/hdc_debug/results/try_grpo25_action-3b'
    # answer_path = os.path.join(data_dir, f'answers.json')
    # outputs_path = os.path.join(data_dir, f'outputs.json')
    # mode ='think'
    # answer = gather_object(load_json(answer_path))
    # outputs = gather_object(load_json(outputs_path))
    # evalute_aitz(answer, outputs, mode)
