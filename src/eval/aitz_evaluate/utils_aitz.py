import re
import numpy as np
import jax
import jax.numpy as jnp
import numpy as np
from action_matching import _yx_in_bounding_boxes, _resize_annotation_bounding_boxes
from process_aitz import action2id_dict
import ast
import json
_TAP_DISTANCE_THRESHOLD = 0.2  # 0.14 Fraction of the screen
ANNOTATION_WIDTH_AUGMENT_FRACTION = 1.4
ANNOTATION_HEIGHT_AUGMENT_FRACTION = 1.4

# Interval determining if an action is a tap or a swipe.
_SWIPE_DISTANCE_THRESHOLD = 0.04


def is_tap_action(normalized_start_yx,
                  normalized_end_yx):
  distance = jnp.linalg.norm(
      jnp.array(normalized_start_yx) - jnp.array(normalized_end_yx))
  return distance <= _SWIPE_DISTANCE_THRESHOLD


def _check_tap_actions_match(
    tap_1_yx,
    tap_2_yx,
    annotation_positions,
    matching_tap_distance_threshold_screen_percentage=_TAP_DISTANCE_THRESHOLD,
    annotation_width_augment_fraction=ANNOTATION_WIDTH_AUGMENT_FRACTION,
    annotation_height_augment_fraction=ANNOTATION_HEIGHT_AUGMENT_FRACTION,
):
  """Determines if two tap actions are the same."""
  resized_annotation_positions = _resize_annotation_bounding_boxes(
      annotation_positions,
      annotation_width_augment_fraction,
      annotation_height_augment_fraction,
  )

  # Check if the ground truth tap action falls in an annotation's bounding box.
  tap1_in_box = _yx_in_bounding_boxes(tap_1_yx, resized_annotation_positions)
  tap2_in_box = _yx_in_bounding_boxes(tap_2_yx, resized_annotation_positions)
  both_in_box = jnp.max(tap1_in_box & tap2_in_box)
#   print("both in box", both_in_box)
  # If the ground-truth tap action falls outside any of the annotation
  # bounding boxes or one of the actions is inside a bounding box and the other
  # is outside bounding box or vice versa, compare the points using Euclidean
  # distance.
#   print("norm distance",jnp.linalg.norm(jnp.array(tap_1_yx) - jnp.array(tap_2_yx)))
  within_threshold = (
      jnp.linalg.norm(jnp.array(tap_1_yx) - jnp.array(tap_2_yx))
      <= matching_tap_distance_threshold_screen_percentage
  )
#   print("within threshold", within_threshold)
#   print("results", jnp.logical_or(both_in_box, within_threshold))
  return jnp.logical_or(both_in_box, within_threshold)


def exact_action(data_str):
    # 正则表达式模式
    # pattern = r"<think>.*?</think>\s*<action>(.*?)</action>"
    pattern = r"<action>(.*?)</action>"

    # breakpoint()
    # 使用 re.search 提取 <action> 标签内的内容
    match = re.search(pattern, data_str, re.DOTALL)
    if match:
        action_content = match.group(1)  # 提取第一个括号内的内容
        # print("Extracted action content:", action_content)
    else:
        raise ValueError("No match found in think action.")
    return action_content

def exact_action_init(data_str):
    # 正则表达式模式
    pattern = r"<action>(.*?)</action>"

    # 使用 re.search 提取 <action> 标签内的内容
    match = re.search(pattern, data_str, re.DOTALL)
    if match:
        action_content = match.group(1)  # 提取第一个括号内的内容
        # print("Extracted action content:", action_content)
    else:
        raise ValueError("No match found in think action.")
    return action_content

def solution2json(data_str, mode):
    # 将字符串转换为字典
    # value里面也有单引号
    if mode == 'think':
        data_str = exact_action(data_str)
    # data_dict = eval(data_str.replace("'", "\""))  # 替换单引号为双引号
    data_dict = exact_action_init(data_str)
    if data_dict.get("position") is not None: 
        if isinstance(data_dict.get("position"), str):
            position=json.loads(data_dict.get("position"))
        elif isinstance(data_dict.get("position"), list):
            position = data_dict.get("position")  
        else: 
            raise ValueError(f"Invalid position format: {data_dict.get('position')}")
        # position=json.loads(data_dict.get("position").replace("'", "\""))
    else:
        position= None
    
    json_data = {
        "action": data_dict.get("action"),
        "value": data_dict.get("value"),
        "position": position  # 将位置字符串转换为列表
    }
    print(json_data)
    return json_data

def answer2json(data_str, mode):
    # 将字符串转换为字典
    # value里面也有单引号
    if mode == 'think':
        data_str = exact_action(data_str)
    else: 
        data_str = exact_action_init(data_str)
    # data_dict = eval(data_str.replace("'", "\""))  # 替换单引号为双引号
    data_dict = eval(data_str)
    if data_dict.get("position") is not None: 
        if isinstance(data_dict.get("position"), str):
            position=json.loads(data_dict.get("position"))
        elif isinstance(data_dict.get("position"), list):
            position = data_dict.get("position")  
        else: 
            raise ValueError(f"Invalid position format: {data_dict.get('position')}")
        # position=json.loads(data_dict.get("position").replace("'", "\""))
    else:
        position= None
    
    json_data = {
        "action": data_dict.get("action"),
        "value": data_dict.get("value"),
        "position": position  # 将位置字符串转换为列表
    }
    print(json_data)
    return json_data

def pred2json(prediction):
    if isinstance(prediction, dict):
        prediction = str(prediction)
    prediction = prediction.replace('\"', '\'')
    pattern = r"'action':\s*'(.*?)',\s*'value':\s*(None|'(.*?)'),\s*'position':\s*(None|\[([0-9.]+),\s*([0-9.]+)\])"
    match = re.search(pattern, prediction)
    try:
        action = match.group(1)
        value = match.group(2)
        if value is None:   
            value = None
        else:
            value = match.group(3)

        position_group = match.group(4)
        if position_group is None:
            position = None
        else:
            position_x = float(match.group(5))
            position_y = float(match.group(6))
            position = [position_x, position_y]

        return {
            "action": action,
            "value": value,
            "position": position
        }
    except:
        raise ValueError(f"Input string '{prediction}' doesn't match the expected format")

def pred2json_post(step_data):
    # {'action': 'ACTION_TYPE', 'value': 'element', 'position': [x,y]}
    action_type = step_data["action"].upper()
    # align with https://github.com/njucckevin/SeeClick/blob/main/agent_tasks/aitw_process.py#L36
    action2id = action2id_dict
    action_id = action2id[action_type]
        
    # click
    if action_id == 4:
        touch_point = step_data["position"]
        lift_point = step_data["position"]
        typed_text = ""
    # scroll
    elif action_id == 0:
        # action_type_new = 4
        touch_point = [0.5, 0.8]
        lift_point = [0.5, 0.2]
        typed_text = ""
    elif action_id == 1:
        # action_type_new = 4
        touch_point = [0.5, 0.2]
        lift_point = [0.5, 0.8]
        typed_text = ""
    elif action_id == 8:
        # action_type_new = 4
        touch_point = [0.2, 0.5]
        lift_point = [0.8, 0.5]
        typed_text = ""
    elif action_id == 9:
        # action_type_new = 4
        touch_point = [0.8, 0.5]
        lift_point = [0.2, 0.5]
        typed_text = ""
    # press; complete; type
    # (select) by aitw
    else:
        # action_type_new = action_id
        touch_point = [-1.0, -1.0]
        lift_point = [-1.0, -1.0]
        typed_text = ""
        if action_id == 3:
            typed_text = step_data["value"]

    action = {"action_type": action_id, "touch_point": touch_point, 
                "lift_point": lift_point, "typed_text": typed_text}

    action["touch_point"] = [action["touch_point"][1], action["touch_point"][0]]
    action["lift_point"] = [action["lift_point"][1], action["lift_point"][0]]
    if action["typed_text"] is not None:
        action["typed_text"] = action["typed_text"].lower()
    return action