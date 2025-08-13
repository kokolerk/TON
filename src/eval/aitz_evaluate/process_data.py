import json
import os
import ast
import imagesize

actions_dict = {
    0: "SCROLL DOWN",
    1: "SCROLL UP",
    3: "TYPE",
    4: "CLICK",
    5: "PRESS BACK",
    6: "PRESS HOME",
    7: "PRESS ENTER",
    8: "SCROLL LEFT",
    9: "SCROLL RIGHT",
    10: "STATUS_TASK_COMPLETE",
    11: "STATUS_TASK_IMPOSSIBLE"
}

action2id_dict = {value: key for key, value in actions_dict.items()}

def get_action(step_data):
    '''
    Here is the action space:
    1. `CLICK`: Click on an element, value is not applicable and the position [x,y] is required. 
    2. `TYPE`: Type a string into an element, value is a string to type and the position is not applicable.
    3. `SCROLL UP`: Scroll up for the screen.
    4. `SCROLL DOWN`: Scroll down for the screen.
    5. `SCROLL LEFT`: Scroll left for the screen.
    6. `SCROLL RIGHT`: Scroll right for the screen.
    7. `PRESS BACK`: Press for returning to the previous step, value and position are not applicable.
    8. `PRESS HOME`: Press for returning to the home screen, value and position are not applicable.
    9. `PRESS ENTER`: Press for submitting the input content, value and position are not applicable.
    10. `STATUS TASK COMPLETE`: Indicate the task is completed, value and position are not applicable.

    Format the action as a dictionary with the following keys:
    {'action': 'ACTION_TYPE', 'value': 'element', 'position': [x,y]}

    If value or position is not applicable, set it as `None`.
    Position represents the relative coordinates on the screenshot and should be scaled to a range of 0-1.
    '''
    action_type_id = int(step_data['action_type_id'])
    action_type_text = actions_dict.get(action_type_id)
    
    # distinguish scroll and click
        
    click_point = None
    type_text = None
    if action_type_id == 4:
        words = step_data['action_desc'].split(" ")
        if 'scroll' in words[0].lower():
            direction = words[1].lower()
            if direction == 'up':
                action_type_id = 1
            elif direction == 'down':
                action_type_id = 0
            elif direction == 'left':
                action_type_id = 8
            elif direction == 'right':
                action_type_id = 9
            action_type_text = actions_dict.get(action_type_id)
        else: # click
            click_point = step_data['lift_yx']
            # 保留两位小数
            click_point = ast.literal_eval(click_point)
            if click_point[click_point[0] > 0]:
                click_point = [round(click_point[0], 2), round(click_point[1], 2)]
    elif action_type_id == 3:
        type_text = step_data['action_text']

    answer = {'action': action_type_text.upper(), 'value': type_text, 'position': click_point}
    return answer  

def load_json(file_path):
    """Load JSON data from a file."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def load_json_from_directory(directory_path):
    """Load all JSON files from a directory."""
    import os
    json_data = []
    # Iterate through all files in the directory
    for dirname in os.listdir(directory_path):
        # print(dirname)
        for filename in os.listdir(os.path.join(directory_path, dirname)):
            if filename.endswith('.json'):
                # print(filename)
                file_path = os.path.join(directory_path, dirname, filename)
                # print(file_path)
                data = load_json(file_path)
                json_data.append(data)
    # breakpoint()
    return json_data

def save_data_to_json(data, file_path):
    """Save data to a JSON file."""
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def reprocess_aitz_data_from_json(file_path):
    datasets = load_json(file_path)
    new_datasets = []
    for episode in datasets: # episode
        new_episode = []
        history_actions = []
        history_images = []
        for data in episode: # data
            new_data = {}
            new_data['ui_positions'] = ast.literal_eval(data['ui_positions'])
            new_data['ui_text'] = ast.literal_eval(data['ui_text'])
            new_data['ui_types'] = ast.literal_eval(data['ui_types']) 
            new_data['episode_id'] = data['episode_id'] 
            new_data['step_id']= data['step_id']
            new_data['episode_length'] = data['episode_length']
            
            # task
            new_data['image_path'] = data['image_path']
            new_data['instruction'] = data['instruction']
            
            # answer
            new_data['action_type_id'] = data['result_action_type']
            new_data['action_text'] = data['result_action_text']
            new_data['action_desc'] = data["coat_action_desc"]
            new_data['touch_yx'] = ast.literal_eval(data['result_touch_yx'])
            if new_data['touch_yx'][0] > 0:
                w,h = imagesize.get(os.path.join('/export3/huangdongchi/hdc_debug/aitz_data/android_in_the_zoo/train',data['image_path']))
                rel_y, rel_x = ast.literal_eval(data['result_touch_yx'])
                abs_y, abs_x = int(rel_y*h), int(rel_x*w)
                gt_action_yx = [abs_y, abs_x]
                new_data['touch_yx'] = gt_action_yx
            
            new_data['lift_yx'] = ast.literal_eval(data['result_lift_yx'])
            if new_data['lift_yx'][0] > 0:
                w,h = imagesize.get(os.path.join('/export3/huangdongchi/hdc_debug/aitz_data/android_in_the_zoo/train',data['image_path']))
                rel_y, rel_x = ast.literal_eval(data['result_lift_yx'])
                abs_y, abs_x = int(rel_y*h), int(rel_x*w)
                gt_action_yx = [abs_y, abs_x]
                new_data['lift_yx'] = gt_action_yx
            new_data['action_think'] = data['coat_action_think']
            
            # process history including action and image list
            new_data['action_history'] = history_actions.copy()
            new_data['image_history'] = history_images.copy()
            history_actions.append(get_action(new_data))
            history_images.append(data['image_path'])
            
            # append new data
            new_episode.append(new_data)
        new_datasets.append(new_episode)
        
    return new_datasets
   

def normalize_ui_position(ui_positions, image_path):
    """
    Normalize the UI positions based on the image size.
    """
    width, height = imagesize.get(os.path.join('/export3/huangdongchi/hdc_debug/other_datasets/aitz_data/android_in_the_zoo/test', image_path))
    ui_positions = ast.literal_eval(ui_positions)
    # [y, x, width, height]
    normalized_positions = []
    for position in ui_positions:
        # breakpoint()
        n_y = float(position[0] / height)
        n_x = float(position[1] / width)
        n_y_w = float(position[2] / height)
        n_x_h = float(position[3] / width)
        normalized_positions.append([n_y, n_x, n_y_w, n_x_h])
    return normalized_positions

def reprocess_aitz_data_from_json_string_01(file_path):
    datasets = load_json(file_path)
    new_datasets = []
    for episode in datasets: # episode
        new_episode = []
        history_actions = []
        history_images = []
        for data in episode: # data
            new_data = {}
             
            new_data['episode_id'] = data['episode_id'] 
            new_data['step_id']= data['step_id']
            new_data['episode_length'] = data['episode_length']
            
            # task
            new_data['image_path'] = data['image_path']
            new_data['instruction'] = data['instruction']
            
            new_data['ui_positions'] = str(normalize_ui_position(data['ui_positions'], data['image_path']))
            new_data['ui_text'] = data['ui_text']
            new_data['ui_types'] = data['ui_types']
            # answer
            new_data['action_type_id'] = data['result_action_type']
            new_data['action_text'] = data['result_action_text']
            new_data['action_desc'] = data["coat_action_desc"]
            new_data['touch_yx'] = data['result_touch_yx']
            
            new_data['lift_yx'] = data['result_lift_yx']
            new_data['action_think'] = data['coat_action_think']
            new_data['solution'] = str(get_action(new_data))
            # process history including action and image list
            new_data['action_history'] = history_actions.copy()
            new_data['image_history'] = history_images.copy()
            history_actions.append(str(get_action(new_data)))
            history_images.append(data['image_path'])
            
            # append new data
            new_episode.append(new_data)
        new_datasets.append(new_episode)
        
    return new_datasets
  
  
def load_jsonl(data_path):
    data = []
    with open(data_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data


if __name__ == "__main__":
    data_path = 'dataset/android_in_the_zoo/train/general' 
    general_aitz = load_json_from_directory(data_path)
    print(len(general_aitz))
    save_path = 'preprocess save directory'
    save_data_to_json(general_aitz, os.path.join(save_path, 'aitz_reprocess.json'))
    
    '''
    attention to the train/train splits
    '''
    # load datasets
    # save_data_to_json(general_aitz, os.path.join(save_path, 'train_general_aitz.json'))
    
    # process format
    new_datasets = reprocess_aitz_data_from_json_string_01(os.path.join(save_path, 'aitz_preprocess.json'))
    save_data_to_json(new_datasets, os.path.join(save_path, 'aitz_reprocess2.json'))
    
    