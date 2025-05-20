import pdb
### AITZ
_AITZ_SYSTEM = """You are an assistant trained to navigate the mobile phone. 
Given a task instruction, a screen observation, and an action history sequence, 
output the next action and wait for the next observation. 
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
"""

_THINK_SYSTEM = "**Please first thinks about the reasoning process in the mind and then provides the user with the action. The reasoning " \
    "process and answer are enclosed within <think> </think> and <action> </action> tags, respectively, i.e., " \
    "<think> reasoning process here </think><action> action here </action>**"

_MIX_SYSTEM = "**Please first evaluates whether the question is simple enough to answer directly. " \
    "If simple, your output is <think>\n\n</think><action>action here</action>. " \
    "If the question is difficult, you need to first about the reasoning process in the mind and then provides the user with the action. " \
    "The output are formatted as <think> reasoning process here </think><action> action here </action>." \
    "You is encouraged to use <think>\n\n</think> while maintaining accuracy." \


_ACTION_SYSTEM = "**Please provides the user with the action. The " \
    "answer are enclosed within <action> </action> tags, i.e., " \
    "<action> action here </action>**"

# where screen_desc, action_result are optional
# _AITZ_USER = """{system}
# Task: {task}
# Observation: <|image_1|>
# {coat_cap}
# Action History: {action_history}
# {coat_res}
# What is the next action?
# {coat_think}
# """

def get_history(action_history, image_history, iterate, mode='tv'):
    """
    Iterate through the action history and image history to create a formatted string.
    """
    assert len(action_history) == len(image_history)
    iterate = min(len(action_history), len(image_history), iterate)
    if iterate == 0:
        return []
    history=[{"type": "text", "text": "Action history:"}]
    for i, (action, image) in enumerate(zip(action_history[-iterate:], image_history[-iterate:])):
        
        if mode == 'tv':
            history.append({"type": "text", "text": f'Step{i}: {action}'})
            history.append({"type": "image", "image": f'{image}'})
        elif mode == 'vt':
            history.append({"type": "image", "image": f'{image}'})
            history.append({"type": "text", "text": f'Step{i}: {action}'})
        if mode == 'tt':
            history.append({"type": "text", "text": f'Step{i}: {action}'})
        elif mode =='vv':
            history.append({"type": "image", "image": f'{image}'})
        
        # tmp_prev = '; '.join(action_prefix)
        # tmp_post = '; '.join(action_history)
        # tmp = tmp_prev + '; ' + tmp_post if tmp_prev != '' else tmp_post
    return history

def aitz_to_openai_qwenvl(task, action_history, image_history,image_path, answer_dict=None, think=None,
                    mode='init'):
    transformed_data = []
    user_content = []
    user_content.append({"type": "text", "text": _AITZ_SYSTEM})
    if mode == 'think':
        user_content.append({"type": "text", "text": _THINK_SYSTEM})
    elif mode == 'action':
        user_content.append({"type": "text", "text": _ACTION_SYSTEM })
    else:
        user_content.append({"type": "text", "text": _MIX_SYSTEM})
    user_content.append({"type": "text", "text": f"Task: {task}"})
    user_content.append({"type": "image","image": f"{image_path}"})
    history= get_history(action_history, image_history, iterate=4, mode='tt')
    if history is not None:
        for h in history:
            user_content.append(h)
    user_content.append({"type": "text", "text": f"\n**Next action:**"})
    # if mode == 'try_think':
    #     user_content.append({'type': "text", "text": f"<think>{think}</think>"})
    transformed_data.append(
                {
                    "role": "user",
                    "content": user_content,
                },
            )
    return transformed_data