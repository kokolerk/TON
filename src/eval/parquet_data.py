import pandas as pd
from PIL import Image
import io
import json
def convert_jsonl_format(input_jsonl_file, output_jsonl_file):
    """
    Converts a JSONL file from one format to another.

    Parameters:
    input_jsonl_file (str): The path to the input JSONL file.
    output_jsonl_file (str): The path to the output JSONL file.
    """
    with open(input_jsonl_file, 'r') as infile, open(output_jsonl_file, 'w') as outfile:
        for line in infile:
            # 读取原始数据
            data = eval(line.strip())  # 使用 eval 解析字典格式
            solution = ''.join(filter(str.isdigit, data['solution']))
            # 创建新的格式
            new_format = {
                "image_path": data['image'],
                "question": data['problem'],
                "ground_truth": solution.strip('<answer> ').strip(' </answer>')  # 提取数字答案
            }

            # 写入新的 JSONL 文件
            outfile.write(json.dumps(new_format) + '\n')


def convert_parquet_to_jsonl_with_images(parquet_file, jsonl_file, image_folder):
    """
    Converts a Parquet file to a JSONL file and saves images.

    Parameters:
    parquet_file (str): The path to the Parquet file.
    jsonl_file (str): The path where the JSONL file will be saved.
    image_folder (str): The folder where images will be saved.
    """
    # 读取 Parquet 文件
    df = pd.read_parquet(parquet_file)
    # breakpoint()
    # 确保只选择所需的列
    df = df[['image', 'problem', 'solution']]
    # 筛选以 "How many" 开头的 questions，并限制为 200 个
    # filtered_df = df[df['problem'].str.startswith('How many')].head(200)
    # 写入 JSONL 文件并保存图片
    with open(jsonl_file, 'w') as f:
        for index, row in df.iterrows():
            # flag += 1
            # if flag >= 200:
            #     break
            # 保存图片
            # 假设 image_data 是一个字典，提取字节数据
            # breakpoint()
            image_data = row['image']['bytes']  # 调整为实际的字典键
            path = row['image']['path']
            image = Image.open(io.BytesIO(image_data))
            image_filename = f"{image_folder}/{path}"
            image.save(image_filename)

            # 写入 JSONL
            record = {
                'image': image_filename,
                'problem': row['problem'],
                'solution': row['solution']
            }
            f.write(f"{record}\n")


train_data = 'your parquet file path'
jsonl_data = 'your save .jsonl file path'    
image_folder = 'your save image folder path'  



convert_parquet_to_jsonl_with_images(
    train_data,
    jsonl_data,
    image_folder
)



