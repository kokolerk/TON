

<p align="center">
<h1 align="center"> <img src="./pics/ai.png" width="70" />R1-V-GUI-Agent: Reinforcement Learning for GUI Agents</h1>
</p>
<p align="center">
  	<a href="https://img.shields.io/badge/version-v0.1.0-blue">
      <img alt="version" src="https://img.shields.io/badge/version-v0.1.0-blue?color=FF8000?color=009922" />
    </a>
  <a >
       <img alt="Status-building" src="https://img.shields.io/badge/Status-building-blue" />
  	</a>
  <a >
       <img alt="PRs-Welcome" src="https://img.shields.io/badge/PRs-Welcome-red" />
  	</a>
   	<a href="https://github.com/kokolerk/R1-V-GUI-agent/stargazers">
       <img alt="stars" src="https://img.shields.io/github/stars/kokolerk/R1-V-GUI-agent" />
  	</a>
  	<a href="https://github.com/kokolerk/R1-V-GUI-agent/network/members">
       <img alt="FORK" src="https://img.shields.io/github/forks/kokolerk/R1-V-GUI-agent?color=FF8000" />
  	</a>
    <a href="https://github.com/kokolerk/R1-V-GUI-agent/issues">
      <img alt="Issues" src="https://img.shields.io/github/issues/kokolerk/R1-V-GUI-agent?color=0088ff"/>
    </a>
    <br />
</p>

<div align="center">
<p align="center">
  <a href="#Abstract">Abstract</a>/
  <a href="#Updates">Updates</a>/
  <a href="#Resources">Resources</a>/
  <a href="#Training">Training</a>/
  <a href="#Evaluation">Evaluation</a>/
</p>
</div>

# <img src="./pics/abstract.png" width="40" />Abstract

Recently, DeepSeek R1 has attracted considerable attention, primarily owing to its innovative application of the Reinforcement Learning (RL) method known as Group Relative Policy Optimization (GRPO). This approach leverages straightforward format and accuracy rewards and has demonstrated significant efficacy in large reasoning models. Following this, numerous studies have investigated the application of this approach in multimodal large language models (LLMs), particularly in the field of visual reasoning. In this paper, we introduce the first application of GRPO in the realm of multimodal agents, particularly focusing on graphic user interface (GUI) agents. By leveraging a limited dataset of just 2k samples from AITW, we achieved a remarkable performance improvement of 7.5 after two epochs of training on Aguvis-7b. Moreover, our findings indicate that omitting the think format reward can still lead to performance gains. We also note that using the format reward on base models that lack GUI fine-tuning (Qwen2.5-VL-3b) may result in reward hacking, resulting in a rapid increase in format rewards while accuracy rewards remain at zero. Additionally, we open our code for implementing GRPO in GUI agents.

# <img src="./pics/update.png" width="40" />Updates

- 2025-03-28: We release the R1-V-GUI-Agent repo.

# <img src="./pics/resource.png" width="40" />Resources 

We will release our training data and the RL model as soon as possible. See:

[🤗 R1-GUI-Agent RL Dataset: AITW-2k](todo)

[🤗 R1-Aguvis-7b](todo)

## <img src="./pics/training.png" width="40" />Training

1. set up the envrionmets:

```
conda create -n r1-v python=3.11 
conda activate r1-v

bash setup.sh
```

2. download the training datasets:

We create our RL data from the [Aguivis-stage2](https://huggingface.co/datasets/xlangai/aguvis-stage2/tree/main) datasets [aitw-l1.json](https://huggingface.co/datasets/xlangai/aguvis-stage2/blob/main/aitw-l1.json), and you can reproduce our training dataset collection following this code:

```bash
python dataset/raw_load_data.py
```

3. download the model:

Our repo currently support two models: 

- Qwen2.5-VL, which support the mobile use and computer use functions.

- Aguvis-7b, which is fine-tuning on Qwen-2-vl-7b with the 2 stage fine-tuning.

4. train the model with GRPO

The training command is as following, you need to modify it by your local model path (QWEN_PATH), dataset path (HF_DATASET), and output save path(OUTPUT_DIR). We also support wandb to moniter the training process by setting the run name (RUN_NAME).

```bash
QWEN_PATH="Your Aguvis-7B-720P path"
HF_DATASET="Your training dataset path" 
OUTPUT_DIR="Your output dir"
if [ ! -d "$OUTPUT_DIR" ]; then
 mkdir -p "$OUTPUT_DIR"
fi
RUN_NAME="gui_agent_Aguvis-7B-720P_2000"
DS_CONFIG="./r1-v/local_scripts/zero1_no_optimizer.json"  # Note that other zero setting would meet bugs related to vllm at current stage, the current version do not support vllm


CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" torchrun \
    --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    ./r1-v/src/open_r1/grpo_gui.py \
    --output_dir ${OUTPUT_DIR} \
    --model_name_or_path ${QWEN_PATH} \
    --dataset_name ${HF_DATASET} \
    --max_prompt_length 4096 \
    --max_completion_length 2048 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-6 \
    --lr_scheduler_type "constant" \
    --logging_steps 1 \
    --bf16 true \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --min_pixels 3136 \
    --max_pixels 501760 \
    --num_train_epochs 2 \
    --run_name ${RUN_NAME} \
    --save_steps 200 \
    --save_total_limit 3 \
    --save_only_model true \
    --report_to wandb \
    --temperature 1.0 \
    --num_generations 8 \
    --deepspeed ${DS_CONFIG} \
    2>&1 | tee "${OUTPUT_DIR}/training_log.txt"

```

We save the before commands on the scripts/run_grpo_gui.sh and you can directlyrun the command:

```
python scripts/run_grpo_gui.sh
```

> [!NOTE] 
>
> 1. To reproduce the result, keep the per_device_train_batch_size to 1 for now, as there is a revealed bug about batched training. As we follow the code from R1-V, as detailed in the [reproduction report](https://github.com/Deep-Agent/R1-V/issues/4#issuecomment-2633348354) here. 
> 2. If you meet **OOM Error**, you can try reduce `--num_generations`

## <img src="./pics/eval.png" width="40" />Evaluation

We currrently give the evalutaion codes of the training datasets on [test-aitw.py](https://github.com/kokolerk/R1-V-GUI-agent/blob/main/eval/test_aitw.py) 

You can test your model by run:

```
python eval/test-aitw.py
```

The test results are as following:

|                       | Type accuracy | parameter accuracy |
| --------------------- | ------------- | ------------------ |
| Aguvis                | 65.26         | 14.64              |
| RI-V-GUI-Agent (ours) | 72.76         | 16.6               |

Please note that we evaluate the parameter accuracy only when the parameters are identical, not in terms of the bounding box. This represents the most rigorous evaluation, and we will assess it in a more reasonable manner soon.

For more comprehensive evalutation, we will relase it as soon as possible

# R1-V-GUI-Agent Team 

[Jiaqi Wang](https://github.com/kokolerk) · [Binghui Xie](https://scholar.google.com/citations?user=krUTLTkAAAAJ&hl=zh-CN)· [Dongchi Huang](https://github.com/hggforget) · [Ming Hu]() · [Xiaojun Guo](https://zero-lab-pku.github.io/personwise/guoxiaojun/)· [Qixun Wang](https://novaglow646.github.io/QixunWang-Homepage.github.io/) · [Qiguang Chen](https://lightchen233.github.io/)  · [James Cheng](https://www.cse.cuhk.edu.hk/~jcheng/)

# Acknowledgements

We sincerely thank [DeepSeek](https://github.com/deepseek-ai/DeepSeek-R1), [Open-R1](https://github.com/huggingface/open-r1), [QwenVL](https://github.com/QwenLM/Qwen2.5-VL), [Open-R1-Multimodal](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal), [R1-V](https://github.com/Deep-Agent/R1-V?tab=readme-ov-file) (our initial codebase),  Specially thanks to [Aguvis](https://github.com/xlang-ai/aguvis) for open-source training datasets and model. 

# Citation

```bib
@misc{Wang2025r1GUI,
  author       = {Wang, Jiaqi and Binghui, Xie and Dongchi, Huang and Guo, xiaojun and Qizhou, Wang and Qiguang, Chen and James, Cheng},
  title        = {R1-V-GUI-Agent: Reinforcement Learning for GUI Agents},
  howpublished = {\url{https://github.com/kokolerk/R1-V-GUI-agent}},
  note         = {Accessed: 2025-03-28},
  year         = {2025}
}
```
