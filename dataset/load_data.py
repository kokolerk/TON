import pathlib
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl
from PIL import ImageFile
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
)

from aguvis.constants import IGNORE_INDEX
from aguvis.dataset import LazySupervisedDataset
from aguvis.trainer import AGUVISTrainer, rank0_print, safe_save_model_for_hf_trainer

apply_liger_kernel_to_qwen2_vl()

torch.multiprocessing.set_sharing_strategy("file_system")

ImageFile.LOAD_TRUNCATED_IMAGES = True
local_rank = None


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None)
    early_mix_text: bool = False
    image_folder: Optional[str] = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    group_by_modality_length: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)
    verbose_logging: bool = field(default=False)
    attn_implementation: str = field(
        default="flash_attention_2", metadata={"help": "Use transformers attention implementation."}
    )
    freeze_visual_encoder: bool = field(default=False)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [_input_ids[: self.tokenizer.model_max_length] for _input_ids in input_ids]
        labels = [_labels[: self.tokenizer.model_max_length] for _labels in labels]
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0  # This gets the best result. Don't know why.
        input_ids = self.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = self.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        batch = {
            "input_ids": input_ids,
            "labels": labels.long() if labels.dtype == torch.int32 else labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
        }

        if "pixel_values" in instances[0]:
            batch["pixel_values"] = torch.concat([instance["pixel_values"] for instance in instances], dim=0)
            batch["image_grid_thw"] = torch.concat([instance["image_grid_thw"] for instance in instances], dim=0)

        return batch


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, processor: transformers.ProcessorMixin, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer, processor=processor, data_path=data_args.data_path, data_args=data_args
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return {"train_dataset": train_dataset, "eval_dataset": None, "data_collator": data_collator}


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.verbose_logging:
        rank0_print("Inspecting experiment hyperparameters:\n")
        rank0_print(f"model_args = {vars(model_args)}\n\n")
        rank0_print(f"data_args = {vars(data_args)}\n\n")
        rank0_print(f"training_args = {vars(training_args)}\n\n")
        # rank0_print(f"evaluation_args = {vars(evaluation_args)}\n\n")

    local_rank = training_args.local_rank

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        low_cpu_mem_usage=False,
    )
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
    )
    additional_special_tokens = tokenizer.additional_special_tokens
    if "<|recipient|>" not in additional_special_tokens:
        additional_special_tokens = additional_special_tokens + ["<|recipient|>"]
    if "<|diff_marker|>" not in additional_special_tokens:
        additional_special_tokens = additional_special_tokens + ["<|diff_marker|>"]
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict={"additional_special_tokens": additional_special_tokens},
        tokenizer=tokenizer,
        model=model,
    )

    # print(f"Training args: {training_args}")
    # if training_args.freeze_visual_encoder:
    #     for p in model.visual.parameters():
    #         p.requires_grad = False
    #     for p in model.visual.merger.parameters():
    #         p.requires_grad = True

    min_pixels = 256 * 28 * 28
    # max_pixels = 31 * 18 * 28 * 28  # 480p
    max_pixels = 46 * 26 * 28 * 28  # 720p
    # max_pixels = 69 * 39 * 28 * 28  # 1080p
    data_args.processor = Qwen2VLProcessor.from_pretrained(
        model_args.model_name_or_path, min_pixels=min_pixels, max_pixels=max_pixels
    )
    data_args.processor.tokenizer = tokenizer

    data_module = make_supervised_data_module(tokenizer=tokenizer, processor=data_args.processor, data_args=data_args)

    # trainer = AGUVISTrainer(
    #     model=model,
    #     processing_class=data_args.processor,
    #     args=training_args,
    #     **data_module,
    # )

    # if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
    #     trainer.train(resume_from_checkpoint=True)
    # else:
    #     trainer.train()
    # trainer.save_state()

    # model.config.use_cache = True

    # safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    # rank0_print(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    train()