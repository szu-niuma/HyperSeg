import torch
import torch.distributed as distributed
import os
from enum import Enum
from tqdm import tqdm
import numpy as np

import cv2
from torch.utils.data import Dataset, DataLoader
from transformers import SiglipImageProcessor
from detectron2.data import MetadataCatalog, DatasetCatalog
from pycocotools import mask
from typing import Dict, Optional, Sequence, List
from dataclasses import dataclass, field
import torch.distributed as dist

from pathlib import Path

from hyperseg.dataset.arguments import TrainingArguments, DataArguments
from hyperseg.dataset import load_dataset
from hyperseg.utils.builder import load_train_model
from hyperseg.utils import conversation as conversation_lib
from hyperseg.eval.eval_dataset.eval_datasets import DataCollatorForCOCODatasetV2, Reason_dataset_test

import transformers
from LLMSDTrainer import LLMSDTrainer, safe_save_model_for_hf_trainer


@dataclass
class ModelArguments:

    local_rank: int = 0

    lora_enable: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"

    vision_tower: str = "google/siglip-so400m-patch14-384"
    vision_tower_mask: str = "./pretrained_model/maskformer2_swin_base_IN21k_384_bs16_50ep.pkl"

    lazy_preprocess: bool = False
    is_multimodal: bool = False
    model_path: Optional[str] = field(default="zhumj34/Mipha-3B")
    mask_config: Optional[str] = field(default="./hyperseg/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml")
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = field(default=None)

    model_map_name: str = "HyperSeg"
    version: str = "llava_phi"
    output_dir: str = "./output/reasonseg"
    segmentation: bool = True
    eval_batch_size: int = 1
    dataloader_num_workers: int = 8
    seg_task: Optional[str] = field(default="referring")

    sum_ref_pred_answer: bool = False

    # enable FVP module in paper
    enable_mgvp_seg_query: bool = field(default=True)

    visualize: bool = True

    # reason seg
    reason_path: str = "../dataset/ReasonSeg"
    reason_seg_data: str = "ReasonSeg|val"
    explanatory: float = -1

    distributed: bool = False


def init_distributed_mode(para):
    if torch.cuda.device_count() <= 1:
        para.distributed = False
        para.local_rank = 0
        para.world_size = 1

    if para.distributed:
        # Init distributed environment
        distributed.init_process_group(backend="nccl")

        local_rank = distributed.get_rank()
        world_size = distributed.get_world_size()
        torch.cuda.set_device(local_rank)
        print("I am rank %d in this world of size %d!" % (local_rank, world_size))
        para.local_rank = local_rank
        para.world_size = world_size


def do_train():
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = model_args.local_rank

    init_distributed_mode(model_args)

    tokenizer, model, image_processor, context_len = load_train_model(
        model_args.model_path,
        model_args=model_args,
        mask_config=model_args.mask_config,
        device="cuda",
    )

    device = torch.device(local_rank if torch.cuda.is_available() else "cpu")
    model.to(dtype=torch.float32, device=device)

    model_args.image_processor = image_processor
    model_args.is_multimodal = True
    conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    clip_image_processor = SiglipImageProcessor.from_pretrained(model_args.vision_tower)

    data_collator = DataCollatorForCOCODatasetV2(tokenizer=tokenizer, clip_image_processor=clip_image_processor)

    # load dataset
    data_module_ = load_dataset(data_args, training_args, model, tokenizer)
    print("数据集载入成功")

    # trainer for pretrained checkpoint or not
    print("开始训练")
    trainer = LLMSDTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module_)
    trainer.train()

    # 保存
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    do_train()
