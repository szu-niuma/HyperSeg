from dataclasses import dataclass
import random
from typing import Dict, Sequence
import torch
import transformers
from torch.utils.data import DataLoader
from .ReasonEdit_dataset import ReasoningEditingDataset
from .ReasonSeg_dataset import ReasoningSegmentation_Dataset
from .dragonEdit_dataset import DragonEditDataset


def load_dataset(data_args, training_args, model_, llm_tokenizer):
    dragon_editing_dataset = DragonEditDataset(
        dataset_path=r"/home/yuyangxin/data/experiment/experiment.json",
        clip_image_processor=model_.image_processor,
        mm_projection_length=training_args.mm_projection_length,
        editing_template=data_args.editing_template,
        editing_max_length=training_args.editing_max_length,
        llm_tokenizer=llm_tokenizer,
    )
    print("Checking Dragon-Editing-Dataset train dataset...")
    print("Dragon-Editing-Dataset length:", len(dragon_editing_dataset))

    # 7. len(ReasoningEditing_train_Dataset)
    ReasoningEditing_train_Dataset = ReasoningEditingDataset(
        dataset_path=data_args.ReasoningEditingDataset_path,
        vit_resolution=data_args.ReasoningEditingDataset_resolution_ViT,
        sd_resolution=data_args.ReasoningEditingDataset_resolution_for_SD,
        clip_image_processor=model_.image_processor,
        mm_projection_length=training_args.mm_projection_length,
        editing_template=data_args.editing_template,
        editing_max_length=training_args.editing_max_length,
        llm_tokenizer=llm_tokenizer,
    )
    print("Checking Reasoning-Editing-Dataset train dataset...")
    print("Reasoning-Editing-Dataset length:", len(ReasoningEditing_train_Dataset))

    # 8. len(ReasoningSegmentation_train_Dataset)
    ReasoningSegmentation_train_Dataset = ReasoningSegmentation_Dataset(
        ReasoningSegmentationDataset_json_path=data_args.ReasoningSegmentationDataset_json_path,
        ReasoningSegmentationDataset_image_path=data_args.ReasoningSegmentationDataset_image_path,
        ReasoningSegmentationDataset_binary_mask_path=data_args.ReasoningSegmentationDataset_binary_mask_path,
        ReasoningSegmentationDataset_resolution_ViT=data_args.ReasoningSegmentationDataset_resolution_ViT,
        ReasoningSegmentationDataset_resolution_for_SD=data_args.ReasoningSegmentationDataset_resolution_for_SD,
        transparency=data_args.ReasoningSegmentation_transparency,
        CLIPImageProcessor=model_.image_processor,
        mm_projection_length=training_args.mm_projection_length,
        editing_template=data_args.editing_template,
        editing_max_length=training_args.editing_max_length,
        llm_tokenizer=llm_tokenizer,
        InstructDiffusion_color_template=data_args.InstructDiffusion_color_template,
        InstructDiffusion_seg_template=data_args.InstructDiffusion_seg_template,
    )
    merged_train_dataset = Merge_Dataset(
        ReasoningEditingDataset=ReasoningEditing_train_Dataset,
        ReasoningSegmentationDataset=ReasoningSegmentation_train_Dataset,
        DragonEditDataset=dragon_editing_dataset,
    )

    # 10. DataCollatorForLLaVADataset
    data_collator_train_dataset = DataCollatorForLLaVADataset(LLM_tokenizer=llm_tokenizer)

    # add data_collator
    data_module = dict(train_dataset=merged_train_dataset, eval_dataset=None, data_collator=data_collator_train_dataset)

    return data_module


########################################################################################################################################################################
# Merge InstructPix2Pix + MagicBrush + LLaVA + RefCOCO + GRefCOCO + COCO-stuff + reasoning segmentation + reasoning editing
class Merge_Dataset(torch.utils.data.Dataset):
    def __init__(self, ReasoningEditingDataset, ReasoningSegmentationDataset, DragonEditDataset):
        # 初始化数据集
        self.ReasoningEditingDataset = ReasoningEditingDataset
        self.ReasoningSegmentationDataset = ReasoningSegmentationDataset
        self.DragonEditDataset = DragonEditDataset
        # 数据集长度
        self.ReasoningEditingDataset_length = len(ReasoningEditingDataset)
        self.ReasoningSegmentationDataset_length = len(ReasoningSegmentationDataset)
        # 选择数据集
        self.total_len = self.ReasoningEditingDataset_length + self.ReasoningSegmentationDataset_length

    def __getitem__(self, index):
        choose_RE = random.random()
        return self.DragonEditDataset[index]
        # # 1. 选择 ReasoningEditingDataset
        # if choose_RE < 0.15:
        #     ReasoningEditingDataset_data = self.ReasoningEditingDataset[random.randint(0, self.ReasoningEditingDataset_length - 1)]
        #     return ReasoningEditingDataset_data
        # else:
        #     # 2. 选择 ReasoningSegmentationDataset
        #     ReasoningSegmentationDataset_data = self.ReasoningSegmentationDataset[
        #         random.randint(0, self.ReasoningSegmentationDataset_length - 1)
        #     ]
        #     return ReasoningSegmentationDataset_data

    def __len__(self):
        return self.total_len


@dataclass
class DataCollatorForLLaVADataset(object):
    """Collate examples for supervised fine-tuning."""

    IGNORE_INDEX = -100
    LLM_tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # LLaVA: len(instances)=batch_size & instances[i].keys() -> dict_keys(['input_ids', 'labels', 'image'])

        original_img = [instance["original_img"] for instance in instances]
        original_img = torch.stack(original_img)

        edited_img = [instance["edited_img"] for instance in instances]
        edited_img = torch.stack(edited_img)

        # for LLaVA processing
        # 1. LLM tokenizer
        input_ids = tuple([instance["input_ids"] for instance in instances])
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.LLM_tokenizer.pad_token_id)
        input_ids = input_ids[:, : self.LLM_tokenizer.model_max_length]

        input_attention_mask = [input_id.ne(self.LLM_tokenizer.pad_token_id) for input_id in input_ids]
        input_attention_mask = torch.stack(input_attention_mask)
        input_attention_mask = input_attention_mask[:, : self.LLM_tokenizer.model_max_length]

        generated_caption_targets = tuple([instance["generated_caption_targets"] for instance in instances])
        generated_caption_targets = torch.nn.utils.rnn.pad_sequence(
            generated_caption_targets, batch_first=True, padding_value=self.IGNORE_INDEX
        )
        generated_caption_targets = generated_caption_targets[:, : self.LLM_tokenizer.model_max_length]
        generated_caption_encoder_attention_mask = tuple([instance["generated_caption_encoder_attention_mask"] for instance in instances])
        generated_caption_encoder_attention_mask = torch.nn.utils.rnn.pad_sequence(
            generated_caption_encoder_attention_mask, batch_first=True, padding_value=False
        )
        generated_caption_encoder_attention_mask = generated_caption_encoder_attention_mask[:, : self.LLM_tokenizer.model_max_length]

        # return from DataCollatorForLLaVADataset
        return {
            "original_img": original_img,
            "edited_img": edited_img,
            "input_ids": input_ids,
            "input_attention_mask": input_attention_mask,
            "generated_caption_targets": generated_caption_targets,
            "generated_caption_encoder_attention_mask": generated_caption_encoder_attention_mask,
        }
