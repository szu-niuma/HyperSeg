import json
import random

import torch
from PIL import Image

from .constants import IGNORE_INDEX
from .conversation_v01 import get_conv_template
from .base import BaseDataset


class ReasoningEditingDataset(BaseDataset):
    """Dataset for reasoning and editing tasks."""

    def __init__(
        self,
        dataset_path: str,
        vit_resolution: int,
        sd_resolution: int,
        clip_image_processor,
        mm_projection_length: int,
        editing_template: str,
        editing_max_length: int,
        llm_tokenizer=None,
    ):
        with open(dataset_path, "r") as file:
            self.data = json.load(file)
        self.vit_resolution = vit_resolution
        self.sd_resolution = sd_resolution
        self.clip_image_processor = clip_image_processor
        self.llm_tokenizer = llm_tokenizer
        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = "right"
        self.editing_template = editing_template
        self.editing_max_length = editing_max_length
        self.mm_projection_length = mm_projection_length
        self.conversation_templates = self.load_conversation_templates(editing_template)
        self.conv_template = get_conv_template("vicuna_v1.3")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        key = f"{index:04d}"
        sample = self.data[key]
        original_img_path = sample["origin_img_path"]
        target_img_path = sample["target_img_path"]
        instruction = random.choice(sample["instruction"])

        # Load and process images
        original_image = self.load_image(original_img_path)
        target_image = self.load_image(target_img_path)
        original_vit_image = original_image.resize((self.vit_resolution, self.vit_resolution), resample=Image.Resampling.BICUBIC)
        vi_image_tensor = self.clip_image_processor.preprocess(original_vit_image, return_tensors="pt")["pixel_values"][0]
        original_sd_image, target_sd_image = self.prepare_sd_images(original_image, target_image, self.sd_resolution)

        # Build conversation
        roles = {"Human": self.conv_template.roles[0], "GPT": self.conv_template.roles[1]}
        num_new_tokens = len(self.llm_tokenizer) - self.llm_tokenizer.vocab_size
        append_str = " ".join(f"<img_{i}>" for i in range(num_new_tokens - 3))
        edited_prompt = f"<im_start> <img_0> <im_end>{instruction}"
        conversation_template = random.choice(self.conversation_templates)
        human_message = conversation_template["Human"].replace("[cap]", f'"{edited_prompt}"')
        assistant_message = conversation_template["GPT"].replace(" [img].", f" {append_str}")
        self.conv_template.messages = []
        self.conv_template.append_message(roles["Human"], human_message)
        self.conv_template.append_message(roles["GPT"], assistant_message)
        conversation = self.conv_template.get_prompt().replace("\n", "")

        # Tokenize conversation
        input_ids_max_len = self.editing_max_length - self.mm_projection_length
        input_ids = self.llm_tokenizer(
            conversation,
            return_tensors="pt",
            padding="max_length",
            max_length=input_ids_max_len,
            truncation=True,
        ).input_ids[0]

        # Generate targets
        generated_caption_targets = input_ids.clone()
        separator = self.conv_template.sep + roles["GPT"] + ": "
        total_padding_len = generated_caption_targets.ne(self.llm_tokenizer.pad_token_id).sum().item()
        parts = conversation.split(separator)
        if len(parts) < 2:
            raise ValueError("Separator not found in conversation.")
        instruction_len = len(self.llm_tokenizer(parts[0], max_length=input_ids_max_len, truncation=True).input_ids) - 2
        generated_caption_targets[:1] = IGNORE_INDEX
        generated_caption_targets[1 : 1 + instruction_len] = IGNORE_INDEX
        generated_caption_targets[total_padding_len:] = IGNORE_INDEX

        # Attention masks
        input_attention_mask = input_ids.ne(self.llm_tokenizer.pad_token_id)
        generated_caption_encoder_attention_mask = input_ids.ge(self.llm_tokenizer.img_start_token_id)
        is_editing_task = torch.tensor(1)

        return {
            "original_img": vi_image_tensor,
            "original_img_for_vae": original_sd_image,
            "edited_img": target_sd_image,
            "input_ids": input_ids,
            "input_attention_mask": input_attention_mask,
            "generated_caption_targets": generated_caption_targets,
            "generated_caption_encoder_attention_mask": generated_caption_encoder_attention_mask,
            "is_editing_task": is_editing_task,
        }
