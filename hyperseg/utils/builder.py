#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings
import shutil
import copy

# from peft import LoraConfig, get_peft_model

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch

from hyperseg.model import *
from hyperseg.eval.eval_dataset.eval_datasets import get_mask_config
from hyperseg.model.language_model.llava_phi import HyperSeg
from hyperseg.model.mipha.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from hyperseg.model.mipha.model.language_model.mipha_phi import MiphaPhiForCausalLM, MiphaPhiModel

import os
import warnings
import shutil

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig,
    CLIPImageProcessor,
    SiglipImageProcessor,
    BitImageProcessor,
)
import torch


def find_linear_layers(model, lora_target_modules=["q_proj", "v_proj"], train_module_list=[]):
    cur_train_module_list = copy.deepcopy(train_module_list)
    cur_train_module_list.extend(["vision_tower", "vision_tower_mask"])
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if (
            isinstance(module, cls)
            and all([x not in name for x in cur_train_module_list])
            and any([x in name for x in lora_target_modules])
        ):
            # names = name.split('.')
            # lora_module_names.add(names[0] if len(names) == 1 else names[-1])
            lora_module_names.add(name)

    return sorted(list(lora_module_names))


def load_pretrained_model(
    model_path,
    model_args,
    mask_config="./mask_config/maskformer2_swin_base_384_bs16_50ep.yaml",
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    device="cuda",
):

    kwargs = {"device_map": "cpu"}

    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4"
        )
    else:
        kwargs["torch_dtype"] = torch.float16

    mask_cfg = get_mask_config(mask_config)
    mask_cfg.MODEL.MASK_FORMER.SEG_TASK = model_args.seg_task if hasattr(model_args, "seg_task") else "instance"

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    # model = HyperSeg.from_pretrained(model_path, mask_decoder_cfg=mask_cfg, cur_vocab_size=len(tokenizer), **kwargs)
    config = AutoConfig.from_pretrained(model_path)
    # model = HyperSeg(config=config, mask_decoder_cfg=mask_cfg, cur_vocab_size=len(tokenizer), **kwargs)
    # model.use_temporal_query = model_args.use_temporal_query if hasattr(model_args, "use_temporal_query") else False

    # mask2former_ckpt = model_args.vision_tower_mask
    # model.initial_mask_module(mask2former_ckpt, model_args)

    # model.get_model().initialize_vision_modules(model_args)

    # vision_tower = model.get_model().get_vision_tower_mask()
    # vision_tower.to(device=device)
    # image_processor = vision_tower.image_processor
    # model.resize_token_embeddings(len(tokenizer))

    # if hasattr(model.config, "max_sequence_length"):
    #     context_len = model.config.max_sequence_length
    # else:
    #     context_len = 2048

    model = None
    image_processor = None
    context_len = 2048
    return tokenizer, model, image_processor, context_len


def load_train_model(
    model_path,
    model_args,
    mask_config="./mask_config/maskformer2_swin_base_384_bs16_50ep.yaml",
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    device="cuda",
    kwargs={"device_map": "cpu"},
):
    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.float16
    print("load model from model_path: ", model_path)

    mask_cfg = get_mask_config(mask_config)
    mask_cfg.MODEL.MASK_FORMER.SEG_TASK = model_args.seg_task if hasattr(model_args, "seg_task") else "instance"

    # tokenizer的内容
    llm_tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if "Mipha" in model_path:
        print("load Mipha-Phi MSLM!!!")
        config = AutoConfig.from_pretrained(model_path)
        model = MiphaPhiForCausalLM.from_pretrained(model_path, config=config, use_safetensors=True, **kwargs).to("cuda")
        # model = HyperSeg(config, mask_decoder_cfg=mask_cfg, cur_vocab_size=len(llm_tokenizer), **kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_path}")

    if "clip" in config.vision_config["vision_tower"]["vision_model_name_or_path"]:
        image_processor = CLIPImageProcessor.from_pretrained(model_path)
    elif "siglip" in config.vision_config["vision_tower"]["vision_model_name_or_path"]:
        image_processor = SiglipImageProcessor.from_pretrained(model_path)
    elif "dinov2" in config.vision_config["vision_tower"]["vision_model_name_or_path"]:
        image_processor = BitImageProcessor.from_pretrained(model_path)
    else:
        return NotImplementedError

    if "Mipha" or "gemma" in model_path:
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        # TODO: the tokenizer length of phi-2 is 50295, but the output class of lm_head is 51200
        # DEFAULT_IMAGE_PATCH_TOKEN：用于表示图像补丁
        if mm_use_im_patch_token:
            llm_tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        # DEFAULT_IM_START_TOKEN和DEFAULT_IM_END_TOKEN：用于标记图像内容的开始和结束
        if mm_use_im_start_end:
            llm_tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            # model.resize_token_embeddings(len(tokenizer))
    else:
        raise ValueError(f"Unsupported model name: {model_path}")

    context_len = getattr(model.config, "max_sequence_length", 2048)
    model.to(device=device)

    print("model loaded to device: ", device)
    print(kwargs)

    return llm_tokenizer, model, image_processor, context_len
