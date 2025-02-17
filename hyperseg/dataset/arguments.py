from dataclasses import dataclass, field
from typing import Optional

import transformers


#############################################################################################################################
@dataclass
class ModelArguments:
    # LLM -> Vicuna
    model_name_or_path: Optional[str] = field(default="lmsys/vicuna-7b-v1.5")

    # config for added token
    num_new_tokens: int = 32

    # config for pretrained clip text model
    clip_path: str = "openai/clip-vit-large-patch14"
    clip_max_length: int = 77

    # config for qformer that link to sd
    sd_qformer_num_layers: int = 6
    sd_qformer_cross_attention_freq: int = 2

    # pretrained model
    LLaVA_model_path: str = "liuhaotian/llava-v1.6-vicuna-7b"
    LLaVA_00001: str = f"{LLaVA_model_path}/pytorch_model-00001-of-00002.bin"
    LLaVA_00002: str = f"{LLaVA_model_path}/pytorch_model-00001-of-00002.bin"

    # pretrain_sd_qformer: str = None
    # pretrained_LLaMA: str = "./LLMSD_exp/stage2_MLLMSD_7b/LLM-5000/adapter_model.bin"
    # pretrained_model: str = "./LLMSD_exp/stage2_MLLMSD_7b/embeddings_qformer/checkpoint-5000.bin"
    # pretrained_unet: str = "./LLMSD_exp/stage2_MLLMSD_7b/unet-5000/adapter_model.bin"


#############################################################################################################################
@dataclass
class DataArguments:
    # ReasoningEditing dataset
    ReasoningEditingDataset_path: str = "./resources/reason_edit/SyntheticData_info_new.json"
    ReasoningEditingDataset_resolution_ViT: int = 224
    ReasoningEditingDataset_resolution_for_SD: int = 336

    # ReasoningSegmentation dataset
    ReasoningSegmentationDataset_json_path: str = "./resources/reason_seg/train"
    ReasoningSegmentationDataset_image_path: str = "./resources/reason_seg/train"
    ReasoningSegmentationDataset_binary_mask_path: str = "./resources/reason_seg/vis"
    ReasoningSegmentationDataset_resolution_ViT: int = 224
    ReasoningSegmentationDataset_resolution_for_SD: int = 336
    ReasoningSegmentation_transparency: float = 0.5

    # InstructDiffusion color and segmentation templates
    InstructDiffusion_color_template: str = "./resources/data/LLMSD_InstructDiffusion_color.txt"
    InstructDiffusion_seg_template: str = "./resources/data/LLMSD_InstructDiffusion_seg.txt"

    # Instruction tuning template
    editing_template: str = "./resources/data/ConversationTemplateEditing_use.txt"


#############################################################################################################################
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")

    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")

    # max_length -> editing=280, LLaVA=2048
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )

    double_quant: bool = field(default=True, metadata={"help": "Compress the quantization statistics through double quantization."})
    quant_type: str = field(default="nf4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    bits: int = field(default=16, metadata={"help": "How many bits to use."})

    # lora相关配置
    lora_enable: bool = True
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"

    editing_max_length: int = 512
    mm_projection_length: int = 256

    # training settings
    llm_loss_weight: float = 1.0
    diffusion_loss_weight: float = 1.0
    gradient_checkpointing: bool = True

    output_dir: str = "./checkpoints"


@dataclass
class TestingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")

    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")

    # max_length -> editing=280, LLaVA=2048
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )

    double_quant: bool = field(default=True, metadata={"help": "Compress the quantization statistics through double quantization."})
    quant_type: str = field(default="nf4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    bits: int = field(default=16, metadata={"help": "How many bits to use."})

    # lora相关配置
    lora_enable: bool = True
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"

    editing_max_length: int = 512
    mm_projection_length: int = 256

    # training settings
    llm_loss_weight: float = 1.0
    diffusion_loss_weight: float = 1.0
    gradient_checkpointing: bool = True

    output_dir: str = "./checkpoints"
