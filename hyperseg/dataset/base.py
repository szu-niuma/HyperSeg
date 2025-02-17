import os
from torch.utils.data import Dataset
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from .constants import IGNORE_TOKEN_ID


class BaseDataset(Dataset):
    IGNORE_TOKEN_ID = IGNORE_TOKEN_ID

    @staticmethod
    def convert_to_np(image: Image.Image, resolution: int) -> np.ndarray:
        """Convert an image to a NumPy array suitable for model input."""
        image = image.convert("RGB")
        image = image.resize((resolution, resolution), resample=Image.Resampling.BICUBIC)
        return np.array(image).transpose(2, 0, 1)

    @staticmethod
    def load_image(image_path: str) -> Image.Image:
        """Load an image from the given path."""
        # 判断文件存不存在
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        return Image.open(image_path).convert("RGB")

    @staticmethod
    def prepare_sd_images(original_image: Image.Image, target_image: Image.Image, resolution: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare images for Stable Diffusion input."""
        original_np = BaseDataset.convert_to_np(original_image, resolution)
        target_np = BaseDataset.convert_to_np(target_image, resolution)
        sd_input = np.concatenate([original_np, target_np])
        sd_input = torch.tensor(sd_input)
        sd_input = 2 * (sd_input / 255.0) - 1
        return sd_input.chunk(2)

    @staticmethod
    def load_conversation_templates(template_path: str) -> List[Dict[str, str]]:
        """Load conversation templates from a file."""
        templates = []
        with open(template_path, "r") as file:
            current_template = {}
            for line in file:
                line = line.strip()
                if line.startswith("Human: "):
                    if current_template:
                        templates.append(current_template)
                        current_template = {}
                    current_template["Human"] = line[len("Human: ") :]
                elif line.startswith("GPT: "):
                    current_template["GPT"] = line[len("GPT: ") :]
            if current_template:
                templates.append(current_template)
        return templates
