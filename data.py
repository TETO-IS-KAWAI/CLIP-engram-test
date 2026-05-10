import os
from typing import Optional, Callable

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
from datasets import load_dataset
from transformers import AutoTokenizer


class clip_dataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        ds=None,
        tokenizer_name: str = "openai/clip-vit-base-patch32",
        max_length: int = 77,
        image_size: int = 224,
        image_transform: Optional[Callable] = None,
        nsfw_filter: bool = False,
        min_similarity: float = 0.28,
    ):

        if nsfw_filter:
            ds = ds.filter(lambda x: str(x.get("NSFW", "UNLIKELY")).upper() == "UNLIKELY")
        if min_similarity > 0:
            ds = ds.filter(lambda x: (x.get("similarity") or 0.0) >= min_similarity)

        self.ds = ds
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.transform = image_transform or self._default_transform(image_size)

    @staticmethod
    def _default_transform(image_size: int) -> Callable:
        return T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(
                mean=[0.48145466, 0.4578275,  0.40821073],
                std= [0.26862954, 0.26130258, 0.27577711],
            ),
        ])

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.ds[idx]

        # 이미지
        image = row.get("image")
        if isinstance(image, Image.Image):
            if image.mode != "RGB":
                image = image.convert("RGB")
        else:
            image = Image.new("RGB", (224, 224))  # 이미지 없는 샘플 fallback

        image_tensor: torch.Tensor = self.transform(image)   # [3, H, W]

        # 텍스트 → token_ids
        text = str(row.get("TEXT") or "")
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        token_ids: torch.Tensor = enc["input_ids"].squeeze(0)  # [seq_len]

        return image_tensor, token_ids