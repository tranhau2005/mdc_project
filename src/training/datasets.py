from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


class PromptClassificationDataset(Dataset):
    def __init__(self, prompts: list[str], labels: list[int]):
        self.prompts = prompts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict:
        return {
            "text": self.prompts[idx],
            "labels": int(self.labels[idx]),
        }


@dataclass
class PromptDataCollator:
    tokenizer: any
    max_length: int = 2048

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        texts = [f["text"] for f in features]
        labels = [int(f["labels"]) for f in features]
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch
