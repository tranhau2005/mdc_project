from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import torch
from transformers import Trainer


class ModelEMA:
    def __init__(self, model: torch.nn.Module, beta: float = 0.9995, update_every: int = 1, update_after_step: int = 50):
        self.beta = beta
        self.update_every = update_every
        self.update_after_step = update_after_step
        self.step = 0
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.step += 1
        if self.step < self.update_after_step:
            self._copy_from(model)
            return
        if self.step % self.update_every != 0:
            return
        ema_state = self.ema_model.state_dict()
        model_state = model.state_dict()
        for key, ema_tensor in ema_state.items():
            model_tensor = model_state[key].detach()
            if not torch.is_floating_point(ema_tensor):
                ema_tensor.copy_(model_tensor)
            else:
                ema_tensor.mul_(self.beta).add_(model_tensor, alpha=(1.0 - self.beta))

    @torch.no_grad()
    def _copy_from(self, model: torch.nn.Module) -> None:
        self.ema_model.load_state_dict(model.state_dict(), strict=True)


class EMATrainer(Trainer):
    def __init__(
        self,
        *args,
        ema_decay: float = 0.9995,
        ema_update_every: int = 1,
        ema_update_after_step: int = 50,
        class_weights: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ema_decay = ema_decay
        self.ema_update_every = ema_update_every
        self.ema_update_after_step = ema_update_after_step
        self.ema: Optional[ModelEMA] = None
        self.class_weights = class_weights

    def _setup_ema(self) -> None:
        if self.ema is None:
            self.ema = ModelEMA(
                self.model,
                beta=self.ema_decay,
                update_every=self.ema_update_every,
                update_after_step=self.ema_update_after_step,
            )

    def training_step(self, model, inputs, num_items_in_batch=None):
        if self.ema is None:
            self._setup_ema()
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        self.ema.update(self.model)
        return loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        if return_outputs:
            return loss, outputs
        return loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        if self.ema is None:
            return super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        original_model = self.model
        self.model = self.ema.ema_model
        try:
            return super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        finally:
            self.model = original_model

    def save_model(self, output_dir=None, _internal_call=False):
        output_dir = output_dir or self.args.output_dir
        output_dir = str(output_dir)
        super().save_model(output_dir, _internal_call=_internal_call)
        ema_output_dir = Path(output_dir) / "ema_model"
        ema_output_dir.mkdir(parents=True, exist_ok=True)
        if self.ema is not None:
            self.ema.ema_model.save_pretrained(ema_output_dir)
        else:
            self.model.save_pretrained(ema_output_dir)
        if self.processing_class is not None:
            self.processing_class.save_pretrained(ema_output_dir)
