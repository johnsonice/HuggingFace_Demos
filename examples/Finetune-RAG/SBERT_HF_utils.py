""" A Trainer that is compatible with Huggingface transformers 
code got from :
https://github.com/matthewfranglen/sentence-transformers/blob/huggingface-trainer/sentence_transformers/huggingface.py
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import PreTrainedTokenizerBase, Trainer
from transformers.utils import is_datasets_available
from transformers.tokenization_utils import BatchEncoding
from transformers.utils.generic import PaddingStrategy
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import datasets

from sentence_transformers import SentenceTransformer

@dataclass
class SentenceTransformersCollator:
    """Collator for a SentenceTransformers model.
    This encodes the text columns to {column}_input_ids and {column}_attention_mask columns.
    This works with the two text dataset that is used as the example in the training overview:
    https://www.sbert.net/docs/training/overview.html"""

    tokenizer: PreTrainedTokenizerBase
    text_columns: List[str]

    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, text_columns: List[str]
    ) -> None:
        self.tokenizer = tokenizer
        self.text_columns = text_columns
        self.max_length=tokenizer.model_max_length
        self.padding=True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        ## label is actually not used in multiple negative ranking loss
        ## just crate a dummy place for comput loss function to work 
        #batch = {"label": torch.tensor([1 for row in features])}
        batch = {}
        for column in self.text_columns:
            padded = self._encode([row[column] for row in features])
            batch[f"{column}_input_ids"] = padded.input_ids
            batch[f"{column}_attention_mask"] = padded.attention_mask
        return batch

    def _encode(self, texts: List[str]) -> BatchEncoding:
        tokens = self.tokenizer(texts, #return_attention_mask=False,
                                padding=self.padding,
                                truncation=True,
                                return_tensors=self.return_tensors)
        return tokens
        # return self.tokenizer.pad(
        #     tokens,
        #     padding=self.padding,
        #     max_length=self.max_length,
        #     pad_to_multiple_of=self.pad_to_multiple_of,
        #     return_tensors=self.return_tensors,
        # )


class SentenceTransformersTrainer(Trainer):
    """Huggingface Trainer for a SentenceTransformers model.

    This works with the two text dataset that is used as the example in the training overview:
    https://www.sbert.net/docs/training/overview.html

    """

    def __init__(
        self,
        *args,
        text_columns: List[str],
        loss: nn.Module,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.text_columns = text_columns
        self.loss = loss
        self.loss.to(self.model.device)

    def compute_loss(
        self,
        model: SentenceTransformer,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        features = self.collect_features(inputs)
        loss = self.loss(features, None)
        if return_outputs:
            output = torch.cat(
                [model(row)["sentence_embedding"][:, None] for row in features], dim=1
            )
            return loss, output
        return loss

    def collect_features(
        self, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> List[Dict[str, torch.Tensor]]:
        """Turn the inputs from the dataloader into the separate model inputs."""
        return [
            {
                "input_ids": inputs[f"{column}_input_ids"],
                "attention_mask": inputs[f"{column}_attention_mask"],
            }
            for column in self.text_columns
        ]

    def evaluate(
            self,
            eval_dataset = None,
            ignore_keys = None,
            metric_key_prefix: str = "eval",
        ):
            # memory metrics - must set up as early as possible
            self._memory_tracker.start()
            metrics = self.compute_metrics(None)
            self.log(metrics)
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
            self._memory_tracker.stop_and_update_metrics(metrics)

            return metrics