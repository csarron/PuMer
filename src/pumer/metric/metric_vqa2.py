from typing import Any, Callable, Optional

import torch
import torchmetrics


class Vqa2Accuracy(torchmetrics.Metric):
    def __init__(
        self,
        threshold: float = 0.5,
        top_k: Optional[int] = None,
        subset_accuracy: bool = False,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

        if not 0 < threshold < 1:
            raise ValueError(f"The `threshold` should be a float in the (0,1)" f" interval, got {threshold}")

        if top_k is not None and (not isinstance(top_k, int) or top_k <= 0):
            raise ValueError(f"The `top_k` should be an integer larger than 0, got {top_k}")

        self.threshold = threshold
        self.top_k = top_k
        self.subset_accuracy = subset_accuracy

    # @staticmethod
    # def compute_accuracy(preds, labels):
    #     scores = 0
    #     for pred, label in zip(preds, labels):
    #         scores += label[pred]
    #     return scores / len(preds)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets.
        See :ref:`extensions/metrics:input types` for more information
        on input types.

        Args:
            preds: Predictions from model (probabilities, or labels)
            target: Ground truth labels, one-hot probabilities
        """

        correct = torch.gather(target, 1, preds.unsqueeze(1)).sum()
        total = torch.tensor(target.shape[0], device=target.device)

        self.correct += correct
        self.total += total

    def compute(self) -> torch.Tensor:
        """
        Computes accuracy based on inputs passed in to ``update`` previously.
        """
        return self.correct / self.total
