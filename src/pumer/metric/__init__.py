import torchmetrics

from ..dataset.dataset_nlvr2 import Nlvr2PredictionWriter
from ..dataset.dataset_ve import VePredictionWriter
from ..dataset.dataset_vqa2 import Vqa2PredictionWriter
from .metric_vqa2 import Vqa2Accuracy

_TASK_METRIC_MAP = {
    "nlvr2": torchmetrics.Accuracy,
    "ve": torchmetrics.Accuracy,
    "vqa2": Vqa2Accuracy,
}

_TASK_PRED_WRITER_MAP = {
    "nlvr2": None,  # no need to use Nlvr2PredictionWriter,
    "ve": None,
    "vqa2": Vqa2PredictionWriter,
}


def get_metric(task_name):
    return _TASK_METRIC_MAP.get(task_name, None)


def get_prediction_writer(task_name):
    return _TASK_PRED_WRITER_MAP.get(task_name, None)
