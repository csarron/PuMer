from .meter.configuration_meter import MeterForPreTrainingConfig
from .meter.modeling_meter import (
    MeterForImageAndTextRetrieval,
    MeterForPreTraining,
    MeterForQuestionAnswering,
    MeterForVisualEntailment,
    MeterForVisualReasoning,
)
from .vilt.configuration_vilt import ViltForPreTrainingConfig
from .vilt.modeling_vilt import (
    ViltForImageAndTextRetrieval,
    ViltForPreTraining,
    ViltForQuestionAnswering,
    ViltForVisualEntailment,
    ViltForVisualReasoning,
)

_MODEL_MAP = {
    "vilt_vqa2": ViltForQuestionAnswering,
    "vilt_nlvr2": ViltForVisualReasoning,
    "vilt_ve": ViltForVisualEntailment,
    "vilt_irtr": ViltForImageAndTextRetrieval,
    "meter_vqa2": MeterForQuestionAnswering,
    "meter_nlvr2": MeterForVisualReasoning,
    "meter_ve": MeterForVisualEntailment,
    "meter_irtr": MeterForImageAndTextRetrieval,
    "vilt_pretrain": ViltForPreTraining,
    "meter_pretrain": MeterForPreTraining,
}

_MODEL_CONFIG_MAP = {
    "meter": MeterForPreTrainingConfig,
    "vilt": ViltForPreTrainingConfig,
}


def get_model(model_name, task_name):
    return _MODEL_MAP["_".join([model_name, task_name])]


def get_model_config(model_name):
    return _MODEL_CONFIG_MAP[model_name]
