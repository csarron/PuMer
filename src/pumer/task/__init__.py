from .task_base import Task
from .task_nlvr2 import Nlvr2Task
from .task_retrieval import RetrievalTask

_TASK_CLASS_MAP = {
    "nlvr2": Nlvr2Task,
    "ve": Task,
    "vqa2": Task,
    "irtr": RetrievalTask,
}


def get_task(task_name):
    return _TASK_CLASS_MAP[task_name]
