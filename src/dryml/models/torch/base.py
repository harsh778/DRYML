from dryml.config import Meta
from dryml.object import Object
from dryml.models import DryComponent
from dryml.models import Trainable
from dryml.models import TrainFunction as BaseTrainFunction
import zipfile
import torch


class TorchObject(Object):
    __dry_compute_context__ = "torch"

    @Meta.collect_args
    @Meta.collect_kwargs
    def __init__(self, cls, *args, **kwargs):
        if type(cls) is not type:
            raise TypeError("first argument must be a class!")
        self.cls = cls
        self.args = args
        self.kwargs = kwargs
        self.obj = None

    def compute_prepare_imp(self):
        self.obj = self.cls(*self.args, **self.kwargs)

    def load_compute_imp(self, file: zipfile.ZipFile) -> bool:
        try:
            with file.open('state.pth', 'r') as f:
                self.obj.load_state_dict(torch.load(f))
            return True
        except Exception:
            return False

    def save_compute_imp(self, file: zipfile.ZipFile) -> bool:
        try:
            with file.open('state.pth', 'w') as f:
                torch.save(self.obj.state_dict(), f)
            return True
        except Exception:
            return False

    def compute_cleanup_imp(self):
        del self.obj
        self.obj = None


class TrainFunction(BaseTrainFunction):
    __dry_compute_context__ = 'torch'


class Model(DryComponent):
    __dry_compute_context__ = 'torch'

    def prep_eval(self):
        pass

    def prep_train(self):
        pass


class Trainable(Trainable):
    __dry_compute_context__ = 'torch'
