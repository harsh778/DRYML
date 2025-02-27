from dryml.config import Meta
from dryml.object import Object
from dryml.data import Dataset
from dryml.models.trainable import Trainable as BaseTrainable
from dryml.models.torch.base import Wrapper
from dryml.models.torch.base import Model as TorchModel
from dryml.models.torch.base import TrainFunction as TorchTrainFunction
from dryml.models.torch.base import Trainable as TorchTrainable
from dryml.context import context
import zipfile
import torch
import tqdm
from dryml.utils import validate_class
from collections import Counter
import re
import numpy as np

class Model(TorchModel):
    def __call__(self, *args, **kwargs):
        return self.mdl.forward(*args, **kwargs)

    def compute_cleanup_imp(self):
        del self.mdl
        self.mdl = None

    def load_compute_imp(self, file: zipfile.ZipFile) -> bool:
        try:
            with file.open('state.pth', 'r') as f:
                self.mdl.load_state_dict(torch.load(f))
            return True
        except Exception:
            return False

    def save_compute_imp(self, file: zipfile.ZipFile) -> bool:
        try:
            with file.open('state.pth', 'w') as f:
                torch.save(self.mdl.state_dict(), f)
            return True
        except Exception:
            return False

    def prep_eval(self):
        devs = context().get_torch_devices()
        self.mdl.to(devs[0])
        self.mdl.train(False)

    def prep_train(self):
        devs = context().get_torch_devices()
        self.mdl.to(devs[0])
        self.mdl.train(True)


class TrainFunction(TorchTrainFunction):
    pass


class ModelWrapper(Model):
    @Meta.collect_args
    @Meta.collect_kwargs
    def __init__(self, cls, *args, **kwargs):
        self.cls = validate_class(cls)
        self.args = args
        self.kwargs = kwargs
        self.mdl = None

    def compute_prepare_imp(self):
        self.mdl = self.cls(*self.args, **self.kwargs)


class Sequential(Model):
    def __init__(self, layer_defs=[]):
        self.layer_defs = layer_defs
        self.mdl = None

    def compute_prepare_imp(self):
        # create_layers
        layers = []
        for layer in self.layer_defs:
            if type(layer[0]) is not type:
                raise TypeError(
                    "First element of a layer definition should be a type")
            layers.append(layer[0](*layer[1], **layer[2]))

        self.mdl = torch.nn.Sequential(
            *layers)


class TorchOptimizer(Object):
    @Meta.collect_args
    @Meta.collect_kwargs
    def __init__(self, cls, model: Model, *args, **kwargs):
        if type(cls) is not type:
            raise TypeError("first argument must be a class!")
        self.cls = cls
        self.model = model
        self.args = args
        self.kwargs = kwargs
        self.opt = None

    def compute_prepare_imp(self):
        self.opt = self.cls(
            self.model.mdl.parameters(),
            *self.args,
            **self.kwargs)

    def load_compute_imp(self, file: zipfile.ZipFile) -> bool:
        try:
            with file.open('state.pth', 'r') as f:
                self.opt.load_state_dict(torch.load(f))
            return True
        except Exception:
            return False

    def save_compute_imp(self, file: zipfile.ZipFile) -> bool:
        try:
            with file.open('state.pth', 'w') as f:
                torch.save(self.opt.state_dict(), f)
            return True
        except Exception:
            return False

    def compute_cleanup_imp(self):
        del self.opt
        self.opt = None


class Trainable(TorchTrainable):
    def __init__(
            self,
            model: Model = None,
            train_fn: TrainFunction = None):
        self.model = model
        self.train_fn = train_fn

    def train(
            self, data, train_spec=None, train_callbacks=[],
            metrics=[]):
        self.train_fn(
            self, data, train_spec=train_spec,
            train_callbacks=train_callbacks)
        self.train_state = BaseTrainable.trained

    def prep_train(self):
        self.model.prep_train()

    def prep_eval(self):
        self.model.prep_eval()

    def eval(self, data: Dataset, *args, eval_batch_size=32, **kwargs):
        # Move variables to same device as model
        devs = context().get_torch_devices()
        if data.batched:
            # We can execute the method directly on the data
            return data.torch() \
                       .map_el(lambda el: el.to(devs[0])) \
                       .apply_X(
                           func=lambda X: self.model(X, *args, **kwargs))
        else:
            # We first need to batch the data, then unbatch to leave
            # The dataset character unchanged.
            return data.torch() \
                       .batch(batch_size=eval_batch_size) \
                       .map_el(lambda el: el.to(devs[0])) \
                       .apply_X(
                            func=lambda X: self.model(X, *args, **kwargs)) \
                       .unbatch()


class BasicTraining(TrainFunction):
    def __init__(
            self,
            optimizer: Wrapper = None,
            loss: Wrapper = None,
            epochs=1):
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs

    def __call__(
            self, trainable: Model, data: Dataset, train_spec=None,
            train_callbacks=[]):

        # Pop the epoch to resume from
        start_epoch = 0
        if train_spec is not None:
            start_epoch = train_spec.level_step()

        # Type checking training data, and converting if necessary
        batch_size = 32
        data = data.torch().batch(batch_size=batch_size)
        total_batches = data.count()

        # Move variables to same device as model
        devs = context().get_torch_devices()
        data = data.map_el(lambda el: el.to(devs[0]))

        # Check data is supervised.
        if not data.supervised:
            raise RuntimeError(
                f"{__class__} requires supervised data")

        optimizer = self.optimizer.opt
        loss = self.loss.obj
        model = trainable.model

        for i in range(start_epoch, self.epochs):

            running_loss = 0.
            num_batches = 0
            t_data = tqdm.tqdm(data, total=total_batches)
            for X, Y in t_data:
                optimizer.zero_grad()

                outputs = model(X)
                loss_val = loss(outputs, Y)
                loss_val.backward()
                optimizer.step()

                running_loss += loss_val.item()
                num_batches += 1
                av_loss = running_loss/(num_batches*batch_size)
                t_data.set_postfix(loss=av_loss)

            print(f"Epoch {i+1} - Average Loss: {av_loss}")

class TextVectorizer(Trainable):
    def __init__(self, max_tokens=10000, sequence_length=250, pad_token="<PAD>", unk_token="<UNK>"):
        super().__init__()
        self.max_tokens = max_tokens
        self.sequence_length = sequence_length
        self.pad_token = pad_token
        self.unk_token = unk_token

        self.vocab = {}
        self.id_to_token = []
        self.trained = False

    def simple_tokenize(self, text):
        return re.findall(r"\w+", text.lower())

    def build_vocab(self, dataset):
        counter = Counter()
        for x, _ in dataset:
            tokens = self.simple_tokenize(x)
            counter.update(tokens)
        most_common = counter.most_common(self.max_tokens - 2)
        self.id_to_token = [self.pad_token, self.unk_token] + [w for w, _ in most_common]
        self.vocab = {w: i for i, w in enumerate(self.id_to_token)}

    def vectorize_text(self, text):
        tokens = self.simple_tokenize(text)
        ids = []
        for t in tokens:
            ids.append(self.vocab.get(t, self.vocab[self.unk_token]))
        if len(ids) < self.sequence_length:
            ids += [self.vocab[self.pad_token]] * (self.sequence_length - len(ids))
        else:
            ids = ids[:self.sequence_length]
        return np.array(ids, dtype=np.int64)

    def train(self, ds, **kwargs):
        if not self.trained:
            self.build_vocab(ds)
            self.trained = True

    def predict(self, x, **kwargs):
        if isinstance(x, str):
            return torch.tensor(self.vectorize_text(x), dtype=torch.long)
        elif isinstance(x, np.ndarray):
            return self.predict(x.tolist(), **kwargs)
        elif isinstance(x, (list, tuple)):
            return torch.stack([
                torch.tensor(self.vectorize_text(xx), dtype=torch.long)
                for xx in x
            ])
        raise ValueError("Unsupported input type for vectorizer.")


    def eval(self, dataset, **kwargs):
        return dataset.map(lambda x: (self.predict(x[0]), x[1]))
    
    def prep_train(self, **kwargs):
        pass

    def save_model(self, *args, **kwargs):
        return {
            'vocab': self.vocab,
            'id_to_token': self.id_to_token,
            'trained': self.trained
        }

    def load_model(self, dct, *args, **kwargs):
        self.vocab = dct['vocab']
        self.id_to_token = dct['id_to_token']
        self.trained = dct['trained']
