from dryml.data.dry_data import DryData, NotIndexedError, \
    NotSupervisedError
from dryml.data.util import nested_batcher, nested_unbatcher, \
    nested_flatten
from dryml.utils import is_iterator
import numpy as np
from typing import Callable


class NumpyDataset(DryData):
    """
    A Numpy based dataset based on a list of numpy elements
    """

    def __init__(
            self, data, indexed=False,
            supervised=False, batch_size=None, size=None):

        if type(data) is np.ndarray or type(data) is tuple:
            data_size = len(data)
            if type(data) is tuple:
                size_set = set(map(lambda d: len(d), nested_flatten(data)))
                if len(size_set) > 1:
                    raise ValueError(
                        "nested elements have different numbers of elements!")
                data_size = size_set.pop()
            super().__init__(
                indexed=indexed, supervised=supervised,
                batch_size=data_size)

            self.data_gen = lambda: [data]
            self.size = data_size

        elif callable(data):
            # We have a method which is supposed to yield
            # A generator.
            super().__init__(
                indexed=indexed, supervised=supervised,
                batch_size=batch_size)
            self.data_gen = data
            if size is None:
                self.size = np.inf
            else:
                self.size = size

        elif is_iterator(data):
            # Can't use consumable iterator
            raise TypeError(
                "Can't use a consumable like an iterator or generator. "
                "Pass a callable which produces it instead.")
        else:
            df_test = False
            try:
                import pandas as pd
                if type(data) is pd.core.frame.DataFrame:
                    df_test = True
            except ImportError:
                pass
            if df_test:
                if indexed is False:
                    super().__init__(
                        indexed=indexed, supervised=supervised,
                        batch_size=len(data))
                    self.data_gen = lambda: [data.to_numpy()]
                    self.size = len(data)
                elif indexed is True:
                    super().__init__(
                        indexed=indexed, supervised=supervised,
                        batch_size=len(data))
                    self.data_gen = lambda: [(data.index.to_numpy(),
                                              data.to_numpy())]
                    self.size = len(data)
            elif type(data) is list:
                data_size = len(data)
                if batch_size is not None:
                    data_size = data_size*batch_size
                super().__init__(
                    indexed=indexed, supervised=supervised,
                    batch_size=batch_size)

                self.data_gen = lambda: data
                if size is None:
                    self.size = data_size
                else:
                    if size != data_size:
                        ValueError("Detected incorrect dataset size")
                    self.size = size
            else:
                super().__init__(
                    indexed=indexed, supervised=supervised,
                    batch_size=batch_size)

                self.data_gen = lambda: data
                if size is None:
                    self.size = np.nan
                else:
                    self.size = size

    def index(self):
        """
        If indexed, return the index of this dataset
        """
        if not self.indexed:
            raise NotIndexedError()

        return map(lambda t: t[0], self.data())

    def as_indexed(self, start=0) -> DryData:
        """
        If not already indexed, return a version of this dataset
        which is indexed.
        """
        if self.indexed:
            return self
        else:
            if not self.batched:
                def enumerate_dataset(gen_func, start=0):
                    i = start
                    it = iter(gen_func())
                    while True:
                        try:
                            d = next(it)
                        except StopIteration:
                            return
                        yield (i, d)
                        i += 1

                return NumpyDataset(
                    lambda: enumerate_dataset(self.data_gen, start=start),
                    indexed=True,
                    supervised=self.supervised,
                    batch_size=self.batch_size,
                    size=self.size)
            else:
                def enumerate_dataset(gen_func, start=0):
                    it = iter(gen_func())
                    i = start
                    while True:
                        try:
                            d = next(it)
                        except StopIteration:
                            return
                        batch_size = len(d)
                        idx = np.array(list(range(i, i+batch_size)))
                        i += batch_size
                        yield (idx, d)

                return NumpyDataset(
                    lambda: enumerate_dataset(self.data_gen, start=start),
                    indexed=True,
                    supervised=self.supervised,
                    batch_size=self.batch_size,
                    size=self.size)

    def as_not_indexed(self):
        """
        Strip index from dataset
        """
        if not self.indexed:
            return self
        else:
            return NumpyDataset(
                lambda: map(lambda t: t[1], self.data()),
                indexed=False,
                supervised=self.supervised,
                batch_size=self.batch_size,
                size=self.size)

    def as_not_supervised(self) -> DryData:
        """
        Strip supervised targets
        """

        if not self.supervised:
            return self
        else:
            if self.indexed:
                return NumpyDataset(
                    map(lambda i, xy: (i, xy[0]), self.data()),
                    indexed=self.indexed,
                    supervised=False,
                    batch_size=self.batch_size,
                    size=self.size)
            else:
                return NumpyDataset(
                    map(lambda x, y: x, self.data()),
                    indexed=self.indexed,
                    supervised=False,
                    batch_size=self.batch_size,
                    size=self.size)

    def intersect(self) -> DryData:
        """
        Intersect this dataset with another
        """
        raise NotImplementedError()

    def data(self):
        """
        Get the internal dataset
        """
        return self.data_gen()

    def batch(self, batch_size=32) -> DryData:
        """
        Batch this data
        """
        if self.batched:
            if self.batch_size != batch_size:
                return self.unbatch().batch(batch_size=32)
            else:
                return self
        else:
            return NumpyDataset(
                lambda: nested_batcher(self.data_gen, batch_size),
                indexed=self.indexed,
                supervised=self.supervised,
                batch_size=batch_size,
                size=self.size)

    def unbatch(self) -> DryData:
        """
        Unbatch this data
        """
        if not self.batched:
            return self
        else:
            return NumpyDataset(
                lambda: nested_unbatcher(self.data_gen),
                indexed=self.indexed,
                supervised=self.supervised,
                size=self.size)

    def apply_X(self, func: Callable = None) -> DryData:
        """
        Apply a function to the X component of DryData
        """

        if self.indexed:
            if self.supervised:
                return NumpyDataset(
                    lambda: map(lambda t: (t[0], (func(t[1][0]), t[1][1])),
                                self.data()),
                    indexed=self.indexed,
                    supervised=self.supervised,
                    batch_size=self.batch_size,
                    size=self.size)
            else:
                return NumpyDataset(
                    lambda: map(lambda t: (t[0], func(t[1])),
                                self.data()),
                    indexed=self.indexed,
                    supervised=self.supervised,
                    batch_size=self.batch_size,
                    size=self.size)
        else:
            if self.supervised:
                return NumpyDataset(
                    lambda: map(lambda t: (func(t[0]), t[1]),
                                self.data()),
                    indexed=self.indexed,
                    supervised=self.supervised,
                    batch_size=self.batch_size,
                    size=self.size)
            else:
                return NumpyDataset(
                    lambda: map(lambda x: func(x),
                                self.data()),
                    indexed=self.indexed,
                    supervised=self.supervised,
                    batch_size=self.batch_size,
                    size=self.size)

    def apply_Y(self, func=None) -> DryData:
        """
        Apply a function to the Y component of DryData
        """

        if not self.supervised:
            raise NotSupervisedError(
                "Can't apply a function to the Y component of "
                "non supervised dataset")

        if self.indexed:
            return NumpyDataset(
                lambda: map(lambda i, xy: (i, (xy[0], func(xy[1]))),
                            self.data()),
                indexed=self.indexed,
                supervised=self.supervised,
                batch_size=self.batch_size,
                size=self.size)
        else:
            return NumpyDataset(
                lambda: map(lambda x, y: (x, func(y)),
                            self.data()),
                indexed=self.indexed,
                supervised=self.supervised,
                batch_size=self.batch_size,
                size=self.size)

    def apply(self, func=None) -> DryData:
        """
        Apply a function to (X, Y)
        """

        if not self.supervised:
            raise NotSupervisedError(
                "Can't apply a function to the Y component of "
                "non supervised dataset")

        if self.indexed:
            return NumpyDataset(
                lambda: map(lambda i, xy: (i, func(*xy)),
                            self.data()),
                indexed=self.indexed,
                supervised=self.supervised,
                batch_size=self.batch_size,
                size=self.size)
        else:
            return NumpyDataset(
                lambda: map(lambda x, y: func(x, y),
                            self.data()),
                indexed=self.indexed,
                supervised=self.supervised,
                batch_size=self.batch_size,
                size=self.size)

    def __iter__(self):
        """
        Create iterator
        """

        return iter(self.data())

    def take(self, n):
        """
        Take only a specific number of examples
        """

        def taker(gen_func, n):
            i = 0
            it = iter(gen_func())
            while i < n:
                try:
                    yield next(it)
                    i += 1
                except StopIteration:
                    return
            return

        new_size = self.size
        if new_size is np.nan:
            new_size = n
        elif new_size is np.inf:
            new_size = n
        else:
            if new_size > n:
                new_size = n

        return NumpyDataset(
            lambda: taker(self.data_gen, n),
            indexed=self.indexed,
            supervised=self.supervised,
            batch_size=self.batch_size,
            size=new_size,
            )

    def skip(self, n):
        """
        Skip a specific number of examples
        """

        def skiper(gen_func, n):
            i = 0
            it = iter(gen_func())
            while i < n:
                try:
                    next(it)
                    i += 1
                except StopIteration:
                    return
            while True:
                try:
                    yield next(it)
                except StopIteration:
                    return

        new_size = self.size
        if new_size is not np.nan and new_size is not np.inf:
            if n > new_size:
                new_size = 0
            else:
                new_size -= n

        return NumpyDataset(
            lambda: skiper(self.data_gen, n),
            indexed=self.indexed,
            supervised=self.supervised,
            batch_size=self.batch_size,
            size=new_size)

    def __len__(self):
        """
        Get length of dataset. Will return Infinite if infinite,
        and unknown if it can't be determined.
        """
        return self.size

    def numpy(self):
        return self

    def tf(self):
        from dryml.data.tf import TFDataset
        import dryml.data.util as util
        import tensorflow as tf

        # Heuristic to determine output_signature
        peek_data = self.peek()

        def get_numpy_array_spec(e):
            e_tens = tf.constant(e)
            e_spec = tf.TensorSpec.from_tensor(e_tens)
            if self.batched:
                # We need to remove the first shape number
                # Since this data is batched
                list_shape = list(e_spec.shape)
                list_shape[0] = None
                e_spec = tf.TensorSpec(
                    tf.TensorShape(list_shape),
                    e_spec.dtype)
            return e_spec
        peek_data_signature = util.nested_apply(
            peek_data, get_numpy_array_spec)

        # Create tf dataset
        dataset = tf.data.Dataset.from_generator(
            lambda: self.data_gen(),
            output_signature=peek_data_signature)

        return TFDataset(
            dataset,
            indexed=self.indexed,
            supervised=self.supervised,
            batch_size=self.batch_size)
