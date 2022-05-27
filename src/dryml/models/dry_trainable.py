import zipfile
import pickle
from dryml.dry_object import DryObject
from dryml.context import cls_method_compute


@cls_method_compute('train')
@cls_method_compute('eval')
class DryTrainable(DryObject):
    untrained = 0
    trained = 2

    def __init__(self, *args, description="", **kwargs):
        self.train_state = DryTrainable.untrained

    def load_object_imp(self, file: zipfile.ZipFile) -> bool:
        # Load parent components first
        with file.open('component_data.pkl', 'r') as f:
            component_data = pickle.load(f)
        self.train_state = component_data['train_state']
        return True

    def save_object_imp(self, file: zipfile.ZipFile) -> bool:
        with file.open('component_data.pkl', 'w') as f:
            f.write(pickle.dumps({'train_state': self.train_state}))
        return True

    def prep_train(self):
        # Configure the model for training
        pass

    def prep_eval(self):
        # Configure the model for evaluation
        pass

    def train(self, *args, train_spec=None, train_callbacks=[], **kwargs):
        # Handle the setting of the train state flag
        self.train_state = DryTrainable.trained
        # This should be the last step in training so no more super is needed

    def eval(self, data, *args, **kwargs):
        raise NotImplementedError("Method not defined for a base DryTrainable")
