import os
from dryml.dry_object import DryObject, DryObjectFactory, DryObjectFile, \
    DryObjectDef, change_object_cls
from dryml.utils import get_current_cls, path_needs_directory
from typing import Optional, Callable
import tqdm


class DryRepoContainer(object):
    @classmethod
    def build_from_filepath(filepath: str):
        new_container = DryRepoContainer()
        new_container.filepath = filepath
        return new_container

    @classmethod
    def build_from_object(obj: DryObject, filepath: Optional[str] = None):
        new_container = DryRepoContainer()
        new_container.obj = obj
        if filepath is not None:
            new_container.filepath = filepath
        return new_container

    def __init__(self):
        self._filepath = None
        self._obj = None

    def load(self, directory: Optional[str] = None,
             update: bool = True, reload: bool = False) -> bool:
        if self._obj is not None:
            # Object is already loaded
            return True
        if self._filepath is None and directory is None:
            raise RuntimeError(
                "No filepath indicated, and no directory! "
                "Can't load from disk.")
        # Get full filepath of file
        # Load object at filepath

    @property
    def obj(self):
        if self.obj is None and self._filepath is None:
            raise RuntimeError(
                "Container has neither filepath nor object defined")

        # Handle loading of file
        # if self._obj is not None:
        #    return self._obj
        # else:
        #    self._obj =


# This type will act as a fascade for the various DryObject* types.
class DryRepo(object):
    def __init__(self, directory: Optional[str] = None, create: bool = False,
                 load_objects: bool = True, **kwargs):
        super().__init__(**kwargs)

        # A dictionary of objects
        self.obj_dict = {}

        # Model list (for iteration)
        self.obj_list = []

        if directory is not None:
            self.link_to_directory(directory, create=create,
                                   load_objects=load_objects)

        self._save_objs_on_deletion = False

    @property
    def save_objs_on_deletion(self):
        return self._save_objs_on_deletion

    @save_objs_on_deletion.setter
    def save_objs_on_deletion(self, val: bool):
        if val and self.directory is None:
            raise RuntimeError(
                "Give the repo a directory if you want it to "
                "save objects upon deletion")
        self._save_objs_on_deletion = val

    def __del__(self):
        if self.save_objs_on_deletion:
            self.save()

    def link_to_directory(self, directory: str, create: bool = False,
                          load_objects: bool = True):
        # Check that directory exists
        if not os.path.exists(directory):
            if create:
                os.makedirs(directory)
        self.directory = directory
        if load_objects:
            self.load_objects_from_directory()

    def number_of_objects(self):
        return len(self.obj_list)

    def add_obj_def(self, cat_hash: str, ind_hash: str, obj_def: dict):
        cat_dict = self.obj_dict.get(cat_hash, {})
        if ind_hash in cat_dict:
            raise ValueError(
                f"Object {ind_hash} in category {cat_hash} "
                "already exists in the repo!")
        cat_dict[ind_hash] = obj_def
        self.obj_dict[cat_hash] = cat_dict
        self.obj_list.append(obj_def)

    def load_objects_from_directory(self, directory: Optional[str] = None,
                                    selector: Optional[Callable] = None,
                                    verbose: bool = False):
        "Method to refresh the internal dictionary of objects."
        need_directory = True
        if directory is None:
            if self.directory is None:
                raise RuntimeError(
                    "No default directory selected for this repo!")
            directory = self.directory
            need_directory = False

        files = os.listdir(directory)

        num_loaded = 0
        for filename in files:
            full_filepath = os.path.join(directory, filename)
            try:
                with DryObjectFile(full_filepath) as f:
                    if selector is not None:
                        if not selector(f):
                            # Skip non selected objects
                            continue
                    obj_def = f.definition()
                    obj_cat_id = obj_def.get_category_id()
                    obj_id = obj_def.get_individual_id()
                    obj_definition = {
                        'val': filename,
                        'filepath': filename
                    }
                    if need_directory:
                        obj_definition['filepath'] = \
                            os.path.join(directory,
                                         obj_definition['filepath'])
                    self.add_obj_def(obj_cat_id, obj_id, obj_definition)
                    num_loaded += 1
            except Exception as e:
                print(f"WARNING! Malformed file found! {full_filepath}"
                      f"skipping load Error: {e}")
        if verbose:
            print(f"Loaded {num_loaded} objects")

    def add_object(self, obj: DryObject, filepath: Optional[str] = None):
        # Add a single object
        obj_def = obj.definition()
        obj_cat_id = obj_def.get_category_id()
        obj_ind_id = obj_def.get_individual_id()
        obj_definition = {'val': obj}
        if filepath is not None:
            obj_definition['filepath'] = filepath
        self.add_obj_def(obj_cat_id, obj_ind_id, obj_definition)

    def add_objects(self, obj_factory: DryObjectFactory, num=1):
        # Create numerous objects from a factory function
        for i in range(num):
            obj = obj_factory()
            self.add_object(obj)

    def make_container_handler(self,
                               load_object: bool = True,
                               open_container: bool = True,
                               update: bool = True):
        def container_handler(obj_container):
            def container_opener(obj_container):
                if open_container:
                    return obj_container['val']
                else:
                    return obj_container
            if isinstance(obj_container['val'], str):
                if load_object:
                    filepath = obj_container['filepath']
                    if path_needs_directory(filepath):
                        if self.directory is None:
                            raise RuntimeError(
                                "Repo is not linked to a directory!")
                        filepath = os.path.join(self.directory, filepath)
                    with DryObjectFile(filepath) as f:
                        obj_container['val'] = f.load_object(update=update)
                return container_opener(obj_container)
            elif isinstance(obj_container['val'], DryObjectFile):
                if load_object:
                    f = obj_container['val']
                    obj_container['val'] = f.load_object(update=update)
                    f.close()
                return container_opener(obj_container)
            elif isinstance(obj_container['val'], DryObject):
                return container_opener(obj_container)
            else:
                raise RuntimeError(
                    f"Unsupported value type: {type(obj_container['val'])}")

        return container_handler

    def make_filter_func(
            self,
            selector: Optional[Callable] = None,
            sel_args=None, sel_kwargs=None,
            only_objs: bool = False):
        if sel_args is None:
            sel_args = []
        if sel_kwargs is None:
            sel_kwargs = {}

        def filter_func(obj_container):
            if isinstance(obj_container['val'], str):
                if only_objs:
                    return False
                # This container just a string, we need to
                # load from disk to check against the selector
                filepath = obj_container['val']
                if path_needs_directory(filepath):
                    if self.directory is None:
                        raise RuntimeError(
                            "Repo is not connected to any directory!")
                    filepath = os.path.join(self.directory, filepath)
                if selector is not None:
                    with DryObjectFile(filepath) as f:
                        if selector(f, *sel_args, **sel_kwargs):
                            return True
                        else:
                            return False
                else:
                    return True
            elif isinstance(obj_container['val'], DryObject):
                if selector is not None:
                    if selector(obj_container['val'], *sel_args, **sel_kwargs):
                        return True
                    else:
                        return False
                else:
                    return True
            elif isinstance(obj_container['val'], DryObjectFile):
                if only_objs:
                    return False
                if selector is not None:
                    if selector(obj_container['val'], *sel_args, **sel_kwargs):
                        return True
                    else:
                        return False
                else:
                    return True
            else:
                raise RuntimeError(
                    f"Unsupported value type: {type(obj_container['val'])}")
        return filter_func

    def get_obj(
            self,
            obj_def: DryObjectDef):
        cat_id = obj_def.get_category_id()
        ind_id = obj_def.get_individual_id()
        return self.get_obj_by_id(cat_id, ind_id)

    def get_obj_by_id(
            self,
            cat_id,
            ind_id):
        obj_container = self.obj_dict[cat_id][ind_id]
        container_handler = self.make_container_handler()
        return container_handler(obj_container)

    def get(self,
            selector: Optional[Callable] = None,
            sel_args=None, sel_kwargs=None,
            load_objects: bool = True,
            update: bool = True,
            open_container: bool = True,
            verbose: bool = True):
        # Handle arguments for filter function
        only_objs = True
        if load_objects:
            only_objs = False

        # Filter the internal object list
        filter_func = \
            self.make_filter_func(
                selector,
                sel_args=sel_args,
                sel_kwargs=sel_kwargs,
                only_objs=only_objs)
        obj_list = list(filter(filter_func, self.obj_list))

        # Build container handler
        container_handler = \
            self.make_container_handler(
                update=update,
                open_container=open_container)
        return list(map(container_handler, obj_list))

    def apply(self,
              func, func_args=None, func_kwargs=None,
              selector: Optional[Callable] = None,
              sel_args=None, sel_kwargs=None,
              update: bool = True,
              verbose: bool = False,
              load_objects: bool = False,
              open_container: bool = True):
        """
        Apply a function to all objects tracked by the repo.
        We can also use a DrySelector to apply only to specific models
        """
        if func_args is None:
            func_args = []
        if func_kwargs is None:
            func_kwargs = {}

        # Create apply function
        def apply_func(obj):
            return func(obj, *func_args, **func_kwargs)

        # Get object list
        objs = self.get(
            selector=selector,
            sel_args=sel_args, sel_kwargs=sel_kwargs,
            update=update,
            load_objects=load_objects,
            open_container=open_container)

        # apply function to objects
        if verbose:
            results = list(map(apply_func, tqdm(objs)))
        else:
            results = list(map(apply_func, objs))

        # Handle results
        if len(list(filter(lambda x: x is None, results))) == len(objs):
            return None
        else:
            return results

    def reload_objs(
            self,
            selector: Optional[Callable] = None,
            sel_args=None, sel_kwargs=None,
            update: bool = False,
            reload: bool = False,
            as_cls=None):
        if self.directory is None:
            raise RuntimeError("Repo directory needs to be set for reloading.")

        def reload_func(obj_container):
            obj = obj_container['val']
            if not isinstance(obj, DryObject):
                raise RuntimeError("Can only reload DryObjects")
            # Get current definition of class
            if as_cls is not None:
                cls = as_cls
            else:
                cls = type(obj)
            new_cls = get_current_cls(cls, reload=reload)
            obj_container['val'] = change_object_cls(obj, new_cls, update=True)
            del obj

        self.apply(
            reload_func,
            selector=selector, sel_args=sel_args, sel_kwargs=sel_kwargs,
            open_container=False, load_objects=False)

    def save(self,
             selector: Optional[Callable] = None,
             sel_args=None, sel_kwargs=None,
             directory: Optional[str] = None):

        def save_func(obj_container):
            obj = obj_container['val']
            if not isinstance(obj, DryObject):
                raise RuntimeError("Can only save currently loaded DryObject")
            if directory is None:
                if 'filepath' not in obj_container:
                    obj_container['filepath'] = \
                        obj.definition().get_individual_id()+'.dry'
                filepath = obj_container['filepath']
                if path_needs_directory(filepath):
                    if self.directory is None:
                        raise RuntimeError(
                            "Repo's directory is not set. Set the directory.")
                    filepath = os.path.join(self.directory, filepath)
            else:
                _, tail = os.path.split(obj_container['filepath'])
                filepath = os.path.join(directory, tail)
            obj.save_self(filepath)

        self.apply(
            save_func,
            selector=selector, sel_args=sel_args, sel_kwargs=sel_kwargs,
            open_container=False, load_objects=False)

    def save_and_cache(
            self,
            selector: Optional[Callable] = None,
            sel_args=None, sel_kwargs=None):
        "Save and then delete objects. Replace their entries with strings"

        def save_func(obj_container):
            obj = obj_container['val']
            if not isinstance(obj, DryObject):
                raise RuntimeError("Can only save currently loaded DryObject")
            if 'filepath' not in obj_container:
                obj_container['filepath'] = \
                    obj.definition().get_individual_id()+'.dry'
            filepath = obj_container['filepath']
            orig_filepath = filepath
            if path_needs_directory(filepath):
                if self.directory is None:
                    raise RuntimeError(
                        "Repo's directory is not set. Set the directory.")
                filepath = os.path.join(self.directory, filepath)
            obj.save_self(filepath)
            del obj
            if orig_filepath == filepath:
                obj_container['val'] = filepath
            else:
                obj_container['val'] = orig_filepath

        self.apply(
            save_func,
            selector=selector, sel_args=sel_args, sel_kwargs=sel_kwargs,
            open_container=False, load_objects=False)

    def delete_objs(
            self,
            selector: Optional[Callable] = None,
            sel_args=None, sel_kwargs=None):
        "Unload and delete from disk selected models"

        # Get all selected objects
        obj_containers = self.get(
            selector=selector, sel_args=sel_args, sel_kwargs=sel_kwargs,
            open_container=False, load_objects=False)

        for obj_container in obj_containers:
            filepath = obj_container['filepath']
            if path_needs_directory(filepath):
                if self.directory is None:
                    raise RuntimeError(
                        "Repo's directory is not set. Set the directory.")
                filepath = os.path.join(self.directory, filepath)

            obj = obj_container['val']
            if type(obj) is not str:
                # Delete the object from internal tracking
                obj_def = obj.definition()
                cat_hash = obj_def.get_category_id()
                ind_hash = obj_def.get_individual_id()
                del self.obj_dict[cat_hash][ind_hash]
                # Delete the object from object list
                del self.obj_list[self.obj_list.index(obj_container)]
                # Delete the object itself.
                del obj

            # delete the container
            del obj_container

            # Delete the object file
            os.remove(filepath)
