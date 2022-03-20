from abc import ABC, abstractmethod
from shutil import copyfile
from typing import TypeVar, Any, Generic, Dict, Set

import torch
import torch.nn

T = TypeVar("T")
S = TypeVar("S")


class SaveLoadHandler(ABC, Generic[T, S]):
    @abstractmethod
    def load(self, instance: T, load_info: S) -> T:
        raise NotImplemented

    @abstractmethod
    def save(self, instance: T) -> S:
        raise NotImplemented

    @abstractmethod
    def not_load(self, instance: T) -> S:
        raise NotImplemented


class IdentitySaveLoadHandler(SaveLoadHandler):
    def load(self, instance, load_info):
        return instance

    def not_load(self, instance):
        return instance

    def save(self, instance):
        return instance


class StateDictLoadHandler(IdentitySaveLoadHandler):
    def load(self, instance, load_info):
        instance.load_state_dict(load_info)
        return instance

    def save(self, instance):
        return instance.state_dict()


class InitZeroLoadHandler(IdentitySaveLoadHandler):
    def not_load(self, instance):
        # Init linear layers with zeros
        def init_linear(m):
            if type(m) == torch.nn.Linear:
                torch.nn.init.zeros_(m.weight)

        instance.apply(init_linear)

        return instance


class PickleLoadHandler(IdentitySaveLoadHandler):
    def load(self, instance, load_info):
        return load_info


class CombinedLoadHandler(SaveLoadHandler[T, S], Generic[T, S]):
    def __init__(self, handler1: SaveLoadHandler[T, S], handler2: SaveLoadHandler[T, S]):
        self.h1 = handler1
        self.h2 = handler2

    def load(self, instance, load_info):
        return self.h2.load(self.h1.load(instance, load_info), load_info)

    def not_load(self, instance: T):
        return self.h2.not_load(self.h1.not_load(instance))

    def save(self, instance: T):
        return self.h2.save(self.h1.save(instance))


class AbstractCheckpointManager(ABC):
    @abstractmethod
    def load(self, name: str, instance: T, handler: SaveLoadHandler[T, S]) -> T:
        raise NotImplementedError

    @abstractmethod
    def save(self, info_dict: Dict[str, Any]):
        raise NotImplementedError


class CheckpointManager(AbstractCheckpointManager):
    def __init__(self, path: str, load_checkpoint: bool, save_checkpoint: bool, device):
        self.path = path
        self.load_checkpoint = load_checkpoint
        self.save_checkpoint = save_checkpoint
        self.handlers: Dict[str, SaveLoadHandler] = dict()

        if self.load_checkpoint:
            self.checkpoint = torch.load(path, map_location=device)

    def load(self, name: str, instance: T, handler: SaveLoadHandler[T, S]) -> T:
        self.handlers[name] = handler

        if self.load_checkpoint:
            info = self.checkpoint.get(name)

            if info:
                return handler.load(instance, info)
            else:
                print("Warning: " + name + " was not loaded from checkpoint")

        return handler.not_load(instance)

    def save(self, info_dict: Dict[str, Any]):
        if self.save_checkpoint:
            processed_dict = {key: self.handlers[key].save(value) for key, value in info_dict.items()}

            torch.save(processed_dict, self.path)
            copyfile(self.path,
                     self.path + "_copy")  # If we are interrupted during the save, we still have the previous file copy


class TransplantCheckpointManager(AbstractCheckpointManager):
    def __init__(self, cman: AbstractCheckpointManager, path_to_transplant: str):
        self.cman = cman

        self.checkpoint = torch.load(path_to_transplant, map_location="cpu")

        self.to_transplant: Set[str] = set()

    def load(self, name: str, instance: T, handler: SaveLoadHandler[T, S]) -> T:

        linst = self.cman.load(name, instance, handler)

        if name in self.to_transplant:
            return handler.load(instance, self.checkpoint[name])
        else:
            return linst

    def save(self, info_dict: Dict[str, Any]):
        self.cman.save(info_dict)

    def transplant(self, name):
        self.to_transplant.add(name)

    def load_from_alt(self, name: str, instance: T, handler: SaveLoadHandler[T, S]) -> T:
        return handler.load(instance, self.checkpoint[name])
