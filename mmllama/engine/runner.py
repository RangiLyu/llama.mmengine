import logging
import time
import warnings
from typing import Dict, List, Optional, Union

import mmengine
from mmengine.config import Config, ConfigDict
from mmengine.dist import master_only
from mmengine.fileio import FileClient, join_path
from mmengine.logging import print_log
from mmengine.model import is_model_wrapper
from mmengine.optim import OptimWrapper, OptimWrapperDict, _ParamScheduler
from mmengine.runner.checkpoint import save_checkpoint, weights_to_cpu
from mmengine.utils import get_git_hash

ConfigType = Union[Dict, Config, ConfigDict]
ParamSchedulerType = Union[List[_ParamScheduler], Dict[str,
                                                       List[_ParamScheduler]]]
OptimWrapperType = Union[OptimWrapper, OptimWrapperDict]


from mmengine.runner import Runner as _Runner


class Runner(_Runner):

    @master_only
    def save_checkpoint(
        self,
        out_dir: str,
        filename: str,
        file_client_args: Optional[dict] = None,
        save_optimizer: bool = True,
        save_param_scheduler: bool = True,
        meta: dict = None,
        by_epoch: bool = True,
        backend_args: Optional[dict] = None,
    ):
        """Save checkpoints.

        ``CheckpointHook`` invokes this method to save checkpoints
        periodically.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename (str): The checkpoint filename.
            file_client_args (dict, optional): Arguments to instantiate a
                FileClient. See :class:`mmengine.fileio.FileClient` for
                details. Defaults to None. It will be deprecated in future.
                Please use `backend_args` instead.
            save_optimizer (bool): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            save_param_scheduler (bool): Whether to save the param_scheduler
                to the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            by_epoch (bool): Whether the scheduled momentum is updated by
                epochs. Defaults to True.
            backend_args (dict, optional): Arguments to instantiate the
                prefix of uri corresponding backend. Defaults to None.
                New in v0.2.0.
        """
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')

        if by_epoch:
            # self.epoch increments 1 after
            # `self.call_hook('after_train_epoch)` but `save_checkpoint` is
            # called by `after_train_epoch`` method of `CheckpointHook` so
            # `epoch` should be `self.epoch + 1`
            meta.update(epoch=self.epoch + 1, iter=self.iter)
        else:
            meta.update(epoch=self.epoch, iter=self.iter + 1)

        if file_client_args is not None:
            warnings.warn(
                '"file_client_args" will be deprecated in future. '
                'Please use "backend_args" instead', DeprecationWarning)
            if backend_args is not None:
                raise ValueError(
                    '"file_client_args" and "backend_args" cannot be set at '
                    'the same time.')

            file_client = FileClient.infer_client(file_client_args, out_dir)
            filepath = file_client.join_path(out_dir, filename)
        else:
            filepath = join_path(  # type: ignore
                out_dir, filename, backend_args=backend_args)

        meta.update(
            cfg=self.cfg.pretty_text,
            seed=self.seed,
            experiment_name=self.experiment_name,
            time=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
            mmengine_version=mmengine.__version__ + get_git_hash())

        if hasattr(self.train_dataloader.dataset, 'metainfo'):
            meta.update(dataset_meta=self.train_dataloader.dataset.metainfo)

        if is_model_wrapper(self.model):
            model = self.model.module
        else:
            model = self.model

        checkpoint = {
            'meta': meta,
            'state_dict': weights_to_cpu(model.state_dict()),
            'message_hub': self.message_hub.state_dict()
        }
        # save optimizer state dict to checkpoint
        if save_optimizer:
            if isinstance(self.optim_wrapper, OptimWrapper):
                checkpoint['optimizer'] = self.optim_wrapper.state_dict()
            else:
                raise TypeError(
                    'self.optim_wrapper should be an `OptimWrapper` '
                    'or `OptimWrapperDict` instance, but got '
                    f'{self.optim_wrapper}')

        # save param scheduler state dict
        if save_param_scheduler and self.param_schedulers is None:
            print_log(
                '`save_param_scheduler` is True but `self.param_schedulers` '
                'is None, so skip saving parameter schedulers',
                logger='current',
                level=logging.WARNING)
            save_param_scheduler = False
        if save_param_scheduler:
            if isinstance(self.param_schedulers, dict):
                checkpoint['param_schedulers'] = dict()
                for name, schedulers in self.param_schedulers.items():
                    checkpoint['param_schedulers'][name] = []
                    for scheduler in schedulers:
                        state_dict = scheduler.state_dict()
                        checkpoint['param_schedulers'][name].append(state_dict)
            else:
                checkpoint['param_schedulers'] = []
                for scheduler in self.param_schedulers:  # type: ignore
                    state_dict = scheduler.state_dict()  # type: ignore
                    checkpoint['param_schedulers'].append(state_dict)

        self.call_hook('before_save_checkpoint', checkpoint=checkpoint)
        save_checkpoint(checkpoint, filepath)
