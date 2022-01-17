import torch
from typing import Any, Callable, List, Tuple, Union
from deep500.lv2.events import TrainingStatistics
from deep500.lv2.events.hyperparameter_schedule import HyperparameterSchedule


class CyclicLRScheduler(HyperparameterSchedule):
    """ A naive implementation of torch.optim.lr_scheduler.CyclicLR in
     Deep500. This only calls the step() of the scheuler which is not desired
      for reproducibility across frameworks"""
    def __init__(self, per_epoch,
                 **hyperparameters: Union[Callable[[int], Any],
                                          List[Tuple[int, Any]]]):
        """ Initializes a epoch-wise hyperparameter schedule Event.
            @param per_epoch: Per epoch or per step scheduler
            @param hyperparameters: A dictionary of hyperparameter name to
                                    either a list of (epoch, value), or a
                                    lambda function that accepts epoch
                                    and returns a value.
        """
        self._params = hyperparameters
        self._torch_scheduler = None
        super().__init__(per_epoch, **hyperparameters)

    def before_optimizer_step(self, executor, optimizer, inputs):
        if self._optimizer is None:
            self._optimizer = optimizer

            self.torch_scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer.op, **self._params)
        if self._per_epoch is False:
            self.torch_scheduler.step()

            self._step += 1

    def before_epoch(self, epoch, runner, training_stats: TrainingStatistics):
        if self._optimizer is None:
            return
        if self._per_epoch:
            self.torch_scheduler.step()
