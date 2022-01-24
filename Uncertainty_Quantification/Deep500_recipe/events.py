import torch
import math

from deep500.lv2.validation.metrics import TestMetric
from deep500.lv2.events import TerminalBarEvent
from deep500.lv2.summaries import TrainingStatistics


class RMSETerminalBarEvent(TerminalBarEvent):
    def __init__(self, ds, sampler, batch_size: int = 64):
        super().__init__()
        self.ds = ds
        self.sample_ds = sampler(self.ds, batch_size=batch_size)
        self.batch_size = batch_size
        self.best_loss = math.inf

    def after_test_batch(self, runner, training_stats: TrainingStatistics,
                         output):
        self.bar.set_postfix(
            mixed_loss=training_stats.current_summary.avg_loss)
        self.bar.update(1)

    def after_test_set(self, runner, training_stats: TrainingStatistics):

        acc_loss = 0

        for idx, inp in enumerate(self.sample_ds):

            out = runner.executor.inference(inp)
            output = out[runner.network_output]
            ground_truth = inp['label']
            batch_loss = torch.nn.MSELoss()(torch.from_numpy(output).to(
                runner.executor.devname), torch.from_numpy(ground_truth).to(
                    runner.executor.devname)).item()
            acc_loss += batch_loss

        mse_epoch_loss = acc_loss
        rmse_epoch_loss = math.sqrt(mse_epoch_loss / len(self.sample_ds))

        if self.best_loss > rmse_epoch_loss:
            self.best_loss = rmse_epoch_loss

        self.sample_ds.reset()

        self.bar.close()

        print('\n Epoch {} RMSE Loss : {} - Best RMSE Loss: {} \n'.format(
            self._epoch, rmse_epoch_loss, self.best_loss))
