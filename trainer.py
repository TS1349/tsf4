import os
import math
import time
import torch
import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel as DDP

class PTrainer:
    def __init__(self,
                 model,
                 optimizer,
                 lr_scheduler,
                 loss_function,
                 training_dataloader,
                 validation_dataloader,
                 gpu_id,
                 checkpoint_dir,
                 logger,
                 experiment_name="",
                 ):

        self.time_stamp = "00000000" 
        self.experiment_name = experiment_name
        self.checkpoint_dir = checkpoint_dir

        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader

        self.loss_function = loss_function
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.logger = logger

        # multi-gpu
        self.gpu_id = gpu_id
        self.model = DDP(model, device_ids=[self.gpu_id])

    def _save_checkpoint(self, epoch) -> None:
        checkpoint = self.model.module.state_dict()
        checkpoint_path =\
            f"{self.checkpoint_dir}/{self.time_stamp}_{self.experiment_name}_{epoch}.pt"

        if self.gpu_id == 0:
            torch.save(
                obj=checkpoint,
                f=checkpoint_path,
            )
        print(f"Epoch {epoch}: checkpoint saved at {checkpoint_path}")

    def _run_batch(self, source, target, batch_number):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = self.loss_function(output, target)
        loss.backward()
        nn.utils.clip_grad_norm_(parameters = self.model.parameters(),
                                 max_norm = 1,
                                 )

        self.optimizer.step()
        self.logger(f"TR[{self.gpu_id}]",
                    {"training_loss": loss.item()},
                    batch_number)

    def _time_stamp(self):
        return str(math.floor(time.time()))


    def _run_epoch(self, epoch) -> None:
        self.model.train()
        self.training_dataloader.sampler.set_epoch(epoch)  # multi-gpu
        for batch_number, (source, target) in enumerate(self.training_dataloader):
            source = source.to(self.gpu_id)
            target = target.to(self.gpu_id)
            self._run_batch(source, target, batch_number)

    def _eval(self) -> None:
        self.model.eval()
        size = len(self.validation_dataloader.dataset)
        num_batches = len(self.validation_dataloader)
        test_loss, correct = 0, 0

        with torch.inference_mode():
            counter = 0
            for x, y in self.validation_dataloader:
                print(f"Y: {y}")
                self.training_dataloader.sampler.set_epoch(counter)  # multi-gpu
                x = x.to(self.gpu_id)
                y = y.to(self.gpu_id)
                predictions = self.model(x)
                test_loss += self.loss_function(predictions, y).item()
                correct +=\
                    (predictions.argmax(1) == y).type(torch.float).sum().item()
                counter += 1

            test_loss /= num_batches
            correct /= size
            if self.gpu_id == 0:
                self.logger("VA",
                            {"validation_accuracy": correct,
                             "validation_loss": test_loss})

    def train(self, epochs: int) -> None:
        self.time_stamp = self._time_stamp()
        for epoch in range(epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0:
                self._eval()
                self._save_checkpoint(epoch)

