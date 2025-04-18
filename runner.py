from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from torchvision.transforms.v2 import Compose, Normalize, Resize, CenterCrop, ToDtype,RandomHorizontalFlip
import os
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

from trainer import PTrainer
from model import BridgedTimeSFormer4C
from dataloader import VERandomDataset, TestDataset, EAVDataset, EmognitionDataset, MDMERDataset
from dataloader.transforms import AbsFFT
from scheduler import SteppedScheduler

import argparse


def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    os.environ["DISTRIBUTED_DEBUG"] = "DETAIL"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def run(
    rank,
    world_size,
    epochs,
    batch_size,
    learning_rate,
    weight_decay,
    csv_file,
    experiment_name,
):

    ddp_setup(rank, world_size)
    video_preprocessor = Compose(
        [CenterCrop(size=(480,480)),
         Resize(size=(224,224)),
         RandomHorizontalFlip(p=0.5),
         ToDtype(torch.float32, scale=True),
         Normalize(
             mean=(0.45, 0.45, 0.45),
             std=(0.225, 0.225, 0.225),
         )]
    )
    training_dataset = MDMERDataset(
        csv_file=csv_file,
        time_window = 5.0, #sec
        video_transform=video_preprocessor,
        eeg_transform = AbsFFT(dim=-2),
        split = "train"
    )
    validation_dataset = MDMERDataset(
        csv_file=csv_file,
        time_window = 5.0, #sec
        video_transform=video_preprocessor,
        eeg_transform = AbsFFT(dim=-2),
        split = "test"
    )

    training_dataloader = DataLoader(
        dataset=training_dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(training_dataset),
    )

    validation_dataloader = DataLoader(
        dataset=validation_dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
    ) if validation_dataset is not None else None

    model = BridgedTimeSFormer4C(
                 output_dim = training_dataset.output_shape,
                 image_size = 224,
                 eeg_channels = 8,
                 frequency_bins = 64,
    )

    model.to(rank)


    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=weight_decay
)
    lr_scheduler = SteppedScheduler(optimizer)


    p_trainer = PTrainer(
        model=model,
        lr_scheduler = lr_scheduler,
        training_dataloader=training_dataloader,
        validation_dataloader=validation_dataloader,
        optimizer=optimizer,
        loss_function=loss_function,
        checkpoint_dir="./",
        gpu_id=rank,
        experiment_name=experiment_name,
    )

    p_trainer.train(epochs)

    destroy_process_group()
    print(f"{rank} process group destroyed")


if "__main__" == __name__:
    parser = argparse.ArgumentParser(description="TimeSFormer")

    parser.add_argument("--epochs", type=int, required = True)
    parser.add_argument("--batch_size", type=int, required = True)
    parser.add_argument("--learning_rate", type=float, required = True)
    parser.add_argument("--weight_decay", type=float, required = True)
    parser.add_argument("--csv_file", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, required=True)

    args = parser.parse_args()

    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    csv_file = args.csv_file
    experiment_name = args.experiment_name

    world_size = torch.cuda.device_count()
    print(f"number of cuda devices {world_size}")

    args = (
        epochs,
        batch_size,
        learning_rate,
        weight_decay,
        csv_file,
        experiment_name,
    )
    mp.spawn(run,
             args=args,
             nprocs=world_size)

    print("TRAINING IS DONE")
