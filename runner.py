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
from dataloader import VERandomDataset, TestDataset
from dataloader.transforms import AbsFFT

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
    training_dataset = VERandomDataset(
        csv_file="datasets/share_datasets/fold_csv_files/MDMER_fold_csv/MDMER_dataset_updated_fold0.csv",
        eeg_sampling_rate=500,
        time_window = 5.0, #sec
        video_transform=video_preprocessor,
        eeg_transform = AbsFFT(dim=-2),
        split = "train"
    )
    validation_dataset = VERandomDataset(
        csv_file="datasets/share_datasets/fold_csv_files/MDMER_fold_csv/MDMER_dataset_updated_fold0.csv",
        eeg_sampling_rate=500,
        time_window = 5.0, #sec
        video_transform=video_preprocessor,
        eeg_transform = AbsFFT(dim=-2),
        split = "test"
    )

    training_dataset = TestDataset()
    validation_dataset = TestDataset()

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
                 output_dim = 4,
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
        weight_decay=1e-4
)


    p_trainer = PTrainer(
        model=model,
        lr_scheduler = None,
        training_dataloader=training_dataloader,
        validation_dataloader=validation_dataloader,
        optimizer=optimizer,
        loss_function=loss_function,
        checkpoint_dir="./",
        gpu_id=rank,
        logger = None,
    )

    p_trainer.train(epochs)

    destroy_process_group()
    print(f"{rank} process group destroyed")


if "__main__" == __name__:
    parser = argparse.ArgumentParser(description="TimeSFormer")

    parser.add_argument("--epochs", type=int, required = True)
    parser.add_argument("--batch_size", type=int, required = True)
    parser.add_argument("--learning_rate", type=float, required = True)
    parser.add_argument("--momentum", type=float, required = True)
    parser.add_argument("--dataset", type=str, required=True)

    args = parser.parse_args()

    batch_size = args.batch_size
    epochs = args.epochs

    optimizer = args.optimizer
    learning_rate = args.learning_rate
    momentum = args.momentum

    world_size = torch.cuda.device_count()
    print(f"number of cuda devices {world_size}")

    # rank,
    # world_size,
    # epochs,
    # batch_size,
    # learning_rate,
    # momentum,
    args = (world_size,
            epochs,
            batch_size,
            learning_rate,
            momentum,)
    mp.spawn(run,
             args=args,
             nprocs=world_size)

    print("TRAINING IS DONE")
