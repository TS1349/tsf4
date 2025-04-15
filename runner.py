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
from dataloader import VERandomDataset
from dataloader.transforms import AbsFFT


def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    os.environ["DISTRIBUTED_DEBUG"] = "DETAIL"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def run(
    rank,
    world_size,
    cnn_adapter_type,
    epochs,
    batch_size,
    learning_rate,
    momentum,
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

    model = BridgedTimeSFormer4C()

    model.to(rank)


    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        params=model.parameters(),
        lr=learning_rate,
        momentum=momentum,
    )

    p_trainer = PTrainer(
        model=model,
        training_dataloader=training_dataloader,
        validation_dataloader=validation_dataloader,
        optimizer=optimizer,
        loss_function=loss_function,
        experiment_name=f"R{rank}_{str(cnn_adapter_type)}",
        checkpoint_dir="./",
        gpu_id=rank,
        logger=logger,
    )

    p_trainer.train(epochs)

    destroy_process_group()


if "__main__" == __name__:
    world_size = 1
    epochs = 15
    momentum = 0.9
    batch_size = 4
    world_size = torch.cuda.device_count()
    args = (world_size,
            epochs,
            batch_size,
            momentum,)
    mp.spawn(run,
             args=args,
             nprocs=world_size)
    print("done")
