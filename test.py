import torch
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Resize, CenterCrop, Compose

from dataloader import VERandomDataset
from dataloader.transforms import AbsFFT
from model import BridgedTimeSFormer4C

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()



ds = VERandomDataset(
    csv_file="datasets/share_datasets/fold_csv_files/MDMER_fold_csv/MDMER_dataset_updated_fold0.csv",
    eeg_sampling_rate=500,
    time_window = 5.0, #sec
    video_transform=Compose(
        [CenterCrop(size=(480,480)),
        Resize(size=(224,224))],
    ),
    eeg_transform = AbsFFT(dim=-2),
)


dl = DataLoader(ds, batch_size=2, shuffle=True)


model = BridgedTimeSFormer4C(output_dim = 3,
                             image_size = 224,
                             eeg_channels=18,
                             frequency_bins=64,)

model.to(device)

x = next(iter(dl))
print(x)
x = { k : v.to(device) for (k,v) in x.items() }
with torch.inference_mode():
    model.eval()
    y = model(x)

print(y.shape)