import torch
from torch.utils.data import Dataset
from torchvision.io import read_video
import pandas as pd

class VAEDataset(Dataset):
    def __init__(
            self,
            csv_file,
            split="train",
            video_output_format = "TCHW",
            transform = None,
            video_transform = None,
            audio_transform = None,
            eeg_transform = None,
            ):
        self.csv_file = csv_file
        self.split = split
        self.video_output_format = video_output_format

        self.transform = transform
        self.video_transform = video_transform
        self.audio_transform = audio_transform
        self.eeg_transform = eeg_transform

        df = pd.read_csv(self.csv_file)
        self.df = df[(df["data_split"] == self.split) & (df["bool_both_file"] == True)]
        sample_row = self.df.iloc[0]
        anno_type = sample_row.anno_type

        if anno_type == "category":
            self._get_label = self._get_ctgr_labels
        else:
            self._get_label = self._get_cont_labels


    def __len__(self):
        return len(self.df)

    def _get_label(self, *args):
        raise NotImplementedError("This fuction shouldn't have been accessed")

    def _get_cont_labels(self, row):
        self_annotation = row.self_annotation[1:-1].split(r",")
        self_annotation = [ float(entry) for entry in self_annotation ]
        return torch.tensor(self_annotation, dtype = torch.float32)

    def _get_ctgr_labels(self, row):
        return torch.tensor(row.label_id, dtype = torch.int32)
    
    def _get_eeg(self, row):
        eeg = pd.read_csv(row.EEG)
        eeg = torch.tensor(eeg.to_numpy().transpose(), dtype = torch.float32)
        return eeg

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        print(row)
        video, _ , metadata = read_video(
            filename = row.facial_video,
            pts_unit = "sec",
            output_format = self.video_output_format,
            )
        
        
        assert(row.fps == metadata["video_fps"])

        eeg = self._get_eeg(row)

        output = self._get_label(row)

        return {"video" : video,
                "eeg" : eeg,
                "output" : output,
               }