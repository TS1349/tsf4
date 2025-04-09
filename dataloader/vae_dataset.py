import torch
from torch.utils.data import Dataset
from torchvision.io import read_video
import pandas as pd

class VAEDataset(Dataset):
    def __init__(self, csv_file, split="train", output_format = "CTHW"):
        self.csv_file = csv_file
        self.split = split
        self.output_format = output_format

        df = pd.read_csv(self.csv_file)
        self.df = df[(df["data_split"] == self.split) & (df["bool_both_files"] == True)]
        sample_row = self.df.iloc[0]
        anno_type = sample_row.anno_type

        if anno_type == "category":
            self.get_label = self._get_ctgr_labels
        else:
            self.get_label = self._get_cont_labels


    def __len__(self):
        return len(self.df)

    def get_label(self, *args):
        raise NotImplementedError("This fuction shouldn't have been accessed")

    def _get_cont_labels(self, row):
        return torch.tensor(row.self_annotation, dtype = torch.float32)

    def _get_ctgr_labels(self, row):
        return torch.tensor(row.label_id, dtype = torch.int32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        video, audio, _ = read_video(
            filename = row.facial_video,
            pts_unit = "sec",
            output_format = self.output_format)

        eeg = pd.read_csv(row.eeg)

        label_id = self.get_label(row)

        return {"video" : video,
                "audio" : audio,
                "eeg" : eeg,
                "label_id" : label_id
               }
                    

