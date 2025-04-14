from dataloader import VAEDataset

location = "./datasets/share_datasets/fold_csv_files/Emognition_fold_csv/Emognition_dataset_updated_fold0.csv"


ds_train = VAEDataset(location, split="train")
ds_test = VAEDataset(location, split="test")

breakpoint()
ds_test[0]
ds_test[10]
ds_train[0]
ds_test[10]