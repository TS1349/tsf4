import pandas as pd

LOCATIONS = [
    "./datasets/share_datasets/fold_csv_files/EAV_fold_csv/EAV_dataset_updated_fold0.csv",
    "./datasets/share_datasets/fold_csv_files/Emognition_fold_csv/Emognition_dataset_updated_fold0.csv",
    "./datasets/share_datasets/fold_csv_files/K_EmoCon_fold_csv/K_EmoCon_dataset_updated_fold0.csv",
    "./datasets/share_datasets/fold_csv_files/MDMER_fold_csv/MDMER_dataset_updated_fold0.csv",
    ]

def check_property(df, prop, default_value):
	test = (df[prop] == default_value)
	return test.all()

def check(path):
	df = pd.read_csv(path)
	properties = ["fps", "frame_size", "EEG_vec_counts", "EEG_ch"]
	default_values = { prop : df.iloc[0][prop] for prop in properties }
	
	for prop in properties:
		default_value = default_values[prop]
		print(f"{prop} is {default_value} for all {check_property(df, prop, default_value)}")

def check_all():
	for loc in LOCATIONS:
		print(loc)
		check(loc)
		print("================\n")
	
if __name__ == "__main__":
	check_all()
