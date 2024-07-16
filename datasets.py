# https://discuss.huggingface.co/t/dataset-expected-by-trainer/148/5
import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd

class SingerMultiTaskDataset(Dataset):
    def __init__(self, audios_parent_path, csv_path, sampling_rate=16000):
        # audios_parent_path = '/media/maximos/9C33-6BBD/data/melos_singers/Rebetika_vowels/train'
        self.audios_parent_path = audios_parent_path
        self.csv_path = csv_path
        self.sampling_rate = sampling_rate
        # get full audio paths as list
        self.audio_paths = list( Path(self.audios_parent_path).glob('**/*.mp3') )
        # load csv
        self.feats = pd.read_csv(self.csv_path, delimiter=',')
    # end init

    def __len__(self):
        return len(self.audio_paths)
    # end len

    def __getitem__(self, idx):
        # load audio with librosa in the desired sample rate as mono
        # 
        pass
    # end getitem
# end class