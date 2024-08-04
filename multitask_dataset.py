# https://discuss.huggingface.co/t/dataset-expected-by-trainer/148/5
import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import librosa
from copy import deepcopy

class SingerMultiTaskDataset(Dataset):
    def __init__(self, audios_parent_path, csv_path, sampling_rate=16000):
        # audios_parent_path = '/media/maximos/9C33-6BBD/data/melos_singers/Rebetika_vowels/train'
        self.audios_parent_path = audios_parent_path
        # new_csv_path = '/media/maximos/9C33-6BBD/data/melos_singers/features/multitask_targets.csv'
        self.csv_path = csv_path
        self.sampling_rate = sampling_rate
        # get full audio paths as list
        self.audio_paths = list( Path(self.audios_parent_path).glob('**/*.mp3') )
        # load csv
        self.feats = pd.read_csv(self.csv_path, delimiter=',')
        # make sure names are lowercase
        self.feats['names'] = self.feats['names'].str.lower()
        # keep file names from audio paths
        self.names = []
        for p in self.audio_paths:
            self.names.append( p.stem.lower().replace('kazantzidis_old', 'kazantzidisold') )
        # make a dict that keeps only number of output labels per task
        # for defining regression or classification
        task_labels_num = {k:1 for k in list(self.feats.columns)}
        task_labels_num['singer_id'] = self.feats['singer_id'].max()+1
        # normalize all regression features
        for c in self.feats.columns:
            if c != 'singer_id' and c != 'names' and 'Unn' not in c:
                # self.feats[c] = (self.feats[c]-self.feats[c].mean())/self.feats[c].std()
                self.feats[c] = (self.feats[c]-self.feats[c].min())/(self.feats[c].max()-self.feats[c].min())
    # end init

    def __len__(self):
        return len(self.audio_paths)
    # end len

    def __getitem__(self, idx):
        # load audio with librosa in the desired sample rate as mono
        audio_var, _ = librosa.load(self.audio_paths[idx], sr=self.sampling_rate)
        # get proper line values from csv
        feats_line = self.feats[ self.feats['names'] == self.names[idx] ]
        # make a dict
        feats_line_lists = feats_line.to_dict('tight')
        feats_line_dict = dict( zip( feats_line_lists['columns'] , feats_line_lists['data'][0] ) )
        return {
            'input_values': audio_var,
            'labels': feats_line_dict
        }
    # end getitem
# end class
