from multitask_dataset import SingerMultiTaskDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch
from models import HuBERTClassifierBaseline # HuBERTFeatureFusion #, HuBERTMultiHead
import pandas as pd
from tqdm import tqdm
import numpy as np
import os
import csv

# globals
epochs = 1000
batch_size = 32

save_name = 'baseline_frozen_hubert'

# keep best validation loss for saving
best_val_loss = np.inf
save_dir = 'saved_models/' + save_name + '/'
model_path = save_dir + save_name + '.pt'
os.makedirs(save_dir, exist_ok=True)

# save results
os.makedirs('results', exist_ok=True)
results_path = 'results/' + save_name + '.csv'
result_fields = ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc']
with open( results_path, 'w' ) as f:
    writer = csv.writer(f)
    writer.writerow( result_fields )

# data folders
train_audio_folder = '/media/maindisk/maximos/data/melos_singers/Rebetika_vowels/train/'
test_audio_folder = '/media/maindisk/maximos/data/melos_singers/Rebetika_vowels/test/'
csv_path = '/media/maindisk/maximos/data/melos_singers/features/multitask_targets.csv'
# train_audio_folder = '/media/maximos/9C33-6BBD/data/melos_singers/Rebetika_vowels/train/'
# test_audio_folder = '/media/maximos/9C33-6BBD/data/melos_singers/Rebetika_vowels/test/'
# csv_path = '/media/maximos/9C33-6BBD/data/melos_singers/features/multitask_targets.csv'

# load csv
feats = pd.read_csv(csv_path, delimiter=',')
# keep feature list which will become the tasks
features_list = list(feats.columns)
# delete unnecessary columns
del(features_list[:2])

# keep number of outputs per task
task_labels_num_out = {}
for i in range(1, len(features_list)-3, 1):
    task_labels_num_out[features_list[i]] = 1
# add singer identification
task_labels_num_out['singer_id'] = feats['singer_id'].max()+1 # accounting for zero

# initialize model
model = HuBERTClassifierBaseline(task_labels_num_out=task_labels_num_out, gpu_index=0)

# make datasets
training_data = SingerMultiTaskDataset(train_audio_folder, csv_path)
testing_data = SingerMultiTaskDataset(test_audio_folder, csv_path)

# make dataloaders
train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, collate_fn=model.collate_fn)
test_loader = DataLoader(testing_data, batch_size=batch_size, shuffle=True, collate_fn=model.collate_fn)

# define parameters to train
# keep hubert frozen
for p in model.hubert.parameters():
    p.requires_grad = False
# train projectors and classifiers
for k in model.projectors.keys():
    model.projectors[k].requires_grad = True
    model.classifiers[k].requires_grad = True

# define optimizer
optimizer = Adam( model.parameters(), lr=0.001 )

for epoch in range(epochs):
    train_loss = 0
    running_loss = 0
    batch_num = 0
    running_accuracy = 0
    train_accuracy = 0
    with tqdm(train_loader, unit='batch') as tepoch:
        tepoch.set_description(f"Epoch {epoch} | trn")
        for b in tepoch:
            y = model(
                audio_normalized=b[0]['input_values'],
                attention_mask=b[0]['attention_mask'],
                labels=b[1],
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True
            )
            optimizer.zero_grad()
            y.loss.backward()
            optimizer.step()
            # update loss
            batch_num += 1
            running_loss += y.loss.item()
            train_loss = running_loss/batch_num
            # accuracy
            prediction = y['logits']['singer_id'][0].argmax(dim=1, keepdim=True).squeeze()
            running_accuracy += (prediction == torch.Tensor(b[1]['singer_id']).to(model.dev)).sum().item()/len(b[1]['singer_id'])
            train_accuracy = running_accuracy/batch_num
            tepoch.set_postfix(loss=train_loss, accuracy=train_accuracy)
            # validation
    with torch.no_grad():
        val_loss = 0
        running_loss = 0
        batch_num = 0
        running_accuracy = 0
        val_accuracy = 0
        print('validation...')
        with tqdm(test_loader, unit='batch') as tepoch:
            tepoch.set_description(f"Epoch {epoch} | tst")
            for b in tepoch:
                y = model(
                    audio_normalized=b[0]['input_values'],
                    attention_mask=b[0]['attention_mask'],
                    labels=b[1],
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True
                )
                # update loss
                batch_num += 1
                running_loss += y.loss.item()
                val_loss = running_loss/batch_num
                # accuracy
                prediction = y['logits']['singer_id'][0].argmax(dim=1, keepdim=True).squeeze()
                running_accuracy += (prediction == torch.Tensor(b[1]['singer_id']).to(model.dev)).sum().item()/len(b[1]['singer_id'])
                val_accuracy = running_accuracy/batch_num
            if best_val_loss > val_loss:
                print('saving!')
                best_val_loss = val_loss
                torch.save(model.state_dict(), model_path)
            print(f'validation: accuracy={val_accuracy}, loss={val_loss}')
            with open( results_path, 'a' ) as f:
                writer = csv.writer(f)
                writer.writerow( [epoch, train_loss, train_accuracy, val_loss, val_accuracy] )