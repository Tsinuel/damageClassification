import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.optim as optim
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import argparse
import time
import matplotlib.pyplot as plt
import os
import shutil

def save_files(files_df, input_path, output_path, skip_rate):
                  
    n_files = len(files_df)
    
    #Create class directories
    classes = sorted(files_df['labels'].unique())
    
    for cls_i in classes: 
        path = output_path + '/' + str(cls_i)
        if os.path.exists(path):
            shutil.rmtree(path)
            print('Creating: ', path)
        os.mkdir(path)
    
    for i in range(n_files):
        if i%skip_rate == 0:
            
            source = input_path + '/' + files_df.iloc[i, 1]
            destination = output_path + '/' \
                + str(files_df.iloc[i, 2]) + '/' + files_df.iloc[i, 1]
        
            try:
                shutil.copyfile(source, destination)
                print( files_df.iloc[i, 1])
             
            # If source and destination are same
            except shutil.SameFileError:
                print("Source and destination represents the same file.")
        



train_valid_csv_path = '../xBD_csv/train.csv'
train_input_path = '../xBD_output/'
train_output_path = '../xBD_processed/train'

valid_input_path = '../xBD_output/'
valid_output_path = '../xBD_processed/validation'

test_csv_path = '../xBD_csv/test.csv'
test_input_path = '../xBD_output/'
test_output_path = '../xBD_processed/test'


train_valid_files = pd.read_csv(train_valid_csv_path)
test_files = pd.read_csv(test_csv_path)


train_valid_size = len(train_valid_files)
test_size = len(test_files)

train_size = int(0.8*(train_valid_size + test_size))
valid_size = train_valid_size - train_size

valid_to_train_ratio = valid_size/(train_size + valid_size)

valid_files = train_valid_files.sample(frac=valid_to_train_ratio, 
                                       random_state=200)

train_files = train_valid_files.drop(valid_files.index)

print(train_valid_size)
print(test_size)

print(train_valid_size)
print(test_size)

print()

skip_rate = 4
save_files(train_files, train_input_path, train_output_path, skip_rate)
save_files(valid_files, valid_input_path, valid_output_path, skip_rate)
save_files(test_files, test_input_path, test_output_path, skip_rate)


