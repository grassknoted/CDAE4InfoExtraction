import os
import cv2
import glob
import ntpath
import numpy as np
import pandas as pd

dataset_path = "../Dataset/"

def get_file_name(path):
    head, tail = ntpath.split(path)
    return str(tail) or str(ntpath.basename(head))

def load_dataset(dataset_path):
    classes = ['cats', 'dogs', 'fox']

    labels_dataframe = pd.read_csv(dataset_path+'labels.csv')
    
    data = []
    labels = []
    for class_name in classes:
        img_dir = dataset_path+str(class_name)+'/'
        data_path = os.path.join(img_dir,'*g')
        files = glob.glob(data_path)
        for current_file in files:
            for index, row in labels_dataframe.iterrows():
                if get_file_name(current_file) == row['File Name']:
                    labels.append(row['Features'])
            img = cv2.imread(current_file)
            data.append(img)

    assert len(labels) == len(data), "Length of the training set does not match length of the labels!"
    x_train = np.array(data).reshape(-1, 28, 28, 1).astype('float32') / 255.

load_dataset(dataset_path)