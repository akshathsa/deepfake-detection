from efficientnet_pytorch import EfficientNet
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from einops import rearrange, repeat
from torch import nn, einsum
import torch.nn as nn
import torch.nn.functional as F
from random import random, randint, choice
from vit_pytorch import ViT
import numpy as np
from torch.optim import lr_scheduler
import os
import json
from os import cpu_count
from multiprocessing.pool import Pool
from functools import partial
from multiprocessing import Manager
from progress.bar import ChargingBar
from efficient_vit import EfficientViT
import uuid
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.metrics import accuracy_score
import cv2
from transforms.albu import IsotropicResize
import glob
import pandas as pd
from tqdm import tqdm
from utils import train_parse, get_method, check_correct, resize, shuffle_dataset, get_n_params, train_val_split, video_to_frames
from sklearn.utils.class_weight import compute_class_weight 
from torch.optim.lr_scheduler import LambdaLR
import collections
from deepfakes_dataset import DeepFakesDataset
import math
import yaml
import argparse
import pickle

BASE_DIR = '/Users/akaashrp/Desktop/GT_Classwork/Spring_2024/CS_7641/deepfake-detection/'
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAINING_DIR = os.path.join(DATA_DIR, "dfdc_train")
VALIDATION_DIR = os.path.join(DATA_DIR, "dfdc_val")
TEST_DIR = os.path.join(DATA_DIR, "test")
MODELS_PATH = os.path.join(BASE_DIR, "models")
METADATA_PATH = os.path.join(DATA_DIR, "metadata") # Folder containing all training metadata for DFDC dataset
TRAIN_LABELS_PATH = os.path.join(DATA_DIR, "dfdc_train_labels.csv")
VALIDATION_LABELS_PATH = os.path.join(DATA_DIR, "dfdc_val_labels.csv")
TEST_LABELS_PATH = os.path.join(DATA_DIR, "dfdc_test_labels.csv")

def load_dataset(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def read_frames(video_path, train_dataset, validation_dataset):
    if TRAINING_DIR in video_path:
        train_df = pd.DataFrame(pd.read_csv(TRAIN_LABELS_PATH))
        video_folder_name = os.path.basename(video_path)
        video_key = video_folder_name + ".mp4"
        label = float(train_df.loc[train_df['filename'] == video_key]['label'].values[0])
    else:
        val_df = pd.DataFrame(pd.read_csv(VALIDATION_LABELS_PATH))
        video_folder_name = os.path.basename(video_path)
        video_key = video_folder_name + ".mp4"
        label = float(val_df.loc[val_df['filename'] == video_key]['label'].values[0])
    
    frames = os.listdir(video_path)
        
    for index, frame_image in enumerate(frames):
        image = cv2.imread(os.path.join(video_path, frame_image))
        if image is not None:
            if TRAINING_DIR in video_path:
                train_dataset.append((image, label))
            else:
                validation_dataset.append((image, label))

if __name__ == "__main__":
    opt = train_parse()
    print(opt)

    with open(opt.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)
 
    if opt.efficient_net == 0:
        channels = 1280
    else:
        channels = 2560
    
    print("EfficientNet B" + str(opt.efficient_net) + " with ViT")
    model = EfficientViT(config=config, channels=channels, selected_efficient_net = opt.efficient_net) # EfficientNet B0
    model.train()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight-decay'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config['training']['step-size'], gamma=config['training']['gamma'])
    starting_epoch = 0
    if os.path.exists(opt.resume):
        print("Loading checkpoint from", opt.resume)
        model.load_state_dict(torch.load(opt.resume))
        starting_epoch = int(opt.resume.split("checkpoint")[1].split("_")[0]) + 1 # The checkpoint's file name format should be "checkpoint_EPOCH"
    else:
        print("No checkpoint loaded.")

    print("Model Parameters:", get_n_params(model))
    
    # read dataset
    # if opt.dataset != "All" and opt.dataset != "DFDC":
    #     folders = ["Original", opt.dataset]
    # else:
    #     folders = ["Original", "DFDC", "Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]
    
    mgr = Manager()
    if not os.path.exists(os.path.join(DATA_DIR, "train.pkl")) or not os.path.exists(os.path.join(DATA_DIR, "val.pkl")):
        train_val_split(DATA_DIR, TRAINING_DIR, VALIDATION_DIR, 0.8)
        sets = [TRAINING_DIR, VALIDATION_DIR]

        print("Extracting frames from videos...")
        for dataset in sets:
            videos = os.listdir(dataset)
            videos = [video for video in videos if video.endswith(".mp4")]
            for index, video in enumerate(videos):
                if TRAINING_DIR in dataset and index == opt.max_train_videos:
                    break
                if VALIDATION_DIR in dataset and index == opt.max_val_videos:
                    break
                video_path = os.path.join(dataset, video)
                frame_path = os.path.join(dataset, video.split(".")[0])
                if not os.path.exists(frame_path):
                    video_to_frames(video_path, frame_path, frame_skip=10)
        print("Frames extracted.")

        paths = []
        for dataset in sets:
            frame_folders = os.listdir(dataset)
            frame_folders = [frame_folder for frame_folder in frame_folders if os.path.isdir(os.path.join(dataset, frame_folder))]
            for index, frame_folder in enumerate(frame_folders):
                if TRAINING_DIR in dataset and index == opt.max_train_videos:
                    break
                if VALIDATION_DIR in dataset and index == opt.max_val_videos:
                    break

                if os.path.isdir(os.path.join(dataset, frame_folder)):
                    paths.append(os.path.join(dataset, frame_folder))

        # if len(paths) == 0:
        #     paths = [TRAINING_DIR, VALIDATION_DIR]
        
        train_dataset = mgr.list()
        validation_dataset = mgr.list()

        # with Pool(processes=opt.workers) as p:
        #     with tqdm(total=len(paths)) as pbar:
        #         for v in p.imap_unordered(partial(read_frames, train_dataset=train_dataset, validation_dataset=validation_dataset), paths):
        #             pbar.update()

        #     p.terminate()

        with Pool(processes=opt.workers) as p:
            with tqdm(total=len(paths)) as pbar:
                # for path in paths:
                #     result = read_frames(path, train_dataset=train_dataset, validation_dataset=validation_dataset)
                #     pbar.update()
                for v in p.imap_unordered(partial(read_frames, train_dataset=train_dataset, validation_dataset=validation_dataset), paths):
                    pbar.update()

                pbar.close()
                p.terminate()
        
        train_dataset = shuffle_dataset(train_dataset)
        print(type(train_dataset))
        validation_dataset = shuffle_dataset(validation_dataset)
        train_samples = len(train_dataset)
        validation_samples = len(validation_dataset)
    else:
        # with open(os.path.join(DATA_DIR, "train.pkl"), "rb") as f:
        #     print(f)
        #     train_dataset = pickle.load(f)
        # with open(os.path.join(DATA_DIR, "val.pkl"), "rb") as f:
        #     validation_dataset = pickle.load(f)

        train_dataset_getter = mgr.Function(load_dataset, [DATA_DIR + "/train.pkl"])
        train_dataset = train_dataset_getter()
        validation_dataset_getter = mgr.Function(load_dataset, [DATA_DIR + "/val.pkl"])
        validation_dataset = validation_dataset_getter()
        
        train_samples = len(train_dataset)
        validation_samples = len(validation_dataset)
    
    # Print some useful statistics
    print("Train images:", train_samples, "Validation images:", validation_samples)
    print("__TRAINING STATS__")
    train_counters = collections.Counter(image[1] for image in train_dataset)
    print(train_counters)
    
    class_weights = train_counters[0] / train_counters[1]
    print("Weights", class_weights)

    print("__VALIDATION STATS__")
    val_counters = collections.Counter(image[1] for image in validation_dataset)
    print(val_counters)
    print("___________________")

    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_weights])) # log loss

    # Create the data loaders
    validation_labels = np.asarray([row[1] for row in validation_dataset])
    labels = np.asarray([row[1] for row in train_dataset])

    train_dataset = DeepFakesDataset(np.asarray([row[0] for row in train_dataset]), labels, config['model']['image-size'])
    dl = torch.utils.data.DataLoader(train_dataset, batch_size=config['training']['bs'], shuffle=True, sampler=None,
                                 batch_sampler=None, num_workers=opt.workers, collate_fn=None,
                                 pin_memory=False, drop_last=False, timeout=0,
                                 worker_init_fn=None, prefetch_factor=2,
                                 persistent_workers=False)
    del train_dataset

    validation_dataset = DeepFakesDataset(np.asarray([row[0] for row in validation_dataset]), validation_labels, config['model']['image-size'], mode='validation')
    val_dl = torch.utils.data.DataLoader(validation_dataset, batch_size=config['training']['bs'], shuffle=True, sampler=None,
                                    batch_sampler=None, num_workers=opt.workers, collate_fn=None,
                                    pin_memory=False, drop_last=False, timeout=0,
                                    worker_init_fn=None, prefetch_factor=2,
                                    persistent_workers=False)
    del validation_dataset
    
    print("Beginning training...")
    if torch.cuda.is_available():
        model = model.cuda()
    counter = 0
    not_improved_loss = 0
    previous_loss = math.inf
    for t in range(starting_epoch, opt.num_epochs + 1):
        if not_improved_loss == opt.patience:
            break
        counter = 0

        total_loss = 0
        total_val_loss = 0
        
        bar = ChargingBar('EPOCH #' + str(t), max=(len(dl)*config['training']['bs'])+len(val_dl))
        train_correct = 0
        positive = 0
        negative = 0
        for index, (images, labels) in enumerate(dl):
            images = np.transpose(images, (0, 3, 1, 2))
            labels = labels.unsqueeze(1)
            if torch.cuda.is_available():
                images = images.cuda()
            
            y_pred = model(images)
            y_pred = y_pred.cpu()
            
            loss = loss_fn(y_pred, labels)
        
            corrects, positive_class, negative_class = check_correct(y_pred, labels)  
            train_correct += corrects
            positive += positive_class
            negative += negative_class
            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()
            counter += 1
            total_loss += round(loss.item(), 2)
            
            if index % 1200 == 0: # Intermediate metrics print
                print("\nLoss: ", total_loss/counter, "Accuracy: ",train_correct/(counter*config['training']['bs']) ,"Train 0s: ", negative, "Train 1s:", positive)

            for i in range(config['training']['bs']):
                bar.next()

        val_correct = 0
        val_positive = 0
        val_negative = 0
        val_counter = 0
        train_correct /= train_samples
        total_loss /= counter
        for index, (val_images, val_labels) in enumerate(val_dl):
            val_images = np.transpose(val_images, (0, 3, 1, 2))
            if torch.cuda.is_available():
                val_images = val_images.cuda()
            val_labels = val_labels.unsqueeze(1)
            val_pred = model(val_images)
            val_pred = val_pred.cpu()
            val_loss = loss_fn(val_pred, val_labels)
            total_val_loss += round(val_loss.item(), 2)
            corrects, positive_class, negative_class = check_correct(val_pred, val_labels)
            val_correct += corrects
            val_positive += positive_class
            val_counter += 1
            val_negative += negative_class
            bar.next()
            
        scheduler.step()
        bar.finish()
        
        total_val_loss /= val_counter
        val_correct /= validation_samples
        if previous_loss <= total_val_loss:
            print("Validation loss did not improve")
            not_improved_loss += 1
        else:
            not_improved_loss = 0
        
        previous_loss = total_val_loss
        print("#" + str(t) + "/" + str(opt.num_epochs) + " loss:" +
            str(total_loss) + " accuracy:" + str(train_correct) +" val_loss:" + str(total_val_loss) + " val_accuracy:" + str(val_correct) + " val_0s:" + str(val_negative) + "/" + str(np.count_nonzero(validation_labels == 0)) + " val_1s:" + str(val_positive) + "/" + str(np.count_nonzero(validation_labels == 1)))
    
        if not os.path.exists(MODELS_PATH):
            os.makedirs(MODELS_PATH)
        torch.save(model.state_dict(), os.path.join(MODELS_PATH,  "efficientnetB"+str(opt.efficient_net)+"_checkpoint" + str(t) + "_" + opt.dataset))
