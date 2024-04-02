from glob import glob
import cv2
from albumentations import Compose, PadIfNeeded
from transforms.albu import IsotropicResize
import numpy as np
import os
import cv2
import torch
from statistics import mean
import argparse
import shutil
from sklearn.model_selection import train_test_split
import pandas as pd
import json

def train_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=300, type=int, help='Number of training epochs.')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loader workers.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='Path to latest checkpoint (default: none).')
    parser.add_argument('--dataset', type=str, default='DFDC', help="Which dataset to use (Deepfakes|Face2Face|FaceShifter|FaceSwap|NeuralTextures|All)")
    parser.add_argument('--max_train_videos', type=int, default=-1, help="Maximum number of videos to use for training (default: all).")
    parser.add_argument('--max_val_videos', type=int, default=-1, help="Maximum number of videos to use for validation (default: all).")
    parser.add_argument('--config', type=str, help="config file to use")
    parser.add_argument('--efficient_net', type=int, default=0, help="Which EfficientNet version to use (0 or 7, default: 0)")
    parser.add_argument('--patience', type=int, default=5, help="How many epochs wait before stopping for validation loss not improving.")
    
    # have not used these yet
    parser.add_argument('--frames_per_video', type=int, default=30, help="How many equidistant frames for each video (default: 30)")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size (default: 32)")
    
    opt = parser.parse_args()
    return opt

def test_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', default=4, type=int, help='Number of data loader workers.')
    parser.add_argument('--model_path', default='', type=str, metavar='PATH', help='Path to model checkpoint (default: none).')
    parser.add_argument('--dataset', type=str, default='DFDC', help="Which dataset to use (Deepfakes|Face2Face|FaceShifter|FaceSwap|NeuralTextures|DFDC)")
    parser.add_argument('--max_videos', type=int, default=-1, help="Maximum number of videos to use for testing (default: all).")
    parser.add_argument('--config', type=str, help="config file to use")
    parser.add_argument('--efficient_net', type=int, default=0, help="Which EfficientNet version to use (0 or 7, default: 0)")
    parser.add_argument('--frames_per_video', type=int, default=30, help="How many equidistant frames for each video (default: 30)")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size (default: 32)")
    
    opt = parser.parse_args()
    return opt

def transform_frame(image, image_size):
    transform_pipeline = Compose([
            IsotropicResize(max_side=image_size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
            PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_REPLICATE)
        ]
    )
    return transform_pipeline(image=image)['image']

def resize(image, image_size):
    try:
        return cv2.resize(image, dsize=(image_size, image_size))
    except:
        return []

def custom_round(values):
    result = []
    for value in values:
        if value > 0.55:
            result.append(1)
        else:
            result.append(0)
    return np.asarray(result)

def custom_video_round(preds):
    for pred_value in preds:
        if pred_value > 0.55:
            return pred_value
    return mean(preds)

def get_method(video, data_path):
    methods = os.listdir(os.path.join(data_path, "manipulated_sequences"))
    methods.extend(os.listdir(os.path.join(data_path, "original_sequences")))
    methods.append("DFDC")
    methods.append("Original")
    selected_method = ""
    for method in methods:
        if method in video:
            selected_method = method
            break
    return selected_method

def shuffle_dataset(dataset):
  import random
  random.seed(69)
  random.shuffle(dataset)
  return dataset

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp
    
def check_correct(preds, labels):
    preds = preds.cpu()
    labels = labels.cpu()
    preds = [np.asarray(torch.sigmoid(pred).detach().numpy()).round() for pred in preds]

    correct = 0
    positive_class = 0
    negative_class = 0
    for i in range(len(labels)):
        pred = int(preds[i])
        if labels[i] == pred:
            correct += 1
        if pred == 1:
            positive_class += 1
        else:
            negative_class += 1
    return correct, positive_class, negative_class

def train_val_test_split(data_path, TRAINING_DIR, VALIDATION_DIR, TEST_DIR, train_size, val_size, test_size): # preprocessing to create train and val splits
    assert train_size + val_size + test_size == 1

    if not os.path.exists(os.path.join(data_path, "dfdc_train_labels.csv")) or not os.path.exists(os.path.join(data_path, "dfdc_val_labels.csv")) or not os.path.exists(os.path.join(data_path, "dfdc_test_labels.csv")):
        video_list = []
        labels = []
        for json_path in glob(os.path.join(data_path, "metadata", "*.json")):
            with open(json_path, "r") as f:
                metadata = json.load(f)
            for k, v in metadata.items():
                video_list.append(k[:-4])
                if v["label"] == "FAKE":
                    labels.append(1)
                else:
                    labels.append(0)
        
        train, test, train_labels, test_labels = train_test_split(video_list, labels, test_size=1-train_size, random_state=69)
        val, test, val_labels, test_labels = train_test_split(test, test_labels, test_size=test_size/(test_size+val_size), random_state=69)
        if not os.path.exists(TRAINING_DIR):
            os.makedirs(TRAINING_DIR)
        if not os.path.exists(VALIDATION_DIR):
            os.makedirs(VALIDATION_DIR)
        if not os.path.exists(TEST_DIR):
            os.makedirs(TEST_DIR)

        for dir in os.listdir(data_path):
            if "dfdc" in dir and dir not in TRAINING_DIR and dir not in VALIDATION_DIR:
                for video in os.listdir(os.path.join(data_path, dir)):
                    if video[:-4] in train:
                        shutil.copyfile(os.path.join(data_path, dir, video), os.path.join(TRAINING_DIR, video))
                    elif video[:-4] in val:
                        shutil.copyfile(os.path.join(data_path, dir, video), os.path.join(VALIDATION_DIR, video))
                    else:
                        shutil.copyfile(os.path.join(data_path, dir, video), os.path.join(TEST_DIR, video))
        
        train_df = pd.DataFrame({"filename": [video + ".mp4" for video in train], "label": train_labels})
        val_df = pd.DataFrame({"filename": [video + ".mp4" for video in val], "label": val_labels})
        train_df.to_csv(os.path.join(data_path, "dfdc_train_labels.csv"), index=False)
        val_df.to_csv(os.path.join(data_path, "dfdc_val_labels.csv"), index=False)
        test_df = pd.DataFrame({"filename": [video + ".mp4" for video in test], "label": test_labels})
        test_df.to_csv(os.path.join(data_path, "dfdc_test_labels.csv"), index=False)

def video_to_frames(video_path, frames_path, frame_skip=10):
    if not os.path.exists(frames_path):
        os.makedirs(frames_path)
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        if count % frame_skip == 0:
            cv2.imwrite(os.path.join(frames_path, f"frame{count}.jpg"), image)
        success, image = vidcap.read()
        count += 1
