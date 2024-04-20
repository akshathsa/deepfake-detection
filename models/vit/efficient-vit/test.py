import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import auc # roc_auc_score
from sklearn.metrics import accuracy_score
import os
import cv2
import numpy as np
import torch
from torch import nn, einsum

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, precision_recall_curve
from albumentations import Compose, RandomBrightnessContrast, \
    HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur, Rotate

from transforms.albu import IsotropicResize
from joblib import Parallel, delayed

from utils import get_method, check_correct, resize, shuffle_dataset, get_n_params, video_to_frames
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from efficient_vit import EfficientViT
from utils import transform_frame, test_parse
import glob
from os import cpu_count
import json
from multiprocessing.pool import Pool
from progress.bar import Bar
import pandas as pd
from tqdm import tqdm
from multiprocessing import Manager
from utils import custom_round, custom_video_round

import yaml
import argparse

BASE_DIR = '/global/cfs/projectdirs/m3641/Akaash/deepfake-detection/'
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAINING_DIR = os.path.join(DATA_DIR, "dfdc_train")
VALIDATION_DIR = os.path.join(DATA_DIR, "dfdc_val")
TEST_DIR = os.path.join(DATA_DIR, "dfdc_test")

MODELS_PATH = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(MODELS_PATH, "output")

METADATA_PATH = os.path.join(DATA_DIR, "metadata") # Folder containing all training metadata for DFDC dataset
TRAIN_LABELS_PATH = os.path.join(DATA_DIR, "dfdc_train_labels.csv")
VALIDATION_LABELS_PATH = os.path.join(DATA_DIR, "dfdc_val_labels.csv")
TEST_LABELS_PATH = os.path.join(DATA_DIR, "dfdc_test_labels.csv")

if not os.path.exists(MODELS_PATH):
    os.makedirs(MODELS_PATH)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def create_base_transform(size):
    return Compose([
        IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
    ])

def save_roc_curves(correct_labels, preds, model_name, accuracy, loss, f1):
  plt.figure(1)
  plt.plot([0, 1], [0, 1], 'k--')

  fpr, tpr, th = metrics.roc_curve(correct_labels, preds)
  model_auc = auc(fpr, tpr)

  plt.plot(fpr, tpr, label="Model_"+ model_name + ' (area = {:.3f})'.format(model_auc))
  plt.xlabel('False positive rate')
  plt.ylabel('True positive rate')
  plt.title('ROC curve')
  plt.legend(loc='best')
  plt.savefig(os.path.join(OUTPUT_DIR, model_name +  "_" + opt.dataset + "_acc" + str(accuracy*100) + "_loss"+str(loss)+"_f1"+str(f1)+".jpg"))
  plt.clf()

def read_frames(video_path, videos, config):
    if TEST_DIR in video_path:
        test_df = pd.DataFrame(pd.read_csv(TEST_LABELS_PATH))
        video_folder_name = os.path.basename(video_path)
        video_key = video_folder_name + ".mp4"
        label = float(test_df.loc[test_df['filename'] == video_key]['label'].values[0])

    frames = os.listdir(video_path)
    
    video = {}
    for index, frame_image in enumerate(frames):
        image = cv2.imread(os.path.join(video_path, frame_image))
        if image is not None:
            transform = create_base_transform(config['model']['image-size'])
            image = transform(image=cv2.imread(os.path.join(video_path, frame_image)))['image']
            if len(image) > 0:
                video[index] = video.get(index, []) + [image]
    
    videos.append((video, label, video_path))

def read_frames_wrapper(video_path, videos, config):
    return read_frames(video_path, videos, config)

# Main body
if __name__ == "__main__":
    opt = test_parse()
    print(opt)

    with open(opt.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)

    if opt.efficient_net == 0:
        channels = 1280
    else:
        channels = 2560

    if os.path.exists(opt.model_path):
        model = EfficientViT(config=config, channels=channels, selected_efficient_net = opt.efficient_net)
        model.load_state_dict(torch.load(opt.model_path))
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
    else:
        print("No model found.")
        exit()

    model_name = os.path.basename(opt.model_path)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    dataset = TEST_DIR
    print("Extracting frames from videos...")
    videos = os.listdir(dataset)
    videos = [video for video in videos if video.endswith(".mp4")]
    for index, video in enumerate(videos):
        if index == opt.max_videos:
            break
        video_path = os.path.join(dataset, video)
        frame_path = os.path.join(dataset, video.split(".")[0])
        if not os.path.exists(frame_path):
            video_to_frames(video_path, frame_path, frame_skip=10)
    print("Frames extracted.")

    paths = []
    frame_folders = os.listdir(dataset)
    frame_folders = [frame_folder for frame_folder in frame_folders if os.path.isdir(os.path.join(dataset, frame_folder))]
    for index, frame_folder in enumerate(frame_folders):
        if index == opt.max_videos:
            break
        if os.path.isdir(os.path.join(dataset, frame_folder)):
            paths.append(os.path.join(dataset, frame_folder))

    real_paths = []
    fake_paths = []
    for json_path in glob.glob(os.path.join(DATA_DIR, "metadata", "*.json")):
        with open(json_path, "r") as f:
            metadata = json.load(f)
        
        frame_folders = os.listdir(dataset)
        frame_folders = [frame_folder for frame_folder in frame_folders if os.path.isdir(os.path.join(dataset, frame_folder))]
        for index, frame_folder in enumerate(frame_folders):
            if os.path.isdir(os.path.join(dataset, frame_folder)):
                if metadata[str(frame_folder) + ".mp4"]["label"] == "REAL":
                    real_paths.append(os.path.join(dataset, frame_folder))
                else:
                    fake_paths.append(os.path.join(dataset, frame_folder))
    
    print(len(real_paths))

    real_paths = real_paths[:opt.max_videos//2]
    fake_paths = fake_paths[:opt.max_videos//2]

    paths = real_paths + fake_paths

    # need to fix
    preds = []
    mgr = Manager()
    videos = mgr.list()

    # paths = []
    # folder = TEST_DIR
    # for index, video_folder in enumerate(os.listdir(method_folder)):
    #     paths.append(os.path.join(method_folder, video_folder))
      
    with tqdm(total=len(paths)) as pbar:
        with Parallel(n_jobs=56) as parallel:
            results = parallel(delayed(read_frames_wrapper)(path, videos, config) for path in paths)
            for _ in results:  # Update progress bar for each completed task
                pbar.update()

    videos = shuffle_dataset(videos)
    video_names = np.asarray([row[2] for row in videos])
    correct_test_labels = np.asarray([row[1] for row in videos])
    videos = np.asarray([row[0] for row in videos])
    preds = []

    bar = Bar('Predicting', max=len(videos))

    # f = open(opt.dataset + "_" + model_name + "_labels.txt", "w+")
    f = open(os.path.join(OUTPUT_DIR, f"dfdc_{model_name}_labels.txt"), "w+")
    for index, video in enumerate(videos):
        video_faces_preds = []
        video_name = video_names[index]
        f.write(video_name)
        for key in video:
            faces_preds = []
            video_faces = video[key]
            for i in range(0, len(video_faces), opt.batch_size):
                faces = video_faces[i:i+opt.batch_size]
                faces = torch.tensor(np.asarray(faces))
                if faces.shape[0] == 0:
                    continue
                faces = np.transpose(faces, (0, 3, 1, 2))
                if torch.cuda.is_available():
                    faces = faces.cuda().float()
                else:
                    faces = faces.float()
                
                pred = model(faces)
                
                scaled_pred = []
                for idx, p in enumerate(pred):
                    scaled_pred.append(torch.sigmoid(p))
                faces_preds.extend(scaled_pred)
                
            current_faces_pred = sum(faces_preds) / len(faces_preds)
            face_pred = current_faces_pred.cpu().detach().numpy()[0]
            f.write(" " + str(face_pred))
            video_faces_preds.append(face_pred)
        bar.next()
        if len(video_faces_preds) > 1:
            video_pred = custom_video_round(video_faces_preds) # ~majority vote over frames (0.55 threshold)
        else:
            video_pred = video_faces_preds[0]
        preds.append([video_pred])
        
        f.write(" --> " + str(video_pred) + "(CORRECT: " + str(correct_test_labels[index]) + ")" +"\n")
        
    f.close()
    bar.finish()

    loss_fn = torch.nn.BCEWithLogitsLoss()
    tensor_labels = torch.tensor([[float(label)] for label in correct_test_labels])
    tensor_preds = torch.tensor(preds)

    # evaluate on auc, precision, log loss

    loss = loss_fn(tensor_preds, tensor_labels).numpy() # log loss
    accuracy = accuracy_score(custom_round(np.asarray(preds)), correct_test_labels)
    f1 = f1_score(correct_test_labels, custom_round(np.asarray(preds)))

    precision = precision_score(correct_test_labels, custom_round(np.asarray(preds)))
    recall = recall_score(correct_test_labels, custom_round(np.asarray(preds)))
    
    prcurve = precision_recall_curve(correct_test_labels, custom_round(np.asarray(preds)))
    plt.plot(prcurve[1], prcurve[0])
    plt.savefig(os.path.join(OUTPUT_DIR, model_name + "_" + opt.dataset + "_prcurve.jpg"))
    plt.cla()
    
    print(model_name, "Test Accuracy:", accuracy, "Log Loss:", loss, "F1", f1)
    print("Precision:", precision, "Recall:", recall)
    save_roc_curves(correct_test_labels, preds, model_name, accuracy, loss, f1)
