import os
import cv2
import time
import torch
import random
import argparse
from glob import glob

from torchvision.models.detection import maskrcnn_resnet50_fpn_v2 as maskrcnn
from torchvision.models.detection.mask_rcnn import MaskRCNN_ResNet50_FPN_V2_Weights
import numpy as np

import ffmpeg

from utils import mask_rcnn_parse, mask_rcnn_detect, output_video, extract_audio

# classes and randomly generated colors
classes = ["BG","person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant",None,"stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",None,"backpack","umbrella",None,None,"handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",None,"wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed",None,"dining table",None,None,"toilet",None,"tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",None,"book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]
colors = [[random.randint(0,255) for c in range(3)] for _ in range(len(classes))]

parser = mask_rcnn_parse()
args = parser.parse_args()

include_classes = classes[1:] if "all" in args.classes else [c for c in args.classes if c in classes]
# mode = args.action

# load model
device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")
model = maskrcnn(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT).eval().to(device)
# if torch.cuda.is_available():
#     model = model.cuda()

def mask_rcnn_inference():
    for dir in os.listdir(args.data_folder):
        if not os.path.isdir(f'{args.data_folder}/{dir}') and not dir.endswith('45'):
            continue
        for video in os.listdir(f'{args.data_folder}/{dir}'):
            print(f'Processing {video}')
            if video in os.listdir("processed_frames"):
                continue

            # print(f'Extracting audio from {video}')
            # extract_audio(video)
            
            output_folder = args.output_folder
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            
            input_video = f'{args.data_folder}/{dir}/{video}'
            source = input_video
            cap = cv2.VideoCapture(source)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # width = 1920
            # height = 1080
            # print(f'Video resolution: {width}x{height}')

            if cap is None or not cap.isOpened():
                raise RuntimeError(f"video (\"{source}\") is not a valid input")
            
            frame_folder = f'video_frames/{video.split(".")[0]}'
            if not os.path.exists(frame_folder):
                os.makedirs(frame_folder)
            
            processed_frame_folder = f'processed_frames/{video.split(".")[0]}'
            if not os.path.exists(processed_frame_folder):
                os.makedirs(processed_frame_folder)
            
            skip_frames = args.skip_frames # change to skip frames
            frame_count = 0
            # extract frames from video
            while True:
                success, image = cap.read()
                if not success:
                    break
                if frame_count % skip_frames == 0:
                    cv2.imwrite(os.path.join(frame_folder, f'frame{int(frame_count / skip_frames)}.jpg'), image)
                frame_count += 1
            cap.release()

            files = []
            folder = frame_folder
            # add a "/" to the end of the folder
            if not folder.endswith("/") and not folder.endswith("\\"):
                folder += "/"
            # create a list of files
            for extension in args.extensions:
                files += glob(f"{folder}**/*.{extension}",recursive=True)

            for path in files:
                # get the path and image
                image = cv2.imread(path)
                
                # run the detection
                image = mask_rcnn_detect(image, model, classes, include_classes, args, colors)
                
                # save output
                # if not args.no_save:
                #     if len(images) > 1: # more than 1 person in video
                #         for idx, image in enumerate(images):
                #             save_path = os.path.join(processed_frame_folder, str(idx), os.path.relpath(path, folder))
                #             # save image
                #             directory = os.path.dirname(save_path)
                #             if directory:
                #                 os.makedirs(directory,exist_ok=True)
                #             cv2.imwrite(save_path,image)
                
                if not args.no_save:
                    save_path = os.path.join(processed_frame_folder, os.path.relpath(path, folder))
                    # save image
                    directory = os.path.dirname(save_path)
                    if directory:
                        os.makedirs(directory,exist_ok=True)
                    cv2.imwrite(save_path,image)

            # output_video(processed_frame_folder, output_folder, video, num_people)
            # output_video(processed_frame_folder, output_folder, video)

mask_rcnn_inference()
