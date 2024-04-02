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

def mask_rcnn_parse():
    # default arguments
    parser = argparse.ArgumentParser(description="Mask-RCNN (segmentation model) implementation in PyTorch")
    output_group = parser.add_mutually_exclusive_group()
    boxes_group = parser.add_mutually_exclusive_group()
    masks_group = parser.add_mutually_exclusive_group()
    labels_group = parser.add_mutually_exclusive_group()
    parser.add_argument("--grey-background","-g",action="store_true",help="make the background monochromatic")
    parser.add_argument("--classes","-c",nargs="+",default=["person"],help="limit to certain classes (all or see classes.txt)")
    parser.add_argument("--detection-threshold",default=0.7,type=float,help="confidence threshold for detection (0-1)")
    parser.add_argument("--mask-threshold",default=0.5,type=float,help="confidence threshold for segmentation mask (0-1)")
    parser.add_argument("--max-detections",default=0,type=int,help="maximum concurrent detections (leave 0 for unlimited)")
    parser.add_argument("--data-folder","--data","-d",default="data",help="data directory")
    output_group.add_argument("--hide-output",action="store_true",help="do not show output")
    output_group.add_argument("--display-title",default="Mask-RCNN",help="window title")
    output_group.add_argument("--skip-frames",default=1,type=int,help="skip frames (1 for no skip)")
    boxes_group.add_argument("--hide-boxes",action="store_true",help="do not show bounding boxes")
    masks_group.add_argument("--hide-masks",action="store_true",help="do not show segmentation masks")
    labels_group.add_argument("--show-labels",action="store_true",help="show labels")
    masks_group.add_argument("--mask-opacity",default=0.4,type=float,help="opacity of segmentation masks")
    parser.add_argument("--show-fps",action="store_true",help="display processing speed (fps)")
    labels_group.add_argument("--text-thickness",default=2,type=int,help="thickness of label text")
    boxes_group.add_argument("--box-thickness",default=3,type=int,help="thickness of boxes")

    folder_group = parser.add_mutually_exclusive_group()
    folder_group.add_argument("--output-folder","--output","-o",default="output",help="output save location")
    folder_group.add_argument("--no-save",action="store_true",help="do not save output images")
    folder_group.add_argument("--extensions","-e",nargs="+",default=["png", "jpeg", "jpg", "bmp", "tiff", "tif"],help="image file extensions")

    return parser

def mask_rcnn_detect(image, model, classes, include_classes, args, colors):
    # feed forward the image
    output = model(torch.tensor(np.expand_dims(image,axis=0)).permute(0,3,1,2) / 255)[0]
    cover = np.zeros(image.shape,dtype=bool)
    i = 0
    for box, label, score, mask in zip(*output.values()):
        # check if we need to keep detecting
        if score < args.detection_threshold or (i >= args.max_detections and args.max_detections != 0):
            break
        # ignore irrelevant classes
        if not classes[label] in include_classes:
            continue
        i += 1

        print(f'{classes[label]}: {score:.2f}')
        
        if not args.hide_masks: # draw mask
            image[mask[0] > args.mask_threshold] = image[mask[0] > args.mask_threshold] * (1 - args.mask_opacity) + args.mask_opacity * np.array(colors[label])

        # update the cover
        cover[mask[0] > args.mask_threshold] = 1

        if not args.hide_boxes: # draw bounding box and make surrounding area black
            cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), colors[label], args.box_thickness)
            mask = np.ones(image.shape[:2], dtype="uint8") * 255
            cv2.rectangle(mask, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 0), -1) # Fill the bounding box area with black in the mask
            mask = cv2.bitwise_not(mask) # Invert the mask (everything outside the box becomes white)
            image = cv2.bitwise_and(image, image, mask=mask) # Apply bitwise AND operation to keep only the desired color (black outside the box)

        if args.show_labels:
            cv2.putText(image, f'{classes[label]}: {score:.2f}', (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[label], args.text_thickness)

        image = image[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
    
    ret, thresh = cv2.threshold(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY), 220, 255, cv2.THRESH_BINARY)
    if args.hide_boxes: # if not showing boxes, make non-masked areas black
        image[~cover] = 0 # np.tile(np.expand_dims(thresh,axis=2),(1,1,3))[~cover]
    image[thresh == 255] = 0
    return image

def extract_audio(input_video):
    if not os.path.exists('audio'):
        os.makedirs('audio')
    input_stream = ffmpeg.input(input_video)
    # input_stream.output('audio.mp3', acodec='mp3').run()
    audio = input_stream.audio
    output_file = f'audio/output_{input_video.split(".")[0]}.mp3'
    print(output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    output_stream = ffmpeg.output(audio, output_file)
    ffmpeg.run(output_stream)

def output_video(processed_frame_folder, output_folder, video):
    print(processed_frame_folder, output_folder, video)
    output_file = f'{output_folder}/output_{video.split(".")[0]}.mp4'
    if os.path.exists(output_file):
        os.remove(output_file)
    input_stream = ffmpeg.input(f'{processed_frame_folder}/frame%d.jpg')
    # audio_stream = ffmpeg.input(f'audio/output_{input_video.split(".")[0]}.mp3')
    output_stream = ffmpeg.output(input_stream, output_file)
    ffmpeg.run(output_stream)