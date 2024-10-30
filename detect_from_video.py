"""
Evaluates a folder of video files or a single file with a Xception binary
classification network.

Usage:
python detect_from_video.py
    -i <folder with video files or path to video file>
    -m <path to model file>
    -o <path to output folder, will write one or multiple output videos there>

Author: Andreas Rössler
"""

import os
import argparse
import cv2
import dlib
import torch
import torch.nn as nn
from PIL import Image as pil_image
from tqdm import tqdm
from network.models import model_selection
from dataset.transform import xception_default_data_transforms


def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    """
    Generates a quadratic bounding box around a detected face.
    """
    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        size_bb = max(size_bb, minsize)
    
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    size_bb = min(size_bb, width - x1, height - y1)

    return x1, y1, size_bb


def preprocess_image(image, cuda=True):
    """
    Preprocesses the image for model input.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    preprocess = xception_default_data_transforms['test']
    preprocessed_image = preprocess(pil_image.fromarray(image)).unsqueeze(0)
    
    return preprocessed_image.cuda() if cuda else preprocessed_image


def predict_with_model(image, model, post_function=nn.Softmax(dim=1), cuda=True):
    """
    Predicts the label of an input image.
    """
    preprocessed_image = preprocess_image(image, cuda)
    output = model(preprocessed_image)
    output = post_function(output)

    _, prediction = torch.max(output, 1)
    return int(prediction.cpu().numpy()), output


def load_model(model_path, cuda):
    """
    Loads the Xception model from the specified path.
    """
    model = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)
    model.load_state_dict(torch.load(model_path))
    return model.cuda() if cuda else model


def process_video(video_path, model_path, output_path, start_frame=0, end_frame=None, cuda=True):
    """
    Processes the video file and evaluates frames using the model.
    """
    print(f'Starting: {video_path}')
    reader = cv2.VideoCapture(video_path)
    video_fn = os.path.splitext(os.path.basename(video_path))[0] + '.avi'
    os.makedirs(output_path, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = reader.get(cv2.CAP_PROP_FPS)
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = None

    face_detector = dlib.get_frontal_face_detector()
    model = load_model(model_path, cuda)
    
    end_frame = end_frame if end_frame else num_frames
    pbar = tqdm(total=end_frame - start_frame)

    for frame_num in range(num_frames):
        ret, image = reader.read()
        if not ret or frame_num < start_frame:
            continue
        
        pbar.update(1)
        height, width = image.shape[:2]

        if writer is None:
            writer = cv2.VideoWriter(join(output_path, video_fn), fourcc, fps, (width, height))

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 1)
        
        if faces:
            face = faces[0]
            x, y, size = get_boundingbox(face, width, height)
            cropped_face = image[y:y + size, x:x + size]

            prediction, output = predict_with_model(cropped_face, model, cuda=cuda)
            label = 'fake' if prediction == 1 else 'real'
            color = (0, 255, 0) if prediction == 0 else (0, 0, 255)

            output_list = ['{:.2f}'.format(float(x)) for x in output.detach().cpu().numpy()[0]]
            cv2.putText(image, f"{output_list} => {label}", (x, y + size + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            cv2.rectangle(image, (x, y), (x + size, y + size), color, 2)

        if frame_num >= end_frame:
            break

        cv2.imshow('test', image)
        writer.write(image)

    pbar.close()
    if writer:
        writer.release()
        print(f'Finished! Output saved under {output_path}')
    else:
        print('Input video file was empty')


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--video_path', '-i', type=str, required=True)
    parser.add_argument('--model_path', '-m', type=str, required=True)
    parser.add_argument('--output_path', '-o', type=str, default='.')
    parser.add_argument('--start_frame', type=int, default=0)
    parser.add_argument('--end_frame', type=int, default=None)
    parser.add_argument('--cuda', action='store_true')
    
    args = parser.parse_args()
    video_path = args.video_path

    if os.path.isfile(video_path) and (video_path.endswith('.mp4') or video_path.endswith('.avi')):
        process_video(video_path, args.model_path, args.output_path, args.start_frame, args.end_frame, args.cuda)
    elif os.path.isdir(video_path):
        videos = [f for f in os.listdir(video_path) if f.endswith(('.mp4', '.avi'))]
        for video in videos:
            process_video(join(video_path, video), args.model_path, args.output_path, args.start_frame, args.end_frame, args.cuda)
    else:
        print("Error: Invalid video path. Please provide a valid file or directory.")


if __name__ == '__main__':
    main()
