import os
import argparse
from os.path import join
import cv2
import dlib
import torch
import torch.nn as nn
from PIL import Image as pil_image
from tqdm import tqdm

from network.models import model_selection
from dataset.transform import xception_default_data_transforms


def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)
    return x1, y1, size_bb


def preprocess_image(image, device):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    preprocess = xception_default_data_transforms['test']
    preprocessed_image = preprocess(pil_image.fromarray(image))
    preprocessed_image = preprocessed_image.unsqueeze(0).to(device)
    return preprocessed_image


def predict_with_model(image, model, post_function=nn.Softmax(dim=1), device='cpu'):
    preprocessed_image = preprocess_image(image, device)
    output = model(preprocessed_image)
    output = post_function(output)
    _, prediction = torch.max(output, 1)
    prediction = int(prediction.cpu().numpy())
    return prediction, output


def test_full_image_network(video_path, model_path, output_path, start_frame=0, end_frame=None):
    print('Starting: {}'.format(video_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reader = cv2.VideoCapture(video_path)
    video_fn = video_path.split('/')[-1].split('.')[0] + '.avi'
    os.makedirs(output_path, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = reader.get(cv2.CAP_PROP_FPS)
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))

    if num_frames < 1:
        print("Error: The video file has too few frames.")
        return

    if start_frame >= num_frames:
        print(f"Warning: start_frame ({start_frame}) is greater than total frames ({num_frames}). Resetting to 0.")
        start_frame = 0

    if end_frame is None or end_frame > num_frames:
        print(f"Warning: end_frame ({end_frame}) exceeds total frames ({num_frames}). Resetting to total frames.")
        end_frame = num_frames

    writer = None
    face_detector = dlib.get_frontal_face_detector()

    model = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)
    model.load_state_dict(torch.load(model_path), strict=True)
    model = model.to(device)
    model.eval()

    font_face = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    font_scale = 1

    frame_num = 0
    pbar = tqdm(total=end_frame - start_frame)

    while reader.isOpened():
        _, image = reader.read()
        if image is None:
            break
        frame_num += 1

        if frame_num < start_frame:
            continue
        pbar.update(1)

        height, width = image.shape[:2]

        if writer is None:
            writer = cv2.VideoWriter(join(output_path, video_fn), fourcc, fps, (height, width)[::-1])

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 1)
        if len(faces):
            face = faces[0]

            x, y, size = get_boundingbox(face, width, height)
            cropped_face = image[y:y + size, x:x + size]

            prediction, output = predict_with_model(cropped_face, model, device=device)

            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y
            label = 'fake' if prediction == 1 else 'real'
            color = (0, 255, 0) if prediction == 0 else (0, 0, 255)
            output_list = ['{0:.2f}'.format(float(x)) for x in output.detach().cpu().numpy()[0]]
            cv2.putText(image, str(output_list) + '=>' + label, (x, y + h + 30), font_face, font_scale, color, thickness, 2)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

        if frame_num >= end_frame:
            break

        cv2.imshow('test', image)
        cv2.waitKey(33)
        writer.write(image)
    pbar.close()
    if writer is not None:
        writer.release()
        print('Finished! Output saved under {}'.format(output_path))
    else:
        print('Input video file was empty')


if __name__ == '__main__':
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--video_path', '-i', type=str, default = './jung1.mov')
    #p.add_argument('--model_path', '-mi', type=str, default='./output/deepfake_xception_kfold/new.pth')
    p.add_argument('--model_path', '-mi', type=str, default='/home/hail/Documents/jjw/output/deepfake_xception_last3/best_model_fold_5.pkl')
    p.add_argument('--output_path', '-o', type=str, default='./')
    p.add_argument('--start_frame', type=int, default=0)
    p.add_argument('--end_frame', type=int, default=None)
    args = p.parse_args()

    video_path = args.video_path
    if video_path.endswith('.mp4') or video_path.endswith('.avi') or video_path.endswith(('.mov')):
        test_full_image_network(**vars(args))
    else:
        videos = os.listdir(video_path)
        for video in videos:
            args.video_path = join(video_path, video)
            test_full_image_network(**vars(args))
