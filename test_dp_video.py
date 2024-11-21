import os
import argparse
from os.path import join
import cv2
import dlib
import torch
import torch.nn as nn
from PIL import Image as pil_image
from tqdm import tqdm
import wandb
from sklearn.metrics import roc_auc_score, average_precision_score

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


def preprocess_image(image, cuda=True):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    preprocess = xception_default_data_transforms['test']
    preprocessed_image = preprocess(pil_image.fromarray(image))
    preprocessed_image = preprocessed_image.unsqueeze(0)
    if cuda:
        preprocessed_image = preprocessed_image.cuda()
    return preprocessed_image


def predict_with_model(image, model, post_function=nn.Softmax(dim=1), cuda=True):
    preprocessed_image = preprocess_image(image, cuda)
    output = model(preprocessed_image)
    output = post_function(output)
    _, prediction = torch.max(output, 1)
    prediction = float(prediction.cpu().numpy())
    return int(prediction), output


def test_full_image_network(video_path, model_path, output_path,
                            start_frame=0, end_frame=None, cuda=True, save_output=True):
    print('Starting: {}'.format(video_path))

    # wandb 초기화
    wandb.init(project="df_video_classification", name="test_run")

    # Read and write
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

    # 모델 불러오기
    model = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)
    model.load_state_dict(torch.load(model_path))
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    if cuda:
        model = model.cuda()

    thickness = 2
    font_scale = 1

    frame_num = 0
    correct_predictions = 0  # 정확도 측정을 위한 변수
    total_predictions = 0  # 총 예측 수
    all_predictions = []  # 모든 예측값을 저장
    all_labels = []  # 모든 정답 레이블을 저장
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

        if writer is None and save_output:
            writer = cv2.VideoWriter(join(output_path, video_fn), fourcc, fps,
                                     (width, height))  # (width, height) 순서로 수정

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 1)
        if len(faces):
            face = faces[0]
            x, y, size = get_boundingbox(face, width, height)
            cropped_face = image[y:y + size, x:x + size]

            # 예측
            prediction, output = predict_with_model(cropped_face, model, cuda=cuda)
            label = 'fake' if prediction == 1 else 'real'
            color = (0, 255, 0) if prediction == 0 else (0, 0, 255)

            # 성능 측정: 예측 정확도 계산
            total_predictions += 1

            # 영상 파일명에 따라 정답 레이블 설정
            if "_real" in video_path.lower():
                correct_label = 0  # 실제 영상일 경우
            elif "_fake" in video_path.lower():
                correct_label = 1  # 페이크 영상일 경우
            else:
                print(f"Error: Could not determine the label for {video_path}. Skipping this video.")
                continue

            all_labels.append(correct_label)
            all_predictions.append(output.detach().cpu().numpy()[0][1])  # fake 확률을 기록

            if prediction == correct_label:
                correct_predictions += 1

            # wandb에 로깅
            wandb.log({"frame": frame_num, "prediction": prediction, "correct_label": correct_label})

        if frame_num >= end_frame:
            break

    pbar.close()
    if writer is not None:
        writer.release()
        print('Finished! Output saved under {}'.format(output_path))

    # 최종 성능 기록 (정확도)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f'Accuracy: {accuracy:.4f}')
    wandb.log({"accuracy": accuracy})  # 최종 정확도를 wandb에 기록

    # # AUC 및 AP 계산
    # if len(all_labels) > 0:
    #     auc_score = roc_auc_score(all_labels, all_predictions)
    #     ap_score = average_precision_score(all_labels, all_predictions)
    #
    #     print(f'AUC: {auc_score:.4f}, AP: {ap_score:.4f}')
    #     wandb.log({"auc": auc_score, "average_precision": ap_score})  # AUC 및 AP 기록
    #
    # # wandb 종료
    # wandb.finish()


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--video_path', '-i', default = 'D:/test_data2',type=str)
    p.add_argument('--model_path', '-mi', default ='./pretrained_model/deepfake_c0_xception.pkl' ,type=str)
    p.add_argument('--output_path', '-o', type=str, default='.')
    p.add_argument('--start_frame', type=int, default=0)
    p.add_argument('--end_frame', type=int, default=None)
    p.add_argument('--cuda',action='store_true')
    p.add_argument('--save_output', action='store_true', default=False, help="Save output video")
    args = p.parse_args()

    # 모든 동영상 파일을 테스트하기 위해 리스트에 추가
    if os.path.isdir(args.video_path):
        videos = [f for f in os.listdir(args.video_path) if f.endswith(('.mp4', '.avi'))]
        print("Testing the following videos:")
        for video in videos:
            video_full_path = join(args.video_path, video)
            print(video_full_path)
            test_full_image_network(video_full_path, args.model_path, args.output_path,
                                    start_frame=args.start_frame, end_frame=args.end_frame,
                                    cuda=args.cuda, save_output=args.save_output)
    else:
        # 단일 파일 테스트
        video_path = args.video_path
        print("Testing the following video:")
        print(video_path)
        test_full_image_network(video_path, args.model_path, args.output_path,
                                start_frame=args.start_frame, end_frame=args.end_frame,
                                cuda=args.cuda, save_output=args.save_output)
