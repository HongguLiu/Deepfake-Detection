import os
import argparse
import cv2
import dlib
import torch
import torch.nn as nn
from PIL import Image as pil_image
from tqdm import tqdm
import wandb
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from network.models import model_selection
from dataset.transform import xception_default_data_transforms

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

def test_images_from_csv(csv_path, model_path, cuda=True):
    # wandb 초기화
    wandb.init(project="df_image_classification", name="east_image_test")

    # CSV 파일 읽기
    labels_df = pd.read_csv(csv_path)
    labels_df.columns = labels_df.columns.str.strip()  # 모든 열 이름의 앞뒤 공백 제거

    # 모델 불러오기
    model = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)
    model.load_state_dict(torch.load(model_path))
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    if cuda:
        model = model.cuda()

    face_detector = dlib.get_frontal_face_detector()

    correct_predictions = 0  # 정확도 측정을 위한 변수
    total_predictions = 0  # 총 예측 수
    all_predictions = []  # 모든 예측값을 저장
    all_labels = []  # 모든 정답 레이블을 저장

    pbar = tqdm(total=len(labels_df))

    # 이미지 처리
    for index, row in labels_df.iterrows():
        image_path = os.path.join(os.path.dirname(csv_path), row['filename'])
        correct_label = row['deepfake']  # 딥페이크 라벨 (1: fake, 0: real)

        if not os.path.exists(image_path):
            print(f"Error: Image {image_path} not found.")
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}.")
            continue

        # 얼굴 검출
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 1)
        if len(faces) == 0:
            print(f"No face detected in image {image_path}. Skipping.")
            continue

        face = faces[0]
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cropped_face = image[y1:y2, x1:x2]

        # 예측
        prediction, output = predict_with_model(cropped_face, model, cuda=cuda)
        all_labels.append(correct_label)
        all_predictions.append(output.detach().cpu().numpy()[0][1])  # fake 확률을 기록

        if prediction == correct_label:
            correct_predictions += 1

        total_predictions += 1
        pbar.update(1)

        # wandb에 로깅
        wandb.log({"image": image_path, "prediction": prediction, "correct_label": correct_label})

    pbar.close()

    # 최종 성능 기록 (정확도)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f'Accuracy: {accuracy:.4f}')
    wandb.log({"accuracy": accuracy})  # 최종 정확도를 wandb에 기록

    # AUC 및 AP 계산
    if len(set(all_labels)) > 1:  # 클래스가 하나뿐인 경우 AUC 계산 불가
        auc_score = roc_auc_score(all_labels, all_predictions)
        ap_score = average_precision_score(all_labels, all_predictions)

        print(f'AUC: {auc_score:.4f}, AP: {ap_score:.4f}')
        wandb.log({"auc": auc_score, "average_precision": ap_score})  # AUC 및 AP 기록
    else:
        print("Skipping AUC and AP calculation due to only one class present.")

    # wandb 종료
    wandb.finish()

if __name__ == '__main__':
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--csv_path', '-c', required=True, type=str, help="Path to the CSV file with image paths and labels.")
    p.add_argument('--model_path', '-mi', default='./pretrained_model/ffpp_c23.pth', type=str)
    p.add_argument('--cuda', action='store_true', help="Use CUDA if available.")
    args = p.parse_args()

    test_images_from_csv(args.csv_path, args.model_path, cuda=args.cuda)
