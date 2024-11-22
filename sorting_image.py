import os
import pandas as pd
from shutil import move

# 데이터셋 폴더 경로
base_dir = 'D:/east_df'

# 레이블 파일 경로 (train, valid, test 각각의 labels.csv 파일이 있다고 가정)
label_files = {
    'train': os.path.join(base_dir, 'train', 'labels.csv'),
    'valid': os.path.join(base_dir, 'valid', 'labels.csv'),
    'test': os.path.join(base_dir, 'test', 'labels.csv')
}

# fake와 real 이미지를 저장할 디렉토리
fake_dir = os.path.join(base_dir, 'fake')
real_dir = os.path.join(base_dir, 'real')

# 폴더 생성
os.makedirs(fake_dir, exist_ok=True)
os.makedirs(real_dir, exist_ok=True)

# 각 폴더의 레이블 파일을 읽고 이미지를 분류
for dataset, label_file in label_files.items():
    # CSV 파일 읽기
    labels_df = pd.read_csv(label_file)

    # 열 이름의 앞뒤 공백 제거
    labels_df.columns = labels_df.columns.str.strip()

    # 이미지 경로 생성 및 이동
    dataset_folder = os.path.join(base_dir, dataset)
    for index, row in labels_df.iterrows():
        image_name = row['filename']
        is_fake = row['deepfake']  # 딥페이크 여부 (1: fake, 0: real)

        # 이미지 경로
        image_path = os.path.join(dataset_folder, image_name)

        # 이미지가 존재하는지 확인
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found. Skipping.")
            continue

        # fake 또는 real 폴더로 이동
        if is_fake == 1:
            move(image_path, os.path.join(fake_dir, image_name))
        elif is_fake == 0:
            move(image_path, os.path.join(real_dir, image_name))

print("이미지 분류 완료!")
