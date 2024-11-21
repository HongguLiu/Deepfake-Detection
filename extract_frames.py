#
# #라벨 이어쓰기
import cv2
import numpy as np
import os
from tqdm import tqdm
import argparse

# Argument Parser 설정
parser = argparse.ArgumentParser(description='Extract frames and label videos')
parser.add_argument('--input_root', help='path to root directory containing fake, real, and real2 directories',default='/home/hail/Documents/train_data2/')
parser.add_argument('--out_root', help='path to output directory', default='/home/hail/Documents/train_frames2/')
parser.add_argument('--num_frames', type=int, help='the number of frames to extract', default=300)
parser.add_argument('--label_file', type=str, help='path to save label file', default='last2_labels.txt')
args = parser.parse_args()

# 출력 디렉토리 생성
if not os.path.exists(args.out_root):
    os.makedirs(args.out_root)

# 기존 라벨 파일이 있으면 추가 모드로 열기 ('a'), 없으면 생성 ('w')
file_mode = 'a' if os.path.exists(args.label_file) else 'w'

# 라벨 파일 열기
with open(args.label_file, file_mode) as label_file:
    # 'fake', 'real', 그리고 'real2' 디렉토리를 순회하며 mp4 파일을 처리
    #for label_name, label in [('fake', 1), ('real', 0), ('real2', 0), ('real3', 0)]:
    for label_name, label in [('fake', 1),('fake3',1),('fake4',1),('real2',0), ('real',0),('real3',0),('real4',0),('fake_video(1114)',1),('real_video(1114)',0),('fake_video',1),('real_video',0)]:
        label_dir = os.path.join(args.input_root, label_name)
        if not os.path.exists(label_dir):
            print(f"Warning: {label_name} 디렉토리가 존재하지 않습니다.")
            continue

        # 현재 레이블의 동영상 파일 처리
        to_iterate = list(os.walk(label_dir))
        for root, dirs, files in tqdm(to_iterate, total=len(to_iterate)):
            for file in files:
                if file.endswith('.mp4') or file.endswith('.mov'):
                    image_path = os.path.join(root, file)
                    vidcap = cv2.VideoCapture(image_path)

                    # 동영상이 제대로 열렸는지 확인
                    if not vidcap.isOpened():
                        print(f"Error: Cannot open video file {image_path}")
                        continue

                    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
                    print(f"{file}: {frame_count} frames found")  # 프레임 수 확인용 로그

                    # 고정된 프레임 개수로 추출할 인덱스 계산
                    frame_idxs = np.linspace(0, frame_count - 1, args.num_frames, endpoint=True, dtype=int)

                    # 출력 경로 설정 및 폴더 생성
                    relative_out_path = os.path.relpath(image_path, args.input_root)[:-4]  # .mp4 확장자 제거
                    out_path = os.path.join(args.out_root, relative_out_path)
                    if not os.path.exists(out_path):
                        os.makedirs(out_path)

                    # 프레임 추출 및 저장
                    success, image = vidcap.read()
                    count = 0
                    while success:
                        if count not in frame_idxs:
                            success, image = vidcap.read()
                            count += 1
                            continue
                        # 프레임 저장 경로 생성 및 저장
                        cur_out_path = os.path.join(out_path, f'frame{count:04d}.jpg')
                        cv2.imwrite(cur_out_path, image)  # JPEG 파일로 프레임 저장

                        # 라벨 파일에 이미지 경로와 레이블 기록
                        label_file.write(f"{cur_out_path} {label}\n")

                        success, image = vidcap.read()
                        count += 1

                    vidcap.release()
                    print(f"Extracted frames for {file}")


# import cv2
# import numpy as np
# import os
# from tqdm import tqdm
# import argparse
#
# # Argument Parser 설정
# parser = argparse.ArgumentParser(description='Extract frames and label videos')
# parser.add_argument('--input_root', help='path to root directory containing fake, real, and real2 directories')
# parser.add_argument('--out_root', help='path to output directory', default='/home/hail/Documents/train_frames/')
# parser.add_argument('--label_file', type=str, help='path to save label file', default='last_labels.txt')
# args = parser.parse_args()
#
# # 출력 디렉토리 생성
# if not os.path.exists(args.out_root):
#     os.makedirs(args.out_root)
#
# # 기존 라벨 파일이 있으면 추가 모드로 열기 ('a'), 없으면 생성 ('w')
# file_mode = 'a' if os.path.exists(args.label_file) else 'w'
#
# # 라벨 파일 열기
# with open(args.label_file, file_mode) as label_file:
#     for label_name, label in [('fake', 1), ('fake3',1),('real', 0), ('real2', 0), ('real3', 0)]:
#         label_dir = os.path.join(args.input_root, label_name)
#         if not os.path.exists(label_dir):
#             print(f"Warning: {label_name} 디렉토리가 존재하지 않습니다.")
#             continue
#
#         # 현재 레이블의 동영상 파일 처리
#         to_iterate = list(os.walk(label_dir))
#         for root, dirs, files in tqdm(to_iterate, total=len(to_iterate)):
#             for file in files:
#                 if file.endswith('.mp4'):
#                     image_path = os.path.join(root, file)
#                     vidcap = cv2.VideoCapture(image_path)
#
#                     # 동영상이 제대로 열렸는지 확인
#                     if not vidcap.isOpened():
#                         print(f"Error: Cannot open video file {image_path}")
#                         continue
#
#                     frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
#                     fps = int(vidcap.get(cv2.CAP_PROP_FPS))
#                     video_length_sec = frame_count / fps
#
#                     if label is 0 :
#                         num_frames = 170
#                     elif label is 1 :
#                         num_frames = 200
#                     # # 동영상 길이에 따라 프레임 수 설정
#                     # if video_length_sec <= 10:
#                     #     num_frames = 10  # 10초 이하인 경우
#                     # elif video_length_sec <= 30:
#                     #     num_frames = 30  # 10~30초
#                     # elif video_length_sec <= 60:
#                     #     num_frames = 60  # 30초~1분
#                     # else:
#                     #     num_frames = 90  # 1분 이상
#
#                     # 고정된 프레임 개수로 추출할 인덱스 계산
#                     frame_idxs = np.linspace(0, frame_count - 1, num_frames, endpoint=True, dtype=int)
#
#                     # 출력 경로 설정 및 폴더 생성
#                     relative_out_path = os.path.relpath(image_path, args.input_root)[:-4]  # .mp4 확장자 제거
#                     out_path = os.path.join(args.out_root, relative_out_path)
#                     if not os.path.exists(out_path):
#                         os.makedirs(out_path)
#
#                     # 프레임 추출 및 저장
#                     success, image = vidcap.read()
#                     count = 0
#                     while success:
#                         if count not in frame_idxs:
#                             success, image = vidcap.read()
#                             count += 1
#                             continue
#                         # 프레임 저장 경로 생성 및 저장
#                         cur_out_path = os.path.join(out_path, f'frame{count:04d}.jpg')
#                         cv2.imwrite(cur_out_path, image)  # JPEG 파일로 프레임 저장
#
#                         # 라벨 파일에 이미지 경로와 레이블 기록
#                         label_file.write(f"{cur_out_path} {label}\n")
#
#                         success, image = vidcap.read()
#                         count += 1
#
#                     vidcap.release()
#                     print(f"Extracted frames for {file}")
#                     print("extrated num frame is : ", num_frames)