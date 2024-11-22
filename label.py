from collections import Counter

def check_data_distribution(txt_path):
    # 라벨 파일 읽기
    with open(txt_path, 'r') as file:
        labels = [int(line.strip().split()[-1]) for line in file]

    # 클래스별 샘플 개수 계산
    label_counts = Counter(labels)

    print("Class distribution:")
    for label, count in label_counts.items():
        print(f"Class {label}: {count} samples")


# 예시로 train 데이터셋의 분포 확인
train_list = './labels2.txt'  # train 데이터셋의 라벨 파일 경로
check_data_distribution(train_list)
