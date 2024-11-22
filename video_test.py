import torch


def save_weights_without_module(weights_path='./twst_w.pth', new_weights_path='./twst_w_nomodule.pth'):
    # 가중치 로드
    checkpoint = torch.load(weights_path)

    # 'module.' 접두어를 제거한 새로운 체크포인트 사전 생성
    new_checkpoint = {}
    for key in checkpoint.keys():
        # 'module.'로 시작하면 제거하고 새로운 키로 설정
        new_key = key.replace('module.', '') if key.startswith('module.') else key
        new_checkpoint[new_key] = checkpoint[key]

    # 수정된 가중치를 새 파일로 저장
    torch.save(new_checkpoint, new_weights_path)
    print(f"새 가중치 파일이 '{new_weights_path}'에 저장되었습니다.")


# 예시 사용법
save_weights_without_module('./try.pth', 'output/deepfake_xception_kfold/new.pth')
