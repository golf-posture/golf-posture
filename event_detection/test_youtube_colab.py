"""
YouTube 영상에서 스윙 이벤트 감지하는 Colab용 스크립트

사용법:
    # YouTube 다운로드 방식 (yt-dlp 필요)
    !pip install yt-dlp
    !python test_youtube_colab.py --url "https://www.youtube.com/watch?v=..." --start 3.0 --end 7.0

    # 로컬 파일 방식 (Drive에서 복사한 영상 등)
    !python test_youtube_colab.py --url "https://www.youtube.com/watch?v=..." --file videoplayback.mp4 --start 3.0 --end 7.0

흐름:
    1. 영상 준비 (로컬 파일 or YouTube 다운로드)
    2. 시작/끝 시간으로 트림 (ffmpeg)
    3. 160x160 리사이즈 + 패딩
    4. SwingNet 모델로 이벤트 감지
    5. 결과 이미지 저장 + matplotlib 출력

출력 폴더: output/{youtube_id}_{start}s-{end}s/
"""

import argparse
import os
import subprocess
import cv2
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from eval import ToTensor, Normalize
from model import EventDetector

event_names = {
    0: 'Address',
    1: 'Toe-up',
    2: 'Mid-backswing (arm parallel)',
    3: 'Top',
    4: 'Mid-downswing (arm parallel)',
    5: 'Impact',
    6: 'Mid-follow-through (shaft parallel)',
    7: 'Finish'
}


def download_video(url, output_path):
    """YouTube 영상 다운로드"""
    print(f"Downloading: {url}")
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",
        "-o", output_path,
        "--no-playlist",
        url
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Download error: {result.stderr}")
        raise RuntimeError("영상 다운로드 실패")
    print(f"Downloaded: {output_path}")


def trim_video(input_path, output_path, start, end):
    """시작/끝 시간으로 영상 트림"""
    print(f"Trimming: {start}s ~ {end}s")
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ss", str(start),
        "-to", str(end),
        "-c:v", "libx264",
        "-an",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Trim error: {result.stderr}")
        raise RuntimeError("영상 트림 실패")
    print(f"Trimmed: {output_path}")


class SampleVideo(Dataset):
    """영상을 160x160으로 리사이즈하여 모델 입력 형태로 변환"""
    def __init__(self, path, input_size=160, transform=None):
        self.path = path
        self.input_size = input_size
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        cap = cv2.VideoCapture(self.path)
        frame_size = [cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)]
        ratio = self.input_size / max(frame_size)
        new_size = tuple([int(x * ratio) for x in frame_size])
        delta_w = self.input_size - new_size[1]
        delta_h = self.input_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        images = []
        for pos in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            ret, img = cap.read()
            if not ret:
                break
            resized = cv2.resize(img, (new_size[1], new_size[0]))
            b_img = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                       value=[0.406 * 255, 0.456 * 255, 0.485 * 255])
            b_img_rgb = cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB)
            images.append(b_img_rgb)
        cap.release()

        if len(images) == 0:
            raise RuntimeError(f"영상을 읽을 수 없습니다: {self.path}")

        labels = np.zeros(len(images))
        sample = {'images': np.asarray(images), 'labels': np.asarray(labels)}
        if self.transform:
            sample = self.transform(sample)
        return sample


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YouTube 골프 스윙 이벤트 감지')
    parser.add_argument('--url', required=True, help='YouTube URL')
    parser.add_argument('--file', default=None, help='로컬 영상 파일 경로 (yt-dlp 대신 사용)')
    parser.add_argument('--start', type=float, required=True, help='스윙 시작 시간 (초)')
    parser.add_argument('--end', type=float, required=True, help='스윙 끝 시간 (초)')
    parser.add_argument('--seq-length', type=int, default=64, help='모델 입력 프레임 수')
    parser.add_argument('-o', '--output', default='output', help='결과 저장 폴더')
    args = parser.parse_args()

    # --- 1. 작업 폴더 생성 ---
    video_id = args.url.split("watch?v=")[-1].split("&")[0]
    folder_name = f"{video_id}_{args.start}s-{args.end}s"
    output_dir = os.path.join(args.output, folder_name)
    os.makedirs(output_dir, exist_ok=True)

    trimmed_path = os.path.join(output_dir, "trimmed.mp4")

    # --- 2. 영상 준비 (로컬 파일 or YouTube 다운로드) ---
    if args.file:
        # 로컬 파일 사용 (Drive에서 복사한 영상 등)
        if not os.path.exists(args.file):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {args.file}")
        print(f"Using local file: {args.file}")
        raw_path = args.file
    else:
        # YouTube 다운로드
        raw_path = os.path.join(output_dir, "raw.mp4")
        if not os.path.exists(raw_path):
            download_video(args.url, raw_path)
        else:
            print(f"이미 다운로드됨: {raw_path}")

    # --- 3. 트림 ---
    trim_video(raw_path, trimmed_path, args.start, args.end)

    # --- 4. 영상 정보 출력 ---
    cap = cv2.VideoCapture(trimmed_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"Trimmed video: {width}x{height}, {fps:.1f}fps, {total_frames} frames, {total_frames/fps:.1f}s")

    # --- 5. 모델 로드 ---
    print("Loading model...")
    ds = SampleVideo(trimmed_path, transform=transforms.Compose([
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]))
    dl = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)

    model = EventDetector(pretrain=True,
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          bidirectional=True,
                          dropout=False)

    try:
        save_dict = torch.load('models/swingnet_1800.pth.tar')
    except FileNotFoundError:
        print("Model weights not found. Download and place in 'models' folder.")
        exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.load_state_dict(save_dict['model_state_dict'])
    model.to(device)
    model.eval()
    print("Loaded model weights")

    # --- 6. 이벤트 감지 ---
    print("Detecting events...")
    seq_length = args.seq_length
    for sample in dl:
        images = sample['images']
        batch = 0
        while batch * seq_length < images.shape[1]:
            if (batch + 1) * seq_length > images.shape[1]:
                image_batch = images[:, batch * seq_length:, :, :, :]
            else:
                image_batch = images[:, batch * seq_length:(batch + 1) * seq_length, :, :, :]
            logits = model(image_batch.to(device))
            if batch == 0:
                probs = F.softmax(logits.data, dim=1).cpu().numpy()
            else:
                probs = np.append(probs, F.softmax(logits.data, dim=1).cpu().numpy(), 0)
            batch += 1

    events = np.argmax(probs, axis=0)[:-1]
    print(f"Predicted event frames: {events}")

    confidence = []
    for i, e in enumerate(events):
        confidence.append(probs[e, i])
    print(f"Confidence: {[np.round(c, 3) for c in confidence]}")

    # --- 7. 결과 저장 및 시각화 ---
    cap = cv2.VideoCapture(trimmed_path)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, e in enumerate(events):
        cap.set(cv2.CAP_PROP_POS_FRAMES, e)
        _, img = cap.read()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        filename = f"{i}_{event_names[i].replace(' ', '_')}_f{e}.jpg"
        cv2.imwrite(os.path.join(output_dir, filename), img)
        print(f"  Saved: {filename}")

        axes[i].imshow(img_rgb)
        axes[i].set_title(f"{event_names[i]}\nframe:{e}  conf:{confidence[i]:.3f}", fontsize=10)
        axes[i].axis('off')

    cap.release()

    plt.suptitle(f'Golf Swing Event Detection\nVideo: {video_id} ({args.start}s ~ {args.end}s)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'events_summary.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nResults saved to '{output_dir}/'")
