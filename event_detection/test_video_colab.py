"""
Colab용 이벤트 감지 스크립트
- cv2.imshow() 대신 matplotlib으로 결과 출력
- 이벤트 프레임을 이미지 파일로 저장 (output/ 폴더)

사용법 (Colab 셀에서):
    !python test_video_colab.py -p test_video.mp4
"""

import argparse
import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from eval import ToTensor, Normalize
from model import EventDetector
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

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


class SampleVideo(Dataset):
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
            _, img = cap.read()
            resized = cv2.resize(img, (new_size[1], new_size[0]))
            b_img = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                       value=[0.406 * 255, 0.456 * 255, 0.485 * 255])
            b_img_rgb = cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB)
            images.append(b_img_rgb)
        cap.release()
        labels = np.zeros(len(images))
        sample = {'images': np.asarray(images), 'labels': np.asarray(labels)}
        if self.transform:
            sample = self.transform(sample)
        return sample


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='Path to video', default='test_video.mp4')
    parser.add_argument('-s', '--seq-length', type=int, help='Frames per forward pass', default=64)
    parser.add_argument('-o', '--output', help='Output directory for event frames', default='output')
    args = parser.parse_args()
    seq_length = args.seq_length

    os.makedirs(args.output, exist_ok=True)

    print('Preparing video: {}'.format(args.path))

    ds = SampleVideo(args.path, transform=transforms.Compose([ToTensor(),
                                Normalize([0.485, 0.456, 0.406],
                                          [0.229, 0.224, 0.225])]))

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
    print('Using device:', device)
    model.load_state_dict(save_dict['model_state_dict'])
    model.to(device)
    model.eval()
    print("Loaded model weights")

    print('Testing...')
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
    print('Predicted event frames: {}'.format(events))

    confidence = []
    for i, e in enumerate(events):
        confidence.append(probs[e, i])
    print('Confidence: {}'.format([np.round(c, 3) for c in confidence]))

    # 원본 영상에서 이벤트 프레임 추출 및 저장
    cap = cv2.VideoCapture(args.path)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, e in enumerate(events):
        cap.set(cv2.CAP_PROP_POS_FRAMES, e)
        _, img = cap.read()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 이미지 파일로 저장
        filename = f"{i}_{event_names[i].replace(' ', '_')}_f{e}.jpg"
        cv2.imwrite(os.path.join(args.output, filename), img)
        print(f"  Saved: {filename}")

        # matplotlib으로 표시
        axes[i].imshow(img_rgb)
        axes[i].set_title(f"{event_names[i]}\nframe:{e}  conf:{confidence[i]:.3f}", fontsize=10)
        axes[i].axis('off')

    cap.release()

    plt.suptitle('Golf Swing Event Detection Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, 'events_summary.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nResults saved to '{args.output}/' folder")
