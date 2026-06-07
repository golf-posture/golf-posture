import os

import cv2
from ultralytics import YOLO


def save_yolo_labels(video_path, output_dir):
    img_dir = os.path.join(output_dir, "images")
    lbl_dir = os.path.join(output_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    model = YOLO("yolov8x-pose.pt")
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 5 == 0:
            results = model(frame)[0]

            if results.keypoints is not None and len(results.keypoints) > 0:
                file_name = f"golf_frame_{frame_count}"
                cv2.imwrite(os.path.join(img_dir, f"{file_name}.jpg"), frame)

                with open(os.path.join(lbl_dir, f"{file_name}.txt"), "w", encoding="utf-8") as f:
                    for i in range(len(results.boxes)):
                        cls = int(results.boxes.cls[i])
                        box = results.boxes.xywhn[i].tolist()
                        kpts = results.keypoints.xyn[i].tolist()

                        line = f"{cls} {' '.join(map(str, box))}"
                        for kp in kpts:
                            line += f" {kp[0]} {kp[1]} 2.0"

                        f.write(line + "\n")

                print(f"Saved: {file_name}")

        frame_count += 1

    cap.release()
    print("YOLO pose label export finished.")


if __name__ == "__main__":
    video_path = "C:/Users/0312u/Desktop/Golf Codes/dataset/raw_video/Tiger Woods Iron Swing Slow Motion 2025 _ Face On.mp4"
    output_dir = "C:/Users/0312u/Desktop/Golf Codes/dataset/tiger_wood_labels"
    save_yolo_labels(video_path, output_dir)
