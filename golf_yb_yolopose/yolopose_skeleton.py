import cv2
from ultralytics import YOLO


def run_yolo_pose(video_path, output_path):
    model = YOLO("yolov8n-pose.pt")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Could not open video. Check the input path.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Writing YOLO pose overlay video: {output_path}")
    print("Press 'q' to stop.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame = frame
        results = model(frame, stream=True)
        for result in results:
            annotated_frame = result.plot()

        out.write(annotated_frame)
        cv2.imshow("YOLOv8 Golf Pose Saving", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("YOLO pose overlay video finished.")


if __name__ == "__main__":
    video_file = "C:/Users/0312u/Desktop/Golf Codes/dataset/raw_video/Tiger Woods Iron Swing Slow Motion 2025 _ Face On.mp4"
    output_file = "C:/Users/0312u/Desktop/Golf Codes/dataset/raw_video/Tiger_Woods_Pose_Result.mp4"
    run_yolo_pose(video_file, output_file)
