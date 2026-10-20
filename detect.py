import cv2
from ultralytics import YOLO

# Завантаження навченої моделі YOLO26
model = YOLO('runs/detect/fpv_drone_yolo26/weights/best.pt')

video_path = "./ir4.mp4"
output_path = "./output_detection.mp4"

cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0 or fps is None:
    fps = 30.0

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Поріг впевненості conf=0.6
    results = model(frame, conf=0.6)
    annotated_frame = results[0].plot()

    # Записуємо оброблений кадр у файл
    out.write(annotated_frame)

    cv2.imshow("YOLO26 FPV Drone Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()