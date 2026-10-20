import cv2
from ultralytics import YOLO

# 1. Завантаження моделі YOLO
model = YOLO('runs/detect/fpv_drone_yolo26/weights/best.pt')

video_path = "./V_DRONE_016.mp4"
output_path = "./output_detection.mp4"

cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0 or fps is None:
    fps = 30.0

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# 2. Стан системи та ініціалізація трекера
STATE_SEARCHING = 0  # Пошук об'єкта через YOLO
STATE_TRACKING = 1   # Супровід об'єкта через трекер

state = STATE_SEARCHING
tracker = None

VERIFY_EVERY = 5
MAX_MISSES = 3
IOU_MIN = 0.2
frame_idx = 0
misses = 0

def iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / (aw * ah + bw * bh - inter + 1e-9)

def crop_around(frame, x, y, w, h, pad=100):
    H, W = frame.shape[:2]
    cx, cy = x + w // 2, y + h // 2
    x1, y1 = max(0, cx - pad), max(0, cy - pad)
    x2, y2 = min(W, cx + pad), min(H, cy + pad)
    return frame[y1:y2, x1:x2], x1, y1


# Функція для створення нового екземпляра трекера (CSRT добре тримає дрібні деталі)
def create_tracker():
    # Для новіших версій opencv-contrib-python
    if hasattr(cv2, 'TrackerCSRT_create'):
        return cv2.TrackerCSRT_create()
    elif hasattr(cv2, 'legacy'):
        return cv2.legacy.TrackerCSRT_create()
    else:
        # Резервний виклик через TrackerKCF або TrackerMIL
        return cv2.TrackerKCF_create()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    frame_idx += 1

    if state == STATE_SEARCHING:
        # Режим детекції: запускаємо YOLO
        results = model(frame, conf=0.6, verbose=False)
        boxes = results[0].boxes

        if len(boxes) > 0:
            # Берем перший виявлений дрон з найбільшою впевненістю
            box = boxes[0].xyxy[0].cpu().numpy()  # [xmin, ymin, xmax, ymax]
            xmin, ymin, xmax, ymax = map(int, box[:4])
            w = xmax - xmin
            h = ymax - ymin

            # Ініціалізуємо трекер координатами Bounding Box
            tracker = create_tracker()
            tracker.init(frame, (xmin, ymin, w, h))
            state = STATE_TRACKING

            # Малюємо рамку первинного захоплення (зелений колір)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, "TARGET LOCKED", (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            # Якщо об'єктів не знайдено, просто показуємо статус
            cv2.putText(frame, "SEARCHING", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    elif state == STATE_TRACKING:
        tracking_success, bbox = tracker.update(frame)

        if tracking_success:
            x, y, w, h = map(int, bbox)

            if frame_idx % VERIFY_EVERY == 0:
                crop, ox, oy = crop_around(frame, x, y, w, h, pad=100)
                matched = False

                if crop.size > 0:
                    res = model(crop, conf=0.3, verbose=False)[0].boxes
                    for b in res:
                        cx1, cy1, cx2, cy2 = map(int, b.xyxy[0].cpu().numpy())
                        # переводим координаты из окна в координаты кадра
                        x1, y1, x2, y2 = cx1 + ox, cy1 + oy, cx2 + ox, cy2 + oy
                        if iou((x, y, w, h), (x1, y1, x2 - x1, y2 - y1)) > IOU_MIN:
                            matched = True
                            tracker = create_tracker()
                            tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
                            x, y, w, h = x1, y1, x2 - x1, y2 - y1
                            break

                misses = 0 if matched else misses + 1
                if misses >= MAX_MISSES:
                    state = STATE_SEARCHING
                    tracker = None
                    misses = 0

            if state == STATE_TRACKING:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, "TRACKING", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        else:
            state = STATE_SEARCHING
            tracker = None
            misses = 0
            cv2.putText(frame, "TARGET LOST! RESCANNING...", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Записуємо оброблений кадр та виводимо на екран
    out.write(frame)
    cv2.imshow("Anti-UAV YOLO + Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Обробку завершено. Результат збережено у: {output_path}")