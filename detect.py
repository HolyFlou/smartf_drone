import time

import cv2
import numpy as np
from ultralytics import YOLO

# ---------- НАСТРОЙКИ ----------
MODEL_PATH = 'runs/detect/fpv_drone_yolo26_2.1/weights/best.pt'
VIDEO_IN, VIDEO_OUT = "./V_DRONE_016.mp4", "./output_detection.mp4"

CONF_FULL, CONF_TILE, CONF_VERIFY = 0.40, 0.30, 0.25  # пороги дрона: весь кадр / тайлы / проверка при сопровождении
TILE = 0.6             # размер тайла в долях кадра (2x2 тайла с перекрытием)
SMALL_PX = 24          # цель меньше этого размера считается «малой»
CONF_OTHER = 0.30      # порог для самолёта/птицы (красная рамка)
VETO_MARGIN = 0.10     # дрон отбрасывается, если на нём рамка самолёта с уверенностью >= (дрон - 0.10)
VERIFY_EVERY = 4       # проверка YOLO каждые N кадров
MAX_MISSES = 5         # сколько проверок подряд без подтверждения до сброса
IOU_MIN = 0.10         # минимальное пересечение рамки трекера и рамки YOLO
RELATCH_DIST = 4       # переприцел на «чужую» рамку, только если она ближе чем 4 размера дрона
F = cv2.FONT_HERSHEY_SIMPLEX

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_IN)
if not cap.isOpened():
    raise SystemExit(f"Не удалось открыть видео: {VIDEO_IN}")
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(VIDEO_OUT, cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS) or 30.0, (width, height))

# Класс дрона. Всё остальное (airplane, bird, ...) считается «не дрон». Если нашёлся не тот — задай вручную: DRONE = 0
DRONE = next((i for i, n in model.names.items() if any(k in str(n).lower() for k in ("drone", "uav", "дрон"))), None)
print(f"Классы модели: {model.names} | дрон = {DRONE}")
if DRONE is None:
    raise SystemExit("В модели нет класса 'drone'. Задай DRONE вручную.")

# CSRT, а если его нет — KCF
make_tracker = next((getattr(m, n) for m in (cv2, getattr(cv2, "legacy", cv2))
                     for n in ("TrackerCSRT_create", "TrackerKCF_create") if hasattr(m, n)), None)
if make_tracker is None:
    raise SystemExit("Нет CSRT/KCF трекера. Установи opencv-contrib-python.")


# ---------- ФУНКЦИИ ----------
def new_kf(cx, cy):
    """Фильтр Калмана: состояние [x, y, vx, vy, ax, ay], измерение [x, y]."""
    k = cv2.KalmanFilter(6, 2)
    k.transitionMatrix = (np.eye(6) + np.eye(6, k=2) + 0.5 * np.eye(6, k=4)).astype(np.float32)
    k.measurementMatrix = np.eye(2, 6, dtype=np.float32)
    k.processNoiseCov = np.eye(6, dtype=np.float32) * 0.03
    k.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
    k.errorCovPost = np.eye(6, dtype=np.float32)
    k.statePost = np.float32([[cx], [cy], [0], [0], [0], [0]])
    return k


def iou(a, b):
    """Пересечение двух рамок (x, y, w, h, ...)."""
    w = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    h = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    inter = max(0, w) * max(0, h)
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def zoom(frame, cx, cy, r):
    """Вырезает окно высотой 2r вокруг (cx, cy) и растягивает до размера кадра.
    Возвращает (картинка, (x0, y0, scale_x, scale_y))."""
    hh, hw = min(r, height // 2), min(round(r * width / height), width // 2)
    x0 = int(np.clip(cx - hw, 0, width - 2 * hw))
    y0 = int(np.clip(cy - hh, 0, height - 2 * hh))
    crop = frame[y0:y0 + 2 * hh, x0:x0 + 2 * hw]
    return cv2.resize(crop, (width, height)), (x0, y0, 2 * hw / width, 2 * hh / height)


def detect(img, conf_drone, crop=(0, 0, 1, 1)):
    """YOLO. Возвращает (drones, others): списки (x, y, w, h, conf) в координатах ПОЛНОГО кадра.
    others = самолёты/птицы. Дрон отбрасывается, если на нём стоит рамка другого класса
    с близкой уверенностью (agnostic_nms у YOLO26 не работает, поэтому делаем это вручную)."""
    global raw_max
    ox, oy, sx, sy = crop
    b = model(img, conf=0.05, verbose=False)[0].boxes  # низкий порог — чтобы видеть самолёты и для диагностики
    rows = [(int(x1 * sx + ox), int(y1 * sy + oy), int((x2 - x1) * sx), int((y2 - y1) * sy), c, k)
            for (x1, y1, x2, y2), c, k in zip(b.xyxy.cpu().numpy(), b.conf.tolist(), b.cls.tolist())]
    others = [r for r in rows if r[5] != DRONE]
    raw_max = max([raw_max] + [r[4] for r in rows if r[5] == DRONE])  # лучшая уверенность дрона, даже ниже порога
    drones = [r[:5] for r in rows
              if r[5] == DRONE and r[4] >= conf_drone and r[3] > 0 and 0.3 <= r[2] / r[3] <= 3.2
              and not any(iou(r, o) > 0.3 and o[4] >= r[4] - VETO_MARGIN for o in others)]
    return sorted(drones, key=lambda d: -d[4]), [o[:5] for o in others if o[4] >= CONF_OTHER]


def search_views(frame):
    """Весь кадр + 4 перекрывающихся тайла, каждый растянут до размера кадра (малая цель становится крупнее)."""
    yield frame, (0, 0, 1, 1), CONF_FULL
    tw, th = int(width * TILE), int(height * TILE)
    for y0 in (0, height - th):
        for x0 in (0, width - tw):
            yield cv2.resize(frame[y0:y0 + th, x0:x0 + tw], (width, height)), (x0, y0, tw / width, th / height), CONF_TILE


def init_tracker(frame, x, y, w, h):
    """Обрезает рамку по кадру и создаёт новый трекер. Возвращает рамку (x, y, w, h) или None."""
    global tracker
    x1, y1, x2, y2 = max(0, x), max(0, y), min(width, x + w), min(height, y + h)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    tracker = make_tracker()
    tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
    return x1, y1, x2 - x1, y2 - y1


# ---------- ОСНОВНОЙ ЦИКЛ ----------
# tracker is None -> режим поиска, иначе -> сопровождение
tracker = kf = None
misses, idx, conf, prev, raw_max = 0, 0, 0.0, time.time(), 0.0

while True:
    success, frame = cap.read()
    if not success:
        break
    idx += 1
    now = time.time()
    fps, prev = 1.0 / max(now - prev, 1e-6), now
    show = frame.copy()

    # --- ПОИСК: весь кадр, затем 4 тайла ---
    if tracker is None:
        raw_max = 0.0
        for img, info, thr in search_views(frame):
            drones, others = detect(img, thr, info)
            for x, y, w, h, c in others:
                cv2.rectangle(show, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(show, f"NOT DRONE {c:.2f}", (x, max(15, y - 8)), F, 0.5, (0, 0, 255), 2)
            for d in drones:
                if (box := init_tracker(frame, *d[:4])):
                    kf, misses, conf = new_kf(box[0] + box[2] // 2, box[1] + box[3] // 2), 0, d[4]
                    break
            if tracker is not None:
                break
        else:
            cv2.putText(show, f"SEARCHING (max drone conf {raw_max:.2f})", (20, 40), F, 0.7, (0, 0, 255), 2)

    # --- СОПРОВОЖДЕНИЕ + ПРОВЕРКА YOLO ---
    if tracker is not None:
        ok, bbox = tracker.update(frame)
        lost = not ok
        if ok:
            x, y, w, h = map(int, bbox)
            px, py = kf.predict()[:2, 0].astype(int)
            cx, cy = kf.correct(np.float32([[x + w // 2], [y + h // 2]]))[:2, 0].astype(int)
            box = (cx - w // 2, cy - h // 2, w, h)  # сглаженная рамка
            small = max(w, h) < SMALL_PX  # малая цель: чаще проверяем и сильнее приближаем
            zoomed, info = zoom(frame, cx, cy, 90 if small else 140)

            if idx % (2 if small else VERIFY_EVERY) == 0:
                drones, others = detect(zoomed, CONF_VERIFY, info)
                # 1. дрон, который пересекается с рамкой трекера
                best = max((d for d in drones if iou(box, d) > IOU_MIN), key=lambda d: iou(box, d), default=None)
                # 2. трекер мог уехать на деревья: берём другой дрон, но только РЯДОМ
                if best is None and drones:
                    d = drones[0]
                    if np.hypot(d[0] + d[2] / 2 - cx, d[1] + d[3] / 2 - cy) < RELATCH_DIST * max(w, h, 15):
                        best = d

                if best:  # YOLO подтвердила дрон — перезапускаем трекер на точной рамке
                    conf, misses = best[4], 0
                    if (new := init_tracker(frame, *best[:4])):
                        box = new
                        kf.correct(np.float32([[box[0] + box[2] // 2], [box[1] + box[3] // 2]]))
                else:     # не подтвердила: сброс, если на цели самолёт/птица или слишком много промахов
                    misses += 1
                    lost = misses >= MAX_MISSES or any(iou(box, o) > IOU_MIN for o in others)

            if not lost:
                ox, oy, sx, sy = info
                lx, ly = int((box[0] - ox) / sx), int((box[1] - oy) / sy)
                cv2.rectangle(zoomed, (lx, ly), (lx + int(box[2] / sx), ly + int(box[3] / sy)), (255, 0, 0), 2)
                cv2.circle(zoomed, (int((px - ox) / sx), int((py - oy) / sy)), 5, (0, 255, 0), -1)
                cv2.putText(zoomed, f"TRACKING: {model.names[DRONE]} ({conf:.2f})",
                            (lx, max(15, ly - 10)), F, 0.5, (255, 0, 0), 2)
                show = zoomed
        if lost:
            tracker = kf = None
            misses = 0

    cv2.putText(show, f"FPS: {fps:.1f}", (20, 80), F, 0.7, (0, 255, 255), 2)
    out.write(show)
    cv2.imshow("Anti-UAV 6D Tracker", show)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Обработка завершена. Результат сохранён в: {VIDEO_OUT}")