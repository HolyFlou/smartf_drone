import cv2
import numpy as np
import time
from ultralytics import YOLO

# ==========================================
# 1. НАЛАШТУВАННЯ ТА ІНІЦІАЛІЗАЦІЯ
# ==========================================

# Завантажуємо навчену модель YOLO для виявлення дронів
model = YOLO('runs/detect/fpv_drone_yolo26_2/weights/best.pt')

# Шляхи до вхідного відео та файлу збереження результату
video_path = "./V_DRONE_016.mp4"
output_path = "./output_detection.mp4"

# Відкриваємо відеопотік
cap = cv2.VideoCapture(video_path)

# Отримуємо параметри вхідного відео (ширина, висота, кадри/сек)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

# Налаштовуємо кодек та об'єкт для запису вихідного відео
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))

# ==========================================
# 2. СТАН) МАРШРУТИЗАЦІЇ ТА ГЛОБАЛЬНІ ПАРАМЕТРИ
# ==========================================

# Визначаємо стани скінченного автомата (FSM)
STATE_SEARCHING = 0  # Стан пошуку (детекція через YOLO)
STATE_TRACKING = 1   # Стан ведення цілі (оптичний трекер + фільтр Калмана)

state = STATE_SEARCHING
tracker = None
kf = None

# Параметри логіки верифікації цілі
VERIFY_EVERY = 5   # Перевіряти ціль за допомогою YOLO кожні N кадрів
MAX_MISSES = 5     # Максимальна кількість поспіль пропущених верифікацій до втрати цілі
IOU_MIN = 0.10     # Мінімальний поріг IoU для підтвердження, що YOLO знайшла ту ж саму ціль
frame_idx = 0      # Лічильник кадрів
misses = 0         # Лічильник пропущених верифікацій

# Змінні для розрахунку FPS
prev_time = time.time()
fps_display = 0.0

# ==========================================
# 3. ДОПОМІЖНІ КЛАСИ ТА ФУНКЦІЇ
# ==========================================

class Kalman6D:
    """
    6D Фільтр Калмана з урахуванням координати, швидкості та прискорення (X, Y).
    Вектор стану: [x, y, vx, vy, ax, ay]^T
    Вектор вимірювань: [x, y]^T
    """
    def __init__(self, init_x, init_y, dt=1.0):
        self.dt = dt
        # 6 елементів стану, 2 елементи вимірювання (координати центру x, y)
        self.kf = cv2.KalmanFilter(6, 2)
        
        # Матриця переходу стану (Transition Matrix - F)
        # Моделює кінематику: x_new = x + vx*dt + 0.5*ax*dt^2
        self.kf.transitionMatrix = np.array([
            [1, 0, dt,  0, 0.5 * (dt**2),            0],
            [0, 1,  0, dt,            0, 0.5 * (dt**2)],
            [0, 0,  1,  0,           dt,            0],
            [0, 0,  0,  1,            0,           dt],
            [0, 0,  0,  0,            1,            0],
            [0, 0,  0,  0,            0,            1]
        ], dtype=np.float32)
        
        # Матриця вимірювань (Measurement Matrix - H)
        # Вказує, що ми спостерігаємо лише координати X та Y
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ], dtype=np.float32)
        
        # Коваріація шуму процесу (Process Noise Covariance - Q)
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.03
        
        # Коваріація шуму вимірювань (Measurement Noise Covariance - R)
        self.kf.measurementNoiseCov = np.array([
            [0.5, 0],
            [0, 0.5]
        ], dtype=np.float32)
        
        # Початкова коваріація помилки (Post-error covariance - P)
        self.kf.errorCovPost = np.eye(6, dtype=np.float32)
        
        # Ініціалізація початкового стану
        self.kf.statePre = np.array([[init_x], [init_y], [0], [0], [0], [0]], dtype=np.float32)
        self.kf.statePost = np.array([[init_x], [init_y], [0], [0], [0], [0]], dtype=np.float32)

    def predict(self):
        """Прогнозує наступну позицію об'єкта на основі кінематики."""
        predicted = self.kf.predict()
        return int(predicted[0][0]), int(predicted[1][0])

    def correct(self, meas_x, meas_y):
        """
        Оновлює оцінку стану на основі реальних отриманих вимірювань (meas_x, meas_y).
        Повертає скориговані координати, швидкість та прискорення.
        """
        measurement = np.array([[np.float32(meas_x)], [np.float32(meas_y)]])
        corrected = self.kf.correct(measurement)
        return (int(corrected[0][0]), int(corrected[1][0])), (corrected[2][0], corrected[3][0]), (corrected[4][0], corrected[5][0])

def iou(a, b):
    """
    Обчислює Intersection over Union (IoU) між двома прямокутниками.
    a, b: кортежі виду (x, y, w, h)
    """
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    
    # Координати перетину
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    
    # Площа перетину
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Перетин / Об'єднання
    return inter / (aw * ah + bw * bh - inter + 1e-9)

def create_tracker():
    """Створює оптичний трекер OpenCV (перевага CSRT, при відсутності — KCF)."""
    if hasattr(cv2, 'TrackerCSRT_create'):
        return cv2.TrackerCSRT_create()
    elif hasattr(cv2, 'legacy'):
        return cv2.legacy.TrackerCSRT_create()
    else:
        return cv2.TrackerKCF_create()

def apply_digital_zoom(frame, cx, cy, zoom_radius=140, target_size=(width, height)):
    """
    Вирізає область (кроп) навколо точки (cx, cy) і масштабує її до розміру target_size (цифровий зум).
    Повертає:
      - zoomed_frame: масштабоване зображення
      - кортеж параметрів кропу: (x1, y1, scale_x, scale_y) для зворотного перерахунку координат
    """
    H, W = frame.shape[:2]
    x1, y1 = max(0, cx - zoom_radius), max(0, cy - zoom_radius)
    x2, y2 = min(W, cx + zoom_radius), min(H, cy + zoom_radius)
    crop = frame[y1:y2, x1:x2]
    
    # Якщо область кропу вийшла порожньою
    if crop.size == 0:
        return frame.copy(), (0, 0, 1.0, 1.0)
    
    crop_h, crop_w = crop.shape[:2]
    zoomed_frame = cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Зберігаємо коефіцієнти масштабування
    return zoomed_frame, (x1, y1, crop_w / target_size[0], crop_h / target_size[1])

# ==========================================
# 4. ОСНОВНИЙ ЦИКЛ ОБРОБКИ ВІДЕО
# ==========================================

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    frame_idx += 1
    
    # Обчислення поточного FPS для відображення
    curr_time = time.time()
    if curr_time - prev_time > 0:
        fps_display = 1.0 / (curr_time - prev_time)
    prev_time = curr_time

    # Копія кадру для малювання графічних елементів
    display_frame = frame.copy()

    # ----------------------------------------------------
    # СТАН 1: ПОШУК ЦІЛІ (STATE_SEARCHING)
    # ----------------------------------------------------
    if state == STATE_SEARCHING:
        # Прохід YOLO на повному кадрі з упевненістю conf=0.6
        results = model(frame, conf=0.6, verbose=False)
        boxes = results[0].boxes

        # Якщо на повному кадрі нічого не знайдено — пробуємо цифровий зум по центру
        # Це допомагає виявити дрібні цілі / дрони на великій відстані
        if len(boxes) == 0:
            center_crop, c_info = apply_digital_zoom(frame, width//2, height//2, zoom_radius=250)
            res_crop = model(center_crop, conf=0.20, verbose=False)[0].boxes
            
            if len(res_crop) > 0:
                # Перераховуємо локальні координати кропу в глобальні координати кадру
                lx1, ly1, lx2, ly2 = map(int, res_crop[0].xyxy[0].cpu().numpy())
                gx1 = int(lx1 * c_info[2] + c_info[0])
                gy1 = int(ly1 * c_info[3] + c_info[1])
                gw, gh = int((lx2 - lx1) * c_info[2]), int((ly2 - ly1) * c_info[3])
                
                # Ініціалізуємо трекер та Калман для знайденої цілі
                tracker = create_tracker()
                tracker.init(frame, (gx1, gy1, gw, gh))
                kf = Kalman6D(init_x=gx1 + gw//2, init_y=gy1 + gh//2)
                state = STATE_TRACKING
                misses = 0

        # Якщо ціль знайдено безпосередньо на повному кадрі
        if len(boxes) > 0 and state == STATE_SEARCHING:
            box = boxes[0].xyxy[0].cpu().numpy()
            xmin, ymin, xmax, ymax = map(int, box[:4])
            w, h = xmax - xmin, ymax - ymin

            # Ініціалізація трекера і Калмана
            tracker = create_tracker()
            tracker.init(frame, (xmin, ymin, w, h))
            kf = Kalman6D(init_x=xmin + w//2, init_y=ymin + h//2)
            state = STATE_TRACKING
            misses = 0
        
        # Індикація статусу пошуку на екрані
        if state == STATE_SEARCHING:
            cv2.putText(display_frame, "SEARCHING (LOW CONF)...", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # ----------------------------------------------------
    # СТАН 2: СУПРОВІД ТА ВЕРИФІКАЦІЯ (STATE_TRACKING)
    # ----------------------------------------------------
    elif state == STATE_TRACKING:
        # Оновлюємо стан оптичного трекера
        tracking_success, bbox = tracker.update(frame)

        if tracking_success:
            x, y, w, h = map(int, bbox)
            raw_cx, raw_cy = x + w // 2, y + h // 2

            # 1. Екстраполяція (прогноз) фільтра Калмана
            pred_cx, pred_cy = kf.predict()

            # 2. Корекція стану Калмана реальними вимірюваннями від оптичного трекера
            (corr_cx, corr_cy), (vx, vy), (ax, ay) = kf.correct(raw_cx, raw_cy)

            # Отримуємо згладжені координати рамки
            smooth_x, smooth_y = corr_cx - w // 2, corr_cy - h // 2

            # 3. Цифровий зум (Digital Zoom) навколо скоригованого центру цілі
            zoomed_frame, crop_info = apply_digital_zoom(frame, corr_cx, corr_cy, zoom_radius=140, target_size=(width, height))
            x1_off, y1_off, scale_x, scale_y = crop_info

            # 4. Пеperiodична верифікація через YOLO кожні N кадрів
            if frame_idx % VERIFY_EVERY == 0:
                # Детекція YOLO на кропі зі зниженим порогом упевненості
                res = model(zoomed_frame, conf=0.20, verbose=False)[0].boxes
                matched = False

                for b in res:
                    # Переведення координат детектованої рамки YOLO в глобальну систему
                    lx1, ly1, lx2, ly2 = map(int, b.xyxy[0].cpu().numpy())
                    gx1 = int(lx1 * scale_x + x1_off)
                    gy1 = int(ly1 * scale_y + y1_off)
                    gw, gh = int((lx2 - lx1) * scale_x), int((ly2 - ly1) * scale_y)

                    # Перевіряємо, чи перекривається детекція YOLO з поточним трекером (IoU)
                    if iou((smooth_x, smooth_y, w, h), (gx1, gy1, gw, gh)) > IOU_MIN:
                        matched = True
                        # Переініціалізуємо трекер точними координатами від YOLO
                        tracker = create_tracker()
                        tracker.init(frame, (gx1, gy1, gw, gh))
                        # Скориговуємо фільтр Калмана новими точними координатами від YOLO
                        kf.correct(gx1 + gw//2, gy1 + gh//2)
                        smooth_x, smooth_y, w, h = gx1, gy1, gw, gh
                        break

                # Скидаємо або збільшуємо лічильник пропущених підтверджень
                misses = 0 if matched else misses + 1
                
                # Якщо втратили ціль на занадто тривалий час — повертаємося до стану пошуку
                if misses >= MAX_MISSES:
                    state = STATE_SEARCHING
                    tracker = None
                    kf = None
                    misses = 0

            if state == STATE_TRACKING:
                # Переведення глобальних координат цілі в локальні координати зумованого кадру для отрисовки
                local_x = int((smooth_x - x1_off) / scale_x)
                local_y = int((smooth_y - y1_off) / scale_y)
                local_w, local_h = int(w / scale_x), int(h / scale_y)
                
                local_pred_cx = int((pred_cx - x1_off) / scale_x)
                local_pred_cy = int((pred_cy - y1_off) / scale_y)

                # Малювання рамки супроводу цілі
                cv2.rectangle(zoomed_frame, (local_x, local_y), (local_x + local_w, local_y + local_h), (255, 0, 0), 2)
                # Зелена крапка — екстрапольована точкова позиція (прогноз 6D Калмана)
                cv2.circle(zoomed_frame, (local_pred_cx, local_pred_cy), 5, (0, 255, 0), -1)
                cv2.putText(zoomed_frame, "TRACKING", (local_x, local_y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                display_frame = zoomed_frame
        else:
            # Якщо трекер збігся/втратив об'єкт на кадрі
            state = STATE_SEARCHING
            tracker = None
            kf = None
            misses = 0

    # Відображення FPS на екрані
    cv2.putText(display_frame, f"FPS: {fps_display:.1f}", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Запис та відображення кадру
    out.write(display_frame)
    cv2.imshow("Anti-UAV 6D Tracker", display_frame)

    # Вихід при натисканні клавіші 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==========================================
# 5. ОЧИЩЕННЯ РЕСУРСІВ
# ==========================================
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Обробку завершено. Результат збережено у: {output_path}")