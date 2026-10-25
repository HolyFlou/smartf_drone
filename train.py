# from ultralytics import YOLO

# model = YOLO('./runs/detect/fpv_drone_yolo26_2/weights/best.pt')

# if __name__ == '__main__':
#     results = model.train(
#         data='./drone_detection/data.yaml',
#         epochs=10,
#         imgsz=640,
#         batch=8,
#         workers=4,
#         device=0,
#         amp=True,
#         cache='ram',
#         exist_ok=False,

#         # --- Налаштування аугментацій під FPV ---
#         scale=0.7,        # Активне масштабування для малих цілей
#         degrees=20.0,     # Крен і повороти дрона
#         perspective=0.0005,
#         mosaic=1.0,       # Мозаїка для урізноманітнення фону
#         mixup=0.0,        # Вимикаємо/мінімізуємо, щоб не губити дрібні силуети
#         hsv_h=0.015,      # Зміна відтінку
#         hsv_s=0.7,        # Зміна насиченості
#         hsv_v=0.4,        # Зміна яскравості
#         fliplr=0.5,       # Дзеркальне відображення по горизонталі

#         name='fpv_drone_yolo26_2.1'
#     )

##### ВІДНОВЛЕННЯ НАВЧАННЯ #####
from ultralytics import YOLO

if __name__ == '__main__':
    # 1. Завантажуємо останній збережений чекпоінт
    model = YOLO(r'runs\detect\fpv_drone_yolo26_2.1\weights\last.pt')

    # 2. Запускаем відновлення навчання
    results = model.train(
        resume=True,
        workers=4 
    )