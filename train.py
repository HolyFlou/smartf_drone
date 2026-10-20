from ultralytics import YOLO

model = YOLO('yolo26n.pt')

if __name__ == '__main__':
    results = model.train(
        data='./drone_detection/data.yaml', # Шлях до датасету
        epochs=50,
        imgsz=640,                          # Для малих дронів можна підняти до 800 або 1024
        batch=8,
        workers=2,
        device=0,
        amp=True,
        exist_ok=True,
        name='fpv_drone_yolo26'
    )