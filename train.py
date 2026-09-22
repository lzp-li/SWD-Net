from ultralytics import YOLO

if __name__ == '__main__':
    # 1. 加载模型架构
    model = YOLO("/home/li/我的大硬盘/ultralytics/RUN/train/yolo11s-diffusion-only/weights/best.pt")
    model.load("yolov10s.pt")
    #model = YOLO("yolov10s.pt")
    # 3. 训练模式
    model.train(
        data="/home/li/我的大硬盘/ultralytics/ultralytics/cfg/datasets/data_drone.yaml",
        epochs=300,
        imgsz=640,
        batch=16,
        device=0,
        workers=8,
        project='RUN/train',
        name='yolov10s-spd-wds-diffusion',
        # optimizer='AdamW',
        # lr0=0.0005,
        # amp=False,
    )