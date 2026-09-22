from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("/home/li/我的大硬盘/ultralytics/RUN/train/yolo11s-diffusion-only/weights/best.pt")

    # 开始验证
    metrics = model.val(
        data="data_drone.yaml",
        split='test',
        device=0,
        project='RUN/test',
        name='yolo11s-diffusion-only',
    )

    # 打印结果
    print(f"测试集 mAP@0.5: {metrics.box.map50:.4f}")
    print(f"测试集 mAP@0.5:0.95: {metrics.box.map:.4f}")