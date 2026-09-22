from ultralytics import YOLO

if __name__ == '__main__':
    # 1. 加载模型架构
    model = YOLO("/home/li/我的大硬盘/ultralytics/ultralytics/cfg/models/11/yolo11s-spd-wds-diffusion.yaml")
    model.load("yolo11s.pt")
    #model = YOLO("yolov10s.pt")
    # 3. 训练模式
    model.train(
        data="/home/li/我的大硬盘/ultralytics/ultralytics/cfg/datasets/gc10.yaml",
        epochs=300,
        imgsz=640,
        batch=16,
        device=0,
        workers=8,
        project='/home/li/我的大硬盘/ultralytics/RUN/train',
        name='yolo11s-spd-wds-diffusion-GC10',
        # optimizer='AdamW',
        # lr0=0.0005,
        # amp=False,
        # hsv_h=0.015, hsv_s=0.7, hsv_v=0.7,  # 调高亮度 (v) 和饱和度 (s) 扰动
        # # 🌟 2. 工业检测神技：Copy-Paste (缓解样本极度不平衡)
        # copy_paste=0.3,  # 30% 概率把一个图里的缺陷抠出来贴到另一个图上
        # # 🌟 3. 开启更暴力的马赛克与混合增强
        # mosaic=1.0,  # 确保 100% 开启 Mosaic
        # mixup=0.15,  # 开启 15% 的图像重影混合 (极度锻炼模型在嘈杂背景中找微弱缺陷的能力)
    )