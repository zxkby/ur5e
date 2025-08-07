import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 解决OpenMP冲突

from ultralytics import YOLO
import torch

# 检查GPU是否可用
if torch.cuda.is_available():
    print(f"使用GPU训练，设备编号: {torch.cuda.current_device()}")
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
else:
    print("未检测到可用GPU，将使用CPU训练")

# 加载模型
model = YOLO(r'E:\homework\practice\robortes\UR5e-CNN-Pick-Place-webots\UR5e-CNN-Pick-Place-webots-main\yolov8s.pt')

# 训练模型：通过device参数指定GPU（0表示第一个GPU，若有多个可指定[0,1]）
model.train(
    data=r'E:\homework\practice\robortes\UR5e-CNN-Pick-Place-webots\UR5e-CNN-Pick-Place-webots-main\ur5eod.v2i.yolov8-obb\data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0  # 关键修改：指定使用第0号GPU（-1表示CPU，默认值）
)

# 加载训练好的模型（推理时也会自动使用GPU）
model = YOLO(r'E:\homework\practice\robortes\UR5e-CNN-Pick-Place-webots\UR5e-CNN-Pick-Place-webots-main\best(1).pt')
