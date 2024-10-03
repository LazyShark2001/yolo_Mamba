from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model 
    model = YOLO('yolov8n.pt')  # 也可以加载你自己的模型

    # Validate the model
    # metrics = model.val(split='val', iou=0.7, batch=16, data='coco/coco.yaml', device=[2])
    metrics = model.val(batch=16, data='cocodataset/coco.yaml', device=[0])
    # metrics = model.val(split='test', iou=0.7, batch=16, data='RZB/RZB.yaml')
    metrics.box.map    # 查看目标检测 map50-95 的性能
    metrics.box.map50  # 查看目标检测 map50 的性能
    metrics.box.map75  # 查看目标检测 map75 的性能
    metrics.box.maps   # 返回一个列表包含每一个类别的 map50-95