from ultralytics import YOLO
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if __name__ == '__main__':
    # Load a pretrained YOLOv8n model
    model = YOLO('runs/detect/train5/weights/best.pt')

    # Define path to the image file
    source = 'coco/images/val2017'

    # Run inference on the source
    results = model.predict(source, save=False, imgsz=640, iou=0.7, visualize=False)  # list of Results objects
    #
    # results = model.predict(source, save=False, imgsz=640, iou=0.5)  # list of Results objects

    # for i in results:
    #     print("\n图片{}检测的结果如下:".format(i.path.split("\\")[-1]))
    #     for box in i.boxes:
    #         x1, y1, x2, y2 = box.xyxy.tolist()[0]
    #         conf = box.conf.item()
    #         class_id = int(box.cls.item())
    #         print("目标左上角坐标为({}, {}), 目标右下角坐标为({}, {}), 置信度为:{}, 类别为：{}".format(int(x1),int(y1),int(x2),
    #                                                                            int(y2),conf,i.names[class_id]))
    #         # print([x1, y1, x2, y2], conf, class_id)