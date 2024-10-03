from ultralytics import YOLO
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
if __name__ == '__main__':

    model = YOLO("yaml/LightMUNet.yaml")

    model.train(data="coco/coco.yaml", batch=16, epochs=50, device=[0], resume=True, optimizer='SGD', lr0=1)
    # model.train(data="data_demo/data_demo.yaml", batch=16, epochs=50, device=[1])
