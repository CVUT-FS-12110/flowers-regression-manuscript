import os
from ultralytics import YOLO

import torch

# torch.cuda.set_device(0)



EXPERIMENTS = range(0, 10)
FOLDS = range(0, 10)

if __name__ == '__main__':

    for experiment in EXPERIMENTS:

        for fold in FOLDS:

            data_yaml = f"./yaml_splits/data_{experiment}_{fold}.yaml"
            name = f"run_{experiment}_{fold}_"

            try:
                os.remove("yolov8n-seg.pt")
            except:
                pass
            model = YOLO("yolov8n-seg.pt")
            model.train(data=data_yaml, project="flowers", name=name, epochs=200, batch=32, imgsz=512)