import yaml
import glob
import cv2

import numpy as np

from ultralytics import YOLO


EXPERIMENTS = range(0, 10)
FOLDS = range(0, 10)


if __name__ == '__main__':

    for experiment in EXPERIMENTS:
        experiment_results = []

        for fold in FOLDS:

            data_yaml = f"./yaml_splits/data_{experiment}_{fold}.yaml"
            name = f"run_{experiment}_{fold}_"

            model = YOLO(f"flowers/{name}/weights/best.pt")

            trees = yaml.safe_load(open(data_yaml, 'r'))["val"]
            for tree in trees:
                for filepath in glob.glob(f"final_data/{tree}/images/*.jpg"):
                    image = cv2.imread(filepath)
                    results = model.predict(image, iou=0.2)
                    for result in results:
                        flower_count = len(result.boxes)
                        experiment_results.append(f"{filepath}\t{flower_count}")

        with open(f"results/result_{experiment}.tsv", "w") as fh:
            fh.write("\n".join(experiment_results))
