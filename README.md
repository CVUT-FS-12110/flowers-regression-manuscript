# Flower cluster regression
Scripts and data accompanying  manuscript about flower cluster regression.

## Regression models

The regression models can be run via script `multivew/run_experiments.py`.
Switching models and other experiment properties is done by arguments in function `main`,
namely: 

    ground_truth = True / False

switch between field-validated (True) and visually-estimated (False) values, and

    model_selector = "FPN" / "without_FPN" / "CountNet"

switch between different models.

## Linear regression with YOLO inputs

The YOLO based experiment can be run via script `yolo/run_experiments.py`.

## Miscellaneous

In folder `paper` are additional helper scripts for results aggregation etc.
