# mlflow-experiment

Trying out mlflow with some dummy models to see how it would work for binary classification and feature importance.

### How to use it

1. Run train_model.py and it will create a new directory where it logs the results of the training.
    - Try running a few times with different parameters:
        - First parameter controls the number of estimators (default 100)
        - Second parameter controls the max depth of each estimator (Default None)

    `train_model.py 5 5`

2. Run `mlflow ui` in command line to bring up the web interface and browse the results of the run(s).