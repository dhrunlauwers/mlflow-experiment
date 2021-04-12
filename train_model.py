import mlflow

import sys
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn import datasets
import pandas as pd
import numpy as np

def eval_metrics(actual, pred):

    """To calculate all metrics that will be logged"""

    recall = recall_score(actual, pred)
    precision = precision_score(actual, pred)
    f1 = f1_score(actual, pred)
    accuracy = accuracy_score(actual, pred)
    return recall, precision, f1, accuracy

def plot_confusion_matrix(actual, pred):

    """To plot artifact that will be logged"""

    cm = confusion_matrix(actual, pred)

    fig, ax = plt.subplots(figsize=(16,9))

    cax = ax.matshow(cm, cmap='Blues')

    for (i, j), z in np.ndenumerate(cm):
        ax.text(j, i, '{:g}'.format(z), ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

    ax.set_title('Confusion matrix')
    ax.set_xlabel('Predicted')
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.set_ylabel('Actual')
    plt.tight_layout() 

    fig.colorbar(cax)

    fig.savefig('./plots/confusion_matrix.jpg')

    return None

def plot_feature_importance(model, feature_names):

    """To plot artifact that will be logged"""
    
    fig, ax = plt.subplots(figsize=(16,9))

    ax.barh(np.arange(len(model.feature_importances_)), model.feature_importances_ )

    ax.set_yticklabels(feature_names)
    ax.set_yticks(np.arange(len(feature_names)))

    ax.set_title('Feature Importance')
    ax.set_xlabel('Relative importance')
    ax.set_ylabel('Feature name')

    fig.savefig('./plots/feature_importance.jpg')

    return None

if __name__ == "__main__":

    # Load binary classification dataset
    bc = datasets.load_breast_cancer()
    X = bc.data
    y = bc.target

    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # read in parameters
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else None

    # train and predict
    rf = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,random_state=42)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)

    # evaluate
    (recall, precision, f1, accuracy) = eval_metrics(y_test, pred)
    plot_confusion_matrix(y_test, pred)
    plot_feature_importance(rf, bc.feature_names)

    # log
    mlflow.log_param('n_estimators', n_estimators)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_artifacts('plots')
    mlflow.sklearn.log_model(rf, "model")






