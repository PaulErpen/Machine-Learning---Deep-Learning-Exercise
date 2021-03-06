import sklearn.metrics as metrics
import seaborn as sns
from matplotlib import pyplot as plt

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

import joblib

def do_classification_report(y_true, y_pred, name, labels, file_path=""):
    print("Classification report for {}".format(name))
    print("Accuracy score: {}".format(metrics.accuracy_score(y_true, y_pred)))
    print("Recall score: {}".format(metrics.recall_score(y_true, y_pred, average="weighted")))
    print("Precision score: {}".format(metrics.precision_score(y_true, y_pred, average="weighted")))
    print("F1-Score score: {}".format(metrics.f1_score(y_true, y_pred, average="weighted")))

    cm = metrics.confusion_matrix(y_true, y_pred)

    ax= plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax)

    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels') 
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(labels)
    plt.xticks(rotation=90, ha='right')
    plt.yticks(rotation=0, ha='right')
    ax.yaxis.set_ticklabels(labels)

    plt.gcf().subplots_adjust(left=0.2, bottom=0.27)

    if(len(file_path) > 0):
        plt.savefig(file_path.format("confusion_matrix.png"))
        pd.DataFrame(
            data={"y_true": y_true, "y_pred": y_pred}).to_csv(
                file_path.format("test_set_predictions.csv"))

    plt.show()

def export_model(model, path):
    if len(path) > 0:
        joblib.dump(model, path.format("model.joblib"))

# ---- USED MODELS ----

def do_logistic_regression(X_train, y_train, X_test, y_test, labels, file_path=""):
    #scaling for logistic regression
    scaler = StandardScaler().fit(X_train)

    model_log_res = LogisticRegression(random_state=0)
    model_log_res.fit(scaler.transform(X_train), y_train)
    predictions_log_res = model_log_res.predict(scaler.transform(X_test))
    do_classification_report(y_test, predictions_log_res, "Logistic Regression", labels, file_path)

    export_model(model_log_res, file_path)

def do_lda(X_train, y_train, X_test, y_test, labels, file_path=""):
    model_lda = LinearDiscriminantAnalysis()
    model_lda.fit(X_train, y_train)
    predictions_lda = model_lda.predict(X_test)
    do_classification_report(y_test, predictions_lda, "LDA", labels, file_path)

    export_model(model_lda, file_path)

def do_svm(X_train, y_train, X_test, y_test, labels, file_path=""):
    #scaling for SVM
    scaler = StandardScaler().fit(X_train)

    model_svm = LinearSVC()
    model_svm.fit(scaler.transform(X_train), y_train)
    predictions_svm = model_svm.predict(scaler.transform(X_test))
    do_classification_report(y_test, predictions_svm, "SVM", labels, file_path)

    export_model(model_svm, file_path)

def do_mlp(X_train, y_train, X_test, y_test, labels, file_path=""):
    model_mlp = MLPClassifier()
    model_mlp.fit(X_train, y_train)
    predictions_mlp = model_mlp.predict(X_test)
    do_classification_report(y_test, predictions_mlp, "MLP", labels, file_path)

    export_model(model_mlp, file_path)