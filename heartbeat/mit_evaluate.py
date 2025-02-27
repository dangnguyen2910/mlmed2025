import numpy as np
import joblib
import sklearn.metrics as metrics 
import pandas as pd 

def evaluate(X, y, model): 
    y_pred = model.predict(X)

    balanced_accuracy = metrics.balanced_accuracy_score(y_pred=y_pred, y_true=y)
    confusion_matrix = np.round(metrics.confusion_matrix(y, y_pred, normalize="true"),2)
    macro_precision = metrics.precision_score(y, y_pred, average="macro")
    macro_recall = metrics.recall_score(y, y_pred, average="macro")

    print("Confusion matrix: \n", confusion_matrix)
    print("Balanced Acc: ", balanced_accuracy)
    print("Macro Precision: ", macro_precision)
    print("Macro Recall: ", macro_recall)


def main(): 
    train = pd.read_csv("data/heartbeat/mitbih_train.csv", header = None)
    X_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1]

    test = pd.read_csv("data/heartbeat/mitbih_test.csv", header = None)
    X_test = test.iloc[:, :-1]
    y_test = test.iloc[:, -1]

    model = joblib.load("model/mit_tree.pkl")
    print(model.get_params())

    print("Evaluate on train set: ")
    print("--------------------------------------")
    evaluate(X_train, y_train, model)
    print("Evaluate on test set: ")
    print("--------------------------------------")
    evaluate(X_test, y_test, model)


if __name__ == "__main__": 
    main()