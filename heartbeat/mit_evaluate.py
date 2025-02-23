import joblib
import sklearn.metrics as metrics 
import pandas as pd 

def main(): 
    test = pd.read_csv("data/heartbeat/mitbih_test.csv", header = None)
    X = test.iloc[:, :-1]
    y_true = test.iloc[:, -1]

    model = joblib.load("model/mit_svm.pkl")
    y_pred = model.predict(X)

    kappa = metrics.cohen_kappa_score(y_true, y_pred)
    balanced_accuracy = metrics.balanced_accuracy_score(y_pred=y_pred, y_true=y_true)
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    mcc = metrics.matthews_corrcoef(y_true, y_pred)
    weighted_precision = metrics.precision_score(y_true, y_pred, average="weighted")
    weighted_recall = metrics.recall_score(y_true, y_pred, average="weighted")

    print("Confusion matrix: \n", confusion_matrix)
    print("Kappa: ", kappa)
    print("Balanced Acc: ", balanced_accuracy)
    print("Matthews correlation coefficient: ", mcc)
    print("Weighted Precision: ", weighted_precision)
    print("Weighted Recall: ", weighted_recall)


if __name__ == "__main__": 
    main()