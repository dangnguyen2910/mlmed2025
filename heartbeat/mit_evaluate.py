import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sklearn.metrics as metrics 
import pandas as pd 

def evaluate(X, y, model): 
    y_pred = model.predict(X)
    
    confusion_matrix = np.round(metrics.confusion_matrix(y, y_pred, normalize="true"),2)
    accuracy = metrics.accuracy_score(y, y_pred)
    macro_precision = metrics.precision_score(y, y_pred, average="macro")
    macro_recall = metrics.recall_score(y, y_pred, average="macro")
    macro_f1 = metrics.f1_score(y, y_pred, average='macro')

    print("Accuracy: ", accuracy)
    print("Macro Precision: ", macro_precision)
    print("Macro Recall: ", macro_recall)
    print("Macro F1: ", macro_f1)

    plt.figure(figsize=(6,6))
    sns.heatmap(
        confusion_matrix, 
        annot=True, 
        cmap='gray_r', 
        xticklabels=['N', "S", "V", "F", "Q"], 
        yticklabels=['N', "S", "V", "F", "Q"],
        cbar=False, 
        annot_kws={"size": 20})
    plt.show()
    plt.close()

def main(): 
    train = pd.read_csv("data/heartbeat/mitbih_train.csv", header = None)
    X_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1]

    test = pd.read_csv("data/heartbeat/mitbih_test.csv", header = None)
    X_test = test.iloc[:, :-1]
    y_test = test.iloc[:, -1]

    model = joblib.load("model/mit_forest3.pkl")
    print(model.get_params())

    print("--------------------------------------")
    print("Evaluate on train set: ")
    evaluate(X_train, y_train, model)
    print("--------------------------------------")
    print("Evaluate on test set: ")
    evaluate(X_test, y_test, model)


if __name__ == "__main__": 
    main()