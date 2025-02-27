from scipy import signal 
import pandas as pd 
import joblib
from joblib import Memory

from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
import os 


def main(): 
    memory = Memory(location = ".cache") 
    train = pd.read_csv("data/heartbeat/mitbih_train.csv", header = None)
    
    X_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1]
    
    # model = LogisticRegression(class_weight='balanced')
    # model = SVC(class_weight='balanced')
    # model = DecisionTreeClassifier(class_weight='balanced')
    model = RandomForestClassifier(
        n_estimators=400,
        class_weight='balanced',
        random_state=42, 
        verbose=1
    )

    max_depth = [i for i in range (1,20,2)]
    max_depth.append(None)

    param_grid = {
        'max_depth': max_depth 
    }

    grid_search = GridSearchCV(
        model, 
        param_grid=param_grid, 
        n_jobs=3, 
        scoring="balanced_accuracy",
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    if (not os.path.exists("model/")): 
        os.makedirs("model") 
        
    model_path = "model/mit_forest.pkl"
    joblib.dump(grid_search.best_estimator_, model_path)
    print("Complete. Model is saved to ", model_path)
    
    
if __name__ == "__main__": 
    main()
