import pandas as pd 
import joblib
from joblib import Memory

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE 
from imblearn.pipeline import Pipeline 
import os 


def main(): 
    memory = Memory(location = ".cache") 
    train = pd.read_csv("data/heartbeat/mitbih_train.csv", header = None)
    
    X_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1]
    
    model = SVC()

    param_grid = {
        'C' : [i for i in range(1,10,2)]
    }
    grid_search = GridSearchCV(model, param_grid=param_grid)

    if (not os.path.exists("model/")): 
        os.makedirs("model") 
        
    joblib.dump(model, "model/mit_svm.pkl")
    print("Complete. Model is saved to model/mit_svm.pkl")
    
    
if __name__ == "__main__": 
    main()
