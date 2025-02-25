from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import os

def showLabelDistribution(y, show=False, save=True): 
    fig = plt.figure(figsize=(12,5))
    counts, bins, patches = plt.hist(y)

    for count, bin_edge in zip(counts, bins[:-1]):
        plt.text(bin_edge + (bins[1] - bins[0]) / 2, count, str(int(count)), 
                ha='center', va='bottom', fontsize=10)

    plt.xticks([0,1,2,3,4])
    plt.title("MIT-BIH label distribution")

    if (show): 
        plt.show()

    if (save): 
        plt.savefig("analysis/heartbeat/mit_label_distribution.png")
        print("Figure saved at analysis/heartbeat/mit_label_distribution.png")
    
    plt.close()

def visualizePCA(X,y, show=False, save=True): 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    fig = plt.figure(figsize=(7,7))
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", alpha = 0.7, label=y)
    sns.scatterplot(x = X_pca[:, 0], y = X_pca[:, 1], hue=y, alpha=0.5)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title("Data distribution when reduced to 2D by PCA")
    plt.legend()

    if (show): 
        plt.show()

    if (save): 
        plt.savefig("analysis/heartbeat/mit_pca_visualize.png")
        print("Figure saved at analysis/heartbeat/mit_pca_visualize.png")

    plt.close()
    

def main(): 
    if (not os.path.exists("analysis/heartbeat")): 
        os.makedirs("analysis/heartbeat")

    df = pd.read_csv("data/heartbeat/mitbih_train.csv")
    print("Data shape: ", df.shape)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    showLabelDistribution(y)
    visualizePCA(X,y )
    

if __name__ == "__main__": 
    main()