import pandas as pd 
import matplotlib.pyplot as plt 
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

def main(): 
    if (not os.path.exists("analysis/heartbeat")): 
        os.makedirs("analysis/heartbeat")

    df = pd.read_csv("data/heartbeat/mitbih_train.csv")
    print("Data shape: ", df.shape)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    showLabelDistribution(y)
    
    

if __name__ == "__main__": 
    main()