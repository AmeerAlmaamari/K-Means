import numpy as np
from numpy.linalg import eig
import pandas as pd
import random as rd
import matplotlib.pyplot as plt
import seaborn as sns
from random import sample
from numpy import linalg as LA
from tqdm import tqdm

animals = pd.read_csv(r'C:\Users\COLD\Desktop\clustering-data\animals',delimiter=' ',header=None)
animals[0] = 0         #"animal"

countries = pd.read_csv(r'C:\Users\COLD\Desktop\clustering-data\countries',delimiter=' ',header=None)
countries[0] = 1         #"country"

fruits = pd.read_csv(r'C:\Users\COLD\Desktop\clustering-data\fruits',delimiter=' ',header=None)
fruits[0] = 2              #"fruit"

veggies = pd.read_csv(r'C:\Users\COLD\Desktop\clustering-data\veggies',delimiter=' ',header=None)
veggies[0] = 3             #"veggies"


df = pd.concat([animals, countries,fruits,veggies], ignore_index=True)
df = df.sample(frac=1).reset_index(drop=True)


df_merged = df.drop(0,axis=1).values
df_merged.shape[1]
P = np.shape(df_merged)[1]

#----------------------------------------------------------#

class KMeans:
    def __init__(self,K,method="euclidean"):
        self.method = method
        self.K = K
        self.centroids = []

    def l2_norm(self,dataset:np.ndarray):
        return LA.norm(dataset,axis=1)

    def distance(self,p1,p2):
        if self.method == "euclidean":
            return np.sqrt(np.sum((p1-p2)**2))
        elif self.method == "manhattan":
            if type(self.centroids[0]) == np.float64:
                    return abs(p1-p2)
            return sum(abs(e1-e2) for e1, e2 in zip(p1,p2))
    
    def fit(self,dataset:np.ndarray, n_iter:int):
        self.centroids = sample(list(dataset),self.K)

        for i in tqdm(range(n_iter)):
            assignments = []
            for p_idx,p in enumerate(dataset):
                assignments.append(np.argmin([self.distance(p,c) for c in self.centroids]))
            for k in range(self.K):
                Sk = [i for i,x in enumerate(assignments) if x==k]
                self.centroids[k] = np.mean(dataset[Sk], axis=0)

    def predict(self, points:list):
        assignments = []
        for _,p in enumerate(points):
            assignments.append(np.argmin([self.distance(p,c) for c in self.centroids]))
        return assignments 

def get_best_cluster(predictions, K):
    cluster_count = [sum([1 if c==k else 0 for c in predictions]) for k in range(K)]
    best_k = np.argmax(cluster_count)
    return best_k

metrics = dict() #dict of dicts, {k:{"Accuracy":acc, "Percision":prc, "Recall":rcl, "F1":f1}}
accuracies = []
percisions = []
recalls = []
F1s = []
# C = Number of real classes
C = 4
is_l2_norm = False
for k in range(1,11):
    # Initiate the model with k clusters
    clf = KMeans(k,method="euclidean")
    if is_l2_norm:
        dataset = clf.l2_norm(df_merged)
        used_df = df.drop([i for i in range(1,301)],axis=1)
        used_df[1] = clf.l2_norm(df_merged)
    else:
        dataset = df_merged
        used_df = df
        
    # Train(fit) the model
    clf.fit(dataset,100)
    # Predictions of all classes using full dataset:
    full_predictions = clf.predict(dataset)
    
    for c in range(C):
        temp_pred = clf.predict(used_df[used_df[0]==c].drop(0,axis=1).values)

        best_k = get_best_cluster(temp_pred, k)
        TP = sum([1 if x==best_k else 0 for x in temp_pred])
        FN = len(temp_pred)-TP
        FP = sum([1 if x==best_k else 0 for x in full_predictions]) - TP

        temp_not_c_preds = clf.predict(used_df[used_df[0]!=c].drop(0,axis=1).values)
        TN = sum([1 if x!=best_k else 0 for x in temp_not_c_preds])


        accuracies.append((TP+TN)/((TP+FN+TN+FP)))
        percisions.append(TP/((TP+FP)))
        recalls.append(TP/((TP+FN)))
        F1s.append((2*percisions[-1]*recalls[-1])/(percisions[-1]+recalls[-1]))

        metrics[k] = {"Accuracy":np.mean(accuracies),
                      "Percision":np.mean(percisions), 
                      "Recall":np.mean(recalls), 
                      "F1":np.mean(F1s)}



plt.plot(metrics.keys(),[d["Accuracy"] for d in metrics.values()],label = "Accuracy")
plt.plot(metrics.keys(),[d["Percision"] for d in metrics.values()],label = "Percision")
plt.plot(metrics.keys(),[d["Recall"] for d in metrics.values()],label = "Recall")
plt.plot(metrics.keys(),[d["F1"] for d in metrics.values()],label = "F1 Score")
plt.legend()
plt.xlabel("K (Number of clusters)")
plt.ylabel("Metrics")
plt.show()

        
    