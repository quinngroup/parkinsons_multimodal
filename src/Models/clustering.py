import umap
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab

from sklearn.cluster import KMeans
import hdbscan

from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.metrics import silhouette_score

def umap_visualization(vae, data):

    X = []
    target = []

    for batch in data:
       _, z, _, _ = vae.forward(batch['image'].float())

       for vector,label in zip(z, batch['group']):

            X.append(vector.cpu().detach().numpy())

            if label == 'PD':
                target.append(1)
            else:
                target.append(0)

    X = np.array(X)

    reducer = umap.UMAP(random_state=42)
    reducer.fit(X)
    X_embedded = reducer.transform(X)

    pylab.scatter(X_embedded[:,0], X_embedded[:,1], c=target)
    pylab.savefig('umap.png')
    pylab.show()
    pylab.close()
    return X, X_embedded

def pca_visualization(vae, data):

    X = []
    target = []

    for batch in data:
       _, z, _, _ = vae.forward(batch['image'].float())

       for vector,label in zip(z, batch['group']):

            X.append(vector.cpu().detach().numpy())

            if label == 'PD':
                target.append(1)
            else:
                target.append(0)

    X = np.array(X)
  
    pca_embedding = PCA(n_components=2).fit_transform(X)

    pylab.scatter(pca_embedding[:,0], pca_embedding[:,1], c=target)
    pylab.savefig('pca.png')
    pylab.show()
    pylab.close()
    return target, pca_embedding
    
def k_means(embedding, labels):
    kmeans_labels = KMeans(n_clusters=2).fit_predict(embedding) 
    rand = adjusted_rand_score(labels, kmeans_labels)
    mutual_info = adjusted_mutual_info_score(labels, kmeans_labels)
    print("Adjusted rand score: ", rand)
    print("Adjusted mutual info score: ", mutual_info)
    return rand, mutual_info
    
def hbdscan(embedding, labels):
    hdbscan_labels = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=20).fit_predict(embedding)
    rand = adjusted_rand_score(labels, hdbscan_labels)
    mutual_info = adjusted_mutual_info_score(labels, hdbscan_labels)
    print("Adjusted rand score: ", rand)
    print("Adjusted mutual info score: ", mutual_info)
    return rand, mutual_info