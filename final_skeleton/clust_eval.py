from hw2skeleton import cluster
from hw2skeleton import io
import os
import random
import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score
from matplotlib import pyplot as plt
import pandas as pd

active_sites = io.read_active_sites('./data')


number_clusters = []
p_sil_scores = []
h_sil_scores = []
r_sil_scores = []

for i in range(2,10):

    # clustering all sites by partition
    print('Finding %d clusters by partitioning'% i)
    p_clusters, p_distances = cluster.cluster_by_partitioning(active_sites, i)

    # clustering all sites hierarchically
    print('Finding %d clusters hierarchically'% i)
    h_clusters, h_distances = cluster.cluster_hierarchically(active_sites, i)

    # generating random clusters labels
    print('Generating %d random cluster labels'% i)
    r_clusters = np.random.choice(range(0,i), 136)

    # to evaluate the clusters I will get the silhouette score for each method of clustering
    # as well as randomly generated cluster labels
    p_sil_scores.append(silhouette_score(p_distances, p_clusters))
    h_sil_scores.append(silhouette_score(h_distances, h_clusters))
    r_sil_scores.append(silhouette_score(p_distances, r_clusters))

fig1 = plt.figure(dpi = 300)
plt.xlabel("Number of Clusters")
plt.ylabel('Silhouette Score')
plt.title('Evaluating Clustering Methods')
plt.plot(range(2,10), p_sil_scores, label = 'Partitioning')
plt.plot(range(2,10), h_sil_scores, label = 'Hierarchical')
plt.plot(range(2,10), r_sil_scores, label = 'Random')
plt.legend()
fig1.savefig('clustering_eval_graph.png')

# because 2 has the highest silhouette score for each method, I will compare
# the methods using 2 clusters in each
p_clusters, distances = cluster.cluster_by_partitioning(active_sites, 2)
h_clusters, distances = cluster.cluster_hierarchically(active_sites, 2)

# making a dataframe containing the cluster label for each method for each site (by index)
compare = pd.DataFrame(index = range(0,136), columns = ("Partitioning_Result", "Hierarchical_Result"))
compare.Partitioning_Result = p_clusters
compare.Hierarchical_Result = h_clusters

labels = np.array(range(0,136))

# I want to see how often a site ends up in the same cluster in both Methods
# I can't just compare the cluster label because it could change
prop_same = []
#
for i in range(0,136):
    # get partitioning cluster label of the site
    p_label = compare.Partitioning_Result.loc[i]
    # find all other members of its cluster
    clust_members = labels[compare.Partitioning_Result == p_label]
    num_clust_members = len(clust_members)
    # get hierarchical cluster label of first point
    h_label = compare.Hierarchical_Result.loc[i]
    # how many members of the cluster by the first method are still in
    # the same cluster as the site in the second method
    same = sum(compare.Hierarchical_Result.loc[clust_members] == h_label)
    # get the proportion that are the the same
    prop_same.append(same / num_clust_members)
# get the average of the proportions for estimate of how often a site is the
# same cluster in both methods
mean_prop_same = np.mean(prop_same)
print(mean_prop_same)

fig2 = plt.figure(dpi = 300)
plt.xlabel('Proportion Sites in Same Cluster')
plt.ylabel('Number')
plt.title('Comparing Clustering Methods')
plt.hist(prop_same)
fig2.savefig('clustering_compare_graph.png')
