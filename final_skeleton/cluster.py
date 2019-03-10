from .utils import Atom, Residue, ActiveSite
import rmsd
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering


def backbone_coords(residue):
    coords = []
    for i in range(3):
        coords.append(residue.atoms[i].coords)
    return coords


def residue_dist(residue_a, residue_b):
    a_coords = backbone_coords(residue_a)
    b_coords = backbone_coords(residue_b)
    dist = rmsd.rmsd(a_coords, b_coords)
    return dist

def shortest_dist(residue, active_site):
    distances = []
    for i in range(len(active_site.residues)):
        distances.append(residue_dist(residue, active_site.residues[i]))
    return min(distances)

def total_dist(site_a, site_b):
    distances = 0
    for i in range(len(site_a.residues)):
        distances += shortest_dist(site_a.residues[i], site_b)
    return distances


def compute_similarity(site_a, site_b):
    """
    Compute the similarity between two given ActiveSite instances.

    Input: two ActiveSite instances
    Output: the similarity between them (a floating point number)
    """
    a_to_b = total_dist(site_a, site_b)
    b_to_a = total_dist(site_b, site_a)
    dist = (a_to_b + b_to_a)/(len(site_a.residues) + len(site_b.residues))
    return dist

    # # extracting the coordinates from the active site
    # coord_a = []
    # # getting the residues
    # for i in range(len(site_a.residues)):
    #     # getting the atoms
    #     for j in range(len(site_a.residues[i].atoms)):
    #         # getting the coordinates of the atoms
    #         a = site_a.residues[i].atoms[j].coords
    #         coord_a.append(a)
    #
    # coord_b = []
    # # getting the residues
    # for i in range(len(site_b.residues)):
    #     # getting the atoms
    #     for j in range(len(site_b.residues[i].atoms)):
    #         # getting the coordinates of the atoms
    #         b = site_b.residues[i].atoms[j].coords
    #         coord_b.append(b)
    #
    # similarity = rmsd.rmsd(coord_a, coord_b)
    #
    # # Fill in your code here!


def cluster_by_partitioning(active_sites, num_clust):
    """
    Cluster a given set of ActiveSite instances using a partitioning method.

    Input: a list of ActiveSite instances
    Output: a clustering of ActiveSite instances
            (this is really a list of clusters, each of which is list of
            ActiveSite instances)
    """
    # make an empty dataframe of dimensions equal to the number of active sites
    p_distances = pd.DataFrame(index=range(len(active_sites)), columns = range(len(active_sites)))
    for i in range(len(active_sites)):
        for j in range(len(active_sites)):
            p_distances[i][j] = compute_similarity(active_sites[i], active_sites[j])

    p_clusters = KMeans(n_clusters = num_clust, random_state = 12).fit(p_distances).labels_



    return p_clusters, p_distances


def cluster_hierarchically(active_sites, num_clust):
    """
    Cluster the given set of ActiveSite instances using a hierarchical algorithm.                                                                  #

    Input: a list of ActiveSite instances
    Output: a list of clusterings
            (each clustering is a list of lists of Sequence objects)
    """

    h_distances = pd.DataFrame(index=range(len(active_sites)), columns = range(len(active_sites)))
    for i in range(len(active_sites)):
        for j in range(len(active_sites)):
            h_distances[i][j] = compute_similarity(active_sites[i], active_sites[j])

    h_clusters = AgglomerativeClustering(n_clusters = num_clust).fit(h_distances).labels_

    return h_clusters, h_distances
