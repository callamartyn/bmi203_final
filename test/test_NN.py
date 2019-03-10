from hw2skeleton import cluster
from hw2skeleton import io
import os
import random
import numpy as np
# adding a test function to make sure I get a distance of 0 for self

def test_similarity():
    filename_a = os.path.join("data", "276.pdb")
    filename_b = os.path.join("data", "4629.pdb")

    activesite_a = io.read_active_site(filename_a)
    activesite_b = io.read_active_site(filename_b)

    # testing that the distance between the two sites is as expected
    assert cluster.compute_similarity(activesite_a, activesite_b) == 26.464581285329228
    # testing the the distance between a site and itself is 0
    assert cluster.compute_similarity(activesite_a, activesite_a) == 0

def test_partition_clustering():
    # tractable subset
    pdb_ids = [276, 4629, 10701]

    active_sites = []
    for id in pdb_ids:
        filepath = os.path.join("data", "%i.pdb"%id)
        active_sites.append(io.read_active_site(filepath))

    # update this assertion
    # checking the the three sites cluster as expected
    assert np.array_equal(cluster.cluster_by_partitioning(active_sites, 2)[0], [1, 1, 0])


def test_hierarchical_clustering():
    # tractable subset
    pdb_ids = [276, 4629, 10701]

    active_sites = []
    for id in pdb_ids:
        filepath = os.path.join("data", "%i.pdb"%id)
        active_sites.append(io.read_active_site(filepath))

    # update this assertion
    # checking the the three sites cluster as expected
    assert np.array_equal(cluster.cluster_hierarchically(active_sites, 2)[0], [0, 0, 1])
