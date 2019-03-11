import glob
import os
from Bio import SeqIO
import numpy as np

def read_fasta(x):
    return x

def get_positives(filepath):
    with open(filepath) as pos:
       positives =  pos.readlines()
    positives = [x.rstrip() for x in positives]
    return(positives)

def seq_to_binary(seq):
    seq_b=np.array([])
    bases={
    'A':np.array([1,0,0,0]),
    'T':np.array([0,1,0,0]),
    'C':np.array([0,0,1,0]),
    'G':np.array([0,0,0,1])}
    for base in seq:
        seq_b=np.concatenate([seq_b,bases[base]])
    return seq_b
