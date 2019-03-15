import glob
import os
from Bio import SeqIO
import numpy as np
import random

def read_txt(filepath):
    with open(filepath) as txt:
        seqs =  txt.readlines()
    seqs = [x.rstrip() for x in seqs]
    return(seqs)

def read_fasta(filename):
    seqs = []
    for record in SeqIO.parse(filename, "fasta"):
        seqs.append(str(record.seq))
    return seqs

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

def translate_list(seq_list):
    bin_list = list(map(seq_to_binary, seq_list))
    bin_list = np.array(bin_list)
    return bin_list

def downsample(sample_list, number_obs, len_obs):
    x = random.sample(range(len(sample_list)), number_obs)
    shortlist = []
    for i in x:
        #print(sample_list[i])
        start = random.randint(0,len(sample_list[i])-len_obs)
        stop = start + len_obs
        substr = sample_list[i][start:stop]
        shortlist.append(substr)
    return shortlist
