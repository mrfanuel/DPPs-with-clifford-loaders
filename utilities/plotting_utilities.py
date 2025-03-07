from itertools import chain, combinations
import numpy as np
import pandas as pd
from numpy import linalg as LA
import matplotlib.pyplot as plt


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))#.__next__()

def get_estimated_correlation_functions(K, counts, num_shots, N):
    for indices in powerset(range(N)):
        rho[indices] = LA.det(K[np.ix_(indices, indices)])
        rho_estimated[indices] = 0.0
        for key in counts: 
            key_as_array = np.array(list(key), dtype=int)[::-1] # reverse because qiskit orders the other way
            if np.all(key_as_array[list(indices)]):
                rho_estimated[indices] += counts[key] 
        rho_estimated[indices] /= num_shots  
    return rho, rho_estimated

def get_estimated_probabilities(K,rank, counts, num_shots):
    N,_ = K.shape
    proba = {} # add empty set for compatibility with empty IBM measurements
    proba_estimated = {}
    num_samples = np.sum([counts[key] for key in counts])
    if not num_samples == num_shots:
        print("Warning: expected num_shots is", num_shots, "but number of observed samples is", num_samples)
    for indices in powerset(range(N)):
        # set expected value
        if len(indices)==rank:
            proba[indices] = LA.det(K[np.ix_(indices, indices)])
        else:
            proba[indices] = 0.0
        # set estimated value
        if indices == ():
            key = "0"*N
        else:
            key_as_array = np.zeros(N, dtype="int")
            key_as_array[N-np.array(indices)-1] = 1
            key = ''
            for entry in key_as_array:
                key+=str(entry)
        if key in counts:
            proba_estimated[indices] = 1.0*counts[key] / num_samples
        else:
            proba_estimated[indices] = 0.0
    return proba, proba_estimated


def get_estimated_probabilities_skew_cutoff(W,max_cardinality, counts, num_shots):
    N,_ = W.shape
    
    Sk = np.triu(W.T @ W)
    Sk = Sk - Sk.T
    
    proba = {} # add empty set for compatibility with empty IBM measurements
    proba_estimated = {}
    num_samples = np.sum([counts[key] for key in counts])
    if not num_samples == num_shots:
        print("Warning: expected num_shots is", num_shots, "but number of observed samples is", num_samples)
    for indices in powerset(range(N)):
        if len(indices) <= max_cardinality:
            # set expected value
            l = len(indices)
            if l>0:
                Z = np.zeros((l, l))
                P = W[indices,:]
                L = np.block([[Z,P],[-P.T, Sk]])
                proba[indices] = np.round(abs(LA.det(L)),decimals=6) # abs is not necessary (only for numerical errors)
            else:
                proba[indices] = np.round(abs(LA.det(Sk)),decimals=6) # abs is not necessary (only for numerical errors)
            # set estimated value
            if indices == ():
                key = "0"*N
            else:
                key_as_array = np.zeros(N, dtype="int")
                key_as_array[N-np.array(indices)-1] = 1 # need to reverse the order
                key = ''
                for entry in key_as_array:
                    key+=str(entry)
            if key in counts:
                proba_estimated[indices] = 1.0*counts[key] / num_samples
            else:
                proba_estimated[indices] = 0.0
    return proba, proba_estimated

def get_estimated_probabilities_skew(W, counts, num_shots):
    N,_ = W.shape
    
    Sk = np.triu(W.T @ W)
    Sk = Sk - Sk.T
    
    proba = {} # add empty set for compatibility with empty IBM measurements
    proba_estimated = {}
    num_samples = np.sum([counts[key] for key in counts])
    if not num_samples == num_shots:
        print("Warning: expected num_shots is", num_shots, "but number of observed samples is", num_samples)
    for indices in powerset(range(N)):
        # set expected value
        l = len(indices)
        if l>0:
            Z = np.zeros((l, l))
            P = W[indices,:]
            L = np.block([[Z,P],[-P.T, Sk]])
            proba[indices] = np.round(abs(LA.det(L)),decimals=6) # abs is not necessary (only for numerical errors)
        else:
            proba[indices] = np.round(abs(LA.det(Sk)),decimals=6) # abs is not necessary (only for numerical errors)
        # set estimated value
        if indices == ():
            key = "0"*N
        else:
            key_as_array = np.zeros(N, dtype="int")
            key_as_array[N-np.array(indices)-1] = 1 # need to reverse the order
            key = ''
            for entry in key_as_array:
                key+=str(entry)
        if key in counts:
            proba_estimated[indices] = 1.0*counts[key] / num_samples
        else:
            proba_estimated[indices] = 0.0
    return proba, proba_estimated

def get_estimated_probabilities_skew_cutoff(W,max_cardinality, counts, num_shots):
    N,_ = W.shape
    
    Sk = np.triu(W.T @ W)
    Sk = Sk - Sk.T
    
    proba = {} # add empty set for compatibility with empty IBM measurements
    proba_estimated = {}
    num_samples = np.sum([counts[key] for key in counts])
    if not num_samples == num_shots:
        print("Warning: expected num_shots is", num_shots, "but number of observed samples is", num_samples)
    for indices in powerset(range(N)):
        if len(indices) <= max_cardinality:
            # set expected value
            l = len(indices)
            if l>0:
                Z = np.zeros((l, l))
                P = W[indices,:]
                L = np.block([[Z,P],[-P.T, Sk]])
                proba[indices] = np.round(abs(LA.det(L)),decimals=6) # abs is not necessary (only for numerical errors)
            else:
                proba[indices] = np.round(abs(LA.det(Sk)),decimals=6) # abs is not necessary (only for numerical errors)
            # set estimated value
            if indices == ():
                key = "0"*N
            else:
                key_as_array = np.zeros(N, dtype="int")
                key_as_array[N-np.array(indices)-1] = 1 # need to reverse the order
                key = ''
                for entry in key_as_array:
                    key+=str(entry)
            if key in counts:
                proba_estimated[indices] = 1.0*counts[key] / num_samples
            else:
                proba_estimated[indices] = 0.0
    return proba, proba_estimated


def rejection_sampling_from_DPP(W, counts, num_shots):
    N,_ = W.shape
    
    Sk = np.triu(W.T @ W)
    Sk = Sk - Sk.T
    
    proba = {} # add empty set for compatibility with empty IBM measurements
    proba_estimated = {}
    num_samples = np.sum([counts[key] for key in counts])
    if not num_samples == num_shots:
        print("Warning: expected num_shots is", num_shots, "but number of observed samples is", num_samples)
    for indices in powerset(range(N)):
        # set expected value
        l = len(indices)
        if l>0:
            Z = np.zeros((l, l))
            P = W[indices,:]
            L = np.block([[Z,P],[-P.T, Sk]])
            proba[indices] = np.round(abs(LA.det(L)),decimals=6) # abs is not necessary (only for numerical errors)
        else:
            proba[indices] = np.round(abs(LA.det(Sk)),decimals=6) # abs is not necessary (only for numerical errors)
        # set estimated value
        if indices == ():
            key = "0"*N
        else:
            key_as_array = np.zeros(N, dtype="int")
            key_as_array[N-np.array(indices)-1] = 1 # need to reverse the order
            key = ''
            for entry in key_as_array:
                key+=str(entry)
        if key in counts:
            proba_estimated[indices] = 1.0*counts[key] / num_samples
        else:
            proba_estimated[indices] = 0.0
    return proba, proba_estimated


def load_ibm_result_csv_file(file_name, column_name, num_shots):
    df = pd.read_csv(file_name, index_col="Measurement outcome")
    transform = lambda ss: tuple(np.where(list(reversed([int(s) for s in str(ss)])))[0])   
    index = list(map(transform, df.index.values))
    df.index = index
    df.index.name = "outcome"  
    df.rename(columns = {"Frequency":column_name}, inplace = True)
    num_samples = df[column_name].sum()
    if not num_samples == num_shots:
        print("Warning: expected num_shots is", num_shots, "but number of observed samples is", num_samples)
    df[column_name] /= num_samples
    return df


def sym_rel_diff(rho_2,rho_2_estimated,):
    return 2 * np.abs(rho_2-rho_2_estimated)/np.abs(rho_2 + rho_2_estimated)

def plot_results(counts,rho_2,num_shots):
    
    
    N = rho_2.shape[0]
    rho_2_estimated = np.array([
        [
            np.sum([1.0*counts[key]/num_shots for key in counts if (key[N-i-1] == '1' and key[N-j-1]== '1')]) 
        for i in range(N)
        ]
    for j in range(N)  
    ])

    fig, ax = plt.subplots(1, 3)
    vmin = 0
    vmax = 1
    cmap_str = "binary"

    ax[0].matshow(rho_2, cmap=cmap_str, vmin=vmin, vmax=vmax)
    ax[0].set_title("true")

    im = ax[1].matshow(rho_2_estimated, cmap=cmap_str, vmin=vmin, vmax=vmax)
    ax[1].set_title("estimated")

    epsilon = 1e-10
    ax[2].matshow(sym_rel_diff(rho_2 + epsilon ,rho_2_estimated + epsilon), cmap=cmap_str, vmin=vmin, vmax=vmax)
    ax[2].set_title("sym. relative err.")

    fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.4)
    plt.savefig('../fig/PfPP_rho.pdf', bbox_inches='tight')
    plt.show()
        
    #return rho_2, rho_2_estimated
    
def sample_size(s):
    nb = 0
    for l in range(len(s)):
        nb += int(s[l])
    return nb