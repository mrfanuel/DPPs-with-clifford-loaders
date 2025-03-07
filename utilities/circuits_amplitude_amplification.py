from qiskit.circuit.library import XGate, MCMT, RZGate,QFT
import qiskit as qk
from qiskit_aer import Aer

from initial_states.parallelclifford import ParallelCliffordLoader
from initial_states.pyramidclifford import PyramidCliffordLoader

import numpy as np

def CircuitForV(n):
    '''
    Creates quantum circuit for V. (Code adapted from Rethinasamy et al., 2024 https://arxiv.org/pdf/2404.07151)
    '''
    # find nearest power of 2 larger than n
    k = int(np.ceil(np.log2(n+1)))
    # there are w extra qubits (eliminated later on)
    w = 2**k - (n+1)
    N = n + w

    # prepare operator V = (IQFT x I^n)(prod_m=1^k CU_m)(H^k x I^n)
    circ = qk.QuantumCircuit(k + n)
    # k control qubits and n main qubits

    # act with Hadamard gates on the control qubits namely (H^k x I^n)
    # superPosCircuit = createEqSuperpos(2**k)
    superPosCircuit = qk.QuantumCircuit(k)
    for i in range(k):
        superPosCircuit.h(i)
    circ.append(superPosCircuit.reverse_bits(), list(range(0, k)))

    # we now apply the controlled unitaries (prod_m=1^k CU_m)
    # The following decomposition of CU_m follows from Remark 3
    # of Rethinasamy et al., 2024

    # This sequence of one-qubit gates on the control register comes 
    # in order to compensate for a 'global phase' on the main register
    for i in range(k):
        y = 2**i 
        circ.append(RZGate(np.pi*n*y/(N+1)), [i])
    # circ.barrier()
    # Below, we simply apply controlled Z-rotations (without the phase then)
    for i in range(k):
        t = k-i-1
        y = 2**t
        for j in range(n):
            tqubit = k+j
            circ.append(RZGate(2*np.pi*y/(N+1)).control(1), [t, tqubit])
        # circ.barrier()

    # add inverse QFT to control qubits
    circ.append(QFT(k).inverse(), list(range(0, k)))
    return circ

def CircuitForO(r,k):

    circ = qk.QuantumCircuit(k)
    circ_r = qk.QuantumCircuit(k)
    bin_r = [int(i) for i in list('{0:0b}'.format(r))]
    bin_r.reverse()
    # warning: at this point the bin_r might no have k entries
    # we append zeroes to bin_r until it has k entries
    while len(bin_r) != k:
        bin_r.append(0)

    id = 0
    for i in bin_r:
        if i == 0:      
            circ_r.append(XGate(), [id])
        id = id + 1

    circ.append(circ_r,list(range(0, k)))
    circ.append(MCMT('cz',k-1,1),list(range(0, k)))
    circ.append(circ_r,list(range(0, k)))

    return circ


def CircuitForS0(n):
    circ = qk.QuantumCircuit(n)
    circ_r = qk.QuantumCircuit(n)

    for i in range(0,n):
        circ_r.append(XGate(), [i])

    circ.append(circ_r,list(range(0, n)))
    circ.append(MCMT('cz',n-1,1),list(range(0, n )))
    circ.append(circ_r,list(range(0, n )))

    return circ

def amplitude_amplification_circuit(psi,n,k,r):
    '''
    Creates quantum circuit for Grover operator Q
    '''
    # now we compose with one Grover operator Q
    circ_Q = qk.QuantumCircuit(k + n)
    # apply V
    circ_V = CircuitForV(n)
    circ_V.name = 'V'
    circ_Q.append(circ_V, list(range(0, k + n)))
    # apply O(r) X I^n
    circ_O = CircuitForO(r,k)
    circ_O.name = 'O_r'
    circ_Q.append(circ_O,list(range(0, k)))
    # apply V^*
    circ_Q.append(circ_V.inverse(), list(range(0, k + n)))
    # apply I^k x C^*
    circ_Q.append(psi.inverse(), list(range(k, k + n)))
    # apply  I^k x S0
    circ_S0 = CircuitForS0(n)
    circ_S0.name = 'S_0'
    circ_Q.append(circ_S0, list(range(k, k + n)))
    # apply I^k x C^*
    circ_Q.append(psi, list(range(k, k + n)))
    circ_Q.name = 'Q'
    
    return circ_Q


def sample_Clifford_circuit(X,num_shots,m_Grover):
    '''
    samples from clifford circuit (computationnal basis) with (m_Grover = 0) or without amplification (m_Grover > 0)
    '''
    W = X / np.sqrt(np.sum(X*X, axis=0)) 
    n = W.shape[0] # number of edges
    r = W.shape[1]
    
    psi = ParallelCliffordLoader(W) # (with FBS gates)
    psi.name = 'C_X'
    # circuit = PyramidCliffordLoader(W) # (with only RBS gates)
    
    if m_Grover > 0:
        # append to circuit Q^m_grover
        # number of control qubits
        k = int(np.ceil(np.log2(n+1)))
        # print("k = ",k, " control qubits")
        # print("n = ",n, " main qubits")

        # basic state appended with k control qubits
        circ = qk.QuantumCircuit(k + n)
        circ.append(psi, list(range(k, k + n)))

        circ_Q = amplitude_amplification_circuit(psi,n,k,r)
        
        circ_Q = circ_Q.repeat(m_Grover)
        circ.append(circ_Q, list(range(0, k + n)))
        # Add measurements
        meas = qk.QuantumCircuit(n + k, n + k) # n + k qubits, n + k classical bits
        meas.barrier(range(n + k)) # the barrier is optional, it is an instruction for the later transpiler
        meas.measure(range(k, k + n), range(k, k + n)) # perform the measurement, record it in the classical bits
        circ.add_register(meas.cregs[0])
        qc = circ.compose(meas)
        
    else:
        # no amplification
        # Add measurements
        meas = qk.QuantumCircuit(n, n) # n qubits, n classical bits
        meas.barrier(range(n)) # the barrier is optional, it is an instruction for the later transpiler
        meas.measure(range(n), range(n)) # perform the measurement, record it in the classical bits
        psi.add_register(meas.cregs[0])
        qc = psi.compose(meas)
    
    # Now add measurements
    backend_sim = Aer.get_backend('qasm_simulator')
    job_sim = backend_sim.run(
        qk.transpile(qc, backend_sim), 
        shots=num_shots
    )
    result_sim = job_sim.result()
    counts = result_sim.get_counts(qc)
    # print(counts)
    # if m_Grover > 0: beware counts has n+k entries (although k controls output zeroes)
    
    # process the measurements to output array containing the samples
    samples = np.zeros((num_shots,n))
    id=0
    for key_rev in counts:
        # reverse order of bistring
        key = key_rev[::-1]
        nb_identical = counts[key_rev]
        # covert to np array
        key_array = np.asarray(list(key), dtype=int)
        for _ in range(0,nb_identical):
            if m_Grover > 0:
                # remove the output of meas on control qubits
                samples[id,:] = key_array[range(k,n+k)]
            else:
                samples[id,:] = key_array
            id = id + 1
        # print("id ", id)

    return samples