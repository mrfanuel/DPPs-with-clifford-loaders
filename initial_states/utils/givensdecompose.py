# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# Alteration notice:
# This code is inspired from the Qiskit project from which it was modified by Michaël Fanuel in March 2025. 

"""Private helper functions for initial states."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np
from qiskit import QuantumRegister
from qiskit.circuit import Gate, Qubit
from qiskit.circuit.library import XGate, ZGate, XXPlusYYGate
from qiskit_nature.utils import apply_matrix_to_slices, givens_matrix

def _givens_decompose_column_sparse(
    register: QuantumRegister, column_vector: np.ndarray
): #->Iterable[Tuple[Gate, Tuple[Qubit, ...]]]:
    """Decomposes a column matrix as a sequence of Givens rotations
    acting on specific pairs of entries and yields gates implementing the decomposition. These rotations act on neighbouring qubits (sequentially, so that they form pyramidal loaders).

    Args:
        register: The register containing the qubits to use.
        column_vector: The unit norm vector to be decomposed.

    Yields:
        (gate, qubits) pairs describing the operations in the order they have to appear in the circuit, where the qubits are provided in a tuple.
    """
    # for convenience: turn col to row
    column_vector = np.array(column_vector)
    row_matrix = column_vector.transpose()
    _, k = row_matrix.shape

    decomposition: List[Tuple[Gate, Tuple[Qubit, ...]]] = []
    # loop backward over row entries starting from the right end
    pyramidion = 0
    for j in range(k-1, 0, -1):
        if np.isclose(row_matrix[0, j], 1.0):
            pyramidion = j
            break
        if not np.isclose(row_matrix[0, j], 0.0):
            next_j_nz = 0
            for l in range(j-1, 0, -1):
                if not np.isclose(row_matrix[0, l], 0.0):
                    next_j_nz = l
                    break
            a = next_j_nz
            b = j
            # compute Givens rotation to put a zero at entry j
            givens_mat = givens_matrix(row_matrix[0, a], row_matrix[0, b])
            # compute scalar parameters of gate (representing Givens rotation)
            theta = np.arccos(np.real(givens_mat[0, 0]))
            phi = np.angle(givens_mat[0, 1])
            
            # lastly, apply Givens rotation so that there is a zero at entry j
            row_matrix = apply_matrix_to_slices(row_matrix, givens_mat, [(0,a), (0,b)])
            

            if b-a == 1: # no need of parity gate since consecutive j and j_
                decomposition.append(
                        (XXPlusYYGate( 2 * theta.item(), phi.item() - np.pi / 2), (register[b], register[a]))
                    )
            else:                    
                # PARITY
                decomposition_parity = _parity_gate(register,a,b)
                for gate,qubit in decomposition_parity:
                    decomposition.append((gate,qubit))
                #
                # flipping sign tkhs to CZ-gate if parity of j_ + 1 is odd
                decomposition.append((ZGate().control(num_ctrl_qubits=1), (register[a],register[a+1])))
                ####################                
                decomposition.append(
                    (XXPlusYYGate( 2 * theta.item(), phi.item() - np.pi / 2), (register[b], register[a]))
                )
                ####################
                # flipping sign tkhs to CZ-gate if parity of j_ + 1 is odd
                decomposition.append((ZGate().control(num_ctrl_qubits=1), (register[a],register[a+1])))
                # PARITY inverse
                decomposition_parity_rev = _parity_gate(register,a,b,reverse=True)
                for gate,qubit in decomposition_parity_rev:
                    decomposition.append((gate,qubit))
                    
    return reversed(decomposition), pyramidion


def _givens_decompose_column_pyramid(
    register: QuantumRegister, column_vector: np.ndarray
) ->Iterable[Tuple[Gate, Tuple[Qubit, ...]]]:
    """Decomposes a column matrix as a sequence of Givens rotations
    acting on specific pairs of entries and yields gates implementing the decomposition. These rotations act on neighbouring qubits (sequentially, so that they form pyramidal loaders).

    Args:
        register: The register containing the qubits to use.
        column_vector: The unit norm vector to be decomposed.

    Yields:
        (gate, qubits) pairs describing the operations in the order they have to appear in the circuit, where the qubits are provided in a tuple.
    """
    # for convenience: turn col to row
    column_vector = np.array(column_vector)
    row_matrix = column_vector.transpose()
    _, k = row_matrix.shape

    decomposition: List[Tuple[Gate, Tuple[Qubit, ...]]] = []
    # loop backward over row entries starting from the right end
    for j in range(k-1, 0, -1):
        if not np.isclose(row_matrix[0, j], 0.0):
            # compute Givens rotation to put a zero at entry j
            givens_mat = givens_matrix(row_matrix[0, j - 1], row_matrix[0, j])
            # compute scalar parameters of gate (representing Givens rotation)
            theta = np.arccos(np.real(givens_mat[0, 0]))
            phi = np.angle(givens_mat[0, 1])
            decomposition.append(
                (XXPlusYYGate( 2 * theta.item(), phi.item() - np.pi / 2), (register[j], register[j - 1]))
            )        
            # lastly, apply Givens rotation so that there is a zero at entry j
            row_matrix = apply_matrix_to_slices(row_matrix, givens_mat, [(0,j - 1), (0,j)])
    yield from reversed(decomposition)


def _givens_decompose_column_parallel(
    register: QuantumRegister, column_vector: np.ndarray, linear_depth_parity_gate:bool = False
) ->Iterable[Tuple[Gate, Tuple[Qubit, ...]]]:
    """Decomposes a column matrix as a sequence of Givens rotations
    acting on specific pairs of entries and yields gates implementing the decomposition. These rotations are not acting on neighbouring qubits, but are parallel (yielding a log-depth loader).

    Args:
        register: The register containing the qubits to use.
        column_vector: The unit norm vector to be decomposed.
        linear_depth_parity_gate: The option to have linear depth parity gate rather than log depth

    Yields:
        Iterator (gate, qubits) pairs describing the operations in the order they have to appear in the circuit, where the qubits are provided in a tuple. 
    """
    # for convenience: turn col to row
    column_vector = np.array(column_vector)
    row_matrix = column_vector.transpose()
    # number of columns is k
    _, k = row_matrix.shape
    # power of 2 next to k
    l = np.int(np.ceil(np.log(k)/np.log(2)))
    nxt_pow_of_2 = pow(2, l)
    # fill rwo with zeros
    z = np.zeros((1,nxt_pow_of_2 - k))
    row_matrix = np.concatenate((row_matrix, z), axis=1)
    
    decomposition: List[Tuple[Gate, Tuple[Qubit, ...]]] = []
    # loop backward over row entries starting from the right end
    for depth in range(l):
        for a in range(0, nxt_pow_of_2, pow(2, depth+1)):
            b = a + pow(2, depth)
            if not np.isclose(row_matrix[0, b], 0.0):
                # compute Givens rotation to put a zero at entry j
                givens_mat = givens_matrix(row_matrix[0, a], row_matrix[0, b])
                # compute scalar parameters of gate (representing Givens rotation)
                theta = np.arccos(np.real(givens_mat[0, 0]))
                phi = np.angle(givens_mat[0, 1])
                # apply Givens rotation so that there is a zero at entry j
                row_matrix = apply_matrix_to_slices(row_matrix, givens_mat, [(0,a), (0,b)])

                # now implement gate
                if depth == 0: # no need of parity gate since consecutive j and j_
                    decomposition.append(
                        (XXPlusYYGate( 2 * theta.item(), phi.item() - np.pi / 2), (register[b], register[a]))
                    )
                else:                    
                    # PARITY
                    decomposition_parity = _parity_gate(register,a,b,linear_depth_parity_gate)
                    for gate,qubit in decomposition_parity:
                        decomposition.append((gate,qubit))
                    #
                    # flipping sign tkhs to CZ-gate if parity of j_ + 1 is odd
                    # decomposition.append((ZGate().control(num_ctrl_qubits=1), (register[b],register[a+1])))
                    decomposition.append((ZGate().control(num_ctrl_qubits=1), (register[a],register[a+1])))
                    ####################                
                    decomposition.append(
                        (XXPlusYYGate( 2 * theta.item(), phi.item() - np.pi / 2), (register[b], register[a]))
                    )
                    ####################
                    # flipping sign tkhs to CZ-gate if parity of j_ + 1 is odd
                    # decomposition.append((ZGate().control(num_ctrl_qubits=1), (register[b],register[a+1])))
                    decomposition.append((ZGate().control(num_ctrl_qubits=1), (register[a],register[a+1])))
                    # PARITY inverse
                    decomposition_parity_rev = _parity_gate(register,a,b,linear_depth_parity_gate,reverse=True)
                    for gate,qubit in decomposition_parity_rev:
                        decomposition.append((gate,qubit))
                    #
    yield from reversed(decomposition)
    
    
def _parity_gate(register: QuantumRegister, a :int, b: int, linear_depth_parity_gate: bool = False, reverse: bool = False
)->Iterable[Tuple[Gate, Tuple[Qubit, ...]]]:
    """Returns a circuit made of CNOT gates which returns the parity of the qubits in the open interval (a,b) with a < b. 
    Assuming that b - a > 1, the parity of these b-a-1 qubits is stored in qubit a+1. See Section 2.3 of https://arxiv.org/pdf/2202.00054.
    If linear_depth_parity_gate is true, this circuit has a linear depth. Otherwise, define k as the largest integer 
    such that b - a - 1 = 2^k + remainder, where remainder is also an integer. In that case, we return a circuit with log depth on the first 2^k qubits and linear depth on the remaining qubits.

    Args:
        register: The register containing the qubits to use.
        a: lower endpoint of open interval.
        b: upper endpoint of open interval.
        linear_depth_parity_gate: if true, yields a linear depth circuit parity calculation.
        reverse: if true, yields a circuit with gates appearing in reverse order.
    Yields:
        Iterator (gate, qubits) where gate is a CNOT.
    """
    if  a > b:
        raise ValueError(
            "indices of qubits in wrong order. We require a < b "
        )
    if  a == b:
        raise ValueError(
            "indices of qubits should not be equal. "
        )
    if a + 1 == b:
        raise ValueError(
            "no qubit between a and b. "
        )
    decomposition: List[Tuple[Gate, Tuple[Qubit, ...]]] = []

    nb_intermediate_qubits = b - a - 1
    # parity gate: computing parity of qubits between a and b and bringing it to qubit a + 1
    # we decompose b - a - 1 = 2^inner_depth + remainder
    if linear_depth_parity_gate or nb_intermediate_qubits < 4:
        if nb_intermediate_qubits > 0 : 
            for c in range(a + 2, b):
                decomposition.append((XGate().control(num_ctrl_qubits=1), (register[c],register[a + 1])))
    else: 
        n_remaining = nb_intermediate_qubits
        start_qb = a
        while n_remaining > 1:
            inner_depth = np.int(np.floor(np.log(n_remaining)/np.log(2)))
            n_remaining = n_remaining - pow(2, inner_depth)
            for depth in range(inner_depth):
                for k_start in range(start_qb + 1, start_qb +  pow(2, inner_depth), pow(2, depth + 1)):
                    k_end = k_start + pow(2, depth)
                    decomposition.append((XGate().control(num_ctrl_qubits=1), (register[k_end],register[k_start])))
            if start_qb > a:
                decomposition.append((XGate().control(num_ctrl_qubits=1), (register[start_qb + 1],register[a + 1])))
            start_qb = start_qb +  pow(2, inner_depth)
        if n_remaining == 1:
            decomposition.append((XGate().control(num_ctrl_qubits=1), (register[start_qb + 1],register[a + 1])))
    if reverse:
        decomposition = reversed(decomposition)
    yield from decomposition