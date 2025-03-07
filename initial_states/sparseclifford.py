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

########################
# Alteration notice:
# This code is inspired from the Qiskit project from which it was modified by Michaël Fanuel in March 2025. 
########################

""" """

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit_nature.second_q.mappers import QubitMapper
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit.circuit.library import XGate

from .utils.givensdecompose import _givens_decompose_column_sparse

def _rcols_are_unit_norm(mat: np.ndarray, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    col_norms = np.sqrt(np.sum(mat*mat, axis=0))    
    return np.allclose(col_norms, np.ones_like(col_norms), rtol=rtol, atol=atol)


def _validate_data_matrix(
    mat: np.ndarray, rtol: float = 1e-5, atol: float = 1e-8
) -> None:
    if not len(mat.shape) == 2:
        raise ValueError(
            "data_matrix must be a 2-dimensional array. "
            f"Instead, got shape {mat.shape}."
        )
    if not _rcols_are_unit_norm(mat, rtol=rtol, atol=atol):
        raise ValueError("data_matrix must have unit-norm columns.")

class SparseCliffordLoader(QuantumCircuit):
    r"""A circuit that prepares a Clifford loader for loading a sparse vector such as a column of an edge-vertex incidence matrix.
    Clifford loader is a state of the form

    .. math::
        \mathcal{C}(x_1) \dots \mathcal{C}(x_r) \lvert \text{vac} \rangle,

    where

    .. math::
        \mathcal{C}(x_i) = \mathcal{U}(x_i) (a_1 + a^\dagger_1) \mathcal{U}(x_i)^\dagger.
    Here .. math::\mathcal{U}(x_i) is implemented with sequencial Givens rotations. The resulting circuit has the shape of a pyramid.

    - :math:`x_i` are :math:`N \times 1` unit norm columns of data matrix.
    - :math:`a^\dagger_1, \ldots, a^\dagger_{N}` are the fermionic creation operators.
    - :math:`\lvert \text{vac} \rangle` is the vacuum state.
      (mutual 0-eigenvector of the fermionic number operators :math:`\{a^\dagger_j a_j\}`)


    Currently, only the Jordan-Wigner transformation is supported.

    """


    def __init__(
        self,
        data_matrix: np.ndarray,
        qubit_mapper: QubitMapper | None = None,
        *,
        validate: bool = True,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        **circuit_kwargs,
    ) -> None:
        # pylint: disable=unused-argument
        r"""
        Args:
            data_matrix: matrix whose columns are loaded.
                The columns of the matrix must be have unit 2-norm.
            qubit_mapper: The ``QubitMapper`` or ``QubitConverter`` (use of the latter is
                deprecated). The default behavior is to create one using the call
                ``JordanWignerMapper()``.
            qubit_converter: DEPRECATED The ``QubitConverter`` or ``QubitMapper``. The default
                behavior is to create one using the call ``JordanWignerMapper()``.
            validate: Whether to validate the inputs.
            rtol: Relative numerical tolerance for input validation.
            atol: Absolute numerical tolerance for input validation.
            circuit_kwargs: Keyword arguments to pass to the ``QuantumCircuit`` initializer.

        Raises:
            ValueError: data_matrix must be a 2-dimensional array.
            ValueError: data_matrix must have unit norm columns.
            NotImplementedError: Currently, only the Jordan-Wigner Transform is supported.
                Please use the :class:`qiskit_nature.second_q.mappers.JordanWignerMapper`.
        """
        if validate:
            _validate_data_matrix(data_matrix, rtol=rtol, atol=atol)

        if qubit_mapper is None:
            qubit_mapper = JordanWignerMapper()

        n, m = data_matrix.shape
        register = QuantumRegister(n)
        super().__init__(register, **circuit_kwargs)

        if isinstance(qubit_mapper, JordanWignerMapper):
            
                for i in range(m-1,-1,-1): # over the columns in reverse order
                    # decomposition of U                    
                    decomposition_reversed, pyramidion  = _givens_decompose_column_sparse(register, data_matrix[:,[i]])
                    # we build U (c + c^*) U^{-1}
                    # Circuit of U
                    qc = QuantumCircuit(register)
                    for gate, qubits in decomposition_reversed:
                        qc.append(gate, qubits)
                    # Append U^{-1}
                    self.append(qc.inverse(),register)
                    # top of the pyramid representing (c + c^*)
                    self.append(XGate(), [register[pyramidion]])
                    # Append U
                    self.append(qc,register)
        else:
            raise NotImplementedError(
                "Currently, only the Jordan-Wigner Transform is supported. "
                "Please use the qiskit_nature.second_q.mappers.JordanWignerMapper."
            )
            
