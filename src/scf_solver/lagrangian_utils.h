/*
 * MRChem, a numerical real-space code for molecular electronic structure
 * calculations within the self-consistent field (SCF) approximations of quantum
 * chemistry (Hartree-Fock and Density Functional Theory).
 * Copyright (C) 2023 Stig Rune Jensen, Luca Frediani, Peter Wind and contributors.
 *
 * This file is part of MRChem.
 *
 * MRChem is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * MRChem is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with MRChem.  If not, see <https://www.gnu.org/licenses/>.
 *
 * For information on the complete list of contributors to MRChem, see:
 * <https://mrchem.readthedocs.io/>
 */

#pragma once

#include "mrchem.h"
#include "qmfunctions/qmfunction_fwd.h"

namespace mrchem {

/** @file lagrangian_utils.h
 *
 * @brief Building blocks of the Lagrangian orbital update that do not depend on the solver objects.
 */
namespace lagrangian {

/** @brief Two-body term of the orbital update (before the Helmholtz kernel)
 *
 * @param prec: precision of the products and potentials
 * @param threshold: occupations/multipliers below this are skipped
 * @param real_orbitals: real mode (coefficients are truncated to their real part)
 * @param orbitals: current orbitals (old basis)
 * @param one_rdm, lag_coeff: 1-RDM and Lagrange multipliers in the new basis
 * @param U_dE2: U_dE2(m,j,k,l) = sum_i U(m,i) two_rdm(i,j,k,l), m in the new basis
 * @param new_Phi: in/out, the term is accumulated into it
 * @param block_size: number of target orbitals per block in the distributed path (<= 0: default from the number of workers)
 *
 * Accumulates 2 sum_{jkl} U_dE2(m,j,k,l) / denom(m) * poisson(phi_j^* phi_l) phi_k into new_Phi[m].
 * Uses the triangular loop l >= j with the symmetric partner (l,j) folded in.
 */
void update_two_body(double prec,
                     double threshold,
                     bool real_orbitals,
                     OrbitalVector &orbitals,
                     const ComplexMatrix &one_rdm,
                     const ComplexMatrix &lag_coeff,
                     const ComplexTensorR4 &U_dE2,
                     OrbitalVector &new_Phi,
                     int block_size = 0);

} // namespace lagrangian
} // namespace mrchem
