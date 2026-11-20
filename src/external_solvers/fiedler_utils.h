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

#include <string>
#include <type_traits>

#include "mrchem.h"

namespace mrchem {

/**
 * @namespace fiedler
 * @brief Fiedler ordering of the orbitals in the DMRG chain (sites)
 *
 * A permutation is stored as an IntVector perm, where perm[s] is the orbital that sits at site s.
 * The identity permutation is the order of the input orbitals.
 */
namespace fiedler {

DoubleMatrix exchange_matrix(const ComplexTensorR4 &g);
IntVector order(const DoubleMatrix &K, double edge_thrs, const IntVector &ref);
double cost(const DoubleMatrix &K, const IntVector &perm);
DoubleVector bond_cuts(const DoubleMatrix &K, const IntVector &perm);
IntVector inverse(const IntVector &perm);
ComplexMatrix permute(const ComplexMatrix &A, const IntVector &perm);
ComplexMatrix unpermute(const ComplexMatrix &A, const IntVector &perm);
ComplexTensorR4 permute(const ComplexTensorR4 &A, const IntVector &perm);
ComplexTensorR4 unpermute(const ComplexTensorR4 &A, const IntVector &perm);

/**
 * @brief Write the integrals in chain order into a row-major tensor: R(s,t,u,v) = g(perm[s],perm[t],perm[u],perm[v])
 *
 * @param g     integrals in orbital order (column-major)
 * @param R     output (row-major), same size as g. Complex or real (real part is taken)
 * @param perm  perm[s] is the orbital at site s
 *
 * The loops follow the (contiguous) memory of g and scatter into R.
 */
template <class T> void fill_rowmajor(const ComplexTensorR4 &g, Eigen::Tensor<T, 4, Eigen::RowMajor> &R, const IntVector &perm) {
    const int N = perm.size();
    IntVector site(N);
    for (int s = 0; s < N; s++) site(perm(s)) = s;
    for (int d = 0; d < N; d++)
        for (int c = 0; c < N; c++)
            for (int b = 0; b < N; b++)
                for (int a = 0; a < N; a++) {
                    if constexpr (std::is_same<T, double>::value)
                        R(site(a), site(b), site(c), site(d)) = g(a, b, c, d).real();
                    else
                        R(site(a), site(b), site(c), site(d)) = g(a, b, c, d);
                }
}

} // namespace fiedler
} // namespace mrchem
