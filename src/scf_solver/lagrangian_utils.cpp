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

#include "lagrangian_utils.h"

#include <MRCPP/Printer>

#include "qmfunctions/Orbital.h"
#include "qmfunctions/density_utils.h"
#include "qmoperators/two_electron/two_electron_utils.h"

namespace mrchem {
namespace lagrangian {

void update_two_body(double prec, double threshold, bool real_orbitals, OrbitalVector &orbitals, const ComplexMatrix &one_rdm, const ComplexMatrix &lag_coeff, const ComplexTensorR4 &U_dE2, OrbitalVector &new_Phi) {
    const int L = orbitals.size();
    auto coeff_ = [real_orbitals](ComplexDouble c) { return real_orbitals ? ComplexDouble(c.real(), 0.0) : c; };

    // Computing total density to use for screening potential computations
    Density rho;
    density::compute(prec, rho, orbitals, DensityType::Total);

    int num_skipped = 0;

    // two-body term: m on the new basis (through U_dE2), j, k, l on the old basis (like two_rdm)
    for (int j = 0; j < L; j++) {
        println(3, "Orbital " << j + 1 << " out of " << L);
        for (int l = j; l < L; l++) { // Triangular loop l >= j
            // g_jl = poisson(Phi[j]^* * Phi[l]) (the MRCPP kernel is already 1/|r-r'|, no 4*pi as in python)
            Orbital g_jl = orbitals[j].paramCopy(true);

            double max_coef = 0.0;

            for (int k = 0; k < L; k++) {
                for (int m = 0; m < L; m++) {
                    max_coef = std::max(max_coef, std::abs(U_dE2(m, j, k, l)));
                    max_coef = std::max(max_coef, std::abs(U_dE2(m, l, k, j)));
                }
            }

            if (max_coef < 1.0e-16 or ij_r12(prec / max_coef, rho, orbitals[j], orbitals[l], g_jl)) {
                // MSG_INFO("Skipped orbital pair (" << j << ", " << l << ")");
                num_skipped++;
                continue;
            }

            for (int k = 0; k < L; k++) {
                // state = g_jl * Phi[k]
                Orbital state;
                mrcpp::multiply(state, g_jl, orbitals[k], prec);

                for (int m = 0; m < L; m++) {
                    ComplexDouble denom;
                    if (std::abs(one_rdm(m, m)) > threshold)
                        denom = one_rdm(m, m);
                    else if (std::abs(lag_coeff(m, m)) > threshold)
                        denom = lag_coeff(m, m);
                    else
                        continue; // both below threshold, skip

                    ComplexDouble u = U_dE2(m, j, k, l);
                    if (l != j) u += U_dE2(m, l, k, j);

                    new_Phi[m].add(coeff_(2.0 * u / denom), state);
                }
            }
        }
    }

    MSG_INFO("Skipped " << num_skipped << " orbital pairs");
}

} // namespace lagrangian
} // namespace mrchem
