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

#include <MRCPP/Parallel>
#include <MRCPP/Printer>

#include "qmfunctions/Orbital.h"
#include "qmfunctions/density_utils.h"
#include "qmoperators/two_electron/two_electron_utils.h"
#include "utils/mpi_bcast.h"

namespace mrchem {
namespace lagrangian {

namespace {

/** @brief Serial path (one worker): all orbitals are local, accumulation directly into new_Phi (today's loop and order) */
void update_two_body_serial(double prec, double threshold, bool real_orbitals, OrbitalVector &orbitals, const ComplexMatrix &one_rdm, const ComplexMatrix &lag_coeff, const ComplexTensorR4 &U_dE2, Density &rho, OrbitalVector &new_Phi) {
    const int L = orbitals.size();
    auto coeff_ = [real_orbitals](ComplexDouble c) { return real_orbitals ? ComplexDouble(c.real(), 0.0) : c; };

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
                Orbital state_conj;
                if (l != j and not real_orbitals) mrcpp::multiply(state_conj, g_jl, orbitals[k], prec, false, false, true); // conjugate is applied on g_jl (first argument)

                for (int m = 0; m < L; m++) {
                    ComplexDouble denom;
                    if (std::abs(one_rdm(m, m)) > threshold)
                        denom = one_rdm(m, m);
                    else if (std::abs(lag_coeff(m, m)) > threshold)
                        denom = lag_coeff(m, m);
                    else
                        continue; // both below threshold, skip

                    ComplexDouble u = U_dE2(m, j, k, l);
                    if (l == j) {
                        new_Phi[m].add(coeff_(2.0 * u / denom), state);
                    } else if (real_orbitals) {
                        u += U_dE2(m, l, k, j); // real orbitals: g_lj = g_jl, one shared product
                        new_Phi[m].add(coeff_(2.0 * u / denom), state);
                    } else {
                        // complex orbitals: g_lj = conj(g_jl), so the (l,j) term needs conj(g_jl) * Phi[k]
                        new_Phi[m].add(coeff_(2.0 * u / denom), state);
                        new_Phi[m].add(coeff_(2.0 * U_dE2(m, l, k, j) / denom), state_conj);
                    }
                }
            }
        }
    }

    MSG_INFO("Skipped " << num_skipped << " orbital pairs");
}

/** @brief Distributed path (several workers + bank): pair-distributed update, see casscf_mpi_plan.md section 2.3
 *
 * MPI: collective over the worker communicator. The orbitals are distributed (orbital i lives on rank i % wrk_size);
 * the pairs (j<=l) are distributed round-robin; the partial sums for every target m go through the bank and are added up
 * by the owner of m in a fixed rank order (reproducible for a given number of ranks).
 */
void update_two_body_distributed(double prec, double threshold, bool real_orbitals, OrbitalVector &orbitals, const ComplexMatrix &one_rdm, const ComplexMatrix &lag_coeff, const ComplexTensorR4 &U_dE2, Density &rho, OrbitalVector &new_Phi) {
    const int L = orbitals.size();
    const int me = mrcpp::mpi::wrk_rank;
    const int P = mrcpp::mpi::wrk_size;
    auto coeff_ = [real_orbitals](ComplexDouble c) { return real_orbitals ? ComplexDouble(c.real(), 0.0) : c; };

    // 0. pair list (j <= l) with max |U_dE2|; decided by the master so that all workers agree
    IntVector use_pair = IntVector::Zero(L * L);
    DoubleVector max_u = DoubleVector::Zero(L * L);
    if (me == 0) {
        for (int j = 0; j < L; j++)
            for (int l = j; l < L; l++) {
                double mc = 0.0;
                for (int k = 0; k < L; k++)
                    for (int m = 0; m < L; m++) mc = std::max(mc, std::max(std::abs(U_dE2(m, j, k, l)), std::abs(U_dE2(m, l, k, j))));
                max_u(j * L + l) = mc;
                use_pair(j * L + l) = (mc >= 1.0e-16) ? 1 : 0;
            }
    }
    mpi_utils::bcast(use_pair);
    mpi_utils::bcast(max_u);
    std::vector<std::pair<int, int>> my_pairs; // round-robin over the pair list
    int ip = 0;
    for (int j = 0; j < L; j++)
        for (int l = j; l < L; l++)
            if (use_pair(j * L + l)) {
                if (ip % P == me) my_pairs.emplace_back(j, l);
                ip++;
            }

    // 1. all orbitals on every worker (own ones are local, the others come from the bank)
    mrcpp::BankAccount PhiBank;
    for (int i = 0; i < L; i++)
        if (mrcpp::mpi::my_func(i)) PhiBank.put_func(i, orbitals[i]);
    std::vector<Orbital> Phi_loc(L);
    for (int i = 0; i < L; i++) {
        if (mrcpp::mpi::my_func(i)) {
            Phi_loc[i] = orbitals[i];
        } else {
            PhiBank.get_func(i, Phi_loc[i], 1);
        }
    }

    // 2. own pair potentials and accumulation of the partial sums for every target m
    std::vector<Orbital> Acc(L);
    int num_skipped = 0;
    for (auto &pr : my_pairs) {
        const int j = pr.first, l = pr.second;
        Orbital g_jl = Phi_loc[j].paramCopy(true);
        if (ij_r12(prec / max_u(j * L + l), rho, Phi_loc[j], Phi_loc[l], g_jl)) {
            num_skipped++;
            continue;
        }
        for (int k = 0; k < L; k++) {
            Orbital state;
            mrcpp::multiply(state, g_jl, Phi_loc[k], prec);
            Orbital state_conj;
            if (l != j and not real_orbitals) mrcpp::multiply(state_conj, g_jl, Phi_loc[k], prec, false, false, true); // conjugate is applied on g_jl (first argument)
            for (int m = 0; m < L; m++) {
                ComplexDouble denom;
                if (std::abs(one_rdm(m, m)) > threshold)
                    denom = one_rdm(m, m);
                else if (std::abs(lag_coeff(m, m)) > threshold)
                    denom = lag_coeff(m, m);
                else
                    continue;
                ComplexDouble u = U_dE2(m, j, k, l);
                if (l == j) {
                    Acc[m].add(coeff_(2.0 * u / denom), state);
                } else if (real_orbitals) {
                    u += U_dE2(m, l, k, j); // real orbitals: g_lj = g_jl, one shared product
                    Acc[m].add(coeff_(2.0 * u / denom), state);
                } else {
                    // complex orbitals: g_lj = conj(g_jl), so the (l,j) term needs conj(g_jl) * Phi[k]
                    Acc[m].add(coeff_(2.0 * u / denom), state);
                    Acc[m].add(coeff_(2.0 * U_dE2(m, l, k, j) / denom), state_conj);
                }
            }
        }
        g_jl.free();
    }
    for (int i = 0; i < L; i++)
        if (not mrcpp::mpi::my_func(i)) Phi_loc[i].free();

    // 3. which worker has a partial sum for which target (from the actual results; one collective)
    IntVector has_part = IntVector::Zero(L * P);
    for (int m = 0; m < L; m++) has_part(m + L * me) = (Acc[m].getNNodes() > 0) ? 1 : 0;
    mrcpp::mpi::allreduce_vector(has_part, mrcpp::mpi::comm_wrk);

    // 4. send the partial sums of foreign targets to the bank, owners add them up in rank order
    mrcpp::BankAccount AccBank;
    for (int m = 0; m < L; m++) {
        if (has_part(m + L * me) and not mrcpp::mpi::my_func(m)) {
            AccBank.put_func(m + L * me, Acc[m]);
            Acc[m].free();
        }
    }
    for (int m = 0; m < L; m++) {
        if (not mrcpp::mpi::my_func(m)) continue;
        for (int r = 0; r < P; r++) {
            if (not has_part(m + L * r)) continue;
            if (r == me) {
                new_Phi[m].add(1.0, Acc[m]);
            } else {
                Orbital f;
                AccBank.get_func(m + L * r, f, 1);
                new_Phi[m].add(1.0, f);
                f.free();
            }
        }
        Acc[m].free();
    }
    // Single block: the bank accounts are freed by their destructors (clear() is only needed to free memory between blocks, S10)
    if (me == 0) MSG_INFO("Skipped " << num_skipped << " orbital pairs on the master");
}

} // namespace

void update_two_body(double prec, double threshold, bool real_orbitals, OrbitalVector &orbitals, const ComplexMatrix &one_rdm, const ComplexMatrix &lag_coeff, const ComplexTensorR4 &U_dE2, OrbitalVector &new_Phi) {
    // Computing total density to use for screening potential computations (collective)
    Density rho;
    density::compute(prec, rho, orbitals, DensityType::Total);

    // wrk_size == 1 also covers np=2 (1 worker + 1 bank): no distribution, no bank accounts
    if (mrcpp::mpi::wrk_size > 1)
        update_two_body_distributed(prec, threshold, real_orbitals, orbitals, one_rdm, lag_coeff, U_dE2, rho, new_Phi);
    else
        update_two_body_serial(prec, threshold, real_orbitals, orbitals, one_rdm, lag_coeff, U_dE2, rho, new_Phi);
}

} // namespace lagrangian
} // namespace mrchem
