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
#include <MRCPP/Timer>

#include "qmfunctions/Orbital.h"
#include "qmfunctions/density_utils.h"
#include "qmoperators/two_electron/two_electron_utils.h"
#include "utils/mpi_bcast.h"

namespace mrchem {
namespace lagrangian {

namespace {

/** @brief Block size B (number of target orbitals m per block) used when `requested` <= 0
 *
 * Same style as `calc_2elintegrals`: L / sqrt(14 W) clamped to 2...16 (W = number of workers), at most L.
 */
int default_block_size(int L, int W) {
    const int B = static_cast<int>(L / std::sqrt(14.0 * W));
    return std::min(L, std::min(16, std::max(2, B)));
}

/** @brief Contribution of one pair potential g = g_jl to the targets m0 <= m < m1 (loop order alpha, casscf_mpi_plan.md 2.1)
 *
 * target[m] += g * (sum_k c1(m,k) phi_k)               (j == l, or real orbitals with the (l,j) term folded into c1)
 * target[m] += conj(g) * (sum_k c2(m,k) phi_k)         (complex orbitals, l != j: the (l,j) term, g_lj = conj(g_jl))
 *
 * @param u1, u2: U_dE2(m,j,k,l) and U_dE2(m,l,k,j), stored as u[m + L*k]
 * @param denom: 1-RDM (or multiplier) diagonal per target, 0 means "skip this target"
 */
template <class Target>
void add_pair_contribution(double prec,
                           bool real_orbitals,
                           int j,
                           int l,
                           Orbital &g,
                           std::vector<Orbital> &Phi_loc,
                           const ComplexDouble *u1,
                           const ComplexDouble *u2,
                           const std::vector<ComplexDouble> &denom,
                           int m0,
                           int m1,
                           Target &target) {
    const int L = Phi_loc.size();
    auto coeff_ = [real_orbitals](ComplexDouble c) { return real_orbitals ? ComplexDouble(c.real(), 0.0) : c; };
    std::vector<ComplexDouble> c1(L), c2(L);
    std::vector<mrcpp::CompFunction<3>> inp(Phi_loc.begin(), Phi_loc.end()); // shallow copies for linear_combination
    for (int m = m0; m < m1; m++) {
        if (denom[m] == ComplexDouble(0.0, 0.0)) continue;
        const bool second = (l != j and not real_orbitals); // complex: separate (l,j) term
        for (int k = 0; k < L; k++) {
            ComplexDouble u = u1[m + L * k];
            if (l != j and real_orbitals) u += u2[m + L * k]; // real orbitals: g_lj = g_jl, one shared product
            c1[k] = coeff_(2.0 * u / denom[m]);
            if (second) c2[k] = coeff_(2.0 * u2[m + L * k] / denom[m]);
        }
        Orbital chi, term;
        mrcpp::linear_combination(chi, c1, inp, prec);
        mrcpp::multiply(term, g, chi, prec);
        target[m].add(1.0, term);
        term.free();
        chi.free();
        if (second) {
            mrcpp::linear_combination(chi, c2, inp, prec);
            mrcpp::multiply(term, g, chi, prec, false, false, true); // conjugate is applied on g (first argument)
            target[m].add(1.0, term);
            term.free();
            chi.free();
        }
    }
}

std::vector<ComplexDouble> make_denominators(double threshold, const ComplexMatrix &one_rdm, const ComplexMatrix &lag_coeff) {
    const int L = one_rdm.rows();
    std::vector<ComplexDouble> denom(L, ComplexDouble(0.0, 0.0));
    for (int m = 0; m < L; m++) {
        if (std::abs(one_rdm(m, m)) > threshold)
            denom[m] = one_rdm(m, m);
        else if (std::abs(lag_coeff(m, m)) > threshold)
            denom[m] = lag_coeff(m, m);
        // else: both below threshold, the target is skipped
    }
    return denom;
}

/** @brief Serial path (one worker): all orbitals are local, accumulation directly into new_Phi, one block */
void update_two_body_serial(double prec,
                            double threshold,
                            bool real_orbitals,
                            OrbitalVector &orbitals,
                            const ComplexMatrix &one_rdm,
                            const ComplexMatrix &lag_coeff,
                            const ComplexTensorR4 &U_dE2,
                            Density &rho,
                            OrbitalVector &new_Phi) {
    const int L = orbitals.size();
    const auto denom = make_denominators(threshold, one_rdm, lag_coeff);
    std::vector<Orbital> Phi_loc(orbitals.begin(), orbitals.end());
    std::vector<ComplexDouble> u1(static_cast<size_t>(L) * L), u2(static_cast<size_t>(L) * L);

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
                    u1[m + L * k] = U_dE2(m, j, k, l);
                    u2[m + L * k] = U_dE2(m, l, k, j);
                    max_coef = std::max(max_coef, std::max(std::abs(u1[m + L * k]), std::abs(u2[m + L * k])));
                }
            }

            if (max_coef < 1.0e-16 or ij_r12(prec / max_coef, rho, orbitals[j], orbitals[l], g_jl)) {
                num_skipped++;
                continue;
            }
            add_pair_contribution(prec, real_orbitals, j, l, g_jl, Phi_loc, u1.data(), u2.data(), denom, 0, L, new_Phi);
            g_jl.free();
        }
    }
    MSG_INFO("Skipped " << num_skipped << " orbital pairs");
}

/** @brief Distributed path (several workers + bank): pair-distributed update, see casscf_mpi_plan.md section 2.3
 *
 * MPI: collective over the worker communicator. The orbitals are distributed (orbital i lives on rank i % wrk_size);
 * the pairs (j<=l) are distributed round-robin and their potentials are kept; the targets m are handled in blocks of B:
 * the partial sums of a block go through the bank and are added up by the owner of m in a fixed rank order
 * (reproducible for a given number of ranks).
 */
void update_two_body_distributed(double prec,
                                 double threshold,
                                 bool real_orbitals,
                                 int block_size,
                                 OrbitalVector &orbitals,
                                 const ComplexMatrix &one_rdm,
                                 const ComplexMatrix &lag_coeff,
                                 const ComplexTensorR4 &U_dE2,
                                 Density &rho,
                                 OrbitalVector &new_Phi) {
    const int L = orbitals.size();
    const int me = mrcpp::mpi::wrk_rank;
    const int P = mrcpp::mpi::wrk_size;
    const auto denom = make_denominators(threshold, one_rdm, lag_coeff);

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
    struct MyPair {
        int j, l, ip;
    };
    std::vector<std::vector<MyPair>> all_pairs(P); // round-robin over the pair list, the same on every rank (replicated data)
    int ip = 0;
    for (int j = 0; j < L; j++)
        for (int l = j; l < L; l++)
            if (use_pair(j * L + l)) {
                all_pairs[ip % P].push_back({j, l, ip});
                ip++;
            }
    const std::vector<MyPair> &my_pairs = all_pairs[me];

    // 0b. U_dE2 lives on the master only: each worker receives the slices of its own pairs,
    //     2*L*L numbers per pair: u1(m,k) = U_dE2(m,j,k,l) and u2(m,k) = U_dE2(m,l,k,j), index m + L*k
    const int slice = 2 * L * L;
    IntVector counts(P);
    for (int r = 0; r < P; r++) counts(r) = slice * static_cast<int>(all_pairs[r].size());
    std::vector<ComplexDouble> send;
    if (me == 0) {
        send.reserve(static_cast<size_t>(counts.sum()));
        for (int r = 0; r < P; r++)
            for (auto &pr : all_pairs[r]) {
                for (int k = 0; k < L; k++)
                    for (int m = 0; m < L; m++) send.push_back(U_dE2(m, pr.j, k, pr.l));
                for (int k = 0; k < L; k++)
                    for (int m = 0; m < L; m++) send.push_back(U_dE2(m, pr.l, k, pr.j));
            }
    }
    std::vector<ComplexDouble> my_U;
    mpi_utils::scatterv(send, counts, my_U);
    send = std::vector<ComplexDouble>(); // free the master's copy

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

    // 2. own pair potentials, kept for all blocks
    int num_skipped = 0;
    DoubleVector pair_time = DoubleVector::Zero(ip); // seconds per pair on its owner (diagnostics)
    std::vector<Orbital> g(my_pairs.size());
    std::vector<char> skip(my_pairs.size(), 0);
    for (size_t p = 0; p < my_pairs.size(); p++) {
        const auto &pr = my_pairs[p];
        mrcpp::Timer t_pair;
        g[p] = Phi_loc[pr.j].paramCopy(true);
        if (ij_r12(prec / max_u(pr.j * L + pr.l), rho, Phi_loc[pr.j], Phi_loc[pr.l], g[p])) {
            skip[p] = 1;
            num_skipped++;
        }
        pair_time(pr.ip) += t_pair.elapsed();
    }

    // 3. blocks of B targets (the same boundaries on all workers)
    const int B = (block_size > 0) ? std::min(block_size, L) : default_block_size(L, P);
    println(3, "2e update: block size " << B << " (" << (L + B - 1) / B << " blocks)");
    mrcpp::BankAccount AccBank; // collective
    for (int m0 = 0; m0 < L; m0 += B) {
        const int m1 = std::min(L, m0 + B);
        std::vector<Orbital> Acc(L);
        for (size_t p = 0; p < my_pairs.size(); p++) {
            if (skip[p]) continue;
            const auto &pr = my_pairs[p];
            mrcpp::Timer t_pair;
            const ComplexDouble *u1 = my_U.data() + p * slice;
            add_pair_contribution(prec, real_orbitals, pr.j, pr.l, g[p], Phi_loc, u1, u1 + L * L, denom, m0, m1, Acc);
            pair_time(pr.ip) += t_pair.elapsed();
        }

        // which worker has a partial sum for which target of this block (from the actual results; one collective)
        IntVector has_part = IntVector::Zero(L * P);
        for (int m = m0; m < m1; m++) has_part(m + L * me) = (Acc[m].getNNodes() > 0) ? 1 : 0;
        mrcpp::mpi::allreduce_vector(has_part, mrcpp::mpi::comm_wrk);

        // send the partial sums of foreign targets to the bank, owners add them up in rank order
        for (int m = m0; m < m1; m++) {
            if (has_part(m + L * me) and not mrcpp::mpi::my_func(m)) {
                AccBank.put_func(m + L * me, Acc[m]);
                Acc[m].free();
            }
        }
        for (int m = m0; m < m1; m++) {
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
        AccBank.clear(); // collective: frees the bank memory of this block and acts as the block barrier
    }
    for (size_t p = 0; p < my_pairs.size(); p++) g[p].free();
    for (int i = 0; i < L; i++)
        if (not mrcpp::mpi::my_func(i)) Phi_loc[i].free();
    mrcpp::mpi::allreduce_vector(pair_time, mrcpp::mpi::comm_wrk); // collective
    {
        // the per-rank sums of all ranks follow from the pair times and the (known) round-robin assignment
        DoubleVector rank_time = DoubleVector::Zero(P);
        for (int r = 0; r < P; r++)
            for (int q = r; q < pair_time.size(); q += P) rank_time(r) += pair_time(q);
        println(3,
                "2e pair loop: " << pair_time.size() << " pairs on " << P << " workers, time per worker min/mean/max = " << rank_time.minCoeff() << " / " << rank_time.mean() << " / "
                                 << rank_time.maxCoeff() << " s");
        if (mrcpp::Printer::getPrintLevel() >= 4) println(4, "2e pair times: " << pair_time.transpose());
    }
    IntVector skipped = IntVector::Constant(1, num_skipped);
    mrcpp::mpi::allreduce_vector(skipped, mrcpp::mpi::comm_wrk); // collective
    if (me == 0) MSG_INFO("Skipped " << skipped(0) << " orbital pairs");
}

} // namespace

void update_two_body(double prec,
                     double threshold,
                     bool real_orbitals,
                     OrbitalVector &orbitals,
                     const ComplexMatrix &one_rdm,
                     const ComplexMatrix &lag_coeff,
                     const ComplexTensorR4 &U_dE2,
                     OrbitalVector &new_Phi,
                     int block_size) {
    // Computing total density to use for screening potential computations (collective)
    Density rho;
    density::compute(prec, rho, orbitals, DensityType::Total);

    // wrk_size == 1 also covers np=2 (1 worker + 1 bank): no distribution, no bank accounts
    if (mrcpp::mpi::wrk_size > 1)
        update_two_body_distributed(prec, threshold, real_orbitals, block_size, orbitals, one_rdm, lag_coeff, U_dE2, rho, new_Phi);
    else
        update_two_body_serial(prec, threshold, real_orbitals, orbitals, one_rdm, lag_coeff, U_dE2, rho, new_Phi);
}

} // namespace lagrangian
} // namespace mrchem
