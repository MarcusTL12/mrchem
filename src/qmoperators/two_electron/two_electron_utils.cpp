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

#include "two_electron_utils.h"
#include "qmfunctions/density_utils.h"
#include <MRCPP/Parallel>
#include "utils/mpi_bcast.h"
#include "MRCPP/MWOperators"
#include "MRCPP/Printer"
#include "MRCPP/Timer"

using mrcpp::Printer;
using mrcpp::Timer;
extern mrcpp::MultiResolutionAnalysis<3> *mrchem::MRA;

namespace mrchem {

Eigen::Tensor<std::complex<double>, 4> calc_2elintegrals(double prec, OrbitalVector &Phi) {
    mrcpp::print::header(3, "Computing two electron integrals");
    Timer t_tot, t_snd(false), t_rho(false), t_ovr(false), t_calc(false), t_add(false), t_get(false);

    mrcpp::BankAccount PhiBank; // to put the orbitals

    int N = Phi.size();
    double precf = prec / (10*std::sqrt(1 * Phi.size()));
    double prec_m2 = precf / 100; // second multiplication
    mrcpp::PoissonOperator P(*MRA, precf); // one operator for all pairs (same precision)

    // all orbitals on every worker (own ones are local, the others come from the bank); no bank: Phi itself
    std::vector<Orbital> Phi_loc(N);
    if (mrcpp::mpi::bank_size > 0) {
        for (int i = 0; i < N; i++)
            if (mrcpp::mpi::my_func(i)) PhiBank.put_func(i, Phi[i]);
    }
    t_get.resume();
    for (int i = 0; i < N; i++) {
        if (mrcpp::mpi::bank_size <= 0 or mrcpp::mpi::my_func(i))
            Phi_loc[i] = Phi[i];
        else
            PhiBank.get_func(i, Phi_loc[i], 1);
    }
    t_get.stop();

    // compute-and-contract: right after V_ij is computed, kl_Vij(k,l) = <k| l V_ij> is made and V_ij is freed.
    // One record per pair (i>=j): [i*N+j, kl_Vij(k + N*l) for all k,l], gathered on the master at the end.
    std::vector<ComplexDouble> records;
    auto contract = [&](int i, int j, Orbital &Vij) {
        const size_t start = records.size();
        records.resize(start + 1 + static_cast<size_t>(N) * N);
        records[start] = ComplexDouble(static_cast<double>(i * N + j), 0.0);
        for (int l = 0; l < N; l++) {
            Orbital lVij = Phi_loc[l].paramCopy(true);
            t_ovr.resume();
            mrcpp::multiply(lVij, Phi_loc[l], Vij, prec_m2, true, true);
            for (int k = 0; k < N; k++) records[start + 1 + k + static_cast<size_t>(N) * l] = mrcpp::dot(Phi_loc[k], lVij);
            t_ovr.stop();
            lVij.free();
        }
    };

    // first we compute density to be used so we know were the orbitals are. This
    // is to avoid computing the potential at positions were there are no orbitals.
    t_rho.resume();
    Density rho;
    density::compute(prec, rho, Phi, DensityType::Total);
    t_rho.stop();

    // Initialize and compute own diagonal elements
    Timer t_diag;
    int i = -1;
    for (auto &phi_i : Phi) {
        i++;
        if (!mrcpp::mpi::my_func(i)) continue;
        Orbital Vij = phi_i.paramCopy(true);
        t_calc.resume();
        bool skipped = ij_r12(precf, rho, phi_i, phi_i, Vij, P);
        t_calc.stop();
        if (not skipped) contract(i, i, Vij);
        Vij.free();
    }
    t_diag.stop();

    Timer t_offd;
    // We divide all the contributions into a fixed number of tasks.
    // all "j" orbitals are fetched and stored, and used together with one "i" orbital

    // make a set of tasks
    // We use symmetry: each pair (i,j) must be used once only. Only j<i
    // Divide into square blocks, with the diagonal blocks taken at the end (because they are faster to compute)
    int block_size; // NB: block_size*block_size intermediate exchange results are stored temporarily
    block_size = std::min(16, std::max(2, static_cast<int>(std::sqrt(N * N / (14 * mrcpp::mpi::wrk_size)))));

    int iblocks = (N + block_size - 1) / block_size;
    int ntasksmax = ((iblocks - 1) * iblocks) / 2 + iblocks * (block_size * (block_size - 1) / 2);
    std::vector<std::vector<int>> itasks(ntasksmax); // the i values (orbitals) of each block
    std::vector<std::vector<int>> jtasks(ntasksmax); // the j values (orbitals) of each block

    int task = 0;
    // make a path for tasks that follow diagonals, in order to maximize the spread of orbitals treated
    for (int ib = 0; ib < (iblocks + 1) / 2; ib++) {
        if (task >= (iblocks * (iblocks - 1) / 2)) break;
        for (int ij = 0; ij < iblocks; ij++) {
            int j0 = ij;
            int i0 = ib + ij + 1;
            if (i0 * block_size >= N) {
                // continue in "symmetrical" part
                j0 = ij + ib + 1 - iblocks;
                i0 = ij;
            }
            for (int jj = 0; jj < block_size; jj++) {
                int jjj = j0 * block_size + jj;
                if ((i0 + j0) % 2 != 0) jjj = j0 * block_size + (block_size - 1 - jj); // reversed order
                if (jjj >= N) continue;
                if ((i0 + j0) % 2 == 0)
                    jtasks[task].push_back(jjj);
                else
                    itasks[task].push_back(jjj);
            }
            for (int ii = 0; ii < block_size; ii++) {
                int iii = i0 * block_size + ii;
                if ((i0 + j0) % 2 == 0) iii = i0 * block_size + (block_size - 1 - ii); // reversed order
                if (iii >= N) continue;

                if ((i0 + j0) % 2 == 0)
                    itasks[task].push_back(iii);
                else
                    jtasks[task].push_back(iii);
            }
            task++;
            if (task >= (iblocks * (iblocks - 1) / 2)) break;
        }
    }

    // add diagonal blocks:
    // we make those tasks smaller (1x1 blocks), in order to minimize the time waiting for the last task.
    // NB: only include j<i within these blocks
    for (int i = 0; i < N; i += block_size) {
        int j = i;
        for (int jj = j; jj < j + block_size and jj < N; jj++) {
            for (int ii = i; ii < i + block_size and ii < N; ii++) {
                if (ii <= jj) continue; // only jj<ii is computed
                itasks[task].push_back(ii);
                jtasks[task].push_back(jj);
                task++;
            }
        }
    }
    assert(task <= ntasksmax);
    int ntasks = task;

    // compute the Int(phi_i^dag*phi_j/|r-r'|) potentials and contract them at once
    mrcpp::TaskManager tasksMaster(ntasks);
    while (true) {
        task = tasksMaster.next_task();
        if (task < 0) break;
        for (size_t j = 0; j < jtasks[task].size(); j++) {
            int jorb = jtasks[task][j];
            for (size_t i = 0; i < itasks[task].size(); i++) {
                int iorb = itasks[task][i];
                Orbital Vij = Phi_loc[iorb].paramCopy(true);
                t_calc.resume();
                // the potential is stored as V_ij = P[phi_i^dagger * phi_j] with i >= j, but iorb < jorb can happen
                // in the off-diagonal blocks: swap the orbitals in that case (V_ji = conj(V_ij) for complex orbitals)
                bool skipped;
                if (iorb >= jorb)
                    skipped = ij_r12(precf, rho, Phi_loc[iorb], Phi_loc[jorb], Vij, P);
                else
                    skipped = ij_r12(precf, rho, Phi_loc[jorb], Phi_loc[iorb], Vij, P);
                t_calc.stop();
                if (not skipped) contract(std::max(iorb, jorb), std::min(iorb, jorb), Vij);
                Vij.free();
            }
        }
    }

    t_offd.stop();
    for (int i = 0; i < N; i++)
        if (not (mrcpp::mpi::bank_size <= 0 or mrcpp::mpi::my_func(i))) Phi_loc[i].free();

    // only the master holds the N^4 tensor
    t_get.resume();
    std::vector<ComplexDouble> all_records;
    mpi_utils::gatherv(records, all_records); // collective
    records = std::vector<ComplexDouble>();
    t_get.stop();
    Eigen::Tensor<std::complex<double>, 4> ikVjl;
    if (mrcpp::mpi::wrk_rank == 0) {
        ikVjl.resize(N, N, N, N);
        ikVjl.setZero();
        const size_t rec_size = 1 + static_cast<size_t>(N) * N;
        for (size_t r = 0; r + rec_size <= all_records.size(); r += rec_size) {
            const int ij = static_cast<int>(std::lround(all_records[r].real()));
            const int i = ij / N, j = ij % N;
            const ComplexDouble *kl_Vij = all_records.data() + r + 1; // kl_Vij(k,l) = kl_Vij[k + N*l]
            for (int k = 0; k < N; k++) {
                for (int l = 0; l < N; l++) {
                    ikVjl(i, k, j, l) = kl_Vij[k + N * l];
                    // V_ji = conj(V_ij), so <k|l V_ji> = conj(<l|k V_ij>) (same as kl_Vij(k,l) for real orbitals)
                    if (i != j) ikVjl(j, k, i, l) = std::conj(kl_Vij[l + N * k]);
                }
            }
        }
    }
    t_tot.stop();

    mrcpp::print::time(3, "Time making density", t_rho);
    mrcpp::print::time(3, "Time receiving orbitals / gathering", t_get);
    mrcpp::print::time(3, "Time sending potentials", t_snd);
    mrcpp::print::time(3, "Time computing integrals", t_calc);
    mrcpp::print::separator(3, '-');
    mrcpp::print::time(3, "Time diagonal terms", t_diag);
    mrcpp::print::time(3, "Time off-diagonal terms", t_offd);
    mrcpp::print::time(3, "Time making products and overlaps", t_ovr);
    mrcpp::print::time(3, "Time making two el integrals", t_tot);
    mrcpp::print::separator(3, '-');

    return ikVjl;
}

bool ij_r12(double prec, Orbital rho, Orbital phi_i, Orbital phi_j, Orbital &V_ij) {
    mrcpp::PoissonOperator P(*MRA, prec);
    return ij_r12(prec, rho, phi_i, phi_j, V_ij, P);
}

bool ij_r12(double prec, Orbital rho, Orbital phi_i, Orbital phi_j, Orbital &V_ij, mrcpp::PoissonOperator &P) {
    Timer timer_tot;

    // set precisions
    double prec_m1 = prec / 10;  // first multiplication
    double prec_p = prec * 10;   // Poisson application

    // compute rho_ij = phi_i^dagger * phi_j
    // if the product is smaller than the target precision,
    // the result is expected to be negligible
    Timer timer_ij;
    Orbital rho_ij = phi_i.paramCopy(true);
    mrcpp::multiply(rho_ij, phi_i, phi_j, prec_m1, true, true, true);
    timer_ij.stop();
    if (rho_ij.norm() < prec) return true;

    // For now we assume all phi are complex or all are real.
    bool RealOrbitals = phi_i.isreal();

    // prepare vector used to steer precision of Poisson application
    mrcpp::FunctionTreeVector<3, double> phi_opt_vec_real;
    mrcpp::FunctionTreeVector<3, ComplexDouble> phi_opt_vec_cplx;
    if (RealOrbitals) {
        if (rho.isreal() and rho.getNNodes() > 0) phi_opt_vec_real.emplace_back(1.0, rho.CompD[0]);
        if (phi_j.isreal() and  phi_j.getNNodes() > 0) phi_opt_vec_real.emplace_back(1.0, phi_j.CompD[0]);
        if (phi_i.isreal() and  &phi_i != &phi_j and phi_i.getNNodes() > 0) phi_opt_vec_real.emplace_back(1.0, phi_i.CompD[0]);
    } else {
        if (rho.iscomplex()) phi_opt_vec_cplx.emplace_back(1.0, rho.CompC[0]);
        if (phi_j.iscomplex()) phi_opt_vec_cplx.emplace_back(1.0, phi_j.CompC[0]);
        if (phi_i.iscomplex() and &phi_i != &phi_j) phi_opt_vec_cplx.emplace_back(1.0, phi_i.CompC[0]);
    }
    // compute V_ij = P[rho_ij]
    Timer timer_p;
    if (RealOrbitals) {
        mrcpp::apply(prec_p, *V_ij.CompD[0], P, *rho_ij.CompD[0], phi_opt_vec_real, -1, true);
    } else {
        mrcpp::apply(prec_p, *V_ij.CompC[0], P, *rho_ij.CompC[0], phi_opt_vec_cplx, -1, true);
    }
    rho_ij.free();
    timer_p.stop();

    return false;
}
}
