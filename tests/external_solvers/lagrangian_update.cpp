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

#include "catch2/catch_all.hpp"

#include <chrono>
#include <cmath>
#include <random>

#include <MRCPP/MWFunctions>
#include <MRCPP/MWOperators>

#include "qmfunctions/Orbital.h"
#include "qmfunctions/density_utils.h"
#include "qmfunctions/orbital_utils.h"
#include "qmoperators/two_electron/two_electron_utils.h"
#include "scf_solver/lagrangian_utils.h"

using namespace mrchem;

namespace lagrangian_update_tests {

namespace {
/** Serial oracle: today's (v0) two-body update loop, kept here verbatim as a reference for the distributed versions. */
void oracle_two_body(double prec,
                     double threshold,
                     bool real_orbitals,
                     OrbitalVector &orbitals,
                     const ComplexMatrix &one_rdm,
                     const ComplexMatrix &lag_coeff,
                     const ComplexTensorR4 &U_dE2,
                     OrbitalVector &new_Phi) {
    const int L = orbitals.size();
    auto coeff = [real_orbitals](ComplexDouble c) { return real_orbitals ? ComplexDouble(c.real(), 0.0) : c; };
    Density rho;
    density::compute(prec, rho, orbitals, DensityType::Total);
    for (int j = 0; j < L; j++) {
        for (int l = j; l < L; l++) {
            Orbital g_jl = orbitals[j].paramCopy(true);
            double max_coef = 0.0;
            for (int k = 0; k < L; k++)
                for (int m = 0; m < L; m++) {
                    max_coef = std::max(max_coef, std::abs(U_dE2(m, j, k, l)));
                    max_coef = std::max(max_coef, std::abs(U_dE2(m, l, k, j)));
                }
            if (max_coef < 1.0e-16 or ij_r12(prec / max_coef, rho, orbitals[j], orbitals[l], g_jl)) continue;
            for (int k = 0; k < L; k++) {
                Orbital state;
                mrcpp::multiply(state, g_jl, orbitals[k], prec);
                for (int m = 0; m < L; m++) {
                    ComplexDouble denom;
                    if (std::abs(one_rdm(m, m)) > threshold)
                        denom = one_rdm(m, m);
                    else if (std::abs(lag_coeff(m, m)) > threshold)
                        denom = lag_coeff(m, m);
                    else
                        continue;
                    ComplexDouble u = U_dE2(m, j, k, l);
                    if (l != j) u += U_dE2(m, l, k, j);
                    new_Phi[m].add(coeff(2.0 * u / denom), state);
                }
            }
        }
    }
}

OrbitalVector make_orbitals(int L) {
    OrbitalVector Phi;
    for (int i = 0; i < L; i++) {
        const double a = 1.0 + 0.4 * i;
        const double x0 = 0.3 * i - 0.3;
        std::function<double(const mrcpp::Coord<3> &)> f = [a, x0](const mrcpp::Coord<3> &r) -> double { return std::exp(-a * ((r[0] - x0) * (r[0] - x0) + r[1] * r[1] + r[2] * r[2])); };
        Orbital phi(SPIN::Paired);
        mrcpp::project(phi, f, 1.0e-4);
        Phi.push_back(phi);
    }
    return Phi;
}
} // namespace

// MPI: collective; all workers run the same sequence. Compared at the worker level (orbitals are replicated here).
TEST_CASE("Lagrangian two-body update vs serial oracle", "[lagrangian_update]") {
    const double prec = 1.0e-4;
    const int L = 3;
    if (mrcpp::mpi::is_bank) return;

    OrbitalVector Phi = make_orbitals(L);
    std::mt19937 gen(7);
    std::normal_distribution<double> d(0.0, 1.0);
    ComplexTensorR4 U_dE2(L, L, L, L);
    for (int a = 0; a < L; a++)
        for (int b = 0; b < L; b++)
            for (int c = 0; c < L; c++)
                for (int e = 0; e < L; e++) U_dE2(a, b, c, e) = ComplexDouble(d(gen), 0.0);
    ComplexMatrix one_rdm = ComplexMatrix::Identity(L, L) * 1.5;
    ComplexMatrix lag = ComplexMatrix::Identity(L, L) * -0.5;

    auto start = [&]() {
        OrbitalVector N;
        for (int m = 0; m < L; m++) {
            Orbital p;
            mrcpp::deep_copy(p, Phi[m]);
            N.push_back(p);
        }
        return N;
    };

    OrbitalVector ref = start();
    OrbitalVector test = start();
    oracle_two_body(prec, 1.0e-10, true, Phi, one_rdm, lag, U_dE2, ref);
    lagrangian::update_two_body(prec, 1.0e-10, true, Phi, one_rdm, lag, U_dE2, test);
    // the block size only changes the grouping of the work: B = 1 and B = L must agree to rounding
    OrbitalVector test_b1 = start();
    OrbitalVector test_bL = start();
    lagrangian::update_two_body(prec, 1.0e-10, true, Phi, one_rdm, lag, U_dE2, test_b1, 1);
    lagrangian::update_two_body(prec, 1.0e-10, true, Phi, one_rdm, lag, U_dE2, test_bL, L);

    for (int m = 0; m < L; m++) {
        // distributed path: only the owner of m receives the sum
        if (mrcpp::mpi::wrk_size > 1 and not mrcpp::mpi::my_func(m)) continue;
        Orbital diff;
        mrcpp::add(diff, 1.0, ref[m], -1.0, test[m], -1.0);
        INFO("orbital " << m);
        // the oracle keeps today's loop order (products first); the update combines the orbitals first: precision-level difference
        CHECK(diff.norm() < 2.0 * prec);
        CHECK(ref[m].norm() > 1.0e-3);
        Orbital d1, dL;
        mrcpp::add(d1, 1.0, test[m], -1.0, test_b1[m], -1.0);
        mrcpp::add(dL, 1.0, test[m], -1.0, test_bL[m], -1.0);
        CHECK(d1.norm() < 1.0e-9);
        CHECK(dL.norm() < 1.0e-9);
    }
}

// Hidden benchmark (not run by ctest): cost of building a PoissonOperator versus a whole pair potential ij_r12 (S11 profile).
TEST_CASE("Pair potential timing: PoissonOperator construction vs ij_r12", "[ij_r12_bench][.]") {
    if (mrcpp::mpi::is_bank) return;
    using clk = std::chrono::steady_clock;
    auto secs = [](clk::time_point a) { return std::chrono::duration<double>(clk::now() - a).count(); };
    OrbitalVector Phi = make_orbitals(3);
    Density rho;
    density::compute(1.0e-4, rho, Phi, DensityType::Total);
    for (double prec : {1.0e-4, 1.0e-5, 1.0e-6}) {
        for (int rep = 0; rep < 3; rep++) {
            auto t0 = clk::now();
            { mrcpp::PoissonOperator P(*MRA, prec); }
            const double t_op = secs(t0);
            Orbital g = Phi[0].paramCopy(true);
            t0 = clk::now();
            ij_r12(prec, rho, Phi[0], Phi[1], g);
            const double t_ij = secs(t0);
            std::cout << "BENCH prec=" << prec << " rep=" << rep << " PoissonOperator=" << t_op << " s, ij_r12(total incl. operator)=" << t_ij << " s" << std::endl;
            g.free();
        }
    }
}

} // namespace lagrangian_update_tests
