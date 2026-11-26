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
#include <cstdlib>
#include <random>
#include <vector>

#include <MRCPP/Printer>

#include "external_solvers/ChemTensorSolver.h"
#include "external_solvers/fiedler_utils.h"
#include "qmfunctions/Orbital.h"
#include "qmoperators/two_electron/FockBuilder.h"

using namespace mrchem;

namespace mrchem {
/** Friend of ChemTensorSolver: gives the tests access to the private steps of set_integrals, without orbitals. */
class ChemTensorSolverTester {
public:
    static void set_integrals(ChemTensorSolver &S, const ComplexMatrix &h, const ComplexTensorR4 &g, bool real) {
        S.real = real;
        S.one_body_integrals = std::make_shared<ComplexMatrix>(h);
        S.two_body_integrals = std::make_shared<ComplexTensorR4>(g);
        S.update_site_order();
        S.set_dense_tensors();
    }
    static void force_perm(ChemTensorSolver &S, const IntVector &p) { S.perm = p; }
    static const IntVector &perm(const ChemTensorSolver &S) { return S.perm; }
};
} // namespace mrchem

namespace chemtensor_solver_tests {

namespace {
using Tester = ChemTensorSolverTester;

struct Ham {
    ComplexMatrix h;
    ComplexTensorR4 g;
};

IntVector make_perm(const std::vector<int> &v) {
    IntVector p(v.size());
    for (size_t i = 0; i < v.size(); i++) p(i) = v[i];
    return p;
}
bool same(const IntVector &a, const IntVector &b) {
    return a.size() == b.size() and (a - b).cwiseAbs().sum() == 0;
}

// g(i,j,k,l) = <ij|kl> = sum_P B_P(i,k) B_P(j,l) with Hermitian B_P that decay along a hidden chain (lab[s] = orbital at site s)
Ham make_ham(int N, const IntVector &lab, bool cplx, unsigned seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<double> d(0.0, 1.0);
    const IntVector pos = fiedler::inverse(lab);
    std::vector<ComplexMatrix> B(2 * N, ComplexMatrix::Zero(N, N));
    for (auto &b : B) {
        for (int i = 0; i < N; i++)
            for (int k = i; k < N; k++) {
                const double w = std::exp(-1.5 * std::abs(pos(i) - pos(k)));
                ComplexDouble x(d(gen) * w, (cplx and i != k) ? d(gen) * w : 0.0);
                if (i == k) x = ComplexDouble(std::abs(d(gen)) + 0.5, 0.0);
                b(i, k) = x;
                b(k, i) = std::conj(x);
            }
        b *= 0.3;
    }
    Ham H;
    H.g = ComplexTensorR4(N, N, N, N);
    H.g.setZero();
    for (auto &b : B)
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                for (int k = 0; k < N; k++)
                    for (int l = 0; l < N; l++) H.g(i, j, k, l) += b(i, k) * b(j, l);
    H.h = ComplexMatrix::Zero(N, N);
    for (int i = 0; i < N; i++)
        for (int k = i; k < N; k++) {
            const double w = std::exp(-1.0 * std::abs(pos(i) - pos(k)));
            ComplexDouble x(d(gen) * w, (cplx and i != k) ? 0.5 * d(gen) * w : 0.0);
            if (i == k) x = ComplexDouble(-2.0 + 0.3 * d(gen), 0.0);
            H.h(i, k) = x;
            H.h(k, i) = std::conj(x);
        }
    return H;
}

json make_dict(const std::string &order, int vdim, bool every = false, double thrs = 0.1, bool warm = false) {
    return {{"prec", 1e-6},
            {"max_vdim", vdim},
            {"num_sweeps", 12},
            {"maxiter_lanczos", 40},
            {"tol_split", 1e-14},
            {"start_from_previous_mps", warm},
            {"site_order", order},
            {"reorder_every_iter", every},
            {"reorder_thrs", thrs},
            {"mps_seed", 42}};
}

/** Owns the (empty) objects the solver is constructed from. */
struct Fixture {
    OrbitalVector Phi;
    FockBuilder F;
    Nuclei nucs;
};

struct Result {
    double E;
    ComplexMatrix g1;
    ComplexTensorR4 g2;
    IntVector perm;
};

Result run(const Ham &H, const std::string &order, int vdim, bool real, const IntVector *force, int Ne) {
    Fixture fx;
    ChemTensorSolver S(fx.Phi, fx.F, fx.nucs, Ne, 0, make_dict(order, vdim));
    if (force) Tester::force_perm(S, *force);
    Tester::set_integrals(S, H.h, H.g, real);
    S.optimize();
    return {S.get_energy(), *S.get_one_rdm(), *S.get_two_rdm(), Tester::perm(S)};
}

// E = sum h(i,j) gamma(i,j) + sum g(i,j,k,l) Gamma(i,j,k,l): no transpose, no conjugation
ComplexDouble energy_from_rdms(const Ham &H, const Result &R) {
    const int N = H.h.rows();
    ComplexDouble e = 0.0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) e += H.h(i, j) * R.g1(i, j);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                for (int l = 0; l < N; l++) e += H.g(i, j, k, l) * R.g2(i, j, k, l);
    return e;
}
double max_diff(const ComplexMatrix &a, const ComplexMatrix &b) {
    return (a - b).cwiseAbs().maxCoeff();
}
double max_diff(const ComplexTensorR4 &a, const ComplexTensorR4 &b) {
    Eigen::Tensor<double, 0> m = (a - b).abs().maximum();
    return m();
}
} // namespace

TEST_CASE("ChemTensorSolver: result does not depend on the site order", "[chemtensor_solver]") {
    const int N = 6;
    const IntVector lab = make_perm({3, 0, 5, 1, 4, 2});
    const IntVector cyc = make_perm({1, 2, 3, 4, 5, 0}); // not an involution
    const double thrs_E = 1.0e-9, thrs_rdm = 1.0e-6;

    for (int Ne : {6, 4}) {
        for (bool cplx : {false, true}) {
            const Ham H = make_ham(N, lab, cplx, 1234);
            const bool real = !cplx;
            const Result A = run(H, "none", 400, real, nullptr, Ne);
            const Result B = run(H, "fiedler", 400, real, &cyc, Ne);
            const Result C = run(H, "fiedler", 400, real, nullptr, Ne);

            SECTION(std::string("Ne=") + std::to_string(Ne) + (cplx ? " complex" : " real")) {
                REQUIRE(same(A.perm, IntVector::LinSpaced(N, 0, N - 1)));
                REQUIRE(same(B.perm, cyc)); // 6-cycle is kept: the hysteresis only accepts a lower cost
                for (const Result *R : {&B, &C}) {
                    REQUIRE(std::abs(R->E - A.E) < thrs_E);
                    REQUIRE(max_diff(R->g1, A.g1) < thrs_rdm);
                    REQUIRE(max_diff(R->g2, A.g2) < thrs_rdm);
                }
                // the Fiedler order finds the hidden chain (or its reverse)
                REQUIRE((same(C.perm, lab) or same(C.perm, lab.reverse().eval())));
                // RDMs reproduce the MPO energy and the electron number
                for (const Result *R : {&A, &B, &C}) {
                    const ComplexDouble e = energy_from_rdms(H, *R);
                    REQUIRE(std::abs(e.real() - R->E) < thrs_E);
                    REQUIRE(std::abs(e.imag()) < thrs_E);
                    REQUIRE(std::abs(R->g1.trace() - ComplexDouble(Ne, 0.0)) < thrs_E);
                }
                if (!cplx) {
                    // real integrals through the complex code path
                    const Result D = run(H, "fiedler", 400, false, &cyc, Ne);
                    REQUIRE(std::abs(D.E - A.E) < thrs_E);
                    REQUIRE(max_diff(D.g1, A.g1) < thrs_rdm);
                    REQUIRE(max_diff(D.g2, A.g2) < thrs_rdm);
                }
            }
        }
    }
}

// Hidden benchmark (not run by ctest): DMRG time of one optimize() on a synthetic Hamiltonian, to measure thread scaling.
// Usage: OMP_NUM_THREADS=t BENCH_N=40 BENCH_VDIM=300 mrchem-tests "[chemtensor_bench]"
TEST_CASE("ChemTensorSolver: DMRG timing benchmark", "[chemtensor_bench][.]") {
    const char *sn = std::getenv("BENCH_N");
    const char *sv = std::getenv("BENCH_VDIM");
    const int N = sn ? std::atoi(sn) : 30;
    const int vdim = sv ? std::atoi(sv) : 300;
    IntVector lab = IntVector::LinSpaced(N, 0, N - 1);
    const Ham H = make_ham(N, lab, false, 99);
    Fixture fx;
    ChemTensorSolver S(fx.Phi, fx.F, fx.nucs, N, 0, make_dict("none", vdim));
    Tester::set_integrals(S, H.h, H.g, true);
    const auto t0 = std::chrono::steady_clock::now();
    S.optimize();
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "BENCH N=" << N << " vdim=" << vdim << " threads=" << (std::getenv("OMP_NUM_THREADS") ? std::getenv("OMP_NUM_THREADS") : "?") << " optimize_seconds=" << sec << " E=" << S.get_energy()
              << std::endl;
    REQUIRE(std::isfinite(S.get_energy()));
}

TEST_CASE("ChemTensorSolver: print paths at level 3 and width 75", "[chemtensor_solver]") {
    const IntVector lab = make_perm({3, 0, 5, 1, 4, 2});
    const Ham H = make_ham(6, lab, false, 1234);
    const int old_level = mrcpp::Printer::getPrintLevel();
    const int old_width = mrcpp::Printer::getWidth();
    mrcpp::Printer::setPrintLevel(3);
    mrcpp::Printer::setWidth(75); // default print_width of the input
    // regression for F14: labels longer than 24 characters threw std::length_error
    REQUIRE_NOTHROW(run(H, "fiedler", 64, true, nullptr, 6));
    const IntVector cyc = make_perm({1, 2, 3, 4, 5, 0});
    REQUIRE_NOTHROW(run(H, "fiedler", 64, true, &cyc, 6)); // forced order is kept: other print branch
    mrcpp::Printer::setPrintLevel(old_level);
    mrcpp::Printer::setWidth(old_width);
}

TEST_CASE("ChemTensorSolver: update_site_order", "[chemtensor_solver]") {
    const int N = 6;
    const IntVector lab = make_perm({3, 0, 5, 1, 4, 2});
    const IntVector lab2 = make_perm({0, 2, 4, 1, 5, 3});
    const Ham H = make_ham(N, lab, false, 1234);
    const Ham H2 = make_ham(N, lab2, false, 99);
    auto is_chain = [](const IntVector &p, const IntVector &l) { return same(p, l) or same(p, l.reverse().eval()); };

    SECTION("none: identity, no change, zero cost") {
        Fixture fx;
        ChemTensorSolver S(fx.Phi, fx.F, fx.nucs, 6, 0, make_dict("none", 64));
        Tester::set_integrals(S, H.h, H.g, true);
        REQUIRE(same(S.get_site_order(), IntVector::LinSpaced(N, 0, N - 1)));
        REQUIRE_FALSE(S.site_order_changed());
        REQUIRE(S.get_site_order_cost() == 0.0);
    }

    SECTION("fiedler: first call, then kept") {
        Fixture fx;
        ChemTensorSolver S(fx.Phi, fx.F, fx.nucs, 6, 0, make_dict("fiedler", 64));
        Tester::set_integrals(S, H.h, H.g, true);
        REQUIRE(S.site_order_changed());
        REQUIRE(is_chain(S.get_site_order(), lab));
        const DoubleMatrix K = fiedler::exchange_matrix(H.g);
        REQUIRE(S.get_site_order_cost() == Catch::Approx(fiedler::cost(K, S.get_site_order())));
        const IntVector p1 = S.get_site_order();
        // second call with other integrals, but reorder_every_iter = false: never reordered
        Tester::set_integrals(S, H2.h, H2.g, true);
        REQUIRE_FALSE(S.site_order_changed());
        REQUIRE(same(S.get_site_order(), p1));
    }

    SECTION("reorder_every_iter: new chain accepted, then stable") {
        Fixture fx;
        ChemTensorSolver S(fx.Phi, fx.F, fx.nucs, 6, 0, make_dict("fiedler", 64, true, 0.1));
        Tester::set_integrals(S, H.h, H.g, true);
        Tester::set_integrals(S, H2.h, H2.g, true);
        REQUIRE(S.site_order_changed());
        REQUIRE(is_chain(S.get_site_order(), lab2));
        const IntVector p2 = S.get_site_order();
        Tester::set_integrals(S, H2.h, H2.g, true); // same integrals: no flip to the reverse
        REQUIRE_FALSE(S.site_order_changed());
        REQUIRE(same(S.get_site_order(), p2));
    }

    SECTION("hysteresis: a small gain is rejected") {
        const DoubleMatrix K2 = fiedler::exchange_matrix(H2.g);
        Fixture fx;
        ChemTensorSolver S(fx.Phi, fx.F, fx.nucs, 6, 0, make_dict("fiedler", 64, true, 0.99));
        Tester::set_integrals(S, H.h, H.g, true);
        const IntVector p1 = S.get_site_order();
        // precondition: the gain must be below 99 %, otherwise nothing is tested
        REQUIRE(fiedler::cost(K2, lab2) >= 0.01 * fiedler::cost(K2, p1));
        Tester::set_integrals(S, H2.h, H2.g, true);
        REQUIRE_FALSE(S.site_order_changed());
        REQUIRE(same(S.get_site_order(), p1));
    }

    SECTION("size change is treated as a first call") {
        Fixture fx;
        ChemTensorSolver S(fx.Phi, fx.F, fx.nucs, 6, 0, make_dict("fiedler", 64));
        Tester::set_integrals(S, H.h, H.g, true);
        const IntVector lab5 = make_perm({2, 0, 4, 1, 3});
        const Ham H5 = make_ham(5, lab5, false, 7);
        Tester::set_integrals(S, H5.h, H5.g, true);
        REQUIRE(S.get_site_order().size() == 5);
        REQUIRE(S.site_order_changed());
        REQUIRE(is_chain(S.get_site_order(), lab5));
    }
}

TEST_CASE("ChemTensorSolver: warm start is not used after a reorder", "[chemtensor_solver]") {
    const int N = 6;
    const IntVector lab = make_perm({3, 0, 5, 1, 4, 2});
    const IntVector lab2 = make_perm({0, 2, 4, 1, 5, 3});
    const Ham H = make_ham(N, lab, false, 1234);
    const Ham H2 = make_ham(N, lab2, false, 99);
    const int vdim = 4; // truncated: the start matters

    Fixture fx1;
    ChemTensorSolver S1(fx1.Phi, fx1.F, fx1.nucs, 6, 0, make_dict("fiedler", vdim, true, 0.1, true));
    Tester::set_integrals(S1, H.h, H.g, true);
    S1.optimize();
    Tester::set_integrals(S1, H2.h, H2.g, true);
    REQUIRE(S1.site_order_changed());
    S1.optimize();
    const double E1 = S1.get_energy();

    // fresh solver with the final order of S1: same seeded random start
    Fixture fx2;
    ChemTensorSolver S2(fx2.Phi, fx2.F, fx2.nucs, 6, 0, make_dict("fiedler", vdim, false, 0.1, false));
    Tester::force_perm(S2, Tester::perm(S1));
    Tester::set_integrals(S2, H2.h, H2.g, true);
    S2.optimize();
    REQUIRE(std::abs(E1 - S2.get_energy()) < 1.0e-10);
}

} // namespace chemtensor_solver_tests
