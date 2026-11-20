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

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "external_solvers/fiedler_utils.h"

using namespace mrchem;

namespace fiedler_tests {

namespace {
IntVector make_perm(const std::vector<int> &v) {
    IntVector p(v.size());
    for (size_t i = 0; i < v.size(); i++) p(i) = v[i];
    return p;
}
IntVector identity(int N) {
    IntVector p(N);
    for (int i = 0; i < N; i++) p(i) = i;
    return p;
}
// Chain coupling between neighbouring sites of the "true" chain, relabelled by lab (lab[s] = orbital at true site s)
DoubleMatrix chain_K(int N, const IntVector &lab) {
    DoubleMatrix K = DoubleMatrix::Zero(N, N);
    for (int s = 0; s + 1 < lab.size(); s++) {
        K(lab(s), lab(s + 1)) = 1.0;
        K(lab(s + 1), lab(s)) = 1.0;
    }
    return K;
}
bool same(const IntVector &a, const IntVector &b) {
    return a.size() == b.size() and (a - b).cwiseAbs().sum() == 0;
}
IntVector reversed(const IntVector &a) {
    return a.reverse();
}
} // namespace

TEST_CASE("Fiedler ordering", "[fiedler]") {
    const double edge_thrs = 1.0e-6;

    SECTION("Shuffled chain is recovered") {
        const IntVector lab = make_perm({3, 0, 5, 1, 4, 2});
        const DoubleMatrix K = chain_K(6, lab);
        const IntVector perm = fiedler::order(K, edge_thrs, identity(6));
        REQUIRE((same(perm, lab) or same(perm, reversed(lab))));
        REQUIRE(fiedler::cost(K, perm) == Catch::Approx(fiedler::cost(K, lab)));
        REQUIRE(fiedler::cost(K, perm) < fiedler::cost(K, identity(6)));
    }

    SECTION("Identity chain with identity reference") {
        const DoubleMatrix K = chain_K(5, identity(5));
        REQUIRE(same(fiedler::order(K, edge_thrs, identity(5)), identity(5)));
    }

    SECTION("Reversed reference gives the reversed chain") {
        const DoubleMatrix K = chain_K(5, identity(5));
        const IntVector ref = reversed(identity(5));
        REQUIRE(same(fiedler::order(K, edge_thrs, ref), ref));
    }

    SECTION("Disconnected blocks stay contiguous, in index order") {
        DoubleMatrix K = DoubleMatrix::Zero(6, 6);
        const IntVector a = make_perm({4, 0, 2}), b = make_perm({1, 5, 3});
        K += chain_K(6, a);
        K += chain_K(6, b);
        K(0, 1) = K(1, 0) = 0.5 * edge_thrs; // weak cross coupling, below threshold
        const IntVector perm = fiedler::order(K, edge_thrs, identity(6));
        // block of orbital 0 = {0,2,4} first, then {1,3,5}
        for (int s = 0; s < 3; s++) REQUIRE(perm(s) % 2 == 0);
        for (int s = 3; s < 6; s++) REQUIRE(perm(s) % 2 == 1);
        REQUIRE(fiedler::cost(K, perm) < fiedler::cost(K, identity(6)));
    }

    SECTION("N = 1 and N = 2 return the reference") {
        REQUIRE(same(fiedler::order(DoubleMatrix::Zero(1, 1), edge_thrs, identity(1)), identity(1)));
        DoubleMatrix K2 = DoubleMatrix::Ones(2, 2);
        const IntVector ref = make_perm({1, 0});
        REQUIRE(same(fiedler::order(K2, edge_thrs, ref), ref));
    }

    SECTION("Meaning of permute") {
        const IntVector perm = make_perm({1, 2, 0});
        ComplexMatrix A(3, 3);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) A(i, j) = ComplexDouble(i, 10 * j);
        const ComplexMatrix B = fiedler::permute(A, perm);
        for (int s = 0; s < 3; s++)
            for (int t = 0; t < 3; t++) REQUIRE(B(s, t) == A(perm(s), perm(t)));
        // inverse
        const IntVector inv = fiedler::inverse(perm);
        for (int s = 0; s < 3; s++) REQUIRE(inv(perm(s)) == s);
    }

    SECTION("Round trip") {
        const IntVector perm = make_perm({2, 4, 0, 3, 1});
        std::mt19937 gen(7);
        std::normal_distribution<double> d(0.0, 1.0);
        ComplexMatrix A(5, 5);
        ComplexTensorR4 T(5, 5, 5, 5);
        for (int i = 0; i < 5; i++)
            for (int j = 0; j < 5; j++) A(i, j) = ComplexDouble(d(gen), d(gen));
        for (int i = 0; i < 5; i++)
            for (int j = 0; j < 5; j++)
                for (int k = 0; k < 5; k++)
                    for (int l = 0; l < 5; l++) T(i, j, k, l) = ComplexDouble(d(gen), d(gen));
        REQUIRE((fiedler::unpermute(fiedler::permute(A, perm), perm) - A).norm() < 1.0e-14);
        ComplexTensorR4 Tr = fiedler::unpermute(fiedler::permute(T, perm), perm);
        double diff = 0.0;
        for (int i = 0; i < 5; i++)
            for (int j = 0; j < 5; j++)
                for (int k = 0; k < 5; k++)
                    for (int l = 0; l < 5; l++) diff += std::abs(Tr(i, j, k, l) - T(i, j, k, l));
        REQUIRE(diff < 1.0e-12);
    }

    SECTION("Energy contraction is invariant") {
        const int N = 4;
        const IntVector perm = make_perm({1, 2, 3, 0});
        std::mt19937 gen(3);
        std::normal_distribution<double> d(0.0, 1.0);
        ComplexMatrix h(N, N), gam(N, N);
        ComplexTensorR4 g(N, N, N, N), Gam(N, N, N, N);
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                h(i, j) = ComplexDouble(d(gen), d(gen));
                gam(i, j) = ComplexDouble(d(gen), d(gen));
            }
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                for (int k = 0; k < N; k++)
                    for (int l = 0; l < N; l++) {
                        g(i, j, k, l) = ComplexDouble(d(gen), d(gen));
                        Gam(i, j, k, l) = ComplexDouble(d(gen), d(gen));
                    }
        auto energy = [&](const ComplexMatrix &h_, const ComplexMatrix &gam_, const ComplexTensorR4 &g_, const ComplexTensorR4 &Gam_) {
            ComplexDouble e = 0.0;
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++) e += h_(i, j) * gam_(i, j);
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    for (int k = 0; k < N; k++)
                        for (int l = 0; l < N; l++) e += g_(i, j, k, l) * Gam_(i, j, k, l);
            return e;
        };
        const ComplexDouble e0 = energy(h, gam, g, Gam);
        const ComplexDouble e1 = energy(fiedler::permute(h, perm), fiedler::permute(gam, perm), fiedler::permute(g, perm), fiedler::permute(Gam, perm));
        REQUIRE(std::abs(e0 - e1) < 1.0e-12);
    }

    SECTION("Orientation tie is deterministic") {
        // chain 0-1-2-3 against ref = (1,3,0,2): both directions have the same score (9),
        // so the lexicographic rule picks the reversed chain
        const DoubleMatrix K = chain_K(4, identity(4));
        const IntVector ref = make_perm({1, 3, 0, 2});
        const IntVector p1 = fiedler::order(K, edge_thrs, ref);
        const IntVector p2 = fiedler::order(K, edge_thrs, ref);
        REQUIRE(same(p1, p2));
        REQUIRE(same(p1, make_perm({3, 2, 1, 0})));
        // with a reference that is not a tie, the direction with the larger score wins
        const IntVector ref2 = make_perm({2, 0, 4, 1, 3});
        const DoubleMatrix K5 = chain_K(5, identity(5));
        const IntVector p5 = fiedler::order(K5, edge_thrs, ref2);
        const IntVector pos = fiedler::inverse(ref2);
        double sp = 0.0, sr = 0.0;
        for (int s = 0; s < 5; s++) {
            sp += s * pos(p5(s));
            sr += s * pos(p5(4 - s));
        }
        REQUIRE(sp >= sr);
    }

    SECTION("Absolute cost") {
        // chain 0-1-..-(N-1) with unit couplings in the identity order: sum_ij K_ij (s_i-s_j)^2 = 2 (N-1)
        for (int N : {3, 5, 8}) REQUIRE(fiedler::cost(chain_K(N, identity(N)), identity(N)) == Catch::Approx(2.0 * (N - 1)));
        // 3-cycle: cost must use the inverse. Orbitals 0-1 coupled; perm = {1,2,0} puts orbital 1 at site 0, 2 at site 1, 0 at site 2
        DoubleMatrix K = DoubleMatrix::Zero(3, 3);
        K(0, 1) = K(1, 0) = 1.0;
        REQUIRE(fiedler::cost(K, make_perm({1, 2, 0})) == Catch::Approx(2.0 * 4.0)); // sites 2 and 0: distance 2
        REQUIRE(fiedler::cost(K, make_perm({2, 0, 1})) == Catch::Approx(2.0 * 1.0)); // sites 1 and 2: distance 1
    }

    SECTION("Bond cuts") {
        const DoubleMatrix K = chain_K(5, identity(5));
        const DoubleVector cuts = fiedler::bond_cuts(K, identity(5));
        REQUIRE(cuts.size() == 4);
        for (int b = 0; b < 4; b++) REQUIRE(cuts(b) == Catch::Approx(1.0));
        // one long link 0-3 across 3 bonds
        DoubleMatrix K2 = DoubleMatrix::Zero(4, 4);
        K2(0, 3) = K2(3, 0) = 2.0;
        const DoubleVector c2 = fiedler::bond_cuts(K2, identity(4));
        REQUIRE(c2.sum() == Catch::Approx(6.0));
        REQUIRE(c2.maxCoeff() == Catch::Approx(2.0));
    }

    SECTION("Direct meaning of unpermute and permute of a rank-4 tensor") {
        const int N = 5;
        const IntVector perm = make_perm({2, 4, 0, 3, 1});
        std::mt19937 gen(11);
        std::normal_distribution<double> d(0.0, 1.0);
        ComplexMatrix A(N, N);
        ComplexTensorR4 T(N, N, N, N);
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) A(i, j) = ComplexDouble(d(gen), d(gen));
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                for (int k = 0; k < N; k++)
                    for (int l = 0; l < N; l++) T(i, j, k, l) = ComplexDouble(d(gen), d(gen));
        const ComplexMatrix Au = fiedler::unpermute(A, perm);
        const ComplexTensorR4 Tp = fiedler::permute(T, perm);
        const ComplexTensorR4 Tu = fiedler::unpermute(T, perm);
        for (int s = 0; s < N; s++)
            for (int t = 0; t < N; t++) {
                REQUIRE(Au(perm(s), perm(t)) == A(s, t));
                for (int u = 0; u < N; u++)
                    for (int v = 0; v < N; v++) {
                        REQUIRE(Tp(s, t, u, v) == T(perm(s), perm(t), perm(u), perm(v)));
                        REQUIRE(Tu(perm(s), perm(t), perm(u), perm(v)) == T(s, t, u, v));
                    }
            }
    }

    SECTION("Row-major fill used by the DMRG solver") {
        const int N = 5;
        const IntVector perm = make_perm({2, 4, 0, 3, 1});
        std::mt19937 gen(5);
        std::normal_distribution<double> d(0.0, 1.0);
        ComplexTensorR4 g(N, N, N, N);
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                for (int k = 0; k < N; k++)
                    for (int l = 0; l < N; l++) g(i, j, k, l) = ComplexDouble(d(gen), d(gen));
        const ComplexTensorR4 ref = fiedler::permute(g, perm);
        Eigen::Tensor<ComplexDouble, 4, Eigen::RowMajor> Rc(N, N, N, N);
        Eigen::Tensor<double, 4, Eigen::RowMajor> Rr(N, N, N, N);
        fiedler::fill_rowmajor(g, Rc, perm);
        fiedler::fill_rowmajor(g, Rr, perm);
        for (int s = 0; s < N; s++)
            for (int t = 0; t < N; t++)
                for (int u = 0; u < N; u++)
                    for (int v = 0; v < N; v++) {
                        REQUIRE(Rc(s, t, u, v) == ref(s, t, u, v));
                        REQUIRE(Rr(s, t, u, v) == ref(s, t, u, v).real());
                    }
        // identity: R(i,j,k,l) = g(i,j,k,l), the same as the old shuffle + swap_layout path
        Eigen::array<int, 4> reverse = {3, 2, 1, 0};
        Eigen::Tensor<ComplexDouble, 4, Eigen::RowMajor> Rold = g.shuffle(reverse).swap_layout();
        fiedler::fill_rowmajor(g, Rc, identity(N));
        REQUIRE(std::memcmp(Rc.data(), Rold.data(), sizeof(ComplexDouble) * N * N * N * N) == 0);
    }

    SECTION("Degenerate and trivial graphs") {
        // ring: degenerate spectrum, must still give a valid permutation, the same every time
        DoubleMatrix ring = DoubleMatrix::Zero(6, 6);
        for (int i = 0; i < 6; i++) ring(i, (i + 1) % 6) = ring((i + 1) % 6, i) = 1.0;
        const IntVector r1 = fiedler::order(ring, edge_thrs, identity(6));
        const IntVector r2 = fiedler::order(ring, edge_thrs, identity(6));
        REQUIRE(same(r1, r2));
        std::vector<int> sorted(r1.data(), r1.data() + 6);
        std::sort(sorted.begin(), sorted.end());
        for (int i = 0; i < 6; i++) REQUIRE(sorted[i] == i);
        // star: centre 0 connected to 1..4
        DoubleMatrix star = DoubleMatrix::Zero(5, 5);
        for (int i = 1; i < 5; i++) star(0, i) = star(i, 0) = 1.0;
        const IntVector s1 = fiedler::order(star, edge_thrs, identity(5));
        REQUIRE(same(s1, fiedler::order(star, edge_thrs, identity(5))));
        std::vector<int> ss(s1.data(), s1.data() + 5);
        std::sort(ss.begin(), ss.end());
        for (int i = 0; i < 5; i++) REQUIRE(ss[i] == i);
        // no couplings: every orbital is its own component, so the components are joined in index order
        // (the cost is 0 for any order, so the hysteresis keeps the current order anyway)
        const IntVector ref = make_perm({3, 1, 0, 2});
        REQUIRE(same(fiedler::order(DoubleMatrix::Zero(4, 4), edge_thrs, ref), identity(4)));
        // N = 3 chain
        const IntVector lab = make_perm({1, 2, 0});
        const IntVector p3 = fiedler::order(chain_K(3, lab), edge_thrs, identity(3));
        REQUIRE((same(p3, lab) or same(p3, reversed(lab))));
        // exchange matrix: NaN and negative values become 0, result symmetric with zero diagonal
        ComplexTensorR4 g(3, 3, 3, 3);
        g.setZero();
        g(0, 1, 1, 0) = std::nan("");
        g(1, 0, 0, 1) = -1.0;
        g(0, 2, 2, 0) = 0.4;
        g(2, 0, 0, 2) = 0.2;
        g(1, 1, 1, 1) = 5.0;
        const DoubleMatrix K = fiedler::exchange_matrix(g);
        REQUIRE(K(0, 1) == 0.0);
        REQUIRE(K(0, 2) == Catch::Approx(0.3));
        REQUIRE(K(2, 0) == K(0, 2));
        REQUIRE(K(1, 1) == 0.0);
    }
}

namespace {
IntVector random_perm(int N, std::mt19937 &gen) {
    IntVector p = identity(N);
    std::shuffle(p.data(), p.data() + N, gen);
    return p;
}
// kind 0: dense, 1: sparse (each pair coupled with probability 0.2), 2: dense with two empty rows
DoubleMatrix random_K(int N, int kind, std::mt19937 &gen) {
    std::uniform_real_distribution<double> u(0.05, 1.0);
    std::bernoulli_distribution coupled(0.2);
    DoubleMatrix K = DoubleMatrix::Zero(N, N);
    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++) {
            if (kind == 1 and !coupled(gen)) continue;
            K(i, j) = K(j, i) = u(gen);
        }
    if (kind == 2 and N > 3) {
        for (int z : {0, N / 2}) {
            K.row(z).setZero();
            K.col(z).setZero();
        }
    }
    return K;
}
} // namespace

TEST_CASE("Fiedler ordering: properties on random data", "[fiedler]") {
    const double edge_thrs = 1.0e-6;

    SECTION("order returns a valid permutation, deterministically (T4)") {
        for (unsigned seed = 0; seed < 200; seed++) {
            std::mt19937 gen(seed);
            const int N = 1 + seed % 30;
            const int kind = (seed / 30) % 3;
            const DoubleMatrix K = random_K(N, kind, gen);
            const IntVector ref = random_perm(N, gen);
            const IntVector p = fiedler::order(K, edge_thrs, ref);
            REQUIRE(p.size() == N);
            std::vector<int> seen(N, 0);
            for (int s = 0; s < N; s++) {
                REQUIRE(p(s) >= 0);
                REQUIRE(p(s) < N);
                seen[p(s)]++;
            }
            for (int i = 0; i < N; i++) REQUIRE(seen[i] == 1);
            REQUIRE(same(p, fiedler::order(K, edge_thrs, ref)));
        }
    }

    SECTION("order is equivariant under relabelling, connected K (T5)") {
        for (unsigned seed = 0; seed < 100; seed++) {
            std::mt19937 gen(1000 + seed);
            const int N = 3 + seed % 14;
            const DoubleMatrix K = random_K(N, 0, gen);
            const IntVector ref = random_perm(N, gen);
            const IntVector sigma = random_perm(N, gen); // orbital i -> sigma(i)
            DoubleMatrix Kp(N, N);
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++) Kp(sigma(i), sigma(j)) = K(i, j);
            IntVector refp(N);
            for (int s = 0; s < N; s++) refp(s) = sigma(ref(s));
            const IntVector p = fiedler::order(K, edge_thrs, ref);
            const IntVector pp = fiedler::order(Kp, edge_thrs, refp);
            for (int s = 0; s < N; s++) REQUIRE(pp(s) == sigma(p(s)));
        }
    }

    SECTION("order of an ordered input is stable (T6)") {
        for (unsigned seed = 0; seed < 100; seed++) {
            std::mt19937 gen(2000 + seed);
            const int N = 3 + seed % 20;
            const DoubleMatrix K = random_K(N, 0, gen);
            const IntVector p = fiedler::order(K, edge_thrs, random_perm(N, gen));
            REQUIRE(same(fiedler::order(K, edge_thrs, p), p));
        }
    }

    SECTION("cost and bond_cuts against brute force, convention of permute (T9)") {
        for (unsigned seed = 0; seed < 100; seed++) {
            std::mt19937 gen(3000 + seed);
            const int N = 2 + seed % 15;
            const DoubleMatrix K = random_K(N, seed % 2, gen);
            const IntVector perm = random_perm(N, gen);
            const IntVector s = fiedler::inverse(perm);
            double c = 0.0, cut = 0.0;
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++) {
                    c += K(i, j) * double(s(i) - s(j)) * double(s(i) - s(j));
                    if (i < j) cut += K(i, j) * std::abs(s(i) - s(j));
                }
            REQUIRE(fiedler::cost(K, perm) == Catch::Approx(c).epsilon(1.0e-12));
            REQUIRE(fiedler::bond_cuts(K, perm).sum() == Catch::Approx(cut).epsilon(1.0e-12));
            // cost uses the same convention as permute: K in chain order has the same cost at the identity
            DoubleMatrix Kc(N, N);
            for (int a = 0; a < N; a++)
                for (int b = 0; b < N; b++) Kc(a, b) = K(perm(a), perm(b));
            REQUIRE(fiedler::cost(Kc, identity(N)) == Catch::Approx(c).epsilon(1.0e-12));
        }
    }

    SECTION("edge threshold is strict, and order is scale invariant (T10)") {
        const IntVector lab = make_perm({2, 0, 3, 1, 4});
        DoubleMatrix K = chain_K(5, lab);
        // the link between sites 1 and 2 of the chain equals the threshold exactly: no edge
        K(lab(1), lab(2)) = K(lab(2), lab(1)) = edge_thrs;
        // two components {lab0, lab1} and {lab2, lab3, lab4}, each contiguous in the result
        const IntVector p = fiedler::order(K, edge_thrs, identity(5));
        std::vector<int> site(5);
        for (int s = 0; s < 5; s++) site[p(s)] = s;
        const int a = std::min(site[lab(0)], site[lab(1)]), b = std::max(site[lab(0)], site[lab(1)]);
        REQUIRE(b - a == 1);
        std::vector<int> c2 = {site[lab(2)], site[lab(3)], site[lab(4)]};
        std::sort(c2.begin(), c2.end());
        REQUIRE(c2[2] - c2[0] == 2);
        // slightly above the threshold: one connected chain
        K(lab(1), lab(2)) = K(lab(2), lab(1)) = 2.0 * edge_thrs;
        const IntVector q = fiedler::order(K, edge_thrs, identity(5));
        REQUIRE((same(q, lab) or same(q, reversed(lab))));

        std::mt19937 gen(4242);
        const DoubleMatrix R = random_K(8, 1, gen);
        const IntVector ref = random_perm(8, gen);
        const IntVector p0 = fiedler::order(R, 0.3, ref);
        for (double c : {1.0e-3, 1.0e3}) REQUIRE(same(fiedler::order(c * R, c * 0.3, ref), p0));
    }
}

} // namespace fiedler_tests
