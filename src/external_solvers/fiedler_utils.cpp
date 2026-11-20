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

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include <Eigen/Eigenvalues>
#include <MRCPP/Printer>

#include "fiedler_utils.h"

namespace mrchem {

/** @brief Exchange matrix K_ij = max(0, Re g(i,j,j,i)), symmetrized, with zero diagonal
 *
 * @param g two-electron integrals g(a,b,c,d) = <ab|cd>
 *
 * K_ij is the repulsion of the overlap density phi_i^* phi_j with itself, so it is large
 * when the orbitals i and j are close in space.
 */
DoubleMatrix fiedler::exchange_matrix(const ComplexTensorR4 &g) {
    const int N = g.dimension(0);
    DoubleMatrix K = DoubleMatrix::Zero(N, N);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) K(i, j) = std::max(0.0, g(i, j, j, i).real());
    K = (0.5 * (K + K.transpose())).eval(); // eval(): K appears on both sides (aliasing)
    K.diagonal().setZero();
    return K;
}

/** @brief Site of each orbital (the inverse of perm) */
IntVector fiedler::inverse(const IntVector &perm) {
    IntVector inv(perm.size());
    for (int s = 0; s < perm.size(); s++) inv(perm(s)) = s;
    return inv;
}

/** @brief Cost sum_ij K_ij (s_i - s_j)^2, where s_i is the site of orbital i (small cost = strongly coupled orbitals close) */
double fiedler::cost(const DoubleMatrix &K, const IntVector &perm) {
    const IntVector site = inverse(perm);
    double c = 0.0;
    for (int i = 0; i < K.rows(); i++)
        for (int j = i + 1; j < K.cols(); j++) {
            const double d = site(i) - site(j);
            c += 2.0 * K(i, j) * d * d;
        }
    return c;
}

/** @brief Coupling that crosses each bond of the chain: cut_b = sum of K_ij over pairs with site(i) <= b < site(j)
 *
 * This is the quantity that limits a DMRG truncation at bond b. The sum of all cuts is sum_{i<j} K_ij |s_i - s_j|.
 */
DoubleVector fiedler::bond_cuts(const DoubleMatrix &K, const IntVector &perm) {
    const int N = K.rows();
    const IntVector site = inverse(perm);
    DoubleVector cuts = DoubleVector::Zero(std::max(N - 1, 0));
    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++) {
            const int lo = std::min(site(i), site(j));
            const int hi = std::max(site(i), site(j));
            for (int b = lo; b < hi; b++) cuts(b) += K(i, j);
        }
    return cuts;
}

/**
 * @brief Fiedler order of the orbitals
 *
 * @param K          exchange matrix (symmetric, non-negative, zero diagonal)
 * @param edge_thrs  K_ij above this value connects orbitals i and j
 * @param ref        previous order (perm[s] = orbital), used to fix the sign of the Fiedler vector
 *
 * The orbitals are split into connected components. Each component is ordered by the eigenvector of the
 * second-lowest eigenvalue of the graph Laplacian L = D - K, and the components are put one after another,
 * sorted by their smallest orbital index. The direction of each component is chosen to be closest to ref.
 * Returns perm, where perm[s] is the orbital at site s.
 *
 * NOTE: the graph used here is K thresholded at edge_thrs (only to find the components), while the Laplacian
 * inside a component uses the full K. Components are joined by their smallest orbital index. cost() uses the full K.
 * Near-degenerate eigenvalues (symmetric systems) make the vector arbitrary inside the degenerate space: the
 * values are rounded to 1e-10 before sorting, so that near-ties fall back to the stable orbital-index order.
 */
IntVector fiedler::order(const DoubleMatrix &K, double edge_thrs, const IntVector &ref) {
    const int N = K.rows();
    if (N <= 2) return ref;

    IntVector pos_ref = inverse(ref);

    // connected components (BFS), visited in increasing orbital index
    std::vector<std::vector<int>> comps;
    std::vector<bool> seen(N, false);
    for (int start = 0; start < N; start++) {
        if (seen[start]) continue;
        std::vector<int> comp{start};
        seen[start] = true;
        for (size_t q = 0; q < comp.size(); q++) {
            const int i = comp[q];
            for (int j = 0; j < N; j++) {
                if (!seen[j] and K(i, j) > edge_thrs) {
                    seen[j] = true;
                    comp.push_back(j);
                }
            }
        }
        std::sort(comp.begin(), comp.end());
        comps.push_back(comp);
    }

    std::vector<int> result;
    for (const auto &comp : comps) {
        const int n = comp.size();
        std::vector<int> o(comp); // orbitals of the component in the new order
        if (n <= 2) {
            std::stable_sort(o.begin(), o.end(), [&](int a, int b) { return pos_ref(a) < pos_ref(b); });
        } else {
            DoubleMatrix L(n, n);
            for (int a = 0; a < n; a++)
                for (int b = 0; b < n; b++) L(a, b) = -K(comp[a], comp[b]);
            for (int a = 0; a < n; a++) L(a, a) = 0.0;
            for (int a = 0; a < n; a++) L(a, a) = -L.row(a).sum();

            Eigen::SelfAdjointEigenSolver<DoubleMatrix> solver(L);
            if (solver.info() != Eigen::Success) MSG_ABORT("Diagonalization of the graph Laplacian failed");
            const DoubleVector &lam = solver.eigenvalues();
            static bool warned = false; // warn only once per run
            if (!warned and lam(2) - lam(1) < 1.0e-3 * (lam(2) - lam(0))) {
                MSG_WARN("Fiedler ordering: near-degenerate eigenvalues, the order may be unstable");
                warned = true;
            }

            DoubleVector f = solver.eigenvectors().col(1);
            for (int a = 0; a < n; a++) f(a) = std::round(f(a) * 1.0e10) / 1.0e10;
            std::vector<int> idx(n);
            std::iota(idx.begin(), idx.end(), 0);
            std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return f(a) < f(b); });
            for (int s = 0; s < n; s++) o[s] = comp[idx[s]];
        }

        // keep the direction (o or reverse(o)) that is closest to the previous order
        std::vector<int> r(o.rbegin(), o.rend());
        double score_o = 0.0, score_r = 0.0;
        for (int s = 0; s < n; s++) {
            score_o += static_cast<double>(s) * pos_ref(o[s]);
            score_r += static_cast<double>(s) * pos_ref(r[s]);
        }
        bool use_r = score_r > score_o;
        if (score_r == score_o) {
            // tie: lexicographically smaller sequence of previous positions
            std::vector<int> po(n), pr(n);
            for (int s = 0; s < n; s++) {
                po[s] = pos_ref(o[s]);
                pr[s] = pos_ref(r[s]);
            }
            use_r = pr < po;
        }
        const auto &best = use_r ? r : o;
        result.insert(result.end(), best.begin(), best.end());
    }

    IntVector perm(N);
    for (int s = 0; s < N; s++) perm(s) = result[s];
    return perm;
}

/** @brief A'(s,t) = A(perm[s], perm[t]) */
ComplexMatrix fiedler::permute(const ComplexMatrix &A, const IntVector &perm) {
    const int N = perm.size();
    ComplexMatrix B(N, N);
    for (int s = 0; s < N; s++)
        for (int t = 0; t < N; t++) B(s, t) = A(perm(s), perm(t));
    return B;
}

/** @brief Inverse of permute: A(perm[s], perm[t]) = A'(s,t) */
ComplexMatrix fiedler::unpermute(const ComplexMatrix &A, const IntVector &perm) {
    const int N = perm.size();
    ComplexMatrix B(N, N);
    for (int s = 0; s < N; s++)
        for (int t = 0; t < N; t++) B(perm(s), perm(t)) = A(s, t);
    return B;
}

/** @brief A'(s,t,u,v) = A(perm[s], perm[t], perm[u], perm[v]) */
ComplexTensorR4 fiedler::permute(const ComplexTensorR4 &A, const IntVector &perm) {
    const int N = perm.size();
    const IntVector site = inverse(perm);
    ComplexTensorR4 B(N, N, N, N);
    // loops follow the memory of A (first index fastest) and scatter: B(site(a),..) = A(a,..)
    for (int d = 0; d < N; d++)
        for (int c = 0; c < N; c++)
            for (int b = 0; b < N; b++)
                for (int a = 0; a < N; a++) B(site(a), site(b), site(c), site(d)) = A(a, b, c, d);
    return B;
}

/** @brief Inverse of permute: A(perm[s], perm[t], perm[u], perm[v]) = A'(s,t,u,v) */
ComplexTensorR4 fiedler::unpermute(const ComplexTensorR4 &A, const IntVector &perm) {
    const int N = perm.size();
    ComplexTensorR4 B(N, N, N, N);
    // loops follow the memory of A (first index fastest) and scatter
    for (int v = 0; v < N; v++)
        for (int u = 0; u < N; u++)
            for (int t = 0; t < N; t++)
                for (int s = 0; s < N; s++) B(perm(s), perm(t), perm(u), perm(v)) = A(s, t, u, v);
    return B;
}

} // namespace mrchem
