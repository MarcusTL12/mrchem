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

// This include must come first due to name clashes

extern "C" {
#include "aligned_memory.h"
#include "dmrg.h"
#include "gradient.h"
#include "hamiltonian.h"
#include "logging.h"
#include "mps.h"
#include "qnumber.h"
#include "rng.h"
}

#include <limits>
#include <random>
#include <sstream>

#include <MRCPP/Parallel>
#include <MRCPP/Printer>

#include "ChemTensorSolver.h"
#include "fiedler_utils.h"
#include "utils/print_utils.h"

namespace mrchem {

ChemTensorSolver::ChemTensorSolver(OrbitalVector &Phi, FockBuilder &F, Nuclei &nucs, int Ne, int spin, json dict_chemtensor)
        : ExternalSolver(F, nucs, dict_chemtensor["prec"]) {
    this->qnum_sector = encode_quantum_number_pair(Ne, spin);
    println(0, "Initialized ChemTensorSolver with Ne = " << Ne << " and spin = " << spin);

    this->max_vdim = dict_chemtensor["max_vdim"];
    this->num_sweeps = dict_chemtensor["num_sweeps"];
    this->maxiter_lanczos = dict_chemtensor["maxiter_lanczos"];
    this->tol_split = dict_chemtensor["tol_split"];
    this->start_from_previous_mps = dict_chemtensor["start_from_previous_mps"];
    this->site_order = dict_chemtensor["site_order"];
    if (this->site_order != "none" and this->site_order != "fiedler") MSG_ABORT("Unknown site_order: " << this->site_order);
    this->reorder_every_iter = dict_chemtensor["reorder_every_iter"];
    this->reorder_thrs = dict_chemtensor["reorder_thrs"];

    // ChemTensor logs to stdout from C: only the grand master should print
    ct_verbosity = mrcpp::mpi::grand_master() ? CT_VERBOSITY_TIMING : CT_VERBOSITY_NONE;

    // seed of the random initial MPS: if negative, draw a random one
    int seed = dict_chemtensor["mps_seed"];
    if (seed < 0) {
        std::random_device rd;
        seed = std::uniform_int_distribution<int>(0, std::numeric_limits<int>::max())(rd);
    }
    this->mps_seed = seed;
}

ChemTensorSolver::~ChemTensorSolver() {
    if (this->tkin_tensor) {
        delete[] this->tkin_tensor->dim;
        delete this->tkin_tensor;
    }
    if (this->velec_tensor) {
        delete[] this->velec_tensor->dim;
        delete this->velec_tensor;
    }
    if (this->assembly) {
        delete_mpo_assembly(this->assembly);
        delete this->assembly;
    }
    if (this->psi) {
        delete_mps(this->psi);
        delete this->psi;
    }
}

void ChemTensorSolver::set_dense_tensors() {
    // free the tensors of the previous iteration (the data is owned by the row-major members)
    if (this->tkin_tensor) {
        delete[] this->tkin_tensor->dim;
        delete this->tkin_tensor;
    }
    if (this->velec_tensor) {
        delete[] this->velec_tensor->dim;
        delete this->velec_tensor;
    }

    // NOTE: the row-major copies must be members, since the dense tensors only point to their data,
    // which is read later in optimize(). In the real mode only the real parts are passed (real DMRG).
    // The integrals are written in chain order: R(s,t,..) = g(perm[s], perm[t], ..), row-major
    const int N = this->one_body_integrals->rows();
    const IntVector &p = this->perm;
    void *tkin_data, *velec_data;
    if (this->real) {
        auto h_rm = std::make_shared<RowMajorMatrixReal>(N, N);
        auto g_rm = std::make_shared<RowMajorTensorReal>(N, N, N, N);
        for (int s = 0; s < N; s++)
            for (int t = 0; t < N; t++) (*h_rm)(s, t) = (*this->one_body_integrals)(p(s), p(t)).real();
        fiedler::fill_rowmajor(*this->two_body_integrals, *g_rm, p);
        this->one_body_integrals_rowmajor_real = h_rm;
        this->two_body_integrals_rowmajor_real = g_rm;
        tkin_data = static_cast<void *>(this->one_body_integrals_rowmajor_real->data());
        velec_data = static_cast<void *>(this->two_body_integrals_rowmajor_real->data());
    } else {
        auto h_rm = std::make_shared<RowMajorMatrix>(N, N);
        auto g_rm = std::make_shared<RowMajorTensor>(N, N, N, N);
        for (int s = 0; s < N; s++)
            for (int t = 0; t < N; t++) (*h_rm)(s, t) = (*this->one_body_integrals)(p(s), p(t));
        fiedler::fill_rowmajor(*this->two_body_integrals, *g_rm, p);
        this->one_body_integrals_rowmajor = h_rm;
        this->two_body_integrals_rowmajor = g_rm;
        tkin_data = static_cast<void *>(this->one_body_integrals_rowmajor->data());
        velec_data = static_cast<void *>(this->two_body_integrals_rowmajor->data());
    }
    const auto dtype = this->real ? CT_DOUBLE_REAL : CT_DOUBLE_COMPLEX;
    const ct_long L = this->one_body_integrals->rows();

    this->tkin_tensor = new dense_tensor;
    this->tkin_tensor->data = tkin_data;
    this->tkin_tensor->dim = new ct_long[2]{L, L};
    this->tkin_tensor->dtype = dtype;
    this->tkin_tensor->ndim = 2;

    this->velec_tensor = new dense_tensor;
    this->velec_tensor->data = velec_data;
    this->velec_tensor->dim = new ct_long[4]{L, L, L, L};
    this->velec_tensor->dtype = dtype;
    this->velec_tensor->ndim = 4;
}

void ChemTensorSolver::set_integrals(OrbitalVector &Phi) {
    ExternalSolver::set_integrals(Phi);
    update_site_order();
    set_dense_tensors();
}

/**
 * @brief Decide the order of the orbitals in the DMRG chain
 *
 * Uses the exchange matrix K of the current integrals. At the first call (or at every call, if reorder_every_iter)
 * the Fiedler order replaces the current one if its cost is lower by the factor (1 - reorder_thrs).
 * With site_order = none the order is never changed (identity).
 */
void ChemTensorSolver::update_site_order() {
    const int L = this->one_body_integrals->rows();
    bool first = false;
    if (this->perm.size() != L) {
        this->perm = IntVector::LinSpaced(L, 0, L - 1);
        first = true;
    }
    this->perm_changed = false;
    this->perm_cost = 0.0;
    if (this->site_order == "none") return;

    DoubleMatrix K = fiedler::exchange_matrix(*this->two_body_integrals);
    IntVector cand = fiedler::order(K, 10.0 * this->prec, this->perm);
    const double c_old = fiedler::cost(K, this->perm);
    const double c_new = fiedler::cost(K, cand);
    if ((first or this->reorder_every_iter) and c_new < (1.0 - this->reorder_thrs) * c_old) {
        this->perm = cand;
        this->perm_changed = true;
    }
    this->perm_cost = fiedler::cost(K, this->perm);

    std::stringstream o_perm;
    for (int s = 0; s < L; s++) o_perm << this->perm(s) << (s + 1 < L ? " " : "");
    print_utils::text(1, "Site order", o_perm.str());
    print_utils::scalar(1, "Site order cost", this->perm_cost, "", 6);
    if (mrcpp::Printer::getPrintLevel() >= 2) {
        print_utils::scalar(2, "Fiedler candidate cost", c_new, "", 6);
        const DoubleVector cuts = fiedler::bond_cuts(K, this->perm);
        if (cuts.size() > 0) {
            print_utils::scalar(2, "Total cut (sum K |ds|)", cuts.sum(), "", 6);
            print_utils::scalar(2, "Largest cut over bonds", cuts.maxCoeff(), "", 6);
        }
        double kmin = std::numeric_limits<double>::max();
        for (int i = 0; i < L; i++)
            for (int j = i + 1; j < L; j++)
                if (K(i, j) > 0.0) kmin = std::min(kmin, K(i, j));
        if (kmin < std::numeric_limits<double>::max()) print_utils::scalar(2, "Smallest nonzero K_ij", kmin, "", 6, true);
    }
}

void ChemTensorSolver::optimize() {
    if (!this->tkin_tensor || !this->velec_tensor) MSG_ABORT("Integrals not set.");

    // free the assembly of the previous iteration before building a new one
    if (this->assembly)
        delete_mpo_assembly(this->assembly);
    else
        this->assembly = new mpo_assembly{};

    mpo hamiltonian;
    // NOTE: optimized assembly (optimize = true) not implemented in chemtensor, and calculate_rdms
    // relies on the fixed coefficient layout of the non-optimized construction
    construct_spin_molecular_hamiltonian_mpo_assembly(this->tkin_tensor, this->velec_tensor, false, this->assembly);
    // this->assembly = std::make_shared<mpo_assembly>(assembly);

    mpo_from_assembly(this->assembly, &hamiltonian);
    if (!mpo_is_consistent(&hamiltonian)) MSG_ABORT("internal consistency check for Molecular Hamiltonian MPO failed");

    // initial state vector as MPS
    // if requested, start from the MPS of the previous iteration (if any, and of the same data type
    // as the Hamiltonian: not the case after switching from real to complex), otherwise from a random one
    if (this->start_from_previous_mps && !this->perm_changed && this->psi && this->psi->a[0].dtype == hamiltonian.a[0].dtype) {
        // nothing to do: dmrg_twosite uses this->psi as initial state
    } else {
        // free the MPS of the previous iteration before building a new one
        if (this->psi) {
            delete_mps(this->psi);
        } else {
            this->psi = new mps{};
        }

        rng_state rng;
        seed_rng_state(this->mps_seed, &rng);

        construct_random_mps(hamiltonian.a[0].dtype, hamiltonian.nsites, hamiltonian.d, hamiltonian.qsite, this->qnum_sector, this->max_vdim, &rng, this->psi);
        if (!mps_is_consistent(this->psi)) MSG_ABORT("internal MPS consistency check failed");
    }

    // #ifdef _OPENMP
    // printf("maximum number of OpenMP threads: %d\n", omp_get_max_threads());
    // #else
    // printf("OpenMP not available\n");
    // #endif

    // run two-site DMRG
    // this->en_sweeps.assign(this->num_sweeps, 0.0);
    // std::vector<double> entropy(hamiltonian.nsites - 1);
    dmrg_twosite_options options;
    dmrg_twosite_default_options(&options);
    options.max_num_sweeps = this->num_sweeps;
    options.maxiter_lanczos = this->maxiter_lanczos;
    options.tol_split = this->tol_split;
    options.max_vdim = this->max_vdim;
    dmrg_twosite_info info;
    if (dmrg_twosite(&hamiltonian, &options, this->psi, &info) < 0) MSG_ERROR("'dmrg_twosite' failed internally");

    if (this->energy_correction) {
        for (auto i = 0; i < info.num_sweeps; i++) info.en_sweeps[i] += this->E_nn;
    }

    this->energy = info.en_sweeps[info.num_sweeps - 1];

    // calculate final bond dimensions
    this->bond_dimensions.assign(hamiltonian.nsites + 1, 0);
    for (int l = 0; l < hamiltonian.nsites + 1; l++) this->bond_dimensions[l] = mps_bond_dim(this->psi, l);

    // the MPO is rebuilt at every call (RDMs only need assembly and psi)
    delete_mpo(&hamiltonian);

    // calculate RDMs
    calculate_rdms();
}

void ChemTensorSolver::calculate_rdms() {
    // number of orbitals
    const int N = this->psi->nsites;

    // energy average and gradients' coefficients (of the data type of the assembly: real in the real mode)
    std::complex<double> avr;
    std::vector<std::complex<double>> dcoeff(this->assembly->num_coeffs);
    if (this->assembly->dtype == CT_DOUBLE_REAL) {
        std::vector<double> dcoeff_real(this->assembly->num_coeffs);
        operator_average_coefficient_gradient(this->assembly, this->psi, this->psi, &avr, dcoeff_real.data());
        for (size_t i = 0; i < dcoeff_real.size(); i++) dcoeff[i] = dcoeff_real[i];
    } else {
        operator_average_coefficient_gradient(this->assembly, this->psi, this->psi, &avr, dcoeff.data());
    }

    // --- one-body RDM ---
    int c = 2;
    ComplexMatrix rdm1(N, N);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) rdm1(i, j) = dcoeff[c++];
    // symmetrize
    rdm1 = 0.5 * (rdm1 + rdm1.conjugate().transpose());
    this->one_rdm = std::make_shared<ComplexMatrix>(std::move(rdm1));

    // --- two-body RDM ---
    const int dim_g0 = (this->assembly->num_coeffs - c) / 2;

    // intermediate tensors
    ComplexTensorR4 dg0(N, N, N, N);
    ComplexTensorR4 dg1(N, N, N, N);
    dg0.setZero();
    dg1.setZero();
    int c0 = c;
    int c1 = c + dim_g0;
    for (int i = 0; i < N; i++)
        for (int j = i; j < N; j++)
            for (int k = 0; k < N; k++)
                for (int l = k; l < N; l++) {
                    dg0(i, j, k, l) = dcoeff[c0];
                    dg1(i, j, k, l) = dcoeff[c1];
                    c0++;
                    c1++;
                }

    // reconstruct two_body from dg0 and dg1
    ComplexTensorR4 rdm2(N, N, N, N);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                for (int l = 0; l < N; l++) rdm2(i, j, k, l) = 0.5 * (dg0(i, j, k, l) + dg0(j, i, l, k) - dg1(j, i, k, l) - dg1(i, j, l, k));

    // enforce hermiticity rdm2(i,j,k,l) = conj(rdm2(k,l,i,j)), with rdm2(i,j,k,l) = 1/2 <a+_i a+_j a_l a_k>
    ComplexTensorR4 rdm2_sym(N, N, N, N);
    Eigen::array<int, 4> swap_pairs = {2, 3, 0, 1};
    rdm2_sym = 0.5 * (rdm2 + rdm2.conjugate().shuffle(swap_pairs));

    // the RDMs are in chain order: bring them back to the order of the orbitals
    if (this->site_order != "none") {
        this->one_rdm = std::make_shared<ComplexMatrix>(fiedler::unpermute(*this->one_rdm, this->perm));
        rdm2_sym = fiedler::unpermute(rdm2_sym, this->perm);
    }

    // check: energy from the RDMs (in orbital order) against the one from the MPO (without E_nn)
    if (mrcpp::Printer::getPrintLevel() >= 3) {
        // in the real mode the MPO only has the real parts of the integrals, so use the same here
        const int L = this->one_body_integrals->rows();
        auto val = [this](ComplexDouble x) { return this->real ? ComplexDouble(x.real(), 0.0) : x; };
        ComplexDouble e_rdm = 0.0;
        for (int i = 0; i < L; i++)
            for (int j = 0; j < L; j++) e_rdm += val((*this->one_body_integrals)(i, j)) * val((*this->one_rdm)(i, j));
        for (int i = 0; i < L; i++)
            for (int j = 0; j < L; j++)
                for (int k = 0; k < L; k++)
                    for (int l = 0; l < L; l++) e_rdm += val((*this->two_body_integrals)(i, j, k, l)) * val(rdm2_sym(i, j, k, l));
        print_utils::scalar(3, "MPO energy (no E_nn)", avr.real(), "(au)", 12);
        print_utils::scalar(3, "RDM energy (no E_nn)", e_rdm.real(), "(au)", 12);
        print_utils::scalar(3, "RDM - MPO energy", e_rdm.real() - avr.real(), "(au)", 12);
    }

    // store two body RDM in variable of parent class
    this->two_rdm = std::make_shared<ComplexTensorR4>(std::move(rdm2_sym));
}
} // namespace mrchem
