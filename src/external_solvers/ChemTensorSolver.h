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

extern "C" {
    struct dense_tensor;
    struct mpo_assembly;
    struct mps;
}

#include <string>

#include "ExternalSolver.h"
#include "mrchem.h"
#include "qmfunctions/Orbital.h"
#include "qmoperators/two_electron/FockBuilder.h"

namespace mrchem {

class ChemTensorSolver : public ExternalSolver {
public:
    ChemTensorSolver(OrbitalVector &Phi, FockBuilder &F, Nuclei &nucs, int Ne, int spin, json dict_chemtensor);
    ~ChemTensorSolver();

    void set_max_vdim(int max_vdim) { this->max_vdim = max_vdim; }
    void set_num_sweeps(int num_sweeps) { this->num_sweeps = num_sweeps; }
    void set_maxiter_lanczos(int maxiter_lanczos) { this->maxiter_lanczos = maxiter_lanczos; }
    void set_tol_split(double tol_split) { this->tol_split = tol_split; }

    void set_integrals(OrbitalVector &Phi);

    const IntVector &get_site_order() const { return this->perm; }
    double get_site_order_cost() const { return this->perm_cost; }
    bool site_order_changed() const { return this->perm_changed; }

    const int* get_bond_dimensions() const { return this->bond_dimensions.data(); }
    // const double* get_en_sweeps() const { return this->en_sweeps.data(); }

    void optimize() override;

private:
    friend class ChemTensorSolverTester; ///< unit tests (tests/external_solvers/chemtensor_solver.cpp)

    using RowMajorMatrix = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using RowMajorTensor = Eigen::Tensor<std::complex<double>, 4, Eigen::RowMajor>;
    using RowMajorMatrixReal = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using RowMajorTensorReal = Eigen::Tensor<double, 4, Eigen::RowMajor>;

    // row-major copies of the integrals: they own the data that tkin_tensor and velec_tensor point to
    // (real copies in the real mode, so that the DMRG runs in real arithmetic)
    std::shared_ptr<RowMajorMatrix> one_body_integrals_rowmajor{};
    std::shared_ptr<RowMajorTensor> two_body_integrals_rowmajor{};
    std::shared_ptr<RowMajorMatrixReal> one_body_integrals_rowmajor_real{};
    std::shared_ptr<RowMajorTensorReal> two_body_integrals_rowmajor_real{};
    dense_tensor* tkin_tensor{};
    dense_tensor* velec_tensor{};
    mpo_assembly* assembly{};
    mps* psi{};

    int max_vdim;
    int num_sweeps;
    int maxiter_lanczos;
    float tol_split;
    bool energy_correction = true;
    bool start_from_previous_mps;
    int mps_seed{};

    std::string site_order{"none"}; ///< order of the orbitals in the chain: none (input order) or fiedler
    bool reorder_every_iter{false}; ///< try to reorder at every cycle (otherwise only at the first one)
    double reorder_thrs{0.1};       ///< relative cost decrease needed to accept a new order
    IntVector perm{};               ///< perm[s] is the orbital at site s
    bool perm_changed{false};       ///< the order changed in the last call of set_integrals
    double perm_cost{0.0};          ///< Fiedler cost of the current order

    int32_t qnum_sector{};

    std::vector<int> bond_dimensions{};
    // std::vector<double> en_sweeps{};

    void update_site_order();
    void set_dense_tensors();
    void calculate_rdms() override;

    
};

} // namespace mrchem