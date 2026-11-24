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

#include <MRCPP/Parallel>

#include "mrchem.h"

namespace mrchem {

/** @file mpi_bcast.h
 *
 * @brief Collective helpers on the worker communicator (mrcpp::mpi::comm_wrk, root = wrk_rank 0).
 *
 * MPI: every function here is COLLECTIVE over `comm`: all workers must call it in the same order.
 * Never call one inside a branch that depends on rank or on rank-local data.
 * Without MPI (MRCHEM_HAS_MPI undefined) all functions are no-ops.
 */
namespace mpi_utils {

void bcast(bool &x, MPI_Comm comm = mrcpp::mpi::comm_wrk);
void bcast(int &x, MPI_Comm comm = mrcpp::mpi::comm_wrk);
void bcast(double &x, MPI_Comm comm = mrcpp::mpi::comm_wrk);
void bcast(IntVector &x, MPI_Comm comm = mrcpp::mpi::comm_wrk);
void bcast(DoubleVector &x, MPI_Comm comm = mrcpp::mpi::comm_wrk);
void bcast(ComplexMatrix &x, MPI_Comm comm = mrcpp::mpi::comm_wrk);

/** @brief Logical OR of `x` over all workers; the result is returned on every worker. */
bool allreduce_or(bool x, MPI_Comm comm = mrcpp::mpi::comm_wrk);

} // namespace mpi_utils
} // namespace mrchem
