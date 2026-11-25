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

#include "mpi_bcast.h"

#include <limits>

#include <MRCPP/Printer>

namespace mrchem {
namespace mpi_utils {

#ifdef MRCHEM_HAS_MPI

void bcast(bool &x, MPI_Comm comm) {
    int tmp = x ? 1 : 0;
    MPI_Bcast(&tmp, 1, MPI_INT, 0, comm);
    x = (tmp != 0);
}

void bcast(int &x, MPI_Comm comm) {
    MPI_Bcast(&x, 1, MPI_INT, 0, comm);
}

void bcast(double &x, MPI_Comm comm) {
    MPI_Bcast(&x, 1, MPI_DOUBLE, 0, comm);
}

void bcast(IntVector &x, MPI_Comm comm) {
    int n = static_cast<int>(x.size());
    MPI_Bcast(&n, 1, MPI_INT, 0, comm);
    if (x.size() != n) x.resize(n);
    if (n > 0) MPI_Bcast(x.data(), n, MPI_INT, 0, comm);
}

void bcast(DoubleVector &x, MPI_Comm comm) {
    int n = static_cast<int>(x.size());
    MPI_Bcast(&n, 1, MPI_INT, 0, comm);
    if (x.size() != n) x.resize(n);
    if (n > 0) MPI_Bcast(x.data(), n, MPI_DOUBLE, 0, comm);
}

void bcast(ComplexMatrix &x, MPI_Comm comm) {
    int dims[2] = {static_cast<int>(x.rows()), static_cast<int>(x.cols())};
    MPI_Bcast(dims, 2, MPI_INT, 0, comm);
    if (x.rows() != dims[0] || x.cols() != dims[1]) x.resize(dims[0], dims[1]);
    // complex numbers are sent as pairs of doubles; the data is column-major and contiguous
    if (x.size() > 0) MPI_Bcast(reinterpret_cast<double *>(x.data()), 2 * static_cast<int>(x.size()), MPI_DOUBLE, 0, comm);
}

void bcast(ComplexTensorR4 &x, MPI_Comm comm) {
    int dims[4];
    for (int i = 0; i < 4; i++) dims[i] = static_cast<int>(x.dimension(i));
    MPI_Bcast(dims, 4, MPI_INT, 0, comm);
    bool same = true;
    for (int i = 0; i < 4; i++) same = same and (x.dimension(i) == dims[i]);
    if (not same) x.resize(dims[0], dims[1], dims[2], dims[3]);
    // complex numbers are sent as pairs of doubles; the data is contiguous (column-major)
    if (x.size() > 0) MPI_Bcast(reinterpret_cast<double *>(x.data()), 2 * static_cast<int>(x.size()), MPI_DOUBLE, 0, comm);
}

void scatterv(const std::vector<ComplexDouble> &send, const IntVector &counts, std::vector<ComplexDouble> &recv, MPI_Comm comm) {
    int size = 0, rank = 0;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
    // complex numbers are sent as pairs of doubles
    std::vector<int> cnt(size), dsp(size);
    int off = 0;
    for (int r = 0; r < size; r++) {
        cnt[r] = 2 * counts(r);
        dsp[r] = off;
        off += cnt[r];
    }
    recv.assign(counts(rank), ComplexDouble(0.0, 0.0));
    const double *sp = (rank == 0 and not send.empty()) ? reinterpret_cast<const double *>(send.data()) : nullptr;
    MPI_Scatterv(sp, cnt.data(), dsp.data(), MPI_DOUBLE, reinterpret_cast<double *>(recv.data()), cnt[rank], MPI_DOUBLE, 0, comm);
}

void gatherv(const std::vector<ComplexDouble> &send, std::vector<ComplexDouble> &recv, MPI_Comm comm) {
    int size = 0, rank = 0;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
    // complex numbers are sent as pairs of doubles
    int mycount = 2 * static_cast<int>(send.size());
    std::vector<int> cnt(size), dsp(size);
    MPI_Allgather(&mycount, 1, MPI_INT, cnt.data(), 1, MPI_INT, comm);
    long long total = 0;
    for (int r = 0; r < size; r++) {
        dsp[r] = static_cast<int>(total);
        total += cnt[r];
    }
    if (total > std::numeric_limits<int>::max()) MSG_ABORT("gatherv: message too large for a single MPI_Gatherv");
    recv.clear();
    if (rank == 0) recv.assign(static_cast<size_t>(total / 2), ComplexDouble(0.0, 0.0));
    const double *sp = send.empty() ? nullptr : reinterpret_cast<const double *>(send.data());
    double *rp = (rank == 0 and total > 0) ? reinterpret_cast<double *>(recv.data()) : nullptr;
    MPI_Gatherv(sp, mycount, MPI_DOUBLE, rp, cnt.data(), dsp.data(), MPI_DOUBLE, 0, comm);
}

bool allreduce_or(bool x, MPI_Comm comm) {
    int in = x ? 1 : 0, out = 0;
    MPI_Allreduce(&in, &out, 1, MPI_INT, MPI_LOR, comm);
    return out != 0;
}

#else

void bcast(bool &, MPI_Comm) {}
void bcast(int &, MPI_Comm) {}
void bcast(double &, MPI_Comm) {}
void bcast(IntVector &, MPI_Comm) {}
void bcast(DoubleVector &, MPI_Comm) {}
void bcast(ComplexMatrix &, MPI_Comm) {}
void bcast(ComplexTensorR4 &, MPI_Comm) {}
void scatterv(const std::vector<ComplexDouble> &send, const IntVector &counts, std::vector<ComplexDouble> &recv, MPI_Comm) {
    recv = send;
    (void)counts;
}
void gatherv(const std::vector<ComplexDouble> &send, std::vector<ComplexDouble> &recv, MPI_Comm) {
    recv = send;
}
bool allreduce_or(bool x, MPI_Comm) {
    return x;
}

#endif

} // namespace mpi_utils
} // namespace mrchem
