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

#include "utils/mpi_bcast.h"

using namespace mrchem;

namespace mpi_bcast_tests {

TEST_CASE("mpi_utils::bcast and allreduce_or", "[mpi_bcast]") {
    const bool master = (mrcpp::mpi::wrk_rank == 0);
    // only workers take part (bank ranks do not run the tests' collectives)
    if (mrcpp::mpi::is_bank) return;

    SECTION("scalars") {
        int i = master ? 42 : -1;
        double d = master ? 3.25 : -1.0;
        bool b = master;
        mpi_utils::bcast(i);
        mpi_utils::bcast(d);
        mpi_utils::bcast(b);
        REQUIRE(i == 42);
        REQUIRE(d == 3.25);
        REQUIRE(b);
    }
    SECTION("vectors are resized on non-roots") {
        IntVector iv;
        DoubleVector dv;
        if (master) {
            iv = IntVector::LinSpaced(5, 1, 5);
            dv = DoubleVector::LinSpaced(7, 0.5, 3.5);
        }
        mpi_utils::bcast(iv);
        mpi_utils::bcast(dv);
        REQUIRE(iv.size() == 5);
        REQUIRE(iv(4) == 5);
        REQUIRE(dv.size() == 7);
        REQUIRE(dv(6) == 3.5);
    }
    SECTION("complex matrix") {
        ComplexMatrix m;
        if (master) {
            m.resize(3, 2);
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 2; c++) m(r, c) = ComplexDouble(r + 1.0, -c - 0.5);
        }
        mpi_utils::bcast(m);
        REQUIRE(m.rows() == 3);
        REQUIRE(m.cols() == 2);
        REQUIRE(m(2, 1) == ComplexDouble(3.0, -1.5));
    }
    SECTION("complex rank-4 tensor") {
        ComplexTensorR4 t;
        if (master) {
            t.resize(2, 3, 1, 2);
            t.setZero();
            t(1, 2, 0, 1) = ComplexDouble(4.0, -2.5);
        }
        mpi_utils::bcast(t);
        REQUIRE(t.dimension(0) == 2);
        REQUIRE(t.dimension(1) == 3);
        REQUIRE(t.dimension(2) == 1);
        REQUIRE(t.dimension(3) == 2);
        REQUIRE(t(1, 2, 0, 1) == ComplexDouble(4.0, -2.5));
        REQUIRE(t(0, 0, 0, 0) == ComplexDouble(0.0, 0.0));
    }
    SECTION("allreduce_or") {
        const bool only_last = (mrcpp::mpi::wrk_rank == mrcpp::mpi::wrk_size - 1);
        REQUIRE(mpi_utils::allreduce_or(only_last));
        REQUIRE_FALSE(mpi_utils::allreduce_or(false));
    }
}

} // namespace mpi_bcast_tests
