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

#include "QMHeatLaplacian.h"

#include "MRCPP/Gaussians"
#include "MRCPP/MWFunctions"
#include "MRCPP/MWOperators"

extern mrcpp::MultiResolutionAnalysis<3> *mrchem::MRA;

namespace mrchem {

static double heat_kernel_coeffs_1[1] = {1.0};
static double heat_kernel_coeffs_2[2] = {2.0, 0.5};

static double *heat_kernel_coeffs[2] = {heat_kernel_coeffs_1, heat_kernel_coeffs_2};
static double constant_coeffs[2] = {-1.0, -1.5};

QMHeatLaplacian::QMHeatLaplacian(double t, int order, double prec)
    : prec(prec) {
    mrcpp::GaussExp<1> kernel;

    constant_coeff = constant_coeffs[order - 1] / t;

    for (int i = 0; i < order; i++) {
        double exponent = 0.25 / (t * (i + 1));
        double coeff = heat_kernel_coeffs[order - 1][i] * std::sqrt(exponent / mrcpp::pi) / t;

        mrcpp::GaussFunc<1> g(exponent, coeff);

        kernel.append(g);
    }

    conv = std::make_shared<mrcpp::ConvolutionOperator<3>>(*MRA, kernel, prec);
}

Orbital QMHeatLaplacian::apply(Orbital inp) {
    Orbital out_conv, out;

    mrcpp::apply(prec, out_conv.real(), *conv.get(), inp.real());

    mrcpp::add(prec, out.real(), 1.0, out_conv.real(), constant_coeff, inp.real());

    return out;
}

} // namespace mrchem
