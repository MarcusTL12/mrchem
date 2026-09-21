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

QMHeatLaplacian::QMHeatLaplacian(double t, int order, double prec, std::string &type)
        : prec(prec) {
    mrcpp::HeatKernel kernel(t, order, 3, type);

    kernel *= 1.0 / t;

    constant_coeff = kernel.constant_coeff / t;

    conv = std::make_shared<mrcpp::ConvolutionOperator<3>>(*MRA, kernel, prec);
}

Orbital QMHeatLaplacian::apply(Orbital inp) {
    Orbital out_conv, out;

    out_conv.defreal();
    out.defreal();

    mrcpp::apply(prec, out_conv.real(), *conv.get(), inp.real());

    mrcpp::add(prec, out.real(), 1.0, out_conv.real(), constant_coeff, inp.real());

    return out;
}

} // namespace mrchem
