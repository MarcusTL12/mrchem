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

#include "XCData.h"

namespace mrdft {

void XCData::computeDensityColumn(const mrcpp::MWNode<3> &rhoNode, int columnIndex) {
    node.attachCoefs(density.col(columnIndex).data());
    for (int j = 0; j < ncoefs; j++) { density(j, columnIndex) = rhoNode.getCoefs()[j]; }
    node.mwTransform(mrcpp::Reconstruction);
    node.cvTransform(mrcpp::Forward);
}

void XCData::computeDensity(const mrcpp::MWNode<3> &rhoNode) {
    if (isSpin) MSG_ABORT("Trying to compute paired density in unrestricted calculation");
    if (hasDensity) return;

    density = Eigen::MatrixXd(nPts, 1);

    computeDensityColumn(rhoNode, 0);

    hasDensity = true;
}

void XCData::computeDensity(const mrcpp::MWNode<3> &alphaNode, const mrcpp::MWNode<3> &betaNode) {
    if (not isSpin) MSG_ABORT("Trying to compute spin density in restricted calculation");
    if (hasDensity) return;

    density = Eigen::MatrixXd(nPts, 2);

    computeDensityColumn(alphaNode, 0);
    computeDensityColumn(betaNode, 1);

    hasDensity = true;
}

void XCData::computeGradientColumn(mrcpp::DerivativeOperator<3> &derivOp, mrcpp::FunctionTree<3> &rho, int columnIndex) {
    for (int d = 0; d < 3; d++) {
        node.attachCoefs(gradient.col(3 * columnIndex).data());

        mrcpp::DerivativeCalculator<3> derivcalc(d, derivOp, rho);
        // derive rho and put result into xclib_inp aka node
        derivcalc.calcNode(rho.getNode(node.getNodeIndex()), node);
        // make cv representation of gradient of density
        node.mwTransform(mrcpp::Reconstruction);
        node.cvTransform(mrcpp::Forward);
    }
}

void XCData::computeGradient(mrcpp::DerivativeOperator<3> &derivOp, mrcpp::FunctionTree<3> &rho) {
    if (isSpin) MSG_ABORT("Trying to compute paired density gradient in unrestricted calculation");
    if (hasGradient) return;

    computeGradientColumn(derivOp, rho, 0);

    hasGradient = true;
}

void XCData::computeGradient(mrcpp::DerivativeOperator<3> &derivOp, mrcpp::FunctionTree<3> &alpha, mrcpp::FunctionTree<3> &beta) {
    if (not isSpin) MSG_ABORT("Trying to compute spin density gradient in restricted calculation");
    if (hasGradient) return;

    computeGradientColumn(derivOp, alpha, 0);
    computeGradientColumn(derivOp, beta, 1);

    hasGradient = true;
}

} // namespace mrdft
