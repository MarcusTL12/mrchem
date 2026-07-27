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

#include <MRCPP/Printer>
#include <MRCPP/trees/MWNode.h>

#include <Eigen/Core>
#include <memory>

namespace mrdft {

class XCData {
public:
    XCData(const mrcpp::MWNode<3> &node, int nPts, bool spin)
            : nPts(nPts)
            , ncoefs(node->getTDim() * node->getKp1_d())
            , spin(spin)
            , hasDensity(false)
            , hasGradient(false)
            , node(node, true, false) {}

    void computeDensity(const mrcpp::MWNode<3> &rhoNode);
    void computeDensity(const mrcpp::MWNode<3> &alphaNode, const mrcpp::MWNode<3> &betaNode);

private:
    int nPts, ncoefs;
    bool spin;

    bool hasDensity, hasGradient;

    mrcpp::MWNode<3> node;

    Eigen::MatrixXd density;
    Eigen::MatrixXd gradient;

    void computeDensityColumn(const mrcpp::MWNode<3> &rhoNode, int columnIndex);
};

} // namespace mrdft
