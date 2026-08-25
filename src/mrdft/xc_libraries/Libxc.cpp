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

#include <memory>
#include <MRCPP/trees/FunctionNode.h>
#include <MRCPP/Printer>

#include "Libxc.h"

namespace mrdft {

Libxc::~Libxc() {
    for (auto *func : libxc_objects) {
        if (func != nullptr) {
            xc_func_free(func);
        }
    }
}

void Libxc::setFunctional(const std::string &name, double c) {
    std::vector<int> ids;
    std::vector<double> coefs;
    double scaled_custom_exx = 0.0;

    addFunctionalSpec(name, c);
    mapFunctionalName(name, ids, coefs, scaled_custom_exx);
    this->setCustomExx(this->getCustomExx() + c * scaled_custom_exx);
    xc_func_type *libxc_obj;
    for (size_t i = 0; i < ids.size(); i++) {
        libxc_obj = xc_func_alloc();
        auto return_code = xc_func_init(libxc_obj, ids[i], spin ? XC_POLARIZED : XC_UNPOLARIZED);
        if (return_code != 0) {
            std::ostringstream oss;
            oss << "Functional id = " << ids[i] << " not found in the employed version of Libxc.\n";
            MSG_ABORT(oss.str());
        }
        xc_func_set_dens_threshold(libxc_obj, cutoff);
        this->libxc_objects.push_back(libxc_obj);
        this->libxc_coefs.push_back(c * coefs[i]);
    }
}

void Libxc::setCutoff(double cutoff) {
    this->cutoff = cutoff;

    for (auto *f : libxc_objects) {
        xc_func_set_dens_threshold(f, cutoff);
    }
}

void getFamilyType(int family_type, bool &lda, bool &gga, bool &mgga) {
    switch (family_type) {
        case XC_FAMILY_LDA:
#ifdef XC_FAMILY_HYB_GGA
        case XC_FAMILY_HYB_LDA:
#endif
            lda = true;
            break;

        case XC_FAMILY_GGA:
#ifdef XC_FAMILY_HYB_GGA
        case XC_FAMILY_HYB_GGA:
#endif
            gga = true;
            break;

        case XC_FAMILY_MGGA:
        case XC_FAMILY_HYB_MGGA:
            mgga = true;
            MSG_ABORT("Meta-GGA functionals are not supported in MRChem.!\n");

        default:
            MSG_ABORT("Libxc functional family not handled in MRChem!\n");
    }
}

void Libxc::initFunctionalLibrary(bool &lda, bool &gga, bool &mgga, int order, bool gamma) {
    gga = false;
    mgga = false;
    (void)order; // TODO: use order to choose which libxc functions to call to obtain higher derivatives (vxc, fxc, kxc, ...)
    (void)gamma;
    for (const auto *f : libxc_objects) {
        getFamilyType(f->info->family, lda, gga, mgga);
        // Check if functional is range separated
        if ((f->info->flags & XC_FLAGS_HYB_CAM)  || (f->info->flags & XC_FLAGS_HYB_LC)) MSG_ABORT("Coulomb attenuated functionals not supported in MRChem!\n");
        if ((f->info->flags & XC_FLAGS_HYB_CAMY) || (f->info->flags & XC_FLAGS_HYB_LCY)) MSG_ABORT("Yukawa attenuated functionals not supported in MRChem!\n");
    }

}

double Libxc::getAmountExx() const {
    double exx = 0;
    for (std::size_t i = 0; i < libxc_objects.size(); ++i) {
        const xc_func_type *f = libxc_objects[i];
        double frac = xc_hyb_exx_coef(f);
        exx += libxc_coefs[i] * frac;
    }
    return exx + customExx;
}

void Libxc::printFunctionalReference(int out_txt_width) const {
    // Print header and provide wrapping utility via XClib helpers
    XCLib::printReferenceHeader();

    std::string libxc_ref_str = "Using Libxc (version " + std::string(xc_version_string()) + ") to evaluate density functionals. Libxc is free software. It is " +
                                "distributed under the Mozilla Public License, version 2.0. For " + "more information, please check the Libxc manual. You should cite\n\n" +
                                xc_reference() + " DOI: " + xc_reference_doi() + "\n\nwhen " + "reporting the results of your calculation in a scientific article.\n";
    XCLib::printWrap(libxc_ref_str, out_txt_width);

    // Avoid printing the same LibXC functional multiple times
    std::set<int> printed_ids;
    std::cout << "\nLibxc functionals used in this calculation:\n";
    for (const auto *func : libxc_objects) {
        if (func->info == nullptr) continue; // safety

        int id = func->info->number;
        if (!printed_ids.insert(id).second) {
            // Already printed this ID
            continue;
        }

        char *name = xc_functional_get_name(id);
        std::string func_id_str = "  - " + std::string(name) + " (ID " + std::to_string(id) + "): " + func->info->name + "\n";
        XCLib::printWrap(func_id_str, out_txt_width);
        free(name);

        for (int number = 0; number < XC_MAX_REFERENCES; number++) {
            auto reference = xc_func_info_get_references(func->info, number);
            if (reference == nullptr) break;
            std::string func_ref_str = "     * " + std::string(xc_func_reference_get_ref(reference)) + ", DOI:" + xc_func_reference_get_doi(reference) + "\n";
            XCLib::printWrap(func_ref_str, out_txt_width, 7);
        }
    }
}

void Libxc::callLibEval(XCData &data, Eigen::MatrixXd &out, int nPts) const {
    const Eigen::MatrixXd &density = data.getDensity();

    Eigen::MatrixXd spin_density, sigma;

    if (spin) {
        spin_density = Eigen::MatrixXd(nPts, 2);

        for (int i = 0; i < nPts; i++) {
            spin_density(0, i) = density(i, 0);
            spin_density(1, i) = density(i, 1);
        }
    }

    const Eigen::MatrixXd &grad = data.getGradient();

    if (data.hasGradient) {
        sigma = Eigen::MatrixXd(spin ? 3 : 1, nPts);

        for (int i = 0; i < nPts; i++) {
            sigma(0, i) = grad(i, 0) * grad(i, 0) +
                          grad(i, 1) * grad(i, 1) +
                          grad(i, 2) * grad(i, 2);

            if (spin) {
                sigma(1, i) = grad(i, 0) * grad(i, 3) +
                              grad(i, 1) * grad(i, 4) +
                              grad(i, 2) * grad(i, 5);

                sigma(2, i) = grad(i, 3) * grad(i, 3) +
                              grad(i, 4) * grad(i, 4) +
                              grad(i, 5) * grad(i, 5);
            }
        }
    }

    Eigen::MatrixXd exc, vxc, sxc;
    for (size_t i = 0; i < libxc_objects.size(); i++) {
        switch (libxc_objects[i]->info->family) {
            case XC_FAMILY_LDA:
            case XC_FAMILY_HYB_LDA:
                if (spin) {
                    exc = Eigen::MatrixXd::Zero(1, nPts);
                    vxc = Eigen::MatrixXd::Zero(2, nPts);
                    xc_lda_exc_vxc(libxc_objects[i], nPts, spin_density.data(), exc.data(), vxc.data());
                    for (int j = 0; j < nPts; ++j) {
                        //    xcfun calculates actual energy density while libxc calculates
                        //    energy density per electron density

                        // rho = rho_alpha + rho_beta
                        out(0, j) += exc(0, j) * libxc_coefs[i] * (spin_density(0, j) + spin_density(1, j));
                        out(1, j) += vxc(0, j) * libxc_coefs[i];
                        out(2, j) += vxc(1, j) * libxc_coefs[i];
                    }
                } else {
                    exc = Eigen::MatrixXd::Zero(1, nPts);
                    vxc = Eigen::MatrixXd::Zero(1, nPts);
                    xc_lda_exc_vxc(libxc_objects[i], nPts, density.data(), exc.data(), vxc.data());
                    for (int j = 0; j < nPts; ++j) {
                        //    xcfun calculates actual energy density while libxc calculates
                        //    energy density per electron density
                        out(0, j) += exc(0, j) * libxc_coefs[i] * density(j, 0);
                        out(1, j) += vxc(0, j) * libxc_coefs[i];
                    }
                }
                break;
            case XC_FAMILY_GGA:
            case XC_FAMILY_HYB_GGA:
                if (spin) {
                    exc = Eigen::MatrixXd::Zero(1, nPts);
                    vxc = Eigen::MatrixXd::Zero(2, nPts);
                    sxc = Eigen::MatrixXd::Zero(3, nPts);
                    xc_gga_exc_vxc(libxc_objects[i], nPts, spin_density.data(), sigma.data(), exc.data(), vxc.data(), sxc.data());

                    for (int j = 0; j < nPts; ++j) {
                        // clang-format off
                        //    xcfun calculates energy density per volume while libxc calculates
                        //    energy density per electron, so we multiply by the density here
                        out(0, j) += exc(0, j) * libxc_coefs[i] * (spin_density(0, j) + spin_density(1, j));
                        out(1, j) += vxc(0, j) * libxc_coefs[i];
                        out(2, j) += vxc(1, j) * libxc_coefs[i];

                        // alpha_i,     coef         * ( 2 * vaa               * grad_a_i  + vab       * grad_b_i ), i = x, y, z
                        out(3, j) += libxc_coefs[i] * ( 2 * sxc(0, j) * grad(j, 0) + sxc(1, j) * grad(j, 3) );
                        out(4, j) += libxc_coefs[i] * ( 2 * sxc(0, j) * grad(j, 1) + sxc(1, j) * grad(j, 4) );
                        out(5, j) += libxc_coefs[i] * ( 2 * sxc(0, j) * grad(j, 2) + sxc(1, j) * grad(j, 5) );
                        // beta_i,       coef        * ( 2 * vbb               * grad_b_i  + vab               * grad_a_i ), i = x, y, z
                        out(6, j) += libxc_coefs[i] * ( 2 * sxc(2, j) * grad(j, 3) + sxc(1, j) * grad(j, 0) );
                        out(7, j) += libxc_coefs[i] * ( 2 * sxc(2, j) * grad(j, 4) + sxc(1, j) * grad(j, 1) );
                        out(8, j) += libxc_coefs[i] * ( 2 * sxc(2, j) * grad(j, 5) + sxc(1, j) * grad(j, 2) );
                        // clang-format on
                    }
                } else {
                    exc = Eigen::MatrixXd::Zero(1, nPts);
                    vxc = Eigen::MatrixXd::Zero(1, nPts);
                    sxc = Eigen::MatrixXd::Zero(1, nPts);
                    sigma = Eigen::MatrixXd::Zero(1, nPts);
                    for (int j = 0; j < nPts; j++) { sigma(j) = grad(j, 1) * grad(j, 1) + grad(j, 2) * grad(j, 2) + grad(j, 3) * grad(j, 3); }
                    xc_gga_exc_vxc(libxc_objects[i], nPts, density.data(), sigma.data(), exc.data(), vxc.data(), sxc.data());

                    for (int j = 0; j < nPts; ++j) {
                        //    xcfun calculates energy density per volume while libxc calculates
                        //    energy density per electron, so we multiply by the density here
                        out(0, j) += exc(0, j) * libxc_coefs[i] * density(j, 0);
                        out(1, j) += vxc(0, j) * libxc_coefs[i];
                        out(2, j) += 2 * sxc(0, j) * grad(j, 0) * libxc_coefs[i];
                        out(3, j) += 2 * sxc(0, j) * grad(j, 1) * libxc_coefs[i];
                        out(4, j) += 2 * sxc(0, j) * grad(j, 2) * libxc_coefs[i];
                    }
                }
                break;
            default:
                break;
        }
    }
}

int Libxc::getnOut() {
    bool lda = false;
    bool gga = false;
    bool mgga = false;
    for (const auto *f : libxc_objects) {
        getFamilyType(f->info->family, lda, gga, mgga);
    }
    if (gga) {
        if (spin) {
            return 9;
        } else {
            return 5;
        }
    } else if (lda) {
        if (spin) {
            return 3;
        } else {
            return 2;
        }
    }
    MSG_ABORT("Unable to determine Libxc output length for the configured functionals!\n");
    return 0;
}

} // mrdft
