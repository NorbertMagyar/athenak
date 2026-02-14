//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd_forced_box.cpp
//  \brief Problem generator for driven MHD turbulence with optional transverse
//  density structure and OU forcing

#include <cmath>
#include <iostream>

#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "diffusion/resistivity.hpp"
#include "diffusion/viscosity.hpp"
#include "eos/eos.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "mhd/mhd.hpp"
#include "pgen.hpp"
#include "srcterms/turb_driver.hpp"

namespace {
Real forced_rho0 = 1.0;
Real forced_gamma = 5.0/3.0;
Real forced_B0 = 1.0;
Real forced_eps = 0.0;
bool forced_density_variations = false;
}  // namespace

void ForcedBoxHistory(HistoryData *pdata, Mesh *pm);

//----------------------------------------------------------------------------------------
// Problem setup

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  user_hist_func = ForcedBoxHistory;
  forced_rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
  forced_eps = pin->GetOrAddReal("problem", "density_eps", 0.0);
  forced_density_variations =
      pin->GetOrAddBoolean("problem", "density_variations", false);
  forced_B0 = pin->GetOrAddReal("problem", "B0", 1.0);
  Real target_beta = pin->GetOrAddReal("problem", "beta", 0.5);
  Real p0_input = pin->GetOrAddReal("problem", "p0", -1.0);

  if (forced_density_variations && std::abs(forced_eps) >= 0.49) {
    std::cout << "### WARNING in " << __FILE__ << " : density_eps close to unity may "
              << "violate positivity; consider reducing |density_eps| < 0.5."
              << std::endl;
  }

  if (restart) {
    return;
  }

  if (pmbp->pmhd == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "mhd_forced_box requires <mhd> block in the input file." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto &u0 = pmbp->pmhd->u0;
  auto &b0 = pmbp->pmhd->b0;
  EOS_Data &eos = pmbp->pmhd->peos->eos_data;
  forced_gamma = eos.gamma;
  Real gm1 = eos.gamma - 1.0;
  Real p0 = p0_input;
  if (p0 <= 0.0) {
    p0 = 0.5*target_beta*forced_B0*forced_B0;
  }

  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int &nx1 = indcs.nx1;
  int &nx2 = indcs.nx2;
  int &nx3 = indcs.nx3;
  auto &size = pmbp->pmb->mb_size;
  Mesh *pm = pmy_mesh_;
  Real lx = pm->mesh_size.x1max - pm->mesh_size.x1min;
  Real ly = pm->mesh_size.x2max - pm->mesh_size.x2min;
  bool has_y = (nx2 > 1);
  Real kx = 2.0*M_PI/lx;
  Real ky = has_y ? 2.0*M_PI/ly : 0.0;

  Real rho0_loc = forced_rho0;
  Real eps_loc = forced_eps;
  bool vary_loc = forced_density_variations;
  Real B0_loc = forced_B0;

  par_for("forced_box_init", DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x = CellCenterX(i-is, nx1, x1min, x1max);
    Real y = CellCenterX(j-js, nx2, x2min, x2max);

    Real rho = rho0_loc;
    if (vary_loc) {
      Real modulation = 1.0 - eps_loc*(cos(2*kx*x) + (has_y ? cos(2*ky*y) : 0.0));
      rho = rho0_loc*modulation;
    }
    Real rho_floor = 1.0e-8*rho0_loc;
    if (rho < rho_floor) {
      rho = rho_floor;
    }

    u0(m,IDN,k,j,i) = rho;
    u0(m,IM1,k,j,i) = 0.0;
    u0(m,IM2,k,j,i) = 0.0;
    u0(m,IM3,k,j,i) = 0.0;
    if (eos.is_ideal) {
      u0(m,IEN,k,j,i) = p0/gm1 + 0.5*B0_loc*B0_loc;
    }

    b0.x1f(m,k,j,i) = 0.0;
    b0.x2f(m,k,j,i) = 0.0;
    b0.x3f(m,k,j,i) = B0_loc;
    if (i == ie) {
      b0.x1f(m,k,j,i+1) = 0.0;
    }
    if (j == je) {
      b0.x2f(m,k,j+1,i) = 0.0;
    }
    if (k == ke) {
      b0.x3f(m,k+1,j,i) = B0_loc;
    }
  });
}

//----------------------------------------------------------------------------------------
// History diagnostics

void ForcedBoxHistory(HistoryData *pdata, Mesh *pm) {
  auto *pack = pm->pmb_pack;
  if (pack->pmhd == nullptr) {
    pdata->nhist = 0;
    return;
  }

  auto &u0 = pack->pmhd->u0;
  auto &w0 = pack->pmhd->w0;
  auto &bcc = pack->pmhd->bcc0;
  auto *pturb = pack->pturb;
  DvceArray5D<Real> force;
  if (pturb != nullptr) {
    force = pturb->force;
  }
  Real nu = 0.0;
  if (pack->pmhd->pvisc != nullptr) {
    nu = pack->pmhd->pvisc->nu_iso;
  }
  Real eta = 0.0;
  if (pack->pmhd->presist != nullptr) {
    eta = pack->pmhd->presist->eta_ohm;
  }

  pdata->nhist = 12;
  pdata->label[0] = "E_kin";
  pdata->label[1] = "E_mag";
  pdata->label[2] = "E_int";
  pdata->label[3] = "E_tot";
  pdata->label[4] = "P_in";
  pdata->label[5] = "eps_nu";
  pdata->label[6] = "eps_eta";
  pdata->label[7] = "v_rms";
  pdata->label[8] = "b_rms";
  pdata->label[9] = "drho_rms";
  pdata->label[10] = "Mach_rms";
  pdata->label[11] = "cross_H";

  auto &size = pack->pmb->mb_size;
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;
  const int nmkji = (pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  bool has_force = (pturb != nullptr);
  bool need_grad = (nu > 0.0);
  bool need_j = (eta > 0.0);
  Real rho0_loc = forced_rho0;
  Real gamma_loc = forced_gamma;
  Real B0_loc = forced_B0;
  Real total_volume = (pm->mesh_size.x1max - pm->mesh_size.x1min) *
                      (pm->mesh_size.x2max - pm->mesh_size.x2min) *
                      (pm->mesh_size.x3max - pm->mesh_size.x3min);

  array_sum::GlobalSum sum_this_mb;
  Kokkos::parallel_reduce("forced_hist",Kokkos::RangePolicy<>(DevExeSpace(),0,nmkji),
  KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &mb_sum) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    Real dx1 = size.d_view(m).dx1;
    Real dx2 = size.d_view(m).dx2;
    Real dx3 = size.d_view(m).dx3;
    Real cell_vol = dx1*dx2*dx3;

    Real rho = u0(m,IDN,k,j,i);
    Real vx = w0(m,IVX,k,j,i);
    Real vy = w0(m,IVY,k,j,i);
    Real vz = w0(m,IVZ,k,j,i);
    Real v2 = vx*vx + vy*vy + vz*vz;
    Real bx = bcc(m,IBX,k,j,i);
    Real by = bcc(m,IBY,k,j,i);
    Real bz = bcc(m,IBZ,k,j,i);
    Real b2 = bx*bx + by*by + bz*bz;
    Real kin = 0.5*rho*v2;
    Real eint = u0(m,IEN,k,j,i) - kin - 0.5*b2;

    mb_sum.the_array[0] += kin*cell_vol;
    mb_sum.the_array[1] += 0.5*b2*cell_vol;
    mb_sum.the_array[2] += eint*cell_vol;
    mb_sum.the_array[3] += (kin + 0.5*b2 + eint)*cell_vol;

    Real pin_term = 0.0;
    if (has_force) {
      Real fx = force(m,0,k,j,i);
      Real fy = force(m,1,k,j,i);
      Real fz = force(m,2,k,j,i);
      pin_term = rho*(vx*fx + vy*fy + vz*fz);
    }
    mb_sum.the_array[4] += pin_term*cell_vol;

    Real grad_sq = 0.0;
    if (need_grad) {
      Real inv_dx1 = (nx1 > 1) ? 0.5/dx1 : 0.0;
      Real inv_dx2 = (nx2 > 1) ? 0.5/dx2 : 0.0;
      Real inv_dx3 = (nx3 > 1) ? 0.5/dx3 : 0.0;
      Real dvx_dx = (nx1 > 1) ? (w0(m,IVX,k,j,i+1) - w0(m,IVX,k,j,i-1))*inv_dx1 : 0.0;
      Real dvx_dy = (nx2 > 1) ? (w0(m,IVX,k,j+1,i) - w0(m,IVX,k,j-1,i))*inv_dx2 : 0.0;
      Real dvx_dz = (nx3 > 1) ? (w0(m,IVX,k+1,j,i) - w0(m,IVX,k-1,j,i))*inv_dx3 : 0.0;
      Real dvy_dx = (nx1 > 1) ? (w0(m,IVY,k,j,i+1) - w0(m,IVY,k,j,i-1))*inv_dx1 : 0.0;
      Real dvy_dy = (nx2 > 1) ? (w0(m,IVY,k,j+1,i) - w0(m,IVY,k,j-1,i))*inv_dx2 : 0.0;
      Real dvy_dz = (nx3 > 1) ? (w0(m,IVY,k+1,j,i) - w0(m,IVY,k-1,j,i))*inv_dx3 : 0.0;
      Real dvz_dx = (nx1 > 1) ? (w0(m,IVZ,k,j,i+1) - w0(m,IVZ,k,j,i-1))*inv_dx1 : 0.0;
      Real dvz_dy = (nx2 > 1) ? (w0(m,IVZ,k,j+1,i) - w0(m,IVZ,k,j-1,i))*inv_dx2 : 0.0;
      Real dvz_dz = (nx3 > 1) ? (w0(m,IVZ,k+1,j,i) - w0(m,IVZ,k-1,j,i))*inv_dx3 : 0.0;
      grad_sq = dvx_dx*dvx_dx + dvx_dy*dvx_dy + dvx_dz*dvx_dz
                + dvy_dx*dvy_dx + dvy_dy*dvy_dy + dvy_dz*dvy_dz
                + dvz_dx*dvz_dx + dvz_dy*dvz_dy + dvz_dz*dvz_dz;
    }
    mb_sum.the_array[5] += nu*rho*grad_sq*cell_vol;

    Real j_sq = 0.0;
    if (need_j) {
      Real inv_dx1 = (nx1 > 1) ? 0.5/dx1 : 0.0;
      Real inv_dx2 = (nx2 > 1) ? 0.5/dx2 : 0.0;
      Real inv_dx3 = (nx3 > 1) ? 0.5/dx3 : 0.0;
      Real dBy_dz = (nx3 > 1) ? (bcc(m,IBY,k+1,j,i) - bcc(m,IBY,k-1,j,i))*inv_dx3 : 0.0;
      Real dBz_dy = (nx2 > 1) ? (bcc(m,IBZ,k,j+1,i) - bcc(m,IBZ,k,j-1,i))*inv_dx2 : 0.0;
      Real dBz_dx = (nx1 > 1) ? (bcc(m,IBZ,k,j,i+1) - bcc(m,IBZ,k,j,i-1))*inv_dx1 : 0.0;
      Real dBx_dz = (nx3 > 1) ? (bcc(m,IBX,k+1,j,i) - bcc(m,IBX,k-1,j,i))*inv_dx3 : 0.0;
      Real dBy_dx = (nx1 > 1) ? (bcc(m,IBY,k,j,i+1) - bcc(m,IBY,k,j,i-1))*inv_dx1 : 0.0;
      Real dBx_dy = (nx2 > 1) ? (bcc(m,IBX,k,j+1,i) - bcc(m,IBX,k,j-1,i))*inv_dx2 : 0.0;
      Real Jx = dBz_dy - dBy_dz;
      Real Jy = dBx_dz - dBz_dx;
      Real Jz = dBy_dx - dBx_dy;
      j_sq = Jx*Jx + Jy*Jy + Jz*Jz;
    }
    mb_sum.the_array[6] += eta*j_sq*cell_vol;

    Real bx_fluc = bx;
    Real by_fluc = by;
    Real bz_fluc = bz - B0_loc;
    mb_sum.the_array[7] += v2*cell_vol;
    mb_sum.the_array[8] += (bx_fluc*bx_fluc + by_fluc*by_fluc + bz_fluc*bz_fluc)*cell_vol;
    Real drho = rho - rho0_loc;
    mb_sum.the_array[9] += drho*drho*cell_vol;
    Real press = w0(m,IPR,k,j,i);
    Real cs2 = (press > 0.0 && rho > 0.0) ? gamma_loc*press/rho : 1.0;
    mb_sum.the_array[10] += (cs2 > 0.0 ? v2/cs2 : 0.0)*cell_vol;
    mb_sum.the_array[11] += (vx*bx + vy*by + vz*bz)*cell_vol;
  }, Kokkos::Sum<array_sum::GlobalSum>(sum_this_mb));

  for (int n=0; n<pdata->nhist; ++n) {
    pdata->hdata[n] = sum_this_mb.the_array[n];
  }

  if (total_volume > 0.0) {
    pdata->hdata[7] = std::sqrt(pdata->hdata[7]/total_volume);
    pdata->hdata[8] = std::sqrt(pdata->hdata[8]/total_volume);
    pdata->hdata[9] = std::sqrt(pdata->hdata[9]/total_volume)/rho0_loc;
    pdata->hdata[10] = std::sqrt(pdata->hdata[10]/total_volume);
  }
}
