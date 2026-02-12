//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file turb_driver.cpp
//  \brief implementation of functions in TurbulenceDriver

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "ion-neutral/ion-neutral.hpp"
#include "driver/driver.hpp"
#include "utils/random.hpp"
#include "eos/eos.hpp"
#include "eos/ideal_c2p_hyd.hpp"
#include "eos/ideal_c2p_mhd.hpp"
#include "turb_driver.hpp"

//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

TurbulenceDriver::TurbulenceDriver(MeshBlockPack *pp, ParameterInput *pin) :
  pmy_pack(pp),
  force("force",1,1,1,1,1),
  force_tmp("force_tmp",1,1,1,1,1),
  force_plus("force_plus",1,1,1,1,1),
  force_minus("force_minus",1,1,1,1,1),
  force_plus_tmp("force_plus_tmp",1,1,1,1,1),
  force_minus_tmp("force_minus_tmp",1,1,1,1,1),
  bdrive("bdrive",1,1,1,1,1),
  emf_drive("emf_drive",1,1,1,1),
  xccc("xccc",1),xccs("xccs",1),xcsc("xcsc",1),xcss("xcss",1),
  xscc("xscc",1),xscs("xscs",1),xssc("xssc",1),xsss("xsss",1),
  yccc("yccc",1),yccs("yccs",1),ycsc("ycsc",1),ycss("ycss",1),
  yscc("yscc",1),yscs("yscs",1),yssc("yssc",1),ysss("ysss",1),
  zccc("zccc",1),zccs("zccs",1),zcsc("zcsc",1),zcss("zcss",1),
  zscc("zscc",1),zscs("zscs",1),zssc("zssc",1),zsss("zsss",1),
  kx_mode("kx_mode",1),ky_mode("ky_mode",1),kz_mode("kz_mode",1),
  xcos("xcos",1,1,1),xsin("xsin",1,1,1),ycos("ycos",1,1,1),
  ysin("ysin",1,1,1),zcos("zcos",1,1,1),zsin("zsin",1,1,1),
  xcos_edge("xcos_edge",1,1,1),xsin_edge("xsin_edge",1,1,1),
  ycos_edge("ycos_edge",1,1,1),ysin_edge("ysin_edge",1,1,1),
  psiccc("psiccc",1),psiccs("psiccs",1),psicsc("psicsc",1),psicss("psicss",1),
  psiscc("psiscc",1),psiscs("psiscs",1),psissc("psissc",1),psisss("psisss",1),
  bxccc("bxccc",1),bxccs("bxccs",1),bxcsc("bxcsc",1),bxcss("bxcss",1),
  bxscc("bxscc",1),bxscs("bxscs",1),bxssc("bxssc",1),bxsss("bxsss",1),
  byccc("byccc",1),byccs("byccs",1),bycsc("bycsc",1),bycss("bycss",1),
  byscc("byscc",1),byscs("byscs",1),byssc("byssc",1),bysss("bysss",1),
  norm_ccc("norm_ccc",1),norm_ccs("norm_ccs",1),norm_csc("norm_csc",1),
  norm_css("norm_css",1),norm_scc("norm_scc",1),norm_scs("norm_scs",1),
  norm_ssc("norm_ssc",1),norm_sss("norm_sss",1) {
  // allocate memory for force registers
  int nmb = pmy_pack->nmb_thispack;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;

  Kokkos::realloc(force, nmb, 3, ncells3, ncells2, ncells1);
  Kokkos::realloc(force_tmp, nmb, 3, ncells3, ncells2, ncells1);
  Kokkos::realloc(force_plus, nmb, 3, ncells3, ncells2, ncells1);
  Kokkos::realloc(force_minus, nmb, 3, ncells3, ncells2, ncells1);
  Kokkos::realloc(force_plus_tmp, nmb, 3, ncells3, ncells2, ncells1);
  Kokkos::realloc(force_minus_tmp, nmb, 3, ncells3, ncells2, ncells1);
  Kokkos::realloc(bdrive, nmb, 2, ncells3, ncells2, ncells1);
  Kokkos::realloc(emf_drive.x1e, nmb, ncells3+1, ncells2+1, ncells1);
  Kokkos::realloc(emf_drive.x2e, nmb, ncells3+1, ncells2, ncells1+1);
  Kokkos::realloc(emf_drive.x3e, nmb, ncells3, ncells2+1, ncells1+1);
  // range of modes including, corresponding to kmin and kmax
  nlow = pin->GetOrAddInteger("turb_driving", "nlow", 1);
  nhigh = pin->GetOrAddInteger("turb_driving", "nhigh", 2);
  // driving type
  driving_type = pin->GetOrAddInteger("turb_driving", "driving_type", 0);
  // power-law exponent for isotropic driving
  expo = pin->GetOrAddReal("turb_driving", "expo", 5.0/3.0);
  exp_prp = pin->GetOrAddReal("turb_driving", "exp_prp", 5.0/3.0);
  exp_prl = pin->GetOrAddReal("turb_driving", "exp_prl", 0.0);
  // energy injection rate
  dedt = pin->GetOrAddReal("turb_driving", "dedt", 0.0);
  // correlation time
  tcorr = pin->GetOrAddReal("turb_driving", "tcorr", 0.0);
  std::string ctrl = pin->GetOrAddString("turb_driving", "control", "power");
  if (ctrl == "accel" || ctrl == "acceleration" || ctrl == "amplitude") {
    control_mode = ForceControl::kAccel;
  } else {
    control_mode = ForceControl::kPower;
  }
  accel_rms = pin->GetOrAddReal("turb_driving", "accel_rms", 0.1);
  zplus_fraction = pin->GetOrAddReal("turb_driving", "zplus_fraction", 0.5);
  if (zplus_fraction < 0.0) zplus_fraction = 0.0;
  if (zplus_fraction > 1.0) zplus_fraction = 1.0;
  alfvenic_drive = pin->GetOrAddBoolean("turb_driving", "alfvenic_drive", false);
  force_perp_only = pin->GetOrAddBoolean("turb_driving", "force_perp_only", false);
  if (alfvenic_drive) force_perp_only = true;
  alfvenic_rho0 = pin->GetOrAddReal("turb_driving", "alfvenic_rho0", 1.0);
  if (alfvenic_rho0 <= 0.0) alfvenic_rho0 = 1.0;
  alfvenic_sqrt_rho0 = std::sqrt(alfvenic_rho0);

  Real nlow_sqr = nlow*nlow;
  Real nhigh_sqr = nhigh*nhigh;

  mode_count = 0;

  int nkx, nky, nkz;
  Real nsqr;
  for (nkx = 0; nkx <= nhigh; nkx++) {
    for (nky = 0; nky <= nhigh; nky++) {
      for (nkz = 0; nkz <= nhigh; nkz++) {
        if (nkx == 0 && nky == 0 && nkz == 0) continue;
        nsqr = 0.0;
        bool flag_prl = true;
        if (driving_type == 0) {
          nsqr = SQR(nkx) + SQR(nky) + SQR(nkz);
        } else if (driving_type == 1) {
          nsqr = SQR(nkx) + SQR(nky);
          Real nprlsqr = SQR(nkz);
          if (nprlsqr >= nlow_sqr && nprlsqr <= nhigh_sqr) {
            flag_prl = true;
          } else {
            flag_prl = false;
          }
        }
        if (nsqr >= nlow_sqr && nsqr <= nhigh_sqr && flag_prl) {
          mode_count++;
        }
      }
    }
  }

  Kokkos::realloc(xccc, mode_count);
  Kokkos::realloc(xccs, mode_count);
  Kokkos::realloc(xcsc, mode_count);
  Kokkos::realloc(xcss, mode_count);
  Kokkos::realloc(xscc, mode_count);
  Kokkos::realloc(xscs, mode_count);
  Kokkos::realloc(xssc, mode_count);
  Kokkos::realloc(xsss, mode_count);

  Kokkos::realloc(yccc, mode_count);
  Kokkos::realloc(yccs, mode_count);
  Kokkos::realloc(ycsc, mode_count);
  Kokkos::realloc(ycss, mode_count);
  Kokkos::realloc(yscc, mode_count);
  Kokkos::realloc(yscs, mode_count);
  Kokkos::realloc(yssc, mode_count);
  Kokkos::realloc(ysss, mode_count);

  Kokkos::realloc(zccc, mode_count);
  Kokkos::realloc(zccs, mode_count);
  Kokkos::realloc(zcsc, mode_count);
  Kokkos::realloc(zcss, mode_count);
  Kokkos::realloc(zscc, mode_count);
  Kokkos::realloc(zscs, mode_count);
  Kokkos::realloc(zssc, mode_count);
  Kokkos::realloc(zsss, mode_count);

  Kokkos::realloc(kx_mode, mode_count);
  Kokkos::realloc(ky_mode, mode_count);
  Kokkos::realloc(kz_mode, mode_count);

  Kokkos::realloc(xcos, nmb, mode_count, ncells1);
  Kokkos::realloc(xsin, nmb, mode_count, ncells1);
  Kokkos::realloc(ycos, nmb, mode_count, ncells2);
  Kokkos::realloc(ysin, nmb, mode_count, ncells2);
  Kokkos::realloc(zcos, nmb, mode_count, ncells3);
  Kokkos::realloc(zsin, nmb, mode_count, ncells3);
  Kokkos::realloc(xcos_edge, nmb, mode_count, ncells1+1);
  Kokkos::realloc(xsin_edge, nmb, mode_count, ncells1+1);
  Kokkos::realloc(ycos_edge, nmb, mode_count, std::max(1, ncells2+1));
  Kokkos::realloc(ysin_edge, nmb, mode_count, std::max(1, ncells2+1));
  Kokkos::realloc(psiccc, mode_count);
  Kokkos::realloc(psiccs, mode_count);
  Kokkos::realloc(psicsc, mode_count);
  Kokkos::realloc(psicss, mode_count);
  Kokkos::realloc(psiscc, mode_count);
  Kokkos::realloc(psiscs, mode_count);
  Kokkos::realloc(psissc, mode_count);
  Kokkos::realloc(psisss, mode_count);
  Kokkos::realloc(bxccc, mode_count);
  Kokkos::realloc(bxccs, mode_count);
  Kokkos::realloc(bxcsc, mode_count);
  Kokkos::realloc(bxcss, mode_count);
  Kokkos::realloc(bxscc, mode_count);
  Kokkos::realloc(bxscs, mode_count);
  Kokkos::realloc(bxssc, mode_count);
  Kokkos::realloc(bxsss, mode_count);
  Kokkos::realloc(byccc, mode_count);
  Kokkos::realloc(byccs, mode_count);
  Kokkos::realloc(bycsc, mode_count);
  Kokkos::realloc(bycss, mode_count);
  Kokkos::realloc(byscc, mode_count);
  Kokkos::realloc(byscs, mode_count);
  Kokkos::realloc(byssc, mode_count);
  Kokkos::realloc(bysss, mode_count);
  Kokkos::realloc(norm_ccc, mode_count);
  Kokkos::realloc(norm_ccs, mode_count);
  Kokkos::realloc(norm_csc, mode_count);
  Kokkos::realloc(norm_css, mode_count);
  Kokkos::realloc(norm_scc, mode_count);
  Kokkos::realloc(norm_scs, mode_count);
  Kokkos::realloc(norm_ssc, mode_count);
  Kokkos::realloc(norm_sss, mode_count);

  Initialize();
}

//----------------------------------------------------------------------------------------
// destructor

TurbulenceDriver::~TurbulenceDriver() {
}

//----------------------------------------------------------------------------------------
//! \fn  noid Initialize
//  \brief Function to initialize the driver

void TurbulenceDriver::Initialize() {
  Mesh *pm = pmy_pack->pmesh;
  int nmb = pmy_pack->nmb_thispack;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  int &nx1 = indcs.nx1;
  int &nx2 = indcs.nx2;
  int &nx3 = indcs.nx3;

  auto force_ = force;
  auto force_tmp_ = force_tmp;
  auto force_plus_ = force_plus;
  auto force_minus_ = force_minus;
  auto force_plus_tmp_ = force_plus_tmp;
  auto force_minus_tmp_ = force_minus_tmp;
  par_for("force_init_pgen",DevExeSpace(),
          0,nmb-1,0,2,0,ncells3-1,0,ncells2-1,0,ncells1-1,
  KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
    force_(m,n,k,j,i) = 0.0;
    force_tmp_(m,n,k,j,i) = 0.0;
    force_plus_(m,n,k,j,i) = 0.0;
    force_minus_(m,n,k,j,i) = 0.0;
    force_plus_tmp_(m,n,k,j,i) = 0.0;
    force_minus_tmp_(m,n,k,j,i) = 0.0;
  });
  auto bdrive_ = bdrive;
  par_for("bdrive_init_pgen",DevExeSpace(),
          0,nmb-1,0,1,0,ncells3-1,0,ncells2-1,0,ncells1-1,
  KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
    bdrive_(m,n,k,j,i) = 0.0;
  });
  auto emf1_ = emf_drive.x1e;
  auto emf2_ = emf_drive.x2e;
  auto emf3_ = emf_drive.x3e;
  par_for("emf_init_x1", DevExeSpace(), 0, nmb-1, 0, ncells3, 0, ncells2, 0, ncells1-1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    emf1_(m,k,j,i) = 0.0;
  });
  par_for("emf_init_x2", DevExeSpace(), 0, nmb-1, 0, ncells3, 0, ncells2-1, 0, ncells1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    emf2_(m,k,j,i) = 0.0;
  });
  par_for("emf_init_x3", DevExeSpace(), 0, nmb-1, 0, ncells3-1, 0, ncells2, 0, ncells1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    emf3_(m,k,j,i) = 0.0;
  });

  rstate.idum = -1;

  auto kx_mode_ = kx_mode;
  auto ky_mode_ = ky_mode;
  auto kz_mode_ = kz_mode;

  auto xcos_ = xcos;
  auto xsin_ = xsin;
  auto ycos_ = ycos;
  auto ysin_ = ysin;
  auto zcos_ = zcos;
  auto zsin_ = zsin;

  Real dkx, dky, dkz, kx, ky, kz;
  Real lx = pm->mesh_size.x1max - pm->mesh_size.x1min;
  Real ly = pm->mesh_size.x2max - pm->mesh_size.x2min;
  Real lz = pm->mesh_size.x3max - pm->mesh_size.x3min;
  dkx = 2.0*M_PI/lx;
  dky = 2.0*M_PI/ly;
  dkz = 2.0*M_PI/lz;

  int nmode = 0;
  int nkx, nky, nkz;
  Real nsqr;
  Real nlow_sqr = nlow*nlow;
  Real nhigh_sqr = nhigh*nhigh;
  for (nkx = 0; nkx <= nhigh; nkx++) {
    for (nky = 0; nky <= nhigh; nky++) {
      for (nkz = 0; nkz <= nhigh; nkz++) {
        if (nkx == 0 && nky == 0 && nkz == 0) continue;
        nsqr = 0.0;
        bool flag_prl = true;
        if (driving_type == 0) {
          nsqr = SQR(nkx) + SQR(nky) + SQR(nkz);
        } else if (driving_type == 1) {
          nsqr = SQR(nkx) + SQR(nky);
          Real nprlsqr = SQR(nkz);
          if (nprlsqr >= nlow_sqr && nprlsqr <= nhigh_sqr) {
            flag_prl = true;
          } else {
            flag_prl = false;
          }
        }
        if (nsqr >= nlow_sqr && nsqr <= nhigh_sqr && flag_prl) {
          kx = dkx*nkx;
          ky = dky*nky;
          kz = dkz*nkz;
          kx_mode_.h_view(nmode) = kx;
          ky_mode_.h_view(nmode) = ky;
          kz_mode_.h_view(nmode) = kz;
          nmode++;
        }
      }
    }
  }

  kx_mode_.template modify<HostMemSpace>();
  kx_mode_.template sync<DevExeSpace>();
  ky_mode_.template modify<HostMemSpace>();
  ky_mode_.template sync<DevExeSpace>();
  kz_mode_.template modify<HostMemSpace>();
  kz_mode_.template sync<DevExeSpace>();

  auto &size = pmy_pack->pmb->mb_size;

  par_for("xsin/xcos", DevExeSpace(),0,nmb-1,0,mode_count-1,is,ie,
  KOKKOS_LAMBDA(int m, int n, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
    Real k1v = kx_mode_.d_view(n);
    xsin_(m,n,i) = sin(k1v*x1v);
    xcos_(m,n,i) = cos(k1v*x1v);
  });

  par_for("ysin/ycos", DevExeSpace(),0,nmb-1,0,mode_count-1,js,je,
  KOKKOS_LAMBDA(int m, int n, int j) {
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
    Real k2v = ky_mode_.d_view(n);
    ysin_(m,n,j) = sin(k2v*x2v);
    ycos_(m,n,j) = cos(k2v*x2v);
    if (ncells2-1 == 0) {
      ysin_(m,n,j) = 0.0;
      ycos_(m,n,j) = 1.0;
    }
  });

  par_for("zsin/zcos", DevExeSpace(),0,nmb-1,0,mode_count-1,ks,ke,
  KOKKOS_LAMBDA(int m, int n, int k) {
    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
    Real k3v = kz_mode_.d_view(n);
    zsin_(m,n,k) = sin(k3v*x3v);
    zcos_(m,n,k) = cos(k3v*x3v);
    if (ncells3-1 == 0) {
      zsin_(m,n,k) = 0.0;
      zcos_(m,n,k) = 1.0;
    }
  });

  auto xcos_edge_ = xcos_edge;
  auto xsin_edge_ = xsin_edge;
  par_for("xedge_trig", DevExeSpace(),0,nmb-1,0,mode_count-1,is,ie+1,
  KOKKOS_LAMBDA(int m, int n, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1f = LeftEdgeX(i-is, nx1, x1min, x1max);
    Real k1v = kx_mode_.d_view(n);
    xsin_edge_(m,n,i) = sin(k1v*x1f);
    xcos_edge_(m,n,i) = cos(k1v*x1f);
  });

  auto ycos_edge_ = ycos_edge;
  auto ysin_edge_ = ysin_edge;
  par_for("yedge_trig", DevExeSpace(),0,nmb-1,0,mode_count-1,js,je+1,
  KOKKOS_LAMBDA(int m, int n, int j) {
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2f = LeftEdgeX(j-js, nx2, x2min, x2max);
    Real k2v = ky_mode_.d_view(n);
    if (ncells2-1 == 0) {
      ysin_edge_(m,n,j) = 0.0;
      ycos_edge_(m,n,j) = 1.0;
    } else {
      ysin_edge_(m,n,j) = sin(k2v*x2f);
      ycos_edge_(m,n,j) = cos(k2v*x2f);
    }
  });

  ComputeBasisNorms();

  return;
}

void TurbulenceDriver::ComputeBasisNorms() {
  auto xcos_ = xcos;
  auto xsin_ = xsin;
  auto ycos_ = ycos;
  auto ysin_ = ysin;
  auto zcos_ = zcos;
  auto zsin_ = zsin;
  auto norm_ccc_ = norm_ccc;
  auto norm_ccs_ = norm_ccs;
  auto norm_csc_ = norm_csc;
  auto norm_css_ = norm_css;
  auto norm_scc_ = norm_scc;
  auto norm_scs_ = norm_scs;
  auto norm_ssc_ = norm_ssc;
  auto norm_sss_ = norm_sss;

  Kokkos::deep_copy(norm_ccc_.d_view, 0.0);
  Kokkos::deep_copy(norm_ccs_.d_view, 0.0);
  Kokkos::deep_copy(norm_csc_.d_view, 0.0);
  Kokkos::deep_copy(norm_css_.d_view, 0.0);
  Kokkos::deep_copy(norm_scc_.d_view, 0.0);
  Kokkos::deep_copy(norm_scs_.d_view, 0.0);
  Kokkos::deep_copy(norm_ssc_.d_view, 0.0);
  Kokkos::deep_copy(norm_sss_.d_view, 0.0);

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int &nmb = pmy_pack->nmb_thispack;

  int mode_count_ = mode_count;
  par_for("basis_norms", DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    for (int n=0; n<mode_count_; ++n) {
      Real xc = xcos_(m,n,i);
      Real xs = xsin_(m,n,i);
      Real yc = ycos_(m,n,j);
      Real ys = ysin_(m,n,j);
      Real zc = zcos_(m,n,k);
      Real zs = zsin_(m,n,k);
      Real v;

      v = xc*yc*zc;
      Kokkos::atomic_add(&(norm_ccc_.d_view(n)), v*v);
      v = xc*yc*zs;
      Kokkos::atomic_add(&(norm_ccs_.d_view(n)), v*v);
      v = xc*ys*zc;
      Kokkos::atomic_add(&(norm_csc_.d_view(n)), v*v);
      v = xc*ys*zs;
      Kokkos::atomic_add(&(norm_css_.d_view(n)), v*v);
      v = xs*yc*zc;
      Kokkos::atomic_add(&(norm_scc_.d_view(n)), v*v);
      v = xs*yc*zs;
      Kokkos::atomic_add(&(norm_scs_.d_view(n)), v*v);
      v = xs*ys*zc;
      Kokkos::atomic_add(&(norm_ssc_.d_view(n)), v*v);
      v = xs*ys*zs;
      Kokkos::atomic_add(&(norm_sss_.d_view(n)), v*v);
    }
  });

  norm_ccc_.template sync<HostMemSpace>();
  norm_ccs_.template sync<HostMemSpace>();
  norm_csc_.template sync<HostMemSpace>();
  norm_css_.template sync<HostMemSpace>();
  norm_scc_.template sync<HostMemSpace>();
  norm_scs_.template sync<HostMemSpace>();
  norm_ssc_.template sync<HostMemSpace>();
  norm_sss_.template sync<HostMemSpace>();

#if MPI_PARALLEL_ENABLED
  auto reduce_norm = [&](DualArray1D<Real> &arr) {
    std::vector<Real> send(mode_count), recv(mode_count);
    for (int n=0; n<mode_count; ++n) send[n] = arr.h_view(n);
    MPI_Allreduce(send.data(), recv.data(), mode_count, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    arr.template modify<HostMemSpace>();
    for (int n=0; n<mode_count; ++n) arr.h_view(n) = recv[n];
  };
  reduce_norm(norm_ccc_);
  reduce_norm(norm_ccs_);
  reduce_norm(norm_csc_);
  reduce_norm(norm_css_);
  reduce_norm(norm_scc_);
  reduce_norm(norm_scs_);
  reduce_norm(norm_ssc_);
  reduce_norm(norm_sss_);
#endif
}
//----------------------------------------------------------------------------------------
//! \fn  void IncludeModeEvolutionTasks
//  \brief Includes task in the operator split task list that constructs new modes with
//  random amplitudes and phases that can be used to evolve the force via an O-U process
//  Called by MeshBlockPack::AddPhysics() function

void TurbulenceDriver::IncludeInitializeModesTask(std::shared_ptr<TaskList> tl,
                                                  TaskID start) {
  auto id_init = tl->AddTask(&TurbulenceDriver::InitializeModes, this, start);
  auto id_add = tl->AddTask(&TurbulenceDriver::AddForcing, this, id_init);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void IncludeAddForcingTask
//  \brief includes task in the stage_run task list for adding random forcing to fluid
//  as an explicit source terms in each stage of integrator
//  Called by MeshBlockPack::AddPhysics() function

void TurbulenceDriver::IncludeAddForcingTask(std::shared_ptr<TaskList> tl, TaskID start) {
  // These must be inserted after update task, but before send_u
  if (pmy_pack->pionn == nullptr) {
    if (pmy_pack->phydro != nullptr) {
      auto id = tl->InsertTask(&TurbulenceDriver::AddForcing, this,
                              pmy_pack->phydro->id.flux, pmy_pack->phydro->id.rkupdt);
    }
    if (pmy_pack->pmhd != nullptr) {
      auto id = tl->InsertTask(&TurbulenceDriver::AddForcing, this,
                              pmy_pack->pmhd->id.flux, pmy_pack->pmhd->id.rkupdt);
    }
  } else {
    auto id = tl->InsertTask(&TurbulenceDriver::AddForcing, this,
                            pmy_pack->pionn->id.n_flux, pmy_pack->pionn->id.n_rkupdt);
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn InitializeModes()
// \brief Initializes driving, and so is only executed once at start of calc.
// Cannot be included in constructor since (it seems) Kokkos::par_for not allowed in cons.

TaskStatus TurbulenceDriver::InitializeModes(Driver *pdrive, int stage) {
  Mesh *pm = pmy_pack->pmesh;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int &nx1 = indcs.nx1;
  int &nx2 = indcs.nx2;
  int &nx3 = indcs.nx3;
  auto &gindcs = pm->mesh_indcs;
  int &gnx1 = gindcs.nx1;
  int &gnx2 = gindcs.nx2;
  int &gnx3 = gindcs.nx3;

  // Now compute new force using new random amplitudes and phases

  // Zero out new force array
  auto force_tmp_ = force_tmp;
  int &nmb = pmy_pack->nmb_thispack;
  par_for("force_init", DevExeSpace(),0,nmb-1,0,2,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
    force_tmp_(m,n,k,j,i) = 0.0;
  });

  int nlow_sqr = SQR(nlow);
  int nhigh_sqr = SQR(nhigh);
  auto mode_count_ = mode_count;

  auto xccc_ = xccc;
  auto xccs_ = xccs;
  auto xcsc_ = xcsc;
  auto xcss_ = xcss;
  auto xscc_ = xscc;
  auto xscs_ = xscs;
  auto xssc_ = xssc;
  auto xsss_ = xsss;

  auto yccc_ = yccc;
  auto yccs_ = yccs;
  auto ycsc_ = ycsc;
  auto ycss_ = ycss;
  auto yscc_ = yscc;
  auto yscs_ = yscs;
  auto yssc_ = yssc;
  auto ysss_ = ysss;

  auto zccc_ = zccc;
  auto zccs_ = zccs;
  auto zcsc_ = zcsc;
  auto zcss_ = zcss;
  auto zscc_ = zscc;
  auto zscs_ = zscs;
  auto zssc_ = zssc;
  auto zsss_ = zsss;

  Real dkx, dky, dkz, kx, ky, kz;
  Real iky, ikz;
  Real lx = pm->mesh_size.x1max - pm->mesh_size.x1min;
  Real ly = pm->mesh_size.x2max - pm->mesh_size.x2min;
  Real lz = pm->mesh_size.x3max - pm->mesh_size.x3min;
  dkx = 2.0*M_PI/lx;
  dky = 2.0*M_PI/ly;
  dkz = 2.0*M_PI/lz;

  Real &ex = expo;
  Real &ex_prp = exp_prp;
  Real &ex_prl = exp_prl;
  Real norm, kprl, kprp, kiso;

  int nmode = 0;
  int nkx, nky, nkz, nsqr;
  for (nkx = 0; nkx <= nhigh; nkx++) {
    for (nky = 0; nky <= nhigh; nky++) {
      for (nkz = 0; nkz <= nhigh; nkz++) {
        if (nkx == 0 && nky == 0 && nkz == 0) continue;
        norm = 0.0;
        nsqr = 0.0;
        bool flag_prl = true;
        if (driving_type == 0) {
          nsqr = SQR(nkx) + SQR(nky) + SQR(nkz);
        } else if (driving_type == 1) {
          nsqr = SQR(nkx) + SQR(nky);
          Real nprlsqr = SQR(nkz);
          if (nprlsqr >= nlow_sqr && nprlsqr <= nhigh_sqr) {
            flag_prl = true;
          } else {
            flag_prl = false;
          }
        }
        if (nsqr >= nlow_sqr && nsqr <= nhigh_sqr && flag_prl) {
          kx = dkx*nkx;
          ky = dky*nky;
          kz = dkz*nkz;

          // Generate Fourier amplitudes
          if (driving_type == 0) {
            kiso = sqrt(SQR(kx) + SQR(ky) + SQR(kz));
            if (kiso > 1e-16) {
              norm = 1.0/pow(kiso,(ex+2.0)/2.0);
            } else {
              norm = 0.0;
            }
            if (nkz != 0) {
              ikz = 1.0/(dkz*((Real) nkz));

              xccc_.h_view(nmode) = RanGaussianSt(&(rstate));
              xccs_.h_view(nmode) = RanGaussianSt(&(rstate));
              xcsc_.h_view(nmode) = (nky==0)           ? 0.0 : RanGaussianSt(&(rstate));
              xcss_.h_view(nmode) = (nky==0)           ? 0.0 : RanGaussianSt(&(rstate));
              xscc_.h_view(nmode) = (nkx==0)           ? 0.0 : RanGaussianSt(&(rstate));
              xscs_.h_view(nmode) = (nkx==0)           ? 0.0 : RanGaussianSt(&(rstate));
              xssc_.h_view(nmode) = (nkx==0 || nky==0) ? 0.0 : RanGaussianSt(&(rstate));
              xsss_.h_view(nmode) = (nkx==0 || nky==0) ? 0.0 : RanGaussianSt(&(rstate));

              yccc_.h_view(nmode) = RanGaussianSt(&(rstate));
              yccs_.h_view(nmode) = RanGaussianSt(&(rstate));
              ycsc_.h_view(nmode) = (nky==0)           ? 0.0 : RanGaussianSt(&(rstate));
              ycss_.h_view(nmode) = (nky==0)           ? 0.0 : RanGaussianSt(&(rstate));
              yscc_.h_view(nmode) = (nkx==0)           ? 0.0 : RanGaussianSt(&(rstate));
              yscs_.h_view(nmode) = (nkx==0)           ? 0.0 : RanGaussianSt(&(rstate));
              yssc_.h_view(nmode) = (nkx==0 || nky==0) ? 0.0 : RanGaussianSt(&(rstate));
              ysss_.h_view(nmode) = (nkx==0 || nky==0) ? 0.0 : RanGaussianSt(&(rstate));

              // imcompressibility
              zccc_.h_view(nmode) =  ikz*( kx*xscs_.h_view(nmode)+ky*ycss_.h_view(nmode));
              zccs_.h_view(nmode) = -ikz*( kx*xscc_.h_view(nmode)+ky*ycsc_.h_view(nmode));
              zcsc_.h_view(nmode) =  ikz*( kx*xsss_.h_view(nmode)-ky*yccs_.h_view(nmode));
              zcss_.h_view(nmode) =  ikz*(-kx*xssc_.h_view(nmode)+ky*yccc_.h_view(nmode));
              zscc_.h_view(nmode) =  ikz*(-kx*xccs_.h_view(nmode)+ky*ysss_.h_view(nmode));
              zscs_.h_view(nmode) =  ikz*( kx*xccc_.h_view(nmode)-ky*yssc_.h_view(nmode));
              zssc_.h_view(nmode) = -ikz*( kx*xcss_.h_view(nmode)+ky*yscs_.h_view(nmode));
              zsss_.h_view(nmode) =  ikz*( kx*xcsc_.h_view(nmode)+ky*yscc_.h_view(nmode));
            } else if (nky != 0) {  // kz == 0
              iky = 1.0/(dky*((Real) nky));

              xccc_.h_view(nmode) = RanGaussianSt(&(rstate));
              xcsc_.h_view(nmode) = RanGaussianSt(&(rstate));
              xscc_.h_view(nmode) = (nkx==0) ? 0.0 : RanGaussianSt(&(rstate));
              xssc_.h_view(nmode) = (nkx==0) ? 0.0 : RanGaussianSt(&(rstate));
              xccs_.h_view(nmode) = 0.0;
              xscs_.h_view(nmode) = 0.0;
              xcss_.h_view(nmode) = 0.0;
              xsss_.h_view(nmode) = 0.0;

              zccc_.h_view(nmode) = RanGaussianSt(&(rstate));
              zcsc_.h_view(nmode) = RanGaussianSt(&(rstate));
              zscc_.h_view(nmode) = (nkx==0) ? 0.0 : RanGaussianSt(&(rstate));
              zssc_.h_view(nmode) = (nkx==0) ? 0.0 : RanGaussianSt(&(rstate));
              zccs_.h_view(nmode) = 0.0;
              zcss_.h_view(nmode) = 0.0;
              zscs_.h_view(nmode) = 0.0;
              zsss_.h_view(nmode) = 0.0;

              // incompressibility
              yccc_.h_view(nmode) =  iky*kx*xssc_.h_view(nmode);
              ycsc_.h_view(nmode) = -iky*kx*xscc_.h_view(nmode);
              yscc_.h_view(nmode) = -iky*kx*xcsc_.h_view(nmode);
              yssc_.h_view(nmode) =  iky*kx*xccc_.h_view(nmode);
              yccs_.h_view(nmode) = 0.0;
              ycss_.h_view(nmode) = 0.0;
              yscs_.h_view(nmode) = 0.0;
              ysss_.h_view(nmode) = 0.0;
            } else {  // kz == ky == 0, kx != 0 by initial if statement
              zccc_.h_view(nmode) = RanGaussianSt(&(rstate));
              zscc_.h_view(nmode) = RanGaussianSt(&(rstate));
              zcsc_.h_view(nmode) = 0.0;
              zssc_.h_view(nmode) = 0.0;
              zccs_.h_view(nmode) = 0.0;
              zcss_.h_view(nmode) = 0.0;
              zscs_.h_view(nmode) = 0.0;
              zsss_.h_view(nmode) = 0.0;

              yccc_.h_view(nmode) = RanGaussianSt(&(rstate));
              yscc_.h_view(nmode) = RanGaussianSt(&(rstate));
              ycsc_.h_view(nmode) = 0.0;
              yssc_.h_view(nmode) = 0.0;
              yccs_.h_view(nmode) = 0.0;
              ycss_.h_view(nmode) = 0.0;
              yscs_.h_view(nmode) = 0.0;
              ysss_.h_view(nmode) = 0.0;

              // incompressibility
              xccc_.h_view(nmode) = 0.0;
              xscc_.h_view(nmode) = 0.0;
              xcsc_.h_view(nmode) = 0.0;
              xssc_.h_view(nmode) = 0.0;
              xccs_.h_view(nmode) = 0.0;
              xscs_.h_view(nmode) = 0.0;
              xcss_.h_view(nmode) = 0.0;
              xsss_.h_view(nmode) = 0.0;
            }
          } else if (driving_type == 1) {
            kprl = sqrt(SQR(kx));
            kprp = sqrt(SQR(ky) + SQR(kz));
            if (kprl > 1e-16 && kprp > 1e-16) {
              norm = 1.0/pow(kprp,(ex_prp+1.0)/2.0)/pow(kprl,ex_prl/2.0);
            } else {
              norm = 0.0;
            }

            if (nky != 0) {
              iky = 1.0/(dky*((Real) nky));

              xccc_.h_view(nmode) = RanGaussianSt(&(rstate));
              xccs_.h_view(nmode) = RanGaussianSt(&(rstate));
              xcsc_.h_view(nmode) = RanGaussianSt(&(rstate));
              xcss_.h_view(nmode) = RanGaussianSt(&(rstate));
              xscc_.h_view(nmode) = (nkx==0) ? 0.0 : RanGaussianSt(&(rstate));
              xscs_.h_view(nmode) = (nkx==0) ? 0.0 : RanGaussianSt(&(rstate));
              xssc_.h_view(nmode) = (nkx==0) ? 0.0 : RanGaussianSt(&(rstate));
              xsss_.h_view(nmode) = (nkx==0) ? 0.0 : RanGaussianSt(&(rstate));

              // incompressibility
              yccc_.h_view(nmode) =  iky*(kx*xssc_.h_view(nmode));
              yccs_.h_view(nmode) =  iky*(kx*xsss_.h_view(nmode));
              ycsc_.h_view(nmode) = -iky*(kx*xscc_.h_view(nmode));
              ycss_.h_view(nmode) = -iky*(kx*xscs_.h_view(nmode));
              yscc_.h_view(nmode) = -iky*(kx*xcsc_.h_view(nmode));
              yscs_.h_view(nmode) = -iky*(kx*xcss_.h_view(nmode));
              yssc_.h_view(nmode) =  iky*(kx*xccc_.h_view(nmode));
              ysss_.h_view(nmode) =  iky*(kx*xccs_.h_view(nmode));

              zccc_.h_view(nmode) = 0.0;
              zccs_.h_view(nmode) = 0.0;
              zcsc_.h_view(nmode) = 0.0;
              zcss_.h_view(nmode) = 0.0;
              zscc_.h_view(nmode) = 0.0;
              zscs_.h_view(nmode) = 0.0;
              zssc_.h_view(nmode) = 0.0;
              zsss_.h_view(nmode) = 0.0;
            } else {  // ky == 0
              yccc_.h_view(nmode) = RanGaussianSt(&(rstate));
              yscc_.h_view(nmode) = RanGaussianSt(&(rstate));
              ycsc_.h_view(nmode) = 0.0;
              yssc_.h_view(nmode) = 0.0;
              yccs_.h_view(nmode) = 0.0;
              ycss_.h_view(nmode) = 0.0;
              yscs_.h_view(nmode) = 0.0;
              ysss_.h_view(nmode) = 0.0;

              // incompressibility
              xccc_.h_view(nmode) = 0.0;
              xscc_.h_view(nmode) = 0.0;
              xcsc_.h_view(nmode) = 0.0;
              xssc_.h_view(nmode) = 0.0;
              xccs_.h_view(nmode) = 0.0;
              xscs_.h_view(nmode) = 0.0;
              xcss_.h_view(nmode) = 0.0;
              xsss_.h_view(nmode) = 0.0;

              zccc_.h_view(nmode) = 0.0;
              zscc_.h_view(nmode) = 0.0;
              zcsc_.h_view(nmode) = 0.0;
              zssc_.h_view(nmode) = 0.0;
              zccs_.h_view(nmode) = 0.0;
              zcss_.h_view(nmode) = 0.0;
              zscs_.h_view(nmode) = 0.0;
              zsss_.h_view(nmode) = 0.0;
            }
          }
          // normalization
          xccc_.h_view(nmode) *= norm;
          xscc_.h_view(nmode) *= norm;
          xcsc_.h_view(nmode) *= norm;
          xssc_.h_view(nmode) *= norm;
          xccs_.h_view(nmode) *= norm;
          xscs_.h_view(nmode) *= norm;
          xcss_.h_view(nmode) *= norm;
          xsss_.h_view(nmode) *= norm;
          yccc_.h_view(nmode) *= norm;
          yscc_.h_view(nmode) *= norm;
          ycsc_.h_view(nmode) *= norm;
          yssc_.h_view(nmode) *= norm;
          yccs_.h_view(nmode) *= norm;
          yscs_.h_view(nmode) *= norm;
          ycss_.h_view(nmode) *= norm;
          ysss_.h_view(nmode) *= norm;
          zccc_.h_view(nmode) *= norm;
          zscc_.h_view(nmode) *= norm;
          zcsc_.h_view(nmode) *= norm;
          zssc_.h_view(nmode) *= norm;
          zccs_.h_view(nmode) *= norm;
          zscs_.h_view(nmode) *= norm;
          zcss_.h_view(nmode) *= norm;
          zsss_.h_view(nmode) *= norm;

          nmode++;
        }
      }
    }
  }

  xccc_.template modify<HostMemSpace>();
  xccc_.template sync<DevExeSpace>();
  xccs_.template modify<HostMemSpace>();
  xccs_.template sync<DevExeSpace>();
  xcsc_.template modify<HostMemSpace>();
  xcsc_.template sync<DevExeSpace>();
  xcss_.template modify<HostMemSpace>();
  xcss_.template sync<DevExeSpace>();
  xscc_.template modify<HostMemSpace>();
  xscc_.template sync<DevExeSpace>();
  xscs_.template modify<HostMemSpace>();
  xscs_.template sync<DevExeSpace>();
  xssc_.template modify<HostMemSpace>();
  xssc_.template sync<DevExeSpace>();
  xsss_.template modify<HostMemSpace>();
  xsss_.template sync<DevExeSpace>();

  yccc_.template modify<HostMemSpace>();
  yccc_.template sync<DevExeSpace>();
  yccs_.template modify<HostMemSpace>();
  yccs_.template sync<DevExeSpace>();
  ycsc_.template modify<HostMemSpace>();
  ycsc_.template sync<DevExeSpace>();
  ycss_.template modify<HostMemSpace>();
  ycss_.template sync<DevExeSpace>();
  yscc_.template modify<HostMemSpace>();
  yscc_.template sync<DevExeSpace>();
  yscs_.template modify<HostMemSpace>();
  yscs_.template sync<DevExeSpace>();
  yssc_.template modify<HostMemSpace>();
  yssc_.template sync<DevExeSpace>();
  ysss_.template modify<HostMemSpace>();
  ysss_.template sync<DevExeSpace>();

  zccc_.template modify<HostMemSpace>();
  zccc_.template sync<DevExeSpace>();
  zccs_.template modify<HostMemSpace>();
  zccs_.template sync<DevExeSpace>();
  zcsc_.template modify<HostMemSpace>();
  zcsc_.template sync<DevExeSpace>();
  zcss_.template modify<HostMemSpace>();
  zcss_.template sync<DevExeSpace>();
  zscc_.template modify<HostMemSpace>();
  zscc_.template sync<DevExeSpace>();
  zscs_.template modify<HostMemSpace>();
  zscs_.template sync<DevExeSpace>();
  zssc_.template modify<HostMemSpace>();
  zssc_.template sync<DevExeSpace>();
  zsss_.template modify<HostMemSpace>();
  zsss_.template sync<DevExeSpace>();

  auto xcos_ = xcos;
  auto xsin_ = xsin;
  auto ycos_ = ycos;
  auto ysin_ = ysin;
  auto zcos_ = zcos;
  auto zsin_ = zsin;

  for (int n=0; n<mode_count_; n++) {
    par_for("force_compute", DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      force_tmp_(m,0,k,j,i) += xccc_.d_view(n)*xcos_(m,n,i)*ycos_(m,n,j)*zcos_(m,n,k);
      force_tmp_(m,0,k,j,i) += xccs_.d_view(n)*xcos_(m,n,i)*ycos_(m,n,j)*zsin_(m,n,k);
      force_tmp_(m,0,k,j,i) += xcsc_.d_view(n)*xcos_(m,n,i)*ysin_(m,n,j)*zcos_(m,n,k);
      force_tmp_(m,0,k,j,i) += xcss_.d_view(n)*xcos_(m,n,i)*ysin_(m,n,j)*zsin_(m,n,k);
      force_tmp_(m,0,k,j,i) += xscc_.d_view(n)*xsin_(m,n,i)*ycos_(m,n,j)*zcos_(m,n,k);
      force_tmp_(m,0,k,j,i) += xscs_.d_view(n)*xsin_(m,n,i)*ycos_(m,n,j)*zsin_(m,n,k);
      force_tmp_(m,0,k,j,i) += xssc_.d_view(n)*xsin_(m,n,i)*ysin_(m,n,j)*zcos_(m,n,k);
      force_tmp_(m,0,k,j,i) += xsss_.d_view(n)*xsin_(m,n,i)*ysin_(m,n,j)*zsin_(m,n,k);

      force_tmp_(m,1,k,j,i) += yccc_.d_view(n)*xcos_(m,n,i)*ycos_(m,n,j)*zcos_(m,n,k);
      force_tmp_(m,1,k,j,i) += yccs_.d_view(n)*xcos_(m,n,i)*ycos_(m,n,j)*zsin_(m,n,k);
      force_tmp_(m,1,k,j,i) += ycsc_.d_view(n)*xcos_(m,n,i)*ysin_(m,n,j)*zcos_(m,n,k);
      force_tmp_(m,1,k,j,i) += ycss_.d_view(n)*xcos_(m,n,i)*ysin_(m,n,j)*zsin_(m,n,k);
      force_tmp_(m,1,k,j,i) += yscc_.d_view(n)*xsin_(m,n,i)*ycos_(m,n,j)*zcos_(m,n,k);
      force_tmp_(m,1,k,j,i) += yscs_.d_view(n)*xsin_(m,n,i)*ycos_(m,n,j)*zsin_(m,n,k);
      force_tmp_(m,1,k,j,i) += yssc_.d_view(n)*xsin_(m,n,i)*ysin_(m,n,j)*zcos_(m,n,k);
      force_tmp_(m,1,k,j,i) += ysss_.d_view(n)*xsin_(m,n,i)*ysin_(m,n,j)*zsin_(m,n,k);

      force_tmp_(m,2,k,j,i) += zccc_.d_view(n)*xcos_(m,n,i)*ycos_(m,n,j)*zcos_(m,n,k);
      force_tmp_(m,2,k,j,i) += zccs_.d_view(n)*xcos_(m,n,i)*ycos_(m,n,j)*zsin_(m,n,k);
      force_tmp_(m,2,k,j,i) += zcsc_.d_view(n)*xcos_(m,n,i)*ysin_(m,n,j)*zcos_(m,n,k);
      force_tmp_(m,2,k,j,i) += zcss_.d_view(n)*xcos_(m,n,i)*ysin_(m,n,j)*zsin_(m,n,k);
      force_tmp_(m,2,k,j,i) += zscc_.d_view(n)*xsin_(m,n,i)*ycos_(m,n,j)*zcos_(m,n,k);
      force_tmp_(m,2,k,j,i) += zscs_.d_view(n)*xsin_(m,n,i)*ycos_(m,n,j)*zsin_(m,n,k);
      force_tmp_(m,2,k,j,i) += zssc_.d_view(n)*xsin_(m,n,i)*ysin_(m,n,j)*zcos_(m,n,k);
      force_tmp_(m,2,k,j,i) += zsss_.d_view(n)*xsin_(m,n,i)*ysin_(m,n,j)*zsin_(m,n,k);
    });
  }

  DvceArray5D<Real> u0, u0_;
  if (pmy_pack->phydro != nullptr) u0 = (pmy_pack->phydro->u0);
  if (pmy_pack->pmhd != nullptr) u0 = (pmy_pack->pmhd->u0);
  bool flag_twofl = false;
  if (pmy_pack->pionn != nullptr) {
    u0 = (pmy_pack->phydro->u0);
    u0_ = (pmy_pack->pmhd->u0);
    flag_twofl = true;
  }

  const int nmkji = nmb*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  Real t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0;

  Kokkos::parallel_reduce("net_mom_1", Kokkos::RangePolicy<>(DevExeSpace(),0,nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &sum_t0, Real &sum_t1,
                                Real &sum_t2, Real &sum_t3) {
    // compute n,k,j,i indices of thread
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;
    Real den = u0(m,IDN,k,j,i);
    if (flag_twofl) {
      den += u0_(m,IDN,k,j,i);
    }
    sum_t0 += den;
    sum_t1 += den*force_tmp_(m,0,k,j,i);
    sum_t2 += den*force_tmp_(m,1,k,j,i);
    sum_t3 += den*force_tmp_(m,2,k,j,i);
  }, Kokkos::Sum<Real>(t0), Kokkos::Sum<Real>(t1),
     Kokkos::Sum<Real>(t2), Kokkos::Sum<Real>(t3));


#if MPI_PARALLEL_ENABLED
  Real m[4], gm[4];
  m[0] = t0; m[1] = t1; m[2] = t2; m[3] = t3;
  MPI_Allreduce(m, gm, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  t0 = gm[0]; t1 = gm[1]; t2 = gm[2]; t3 = gm[3];
#endif

  par_for("force_remove_net_mom", DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    force_tmp_(m,0,k,j,i) -= t1/t0;
    force_tmp_(m,1,k,j,i) -= t2/t0;
    force_tmp_(m,2,k,j,i) -= t3/t0;
  });

  t0 = 0.0;
  t1 = 0.0;
  Kokkos::parallel_reduce("net_mom_2", Kokkos::RangePolicy<>(DevExeSpace(),0,nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &sum_t0, Real &sum_t1) {
    // compute n,k,j,i indices of thread
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    Real den  = u0(m,IDN,k,j,i);
    Real mom1 = u0(m,IM1,k,j,i);
    Real mom2 = u0(m,IM2,k,j,i);
    Real mom3 = u0(m,IM3,k,j,i);
    if (flag_twofl) {
      den  += u0_(m,IDN,k,j,i);
      mom1 += u0_(m,IM1,k,j,i);
      mom2 += u0_(m,IM2,k,j,i);
      mom3 += u0_(m,IM3,k,j,i);
    }
    Real v1 = force_tmp_(m,0,k,j,i);
    Real v2 = force_tmp_(m,1,k,j,i);
    Real v3 = force_tmp_(m,2,k,j,i);

    sum_t0 += den*(v1*v1+v2*v2+v3*v3);
    sum_t1 += mom1*v1+mom2*v2+mom3*v3;
  }, Kokkos::Sum<Real>(t0), Kokkos::Sum<Real>(t1));

#if MPI_PARALLEL_ENABLED
  m[0] = t0; m[1] = t1;
  MPI_Allreduce(m, gm, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  t0 = gm[0]; t1 = gm[1];
#endif

  t0 = std::max(t0, 1.0e-20);
  t1 = std::max(t1, 1.0e-20);

  Real m0 = t0, m1 = t1;
  Real dt = pm->dt;
  Real dvol = 1.0/(gnx1*gnx2*gnx3);
  Real s = 1.0;
  if (control_mode == ForceControl::kPower) {
    m0 = 0.5*m0*dvol*dt;
    m1 = m1*dvol;
    if (m0 != 0.0) {
      if (m1 >= 0) {
        s = -m1/2./m0 + sqrt(m1*m1/4./m0/m0 + dedt/m0);
      } else {
        s = m1/2./m0 + sqrt(m1*m1/4./m0/m0 + dedt/m0);
      }
    } else {
      s = 0.0;
    }
  } else {
    Real tforce = 0.0;
    Kokkos::parallel_reduce("force_rms", Kokkos::RangePolicy<>(DevExeSpace(),0,nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &sum_tf) {
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;
      Real v1 = force_tmp_(m,0,k,j,i);
      Real v2 = force_tmp_(m,1,k,j,i);
      Real v3 = force_tmp_(m,2,k,j,i);
      sum_tf += v1*v1 + v2*v2 + v3*v3;
    }, Kokkos::Sum<Real>(tforce));
#if MPI_PARALLEL_ENABLED
    Real mt[1], gmt[1];
    mt[0] = tforce;
    MPI_Allreduce(mt, gmt, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    tforce = gmt[0];
#endif
    Real frms_sq = tforce*dvol;
    Real accel_target = std::abs(accel_rms);
    if (frms_sq > 0.0) {
      s = accel_target/std::sqrt(frms_sq);
    } else {
      s = 0.0;
    }
  }
  if (m0 == 0.0) s = 0.0;

  par_for("force_norm", DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    force_tmp_(m,0,k,j,i) *= s;
    force_tmp_(m,1,k,j,i) *= s;
    force_tmp_(m,2,k,j,i) *= s;
  });

  if (force_perp_only) {
    par_for("force_zero_parallel_tmp", DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      force_tmp_(m,2,k,j,i) = 0.0;
    });
  }

  if (alfvenic_drive) {
    auto force_plus_tmp_ = force_plus_tmp;
    auto force_minus_tmp_ = force_minus_tmp;
    par_for("store_plus_minus_templates", DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real fx = force_tmp_(m,0,k,j,i);
      Real fy = force_tmp_(m,1,k,j,i);
      force_plus_tmp_(m,0,k,j,i) = fx;
      force_plus_tmp_(m,1,k,j,i) = fy;
      force_plus_tmp_(m,2,k,j,i) = 0.0;
      force_minus_tmp_(m,0,k,j,i) = -fy;
      force_minus_tmp_(m,1,k,j,i) =  fx;
      force_minus_tmp_(m,2,k,j,i) = 0.0;
    });
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn apply forcing

TaskStatus TurbulenceDriver::AddForcing(Driver *pdrive, int stage) {
  Mesh *pm = pmy_pack->pmesh;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int &nmb = pmy_pack->nmb_thispack;
  int &nx1 = indcs.nx1;
  int &nx2 = indcs.nx2;
  int &nx3 = indcs.nx3;

  Real dt = pm->dt;
  Real fcorr, gcorr;
  if (tcorr <= 1e-6) {  // use whitenoise
    fcorr = 0.0;
    gcorr = 1.0;
  } else {
    fcorr = std::exp(-dt/tcorr);
    gcorr = std::sqrt(1.0 - fcorr*fcorr);
  }

  EquationOfState *peos;

  DvceArray5D<Real> u0, u0_;
  DvceArray5D<Real> w0;
  DvceFaceFld4D<Real> *bcc0;
  if (pmy_pack->phydro != nullptr) u0 = (pmy_pack->phydro->u0);
  if (pmy_pack->phydro != nullptr) peos = (pmy_pack->phydro->peos);
  if (pmy_pack->pmhd != nullptr) u0 = (pmy_pack->pmhd->u0);
  if (pmy_pack->pmhd != nullptr) bcc0 = &(pmy_pack->pmhd->b0);
  if (pmy_pack->pmhd != nullptr) peos = pmy_pack->pmhd->peos;
  bool flag_twofl = false;
  if (pmy_pack->pionn != nullptr) {
    u0 = (pmy_pack->phydro->u0);
    u0_ = (pmy_pack->pmhd->u0);
    flag_twofl = true;
  }

  bool flag_relativistic = pmy_pack->pcoord->is_special_relativistic;
  if (flag_relativistic) {
    if (pmy_pack->phydro != nullptr) w0 = (pmy_pack->phydro->w0);
    if (pmy_pack->pmhd != nullptr) w0 = (pmy_pack->pmhd->w0);
  }

  const bool need_alfvenic = alfvenic_drive;
  const Real frac_plus = zplus_fraction;
  const Real frac_minus = std::max(static_cast<Real>(0.0),
                                   static_cast<Real>(1.0) - frac_plus);
  const Real w_plus = std::sqrt(frac_plus);
  const Real w_minus = std::sqrt(frac_minus);
  const Real sqrt_rho0 = alfvenic_sqrt_rho0;
  auto force_ = force;
  auto force_tmp_ = force_tmp;
  auto bdrive_ = bdrive;
  if (need_alfvenic) {
    auto force_plus_ = force_plus;
    auto force_minus_ = force_minus;
    auto force_plus_tmp_ = force_plus_tmp;
    auto force_minus_tmp_ = force_minus_tmp;

    par_for("force_OU_elsasser",DevExeSpace(),0,nmb-1,0,2,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
      force_plus_(m,n,k,j,i)  = fcorr*force_plus_(m,n,k,j,i)
                              + gcorr*force_plus_tmp_(m,n,k,j,i);
      force_minus_(m,n,k,j,i) = fcorr*force_minus_(m,n,k,j,i)
                              + gcorr*force_minus_tmp_(m,n,k,j,i);
    });

    par_for("force_from_zpm",DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real zp_x = force_plus_(m,0,k,j,i);
      Real zp_y = force_plus_(m,1,k,j,i);
      Real zm_x = force_minus_(m,0,k,j,i);
      Real zm_y = force_minus_(m,1,k,j,i);
      Real v_fac = 0.5;
      Real b_fac = 0.5*sqrt_rho0;
      Real v1 = v_fac*(w_plus*zp_x + w_minus*zm_x);
      Real v2 = v_fac*(w_plus*zp_y + w_minus*zm_y);
      force_(m,0,k,j,i) = v1;
      force_(m,1,k,j,i) = v2;
      force_(m,2,k,j,i) = 0.0;
      bdrive_(m,0,k,j,i) = b_fac*(w_plus*zp_x - w_minus*zm_x);
      bdrive_(m,1,k,j,i) = b_fac*(w_plus*zp_y - w_minus*zm_y);
    });
    ComputeMagneticEMF();
  } else {
    par_for("force_OU_process",DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      force_(m,0,k,j,i) = fcorr*force_(m,0,k,j,i) + gcorr*force_tmp_(m,0,k,j,i);
      force_(m,1,k,j,i) = fcorr*force_(m,1,k,j,i) + gcorr*force_tmp_(m,1,k,j,i);
      force_(m,2,k,j,i) = fcorr*force_(m,2,k,j,i) + gcorr*force_tmp_(m,2,k,j,i);
    });
    if (force_perp_only) {
      par_for("force_zero_parallel",DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        force_(m,2,k,j,i) = 0.0;
      });
    }
    par_for("bdrive_reset",DevExeSpace(),0,nmb-1,0,1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
      bdrive_(m,n,k,j,i) = 0.0;
    });
    emf_ready_ = false;
  }

  par_for("push",DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real v1 = force_(m,0,k,j,i);
    Real v2 = force_(m,1,k,j,i);
    Real v3 = force_(m,2,k,j,i);

    Real den = u0(m,IDN,k,j,i);
    if (flag_relativistic) {
      // Compute Lorentz factor
      auto &ux = w0(m,IVX,k,j,i);
      auto &uy = w0(m,IVY,k,j,i);
      auto &uz = w0(m,IVZ,k,j,i);

      Real ut = 1. + ux*ux + uy*uy + uz*uz;
      ut = sqrt(ut);
      den /= ut;

      Real Fv = (v1*ux + v2*uy + v3*uz)/ut;

      u0(m,IEN,k,j,i) += Fv*den*dt;
    }
    u0(m,IM1,k,j,i) += den*v1*dt;
    u0(m,IM2,k,j,i) += den*v2*dt;
    u0(m,IM3,k,j,i) += den*v3*dt;

    if (flag_twofl) {
      den = u0_(m,IDN,k,j,i);
      u0_(m,IM1,k,j,i) += den*v1*dt;
      u0_(m,IM2,k,j,i) += den*v2*dt;
      u0_(m,IM3,k,j,i) += den*v3*dt;
    }
  });

  const int nmkji = nmb*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;

  // Relativistic case will require a Lorentz transformation
  if (flag_relativistic) {
    if (pmy_pack->pmhd != nullptr) {
      auto &b = *bcc0;
      auto &eos = peos->eos_data;

      par_for("net_mom_4",DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        // load single state conserved variables
        MHDCons1D u;
        u.d = u0(m,IDN,k,j,i);
        u.mx = u0(m,IM1,k,j,i);
        u.my = u0(m,IM2,k,j,i);
        u.mz = u0(m,IM3,k,j,i);
        u.e = u0(m,IEN,k,j,i);

        u.bx = 0.5*(b.x1f(m,k,j,i) + b.x1f(m,k,j,i+1));
        u.by = 0.5*(b.x2f(m,k,j,i) + b.x2f(m,k,j+1,i));
        u.bz = 0.5*(b.x3f(m,k,j,i) + b.x3f(m,k+1,j,i));

        // Compute (S^i S_i) (eqn C2)
        Real s2 = SQR(u.mx) + SQR(u.my) + SQR(u.mz);
        Real b2 = SQR(u.bx) + SQR(u.by) + SQR(u.bz);
        Real rpar = (u.bx*u.mx + u.by*u.my + u.bz*u.mz)/u.d;

        // call c2p function
        // (inline function in ideal_c2p_mhd.hpp file)
        HydPrim1D w;
        bool dfloor_used = false, efloor_used = false;
        //bool vceiling_used = false;
        bool c2p_failure = false;
        int iter_used = 0;
        SingleC2P_IdealSRMHD(u, eos, s2, b2, rpar, w, dfloor_used,
                             efloor_used, c2p_failure, iter_used);
        // apply velocity ceiling if necessary
        Real lor = sqrt(1.0 + SQR(w.vx) + SQR(w.vy) + SQR(w.vz));
        if (lor > eos.gamma_max) {
          //vceiling_used = true;
          Real factor = sqrt((SQR(eos.gamma_max) - 1.0) / (SQR(lor) - 1.0));
          w.vx *= factor;
          w.vy *= factor;
          w.vz *= factor;
        }

        // Temporarily store primitives in conserved state
        u0(m,IDN,k,j,i) = w.d;
        u0(m,IM1,k,j,i) = w.vx;
        u0(m,IM2,k,j,i) = w.vy;
        u0(m,IM3,k,j,i) = w.vz;
        u0(m,IEN,k,j,i) = w.e;
      });
    } else {
      auto &eos = peos->eos_data;

      par_for("net_mom_4",DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        // load single state conserved variables
        HydCons1D u;
        u.d = u0(m,IDN,k,j,i);
        u.mx = u0(m,IM1,k,j,i);
        u.my = u0(m,IM2,k,j,i);
        u.mz = u0(m,IM3,k,j,i);
        u.e = u0(m,IEN,k,j,i);

        // Compute (S^i S_i) (eqn C2)
        Real s2 = SQR(u.mx) + SQR(u.my) + SQR(u.mz);

        // call c2p function
        // (inline function in ideal_c2p_mhd.hpp file)
        HydPrim1D w;
        bool dfloor_used = false, efloor_used = false;
        //bool vceiling_used = false;
        bool c2p_failure = false;
        int iter_used = 0;
        SingleC2P_IdealSRHyd(u, eos, s2, w, dfloor_used, efloor_used,
                             c2p_failure, iter_used);
        // apply velocity ceiling if necessary
        Real lor = sqrt(1.0 + SQR(w.vx) + SQR(w.vy) + SQR(w.vz));
        if (lor > eos.gamma_max) {
          //vceiling_used = true;
          Real factor = sqrt((SQR(eos.gamma_max) - 1.0) / (SQR(lor) - 1.0));
          w.vx *= factor;
          w.vy *= factor;
          w.vz *= factor;
        }

        u0(m,IDN,k,j,i) = w.d;
        u0(m,IM1,k,j,i) = w.vx;
        u0(m,IM2,k,j,i) = w.vy;
        u0(m,IM3,k,j,i) = w.vz;
        u0(m,IEN,k,j,i) = w.e;
      });
    }

    // remove net momentum
    Real t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0;
    Kokkos::parallel_reduce("net_mom_3", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &sum_t0, Real &sum_t1, Real &sum_t2,
                  Real &sum_t3) {
      // compute n,k,j,i indices of thread
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;

      Real u_t = sqrt(1. + u0(m,IVX,k,j,i)*u0(m,IVX,k,j,i) +
                           u0(m,IVY,k,j,i)*u0(m,IVY,k,j,i) +
                           u0(m,IVZ,k,j,i)*u0(m,IVZ,k,j,i));

      Real den = u0(m,IDN,k,j,i)*u_t;
      Real mom1 = den*u0(m,IVX,k,j,i);
      Real mom2 = den*u0(m,IVY,k,j,i);
      Real mom3 = den*u0(m,IVZ,k,j,i);

      sum_t0 += den;
      sum_t1 += mom1;
      sum_t2 += mom2;
      sum_t3 += mom3;
    }, Kokkos::Sum<Real>(t0), Kokkos::Sum<Real>(t1),
       Kokkos::Sum<Real>(t2), Kokkos::Sum<Real>(t3));

#if MPI_PARALLEL_ENABLED
    Real m[4], gm[4];
    m[0] = t0; m[1] = t1; m[2] = t2; m[3] = t3;
    MPI_Allreduce(m, gm, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    t0 = gm[0]; t1 = gm[1]; t2 = gm[2]; t3 = gm[3];
#endif

    // Compute average velocity
    Real uA_x = t1/t0;
    Real uA_y = t2/t0;
    Real uA_z = t3/t0;

    Real uA_0 = sqrt(1. + uA_x*uA_x + uA_y*uA_y + uA_z*uA_z);
    Real betaA = sqrt(uA_x*uA_x + uA_y*uA_y + uA_z*uA_z)/uA_0;

    Real vx = uA_x/uA_0;
    Real vy = uA_y/uA_0;
    Real vz = uA_z/uA_0;

    // LIMIT temp

    if (pmy_pack->pmhd != nullptr) {
      auto &b = *bcc0;
      auto &eos = peos->eos_data;

      par_for("net_mom_4",DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        u0(m,IEN,k,j,i) = fmin(u0(m,IEN,k,j,i), 40.*u0(m,IDN,k,j,i));

        // load single state conserved variables
        MHDPrim1D u;
        u.d = u0(m,IDN,k,j,i);
        u.vx = u0(m,IM1,k,j,i);
        u.vy = u0(m,IM2,k,j,i);
        u.vz = u0(m,IM3,k,j,i);
        u.e = u0(m,IEN,k,j,i);

        u.bx = 0.5*(b.x1f(m,k,j,i) + b.x1f(m,k,j,i+1));
        u.by = 0.5*(b.x2f(m,k,j,i) + b.x2f(m,k,j+1,i));
        u.bz = 0.5*(b.x3f(m,k,j,i) + b.x3f(m,k+1,j,i));

        HydCons1D u_out;
        SingleP2C_IdealSRMHD(u, eos.gamma, u_out);

        Real en = u_out.d + u_out.e;
        Real sx = u_out.mx;
        Real sy = u_out.my;
        Real sz = u_out.mz;

        Real dens = u_out.d;

        auto &w = u;

        Real lorentz = sqrt(1. + w.vx*w.vx + w.vy*w.vy + w.vz*w.vz);
        Real beta = sqrt(w.vx*w.vx + w.vy*w.vy + w.vz*w.vz)/lorentz;

        u0(m,IDN,k,j,i) = dens;  // *uA_0*(1.-beta*betaA);

        // Does not require knowledge of v
        u0(m,IEN,k,j,i) = uA_0*en - uA_0*(sx*vx + sy*vy + sz*vz);
        u0(m,IEN,k,j,i) -= u0(m,IDN,k,j,i);

        u0(m,IM1,k,j,i) = sx + (uA_0 - 1.)/(betaA*betaA)*(sx*vx + sy*vy + sz*vz)*vx;
        u0(m,IM2,k,j,i) = sy + (uA_0 - 1.)/(betaA*betaA)*(sx*vx + sy*vy + sz*vz)*vy;
        u0(m,IM3,k,j,i) = sz + (uA_0 - 1.)/(betaA*betaA)*(sx*vx + sy*vy + sz*vz)*vz;

        u0(m,IM1,k,j,i) -= uA_0*en*vx;
        u0(m,IM2,k,j,i) -= uA_0*en*vy;
        u0(m,IM3,k,j,i) -= uA_0*en*vz;
      });
    } else {
      auto &eos = peos->eos_data;

      par_for("net_mom_4",DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        u0(m,IEN,k,j,i) = fmin(u0(m,IEN,k,j,i), 40.*u0(m,IDN,k,j,i));

        // load single state conserved variables
        HydPrim1D u;
        u.d = u0(m,IDN,k,j,i);
        u.vx = u0(m,IM1,k,j,i);
        u.vy = u0(m,IM2,k,j,i);
        u.vz = u0(m,IM3,k,j,i);
        u.e = u0(m,IEN,k,j,i);

        HydCons1D u_out;
        SingleP2C_IdealSRHyd(u, eos.gamma, u_out);

        Real en = u_out.d + u_out.e;
        Real sx = u_out.mx;
        Real sy = u_out.my;
        Real sz = u_out.mz;

        Real dens = u_out.d;

        auto &w = u;

        Real lorentz = sqrt(1. + w.vx*w.vx + w.vy*w.vy + w.vz*w.vz);
        Real beta = sqrt(w.vx*w.vx + w.vy*w.vy + w.vz*w.vz)/lorentz;

        u0(m,IDN,k,j,i) = dens;  //*uA_0*(1.-beta*betaA);

        // Does not require knowledge of v
        u0(m,IEN,k,j,i) = uA_0*en - uA_0*(sx*vx + sy*vy + sz*vz);
        u0(m,IEN,k,j,i) -= u0(m,IDN,k,j,i);
        u0(m,IM1,k,j,i) = sx + (uA_0 - 1.)/(betaA*betaA)*(sx*vx + sy*vy + sz*vz)*vx;
        u0(m,IM2,k,j,i) = sy + (uA_0 - 1.)/(betaA*betaA)*(sx*vx + sy*vy + sz*vz)*vy;
        u0(m,IM3,k,j,i) = sz + (uA_0 - 1.)/(betaA*betaA)*(sx*vx + sy*vy + sz*vz)*vz;
        u0(m,IM1,k,j,i) -= uA_0*en*vx;
        u0(m,IM2,k,j,i) -= uA_0*en*vy;
        u0(m,IM3,k,j,i) -= uA_0*en*vz;
      });
    }

  } else {
    // remove net momentum
    Real t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0;
    Kokkos::parallel_reduce("net_mom_3", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &sum_t0, Real &sum_t1, Real &sum_t2,
                  Real &sum_t3) {
      // compute n,k,j,i indices of thread
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;

      Real den = u0(m,IDN,k,j,i);
      Real mom1 = u0(m,IM1,k,j,i);
      Real mom2 = u0(m,IM2,k,j,i);
      Real mom3 = u0(m,IM3,k,j,i);
      if (flag_twofl) {
        den += u0_(m,IDN,k,j,i);
        mom1 += u0_(m,IM1,k,j,i);
        mom2 += u0_(m,IM2,k,j,i);
        mom3 += u0_(m,IM3,k,j,i);
      }

      sum_t0 += den;
      sum_t1 += mom1;
      sum_t2 += mom2;
      sum_t3 += mom3;
    }, Kokkos::Sum<Real>(t0), Kokkos::Sum<Real>(t1),
       Kokkos::Sum<Real>(t2), Kokkos::Sum<Real>(t3));

#if MPI_PARALLEL_ENABLED
    Real m[4], gm[4];
    m[0] = t0; m[1] = t1; m[2] = t2; m[3] = t3;
    MPI_Allreduce(m, gm, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    t0 = gm[0]; t1 = gm[1]; t2 = gm[2]; t3 = gm[3];
#endif

    par_for("net_mom_4",DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real den = u0(m,IDN,k,j,i);

      if (flag_relativistic) {
        auto &ux = w0(m,IVX,k,j,i);
        auto &uy = w0(m,IVY,k,j,i);
        auto &uz = w0(m,IVZ,k,j,i);

        Real ut = 1. + ux*ux + uy*uy + uz*uz;
        ut = sqrt(ut);
        den /= ut;

        Real Fv_avg = den*(t1*ux + t2*uy + t3*uz)/ut/t0;

        u0(m,IEN,k,j,i) -= Fv_avg;
      }
      u0(m,IM1,k,j,i) -= den*t1/t0;
      u0(m,IM2,k,j,i) -= den*t2/t0;
      u0(m,IM3,k,j,i) -= den*t3/t0;
      if (flag_twofl) {
        den = u0_(m,IDN,k,j,i);
        u0_(m,IM1,k,j,i) -= den*t1/t0;
        u0_(m,IM2,k,j,i) -= den*t2/t0;
        u0_(m,IM3,k,j,i) -= den*t3/t0;
      }
    });
  }

  return TaskStatus::complete;
}

void TurbulenceDriver::ApplyEMF(DvceEdgeFld4D<Real> &efld) {
  if (!emf_ready_) return;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int nmb = pmy_pack->nmb_thispack;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  auto emf1 = emf_drive.x1e;
  auto emf2 = emf_drive.x2e;
  auto emf3 = emf_drive.x3e;
  auto e1 = efld.x1e;
  auto e2 = efld.x2e;
  auto e3 = efld.x3e;

  par_for("turb_add_emf_x1", DevExeSpace(), 0, nmb-1, 0, ncells3, 0, ncells2, 0, ncells1-1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    e1(m,k,j,i) += emf1(m,k,j,i);
  });
  par_for("turb_add_emf_x2", DevExeSpace(), 0, nmb-1, 0, ncells3, 0, ncells2-1, 0, ncells1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    e2(m,k,j,i) += emf2(m,k,j,i);
  });
  par_for("turb_add_emf_x3", DevExeSpace(), 0, nmb-1, 0, ncells3-1, 0, ncells2, 0, ncells1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    e3(m,k,j,i) += emf3(m,k,j,i);
  });
  emf_ready_ = false;
}

void TurbulenceDriver::ComputeMagneticEMF() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb = pmy_pack->nmb_thispack;

  Kokkos::deep_copy(emf_drive.x1e, 0.0);
  Kokkos::deep_copy(emf_drive.x2e, 0.0);
  Kokkos::deep_copy(emf_drive.x3e, 0.0);

  if (!alfvenic_drive || mode_count == 0) {
    emf_ready_ = false;
    return;
  }

  auto zero_coeff = [](DualArray1D<Real> &arr) {
    Kokkos::deep_copy(arr.d_view, 0.0);
  };
  zero_coeff(bxccc);
  zero_coeff(bxccs);
  zero_coeff(bxcsc);
  zero_coeff(bxcss);
  zero_coeff(bxscc);
  zero_coeff(bxscs);
  zero_coeff(bxssc);
  zero_coeff(bxsss);
  zero_coeff(byccc);
  zero_coeff(byccs);
  zero_coeff(bycsc);
  zero_coeff(bycss);
  zero_coeff(byscc);
  zero_coeff(byscs);
  zero_coeff(byssc);
  zero_coeff(bysss);

  auto bdrive_ = bdrive;
  auto xcos_ = xcos;
  auto xsin_ = xsin;
  auto ycos_ = ycos;
  auto ysin_ = ysin;
  auto zcos_ = zcos;
  auto zsin_ = zsin;
  auto bxccc_ = bxccc;
  auto bxccs_ = bxccs;
  auto bxcsc_ = bxcsc;
  auto bxcss_ = bxcss;
  auto bxscc_ = bxscc;
  auto bxscs_ = bxscs;
  auto bxssc_ = bxssc;
  auto bxsss_ = bxsss;
  auto byccc_ = byccc;
  auto byccs_ = byccs;
  auto bycsc_ = bycsc;
  auto bycss_ = bycss;
  auto byscc_ = byscc;
  auto byscs_ = byscs;
  auto byssc_ = byssc;
  auto bysss_ = bysss;

  int mode_count_ = mode_count;
  par_for("proj_bdrive", DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real bx = bdrive_(m,0,k,j,i);
    Real by = bdrive_(m,1,k,j,i);
    if (bx == 0.0 && by == 0.0) return;
    for (int n=0; n<mode_count_; ++n) {
      Real xc = xcos_(m,n,i);
      Real xs = xsin_(m,n,i);
      Real yc = ycos_(m,n,j);
      Real ys = ysin_(m,n,j);
      Real zc = zcos_(m,n,k);
      Real zs = zsin_(m,n,k);

      if (bx != 0.0) {
        Kokkos::atomic_add(&(bxccc_.d_view(n)), bx*xc*yc*zc);
        Kokkos::atomic_add(&(bxccs_.d_view(n)), bx*xc*yc*zs);
        Kokkos::atomic_add(&(bxcsc_.d_view(n)), bx*xc*ys*zc);
        Kokkos::atomic_add(&(bxcss_.d_view(n)), bx*xc*ys*zs);
        Kokkos::atomic_add(&(bxscc_.d_view(n)), bx*xs*yc*zc);
        Kokkos::atomic_add(&(bxscs_.d_view(n)), bx*xs*yc*zs);
        Kokkos::atomic_add(&(bxssc_.d_view(n)), bx*xs*ys*zc);
        Kokkos::atomic_add(&(bxsss_.d_view(n)), bx*xs*ys*zs);
      }

      if (by != 0.0) {
        Kokkos::atomic_add(&(byccc_.d_view(n)), by*xc*yc*zc);
        Kokkos::atomic_add(&(byccs_.d_view(n)), by*xc*yc*zs);
        Kokkos::atomic_add(&(bycsc_.d_view(n)), by*xc*ys*zc);
        Kokkos::atomic_add(&(bycss_.d_view(n)), by*xc*ys*zs);
        Kokkos::atomic_add(&(byscc_.d_view(n)), by*xs*yc*zc);
        Kokkos::atomic_add(&(byscs_.d_view(n)), by*xs*yc*zs);
        Kokkos::atomic_add(&(byssc_.d_view(n)), by*xs*ys*zc);
        Kokkos::atomic_add(&(bysss_.d_view(n)), by*xs*ys*zs);
      }
    }
  });

  bxccc_.template sync<HostMemSpace>();
  bxccs_.template sync<HostMemSpace>();
  bxcsc_.template sync<HostMemSpace>();
  bxcss_.template sync<HostMemSpace>();
  bxscc_.template sync<HostMemSpace>();
  bxscs_.template sync<HostMemSpace>();
  bxssc_.template sync<HostMemSpace>();
  bxsss_.template sync<HostMemSpace>();
  byccc_.template sync<HostMemSpace>();
  byccs_.template sync<HostMemSpace>();
  bycsc_.template sync<HostMemSpace>();
  bycss_.template sync<HostMemSpace>();
  byscc_.template sync<HostMemSpace>();
  byscs_.template sync<HostMemSpace>();
  byssc_.template sync<HostMemSpace>();
  bysss_.template sync<HostMemSpace>();

  auto gather = [&](DualArray1D<Real> &arr) {
#if MPI_PARALLEL_ENABLED
    std::vector<Real> send(mode_count), recv(mode_count);
    for (int n=0; n<mode_count; ++n) send[n] = arr.h_view(n);
    MPI_Allreduce(send.data(), recv.data(), mode_count, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    arr.template modify<HostMemSpace>();
    for (int n=0; n<mode_count; ++n) arr.h_view(n) = recv[n];
#else
    (void) arr;
#endif
  };

  gather(bxccc_);
  gather(bxccs_);
  gather(bxcsc_);
  gather(bxcss_);
  gather(bxscc_);
  gather(bxscs_);
  gather(bxssc_);
  gather(bxsss_);
  gather(byccc_);
  gather(byccs_);
  gather(bycsc_);
  gather(bycss_);
  gather(byscc_);
  gather(byscs_);
  gather(byssc_);
  gather(bysss_);

  psiccc.template modify<HostMemSpace>();
  psiccs.template modify<HostMemSpace>();
  psicsc.template modify<HostMemSpace>();
  psicss.template modify<HostMemSpace>();
  psiscc.template modify<HostMemSpace>();
  psiscs.template modify<HostMemSpace>();
  psissc.template modify<HostMemSpace>();
  psisss.template modify<HostMemSpace>();

  for (int n=0; n<mode_count; ++n) {
    Real kx = kx_mode.h_view(n);
    Real ky = ky_mode.h_view(n);
    Real tiny = 1.0e-20;

    auto coeff = [&](DualArray1D<Real> &num, DualArray1D<Real> &norm) -> Real {
      Real denom = norm.h_view(n);
      if (denom <= tiny) return 0.0;
      return num.h_view(n)/denom;
    };

    Real cy_ccc = coeff(byccc_, norm_ccc);
    Real cy_ccs = coeff(byccs_, norm_ccs);
    Real cy_csc = coeff(bycsc_, norm_csc);
    Real cy_css = coeff(bycss_, norm_css);
    Real cy_scc = coeff(byscc_, norm_scc);
    Real cy_scs = coeff(byscs_, norm_scs);
    Real cy_ssc = coeff(byssc_, norm_ssc);
    Real cy_sss = coeff(bysss_, norm_sss);

    Real cx_ccc = coeff(bxccc_, norm_ccc);
    Real cx_ccs = coeff(bxccs_, norm_ccs);
    Real cx_csc = coeff(bxcsc_, norm_csc);
    Real cx_css = coeff(bxcss_, norm_css);
    Real cx_scc = coeff(bxscc_, norm_scc);
    Real cx_scs = coeff(bxscs_, norm_scs);
    Real cx_ssc = coeff(bxssc_, norm_ssc);
    Real cx_sss = coeff(bxsss_, norm_sss);

    auto from_by = [&](Real coeff_y, Real sign) -> std::pair<Real, Real> {
      if (std::abs(kx) <= tiny) return std::make_pair(0.0, 0.0);
      Real val = sign*coeff_y/kx;
      Real wt = kx*kx;
      return std::make_pair(val, wt);
    };

    auto from_bx = [&](Real coeff_x, Real sign) -> std::pair<Real, Real> {
      if (std::abs(ky) <= tiny) return std::make_pair(0.0, 0.0);
      Real val = sign*coeff_x/ky;
      Real wt = ky*ky;
      return std::make_pair(val, wt);
    };

    auto blend = [](const std::pair<Real, Real> &a,
                    const std::pair<Real, Real> &b) -> Real {
      Real denom = a.second + b.second;
      if (denom <= 0.0) return 0.0;
      return (a.first*a.second + b.first*b.second)/denom;
    };

    psiccc.h_view(n) = blend(from_by(cy_scc,  1.0), from_bx(cx_csc, -1.0));
    psiccs.h_view(n) = blend(from_by(cy_scs,  1.0), from_bx(cx_css, -1.0));
    psicsc.h_view(n) = blend(from_by(cy_ssc,  1.0), from_bx(cx_ccc,  1.0));
    psicss.h_view(n) = blend(from_by(cy_sss,  1.0), from_bx(cx_ccs,  1.0));
    psiscc.h_view(n) = blend(from_by(cy_ccc, -1.0), from_bx(cx_ssc, -1.0));
    psiscs.h_view(n) = blend(from_by(cy_ccs, -1.0), from_bx(cx_sss, -1.0));
    psissc.h_view(n) = blend(from_by(cy_csc, -1.0), from_bx(cx_scc,  1.0));
    psisss.h_view(n) = blend(from_by(cy_css, -1.0), from_bx(cx_scs,  1.0));
  }

  psiccc.template sync<DevExeSpace>();
  psiccs.template sync<DevExeSpace>();
  psicsc.template sync<DevExeSpace>();
  psicss.template sync<DevExeSpace>();
  psiscc.template sync<DevExeSpace>();
  psiscs.template sync<DevExeSpace>();
  psissc.template sync<DevExeSpace>();
  psisss.template sync<DevExeSpace>();

  auto xcos_edge_ = xcos_edge;
  auto xsin_edge_ = xsin_edge;
  auto ycos_edge_ = ycos_edge;
  auto ysin_edge_ = ysin_edge;
  auto e3 = emf_drive.x3e;
  auto psiccc_ = psiccc;
  auto psiccs_ = psiccs;
  auto psicsc_ = psicsc;
  auto psicss_ = psicss;
  auto psiscc_ = psiscc;
  auto psiscs_ = psiscs;
  auto psissc_ = psissc;
  auto psisss_ = psisss;
  par_for("build_emf_z", DevExeSpace(),0,nmb-1,ks,ke,js,je+1,is,ie+1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real sum = 0.0;
    for (int n=0; n<mode_count_; ++n) {
      Real xc = xcos_edge_(m,n,i);
      Real xs = xsin_edge_(m,n,i);
      Real yc = ycos_edge_(m,n,j);
      Real ys = ysin_edge_(m,n,j);
      Real zc = zcos_(m,n,k);
      Real zs = zsin_(m,n,k);

      sum += psiccc_.d_view(n)*xc*yc*zc;
      sum += psiccs_.d_view(n)*xc*yc*zs;
      sum += psicsc_.d_view(n)*xc*ys*zc;
      sum += psicss_.d_view(n)*xc*ys*zs;
      sum += psiscc_.d_view(n)*xs*yc*zc;
      sum += psiscs_.d_view(n)*xs*yc*zs;
      sum += psissc_.d_view(n)*xs*ys*zc;
      sum += psisss_.d_view(n)*xs*ys*zs;
    }
    e3(m,k,j,i) += sum;
  });

  emf_ready_ = true;
}
