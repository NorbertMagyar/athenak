#ifndef SRCTERMS_TURB_DRIVER_HPP_
#define SRCTERMS_TURB_DRIVER_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file turb_driver.hpp
//  \brief defines turbulence driver class, which implements data and functions for
//  randomly forced turbulence which evolves via an Ornstein-Uhlenbeck stochastic process

#include <memory>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "utils/random.hpp"

//----------------------------------------------------------------------------------------
//! \class TurbulenceDriver

class TurbulenceDriver {
 public:
  TurbulenceDriver(MeshBlockPack *pp, ParameterInput *pin);
  ~TurbulenceDriver();

  DvceArray5D<Real> force, force_tmp;  // arrays used for turb forcing
  DvceArray5D<Real> force_plus, force_minus;
  DvceArray5D<Real> force_plus_tmp, force_minus_tmp;
  DvceArray5D<Real> bdrive;           // stores cell-centered magnetic driving
  DvceEdgeFld4D<Real> emf_drive;      // edge-centered EMF contribution from driving
  RNG_State rstate;                    // random state

  DualArray1D<Real> xccc, xccs, xcsc, xcss, xscc, xscs, xssc, xsss;
  DualArray1D<Real> yccc, yccs, ycsc, ycss, yscc, yscs, yssc, ysss;
  DualArray1D<Real> zccc, zccs, zcsc, zcss, zscc, zscs, zssc, zsss;
  DualArray1D<Real> kx_mode, ky_mode, kz_mode;
  DvceArray3D<Real> xcos, xsin, ycos, ysin, zcos, zsin;
  DvceArray3D<Real> xcos_edge, xsin_edge, ycos_edge, ysin_edge;
  DualArray1D<Real> psiccc, psiccs, psicsc, psicss;
  DualArray1D<Real> psiscc, psiscs, psissc, psisss;
  DualArray1D<Real> bxccc, bxccs, bxcsc, bxcss, bxscc, bxscs, bxssc, bxsss;
  DualArray1D<Real> byccc, byccs, bycsc, bycss, byscc, byscs, byssc, bysss;
  DualArray1D<Real> norm_ccc, norm_ccs, norm_csc, norm_css;
  DualArray1D<Real> norm_scc, norm_scs, norm_ssc, norm_sss;

  // parameters of driving
  int nlow, nhigh;
  int mode_count;
  Real tcorr, dedt;
  Real expo, exp_prl, exp_prp;
  int driving_type;
  Real accel_rms;
  Real solenoidal_fraction;
  Real sol_weight;
  Real comp_weight;
  Real zplus_fraction;
  Real alfvenic_rho0;
  Real alfvenic_sqrt_rho0;
  bool alfvenic_drive;
  bool force_perp_only;
  enum class ForceControl {kPower, kAccel};
  ForceControl control_mode;

  // functions
  void IncludeInitializeModesTask(std::shared_ptr<TaskList> tl, TaskID start);
  void IncludeAddForcingTask(std::shared_ptr<TaskList> tl, TaskID start);
  TaskStatus InitializeModes(Driver *pdrive, int stage);
  TaskStatus AddForcing(Driver *pdrive, int stage);
  void ApplyEMF(DvceEdgeFld4D<Real> &efld);
  void ComputeMagneticEMF();
  void Initialize();
  void ComputeBasisNorms();

 private:
  bool first_time = true;   // flag to enable initialization on first call
  bool emf_ready_ = false;
  MeshBlockPack *pmy_pack;  // ptr to MeshBlockPack containing this TurbulenceDriver
};

#endif  // SRCTERMS_TURB_DRIVER_HPP_
