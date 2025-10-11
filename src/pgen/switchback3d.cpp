// switchback3d_superradial_fixed.cpp
// Axisymmetric switchback atop a user super-radial background; CT via vector potential.
//
// Notes:
//  * No host-pointer dereferences inside KOKKOS_LAMBDA — all scalars captured.
//  * Face-centered B from curl(A) using centered differences on faces.
//  * Cell-centered B used for energy via simple face-averaging with edge clamping.
//  * v ∝ (B − B0) / sqrt(ρ0) with sign = sign_z.

#include <cmath>
#include <iostream>
#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "pgen/pgen.hpp"
#include "globals.hpp"

namespace {
struct StratificationParams {
  Real rho0;
  Real rsp;
  Real temperature;
  Real g_over_rspT;
  Real Rsun;
  Real base_z;
  Real length_unit;
  Real temperature_unit;
  Real density_unit;
  Real time_unit;
  Real magnetic_unit;
  Real gm1;
} strat_params;
} 

void Switchback3DStratifiedBCs(Mesh *pm);

KOKKOS_INLINE_FUNCTION Real StratDensityProfile(const Real z,
                                                const Real base_z,
                                                const Real length_unit,
                                                const Real Rsun,
                                                const Real g_over_rspT,
                                                const Real rho0) {
  const Real z_phys = (z - base_z) * length_unit;
  const Real denom = Rsun + z_phys;
  const Real bracket = (denom > 0.0) ? ((Rsun * Rsun) / denom - Rsun) : 0.0;
  return rho0 * exp(-g_over_rspT * bracket);
}

KOKKOS_INLINE_FUNCTION Real StratPressureFromDensity(const Real rho,
                                                     const Real rsp,
                                                     const Real temperature) {
  return rho * rsp * temperature;
}

KOKKOS_INLINE_FUNCTION static Real sgn(const Real z){
  return (z>0.0?1.0:(z<0.0?-1.0:0.0));
}

// --------------------------- Switchback envelopes ---------------------------
KOKKOS_INLINE_FUNCTION Real G_fun(Real z, Real a){
  Real za=z/a, ez=exp(-(za*za)); return (z*z/(a*a))*ez;
}
KOKKOS_INLINE_FUNCTION Real Gp_fun(Real z, Real a){
  Real za=z/a, ez=exp(-(za*za)); return (2.0*z/(a*a))*ez*(1.0 - (z*z)/(a*a));
}
KOKKOS_INLINE_FUNCTION Real S_fun(Real r, Real b){
  Real b2=b*b, r2=r*r; Real den=b2+r2; Real den3=den*den*den;
  return (den3>0.0)? (4.0*b2*r2/den3) : 0.0;
}
KOKKOS_INLINE_FUNCTION Real F_fun(Real r, Real b){
  Real b2=b*b, r2=r*r; Real den=b2+r2; Real den2=den*den;
  return (den2>0.0)? ((r2*r)/den2) : 0.0;
}

// --------------------------- 16-pt Gauss–Legendre [0,1] ---------------------
KOKKOS_INLINE_FUNCTION Real gl16_x(int i){
  const Real x[16] = {
    0.0483076656877383, 0.1444719615827965, 0.2392873622521371, 0.3318686022821277,
    0.4213512761306353, 0.5068999089322294, 0.5877157572407623, 0.6630442669302152,
    0.7321821187402897, 0.7944837959679424, 0.8493676137325700, 0.8963211557660521,
    0.9349060759377397, 0.9647622555875064, 0.9856115115452684, 0.9972638618494816 };
  return x[i];
}
KOKKOS_INLINE_FUNCTION Real gl16_w(int i){
  const Real w[16] = {
    0.0965400885147278, 0.0956387200792749, 0.0938443990808046, 0.0911738786957639,
    0.0876520930044038, 0.0833119242269467, 0.0781938957870703, 0.0723457941088485,
    0.0658222227763618, 0.0586840934785355, 0.0509980592623762, 0.0428358980222267,
    0.0342738629130214, 0.0253920653092621, 0.0162743947309057, 0.0070186100094701 };
  return w[i];
}

// --------------------------- Super-radial background -------------------------
KOKKOS_INLINE_FUNCTION
void B_superradial(const Real x, const Real y, const Real z,
                   const Real B0, const Real r0, const Real gamma,
                   const Real kD, const Real kQ,
                   const Real z0,                 
                   Real &Bx, Real &By, Real &Bz) {
  const Real Z = z - z0;                            
  const Real r2 = x*x + y*y + Z*Z;                     
  const Real r  = sqrt(r2 + 1e-300);
  const Real invr2 = 1.0/(r2 + 1e-300);
  const Real invr = 1.0/(r + 1e-300);
  const Real invr3 = invr2 * invr;
  const Real invr5 = invr3 * invr2;
  const Real invr7 = invr5 * invr2;

  const Real th = atan(gamma);
  const Real A  = B0*r0*r0 * cos(th);
  const Real D  = B0*r0*r0*r0 * kD * sin(th);
  const Real Q  = B0*r0*r0*r0*r0 * kQ * sin(th);
  const Real S  = (15.0*Z*Z - 3.0*r2);

  Bx =  A * x * invr3
      + D * (-3.0*x*Z * invr5)
      + Q * (-2.0 * S * x * invr7);
  By =  A * y * invr3
      + D * (-3.0*y*Z * invr5)
      + Q * (-2.0 * S * y * invr7);
  Bz =  A * Z * invr3
      + D * ((r2 - 3.0*Z*Z) * invr5)
      + Q * (-2.0 * Z * (15.0*Z*Z - 9.0*r2) * invr7);
}


KOKKOS_INLINE_FUNCTION
void B0_poloidal_rz(const Real r, const Real z,
                    const Real B0, const Real r0, const Real gamma,
                    const Real kD, const Real kQ,
                    const Real z0,               // NEW
                    Real &Br, Real &Bzz){
  Real Bx, By;
  B_superradial(r, 0.0, z, B0, r0, gamma, kD, kQ, z0, Bx, By, Bzz); 
  Br = Bx; 
}

KOKKOS_INLINE_FUNCTION
Real Aphi_bg_fun(const Real r, const Real z, const Real zref,
                 const Real B0, const Real r0, const Real gamma,
                 const Real kD, const Real kQ, const Real z0) {
  (void)zref;

  if (r <= 1e-14) {
    Real Br0, Bz0;
    B0_poloidal_rz(0.0, z, B0, r0, gamma, kD, kQ, z0, Br0, Bz0);
    return 0.5 * r * Bz0;
  }

  Real integral_r = 0.0;
  for (int q=0; q<16; ++q) {
    const Real xi = gl16_x(q);
    const Real wi = gl16_w(q);
    const Real rp = r * xi;
    Real Brp, Bzp;
    B0_poloidal_rz(rp, z, B0, r0, gamma, kD, kQ, z0, Brp, Bzp);
    integral_r += wi * (rp * Bzp);
  }
  integral_r *= r;

  return integral_r / r;
}

// --------------------------- Switchback components ---------------------------
KOKKOS_INLINE_FUNCTION
void Br_Bz_Bphi_sw(const Real r, const Real z,
                   const Real a, const Real b, const Real eps,
                   const Real B0, const Real r0, const Real gamma,
                   const Real kD, const Real kQ,
                   const Real z0,
                   Real &Br_sw, Real &Bz_sw, Real &Bphi_sw){
  const Real G  = G_fun(z,a);
  const Real Gp = Gp_fun(z,a);
  const Real S  = S_fun(r,b);
  const Real F  = F_fun(r,b);

  Br_sw = eps * F * Gp;
  Bz_sw = -eps * S * G;

  // Choose toroidal component so that |B0 + B_sw| ≈ |B0|
  Real Br0, Bz0; B0_poloidal_rz(r,z,B0,r0,gamma,kD,kQ, z0, Br0, Bz0);
  Real rad = - (Br_sw*Br_sw + Bz_sw*Bz_sw + 2.0*(Br0*Br_sw + Bz0*Bz_sw));
  if (rad < 0.0) rad = 0.0;
  Bphi_sw = sgn(z) * sqrt(rad);
}

KOKKOS_INLINE_FUNCTION Real Aphi_sw_fun(const Real r, const Real z,
                                        const Real a, const Real b, const Real eps){
  return -eps * F_fun(r,b) * G_fun(z,a);
}

KOKKOS_INLINE_FUNCTION
Real Az_sw_fun(const Real r, const Real z, const Real a, const Real b, const Real eps,
               const Real B0, const Real r0, const Real gamma, const Real kD, const Real kQ,
               const Real z0)  
{
  if (r <= 1e-14) return 0.0;
  Real acc = 0.0;
  for (int q=0; q<16; ++q){
    const Real rho = r * gl16_x(q);
    Real Brs, Bzs, Bphs;
    Br_Bz_Bphi_sw(rho, z, a, b, eps, B0, r0, gamma, kD, kQ, z0, Brs, Bzs, Bphs); 
    acc += gl16_w(q) * Bphs;
  }
  return -r * acc;
}

// --------------------------- Vector potential in Cartesian -------------------
KOKKOS_INLINE_FUNCTION
void A_cart(const Real x, const Real y, const Real z,
            const Real a, const Real b, const Real eps,
            const Real B0, const Real r0, const Real gamma, const Real kD, const Real kQ,
            const Real zref, const Real z0,
            Real &Ax, Real &Ay, Real &Az){
  const Real rho = sqrt(x*x + y*y);
  const Real inv_rho = (rho > 1e-14)? 1.0/rho : 0.0;

  const Real Aphi_bg = Aphi_bg_fun(rho, z, zref, B0, r0, gamma, kD, kQ, z0);
  const Real Aphi_sw = Aphi_sw_fun(rho, z, a, b, eps);
  const Real Az_sw   = Az_sw_fun(rho, z, a, b, eps, B0,r0,gamma,kD,kQ, z0);

  const Real Aphi = Aphi_bg + Aphi_sw;

  Az = Az_sw;
  if (rho > 1e-14) {
    Ax = Aphi * (-y * inv_rho);
    Ay = Aphi * ( x * inv_rho);
  } else {
    Ax = 0.0;
    Ay = 0.0;
  }
}

// =============================== Problem Generator ===========================
void ProblemGenerator::Switchback3D(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pmhd == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Switchback3D requires <mhd> block in input file." << std::endl;
    exit(EXIT_FAILURE);
  }

  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is;
  const int js = indcs.js;
  const int ks = indcs.ks;
  const int nx1 = indcs.nx1;
  const int nx2 = (indcs.nx2 > 0) ? indcs.nx2 : 1;
  const int nx3 = (indcs.nx3 > 0) ? indcs.nx3 : 1;

  // EOS gamma (host) -> gm1 capture
  const Real gm1_h = pmbp->pmhd->peos->eos_data.gamma - 1.0;

  // Stratification controls
  const Real rho0_base = pin->GetOrAddReal("problem","rho0", 1.0);
  const Real temperature = pin->GetOrAddReal("problem","temperature", 1.0);
  const Real surface_gravity = pin->GetOrAddReal("problem","surface_gravity", 274.0);
  const Real solar_radius = pin->GetOrAddReal("problem","R_sun", 6.957e8);
  const Real mu = pin->GetOrAddReal("problem","mu", 0.62);
  const Real length_unit = pin->GetOrAddReal("problem","length_unit", 1.0e6);
  const Real density_unit = pin->GetOrAddReal("problem","density_unit", 1.0e-12);
  const Real temperature_unit = pin->GetOrAddReal("problem","temperature_unit", 1.0e6);
  const Real base_z = pin->GetOrAddReal("problem","strat_base_z",
                                        pmy_mesh_->mesh_size.x3min);

  const Real k_b = 1.380649e-23;
  const Real m_p = 1.67262192369e-27;
  const Real mean_particle_mass = mu * m_p;
  const Real rsp_phys = k_b / mean_particle_mass;
  const Real time_unit = std::sqrt((length_unit*length_unit)/(rsp_phys * temperature_unit));
  const Real rsp_check = rsp_phys * temperature_unit * (time_unit*time_unit) /
                         (length_unit*length_unit);
  if (std::abs(rsp_check - 1.0) > 1e-10) {
    std::cout << "### WARNING Switchback3D: simulated Rsp deviates from unity by "
              << (rsp_check - 1.0) << std::endl;
  }
  const Real rsp = 1.0;
  const Real temperature_phys = temperature * temperature_unit;
  const Real g_over_rspT = surface_gravity / (rsp_phys * temperature_phys);
  const Real pressure_unit = density_unit * rsp_phys * temperature_unit;
  const Real mu0_phys = 4.0 * acos(-1.0) * 1.0e-7;
  const Real magnetic_unit = std::sqrt(mu0_phys * pressure_unit);
  const Real velocity_unit = length_unit / time_unit;

  strat_params = {rho0_base, rsp, temperature, g_over_rspT,
                  solar_radius, base_z, length_unit, temperature_unit,
                  density_unit, time_unit, magnetic_unit, gm1_h};

  user_bcs_func = Switchback3DStratifiedBCs;

  if (restart) return;

  if (global_variable::my_rank == 0) {
    std::cout << "Switchback3D units: rho_unit=" << density_unit << " kg/m^3, "
              << "T_unit=" << temperature_unit << " K, "
              << "time_unit=" << time_unit << " s, "
              << "velocity_unit=" << velocity_unit << " m/s, "
              << "B_unit=" << magnetic_unit << " T ("
              << magnetic_unit*1.0e4 << " G)" << std::endl;
  }

  // Params
  const Real a     = pin->GetOrAddReal   ("problem","a",     0.25);
  const Real b     = pin->GetOrAddReal   ("problem","b",     0.25);
  const Real eps   = pin->GetOrAddReal   ("problem","eps",   0.5);
  const int  signz = pin->GetOrAddInteger("problem","sign_z",-1);

  // Background field controls
  const Real B0mag = pin->GetOrAddReal("problem","B0",   1.0);
  const Real r0    = pin->GetOrAddReal("problem","r0",   1.0);
  const Real gamma_bg = pin->GetOrAddReal("problem","gamma",0.0);
  const Real kD    = pin->GetOrAddReal("problem","kD",  1.0);
  const Real kQ    = pin->GetOrAddReal("problem","kQ",  0.0);
  const Real z0 = pin->GetOrAddReal("problem","z0", -1.0);

  // Reference z for Aphi_bg integral (take domain min for consistency)
  const Real zref = pmy_mesh_->mesh_size.x3min;

  auto &u0  = pmbp->pmhd->u0;     // conserved: [m,v,k,j,i]
  auto &b0  = pmbp->pmhd->b0;     // face B:    [m,k,j,i] on x1/x2/x3
  auto dsize = pmbp->pmb->mb_size.d_view; // per-block sizes (device)
  auto size_h = Kokkos::create_mirror_view(dsize);
  Kokkos::deep_copy(size_h, dsize);

  // Authoritative number of MBs to iterate on device:
  const int nmb = pmbp->nmb_thispack;

  {
    auto f = b0.x1f; const int Nk=f.extent_int(1), Nj=f.extent_int(2), Ni=f.extent_int(3);
    Kokkos::parallel_for("faces_x1_mdr",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,0,0,0},{nmb,Nk,Nj,Ni}),
      KOKKOS_LAMBDA(int m,int k,int j,int i){
        const auto sz = dsize(m);
        const Real x1min=sz.x1min, x1max=sz.x1max, dx1=sz.dx1;
        const Real x2min=sz.x2min, x2max=sz.x2max, dx2=sz.dx2;
        const Real x3min=sz.x3min, x3max=sz.x3max, dx3=sz.dx3;

        const Real xf = LeftEdgeX(i - is, nx1, x1min, x1max);
        const Real yc = CellCenterX(j - js, nx2, x2min, x2max);
        const Real zc = CellCenterX(k - ks, nx3, x3min, x3max);

        Real Ax,Ay,Az, Az_p,Az_m, Ay_p,Ay_m;
        A_cart(xf, yc+0.5*dx2, zc, a,b,eps, B0mag,r0,gamma_bg,kD,kQ, zref, z0, Ax,Ay,Az_p);
        A_cart(xf, yc-0.5*dx2, zc, a,b,eps, B0mag,r0,gamma_bg,kD,kQ, zref, z0, Ax,Ay,Az_m);
        A_cart(xf, yc, zc+0.5*dx3, a,b,eps, B0mag,r0,gamma_bg,kD,kQ, zref, z0, Ax,Ay_p,Az);
        A_cart(xf, yc, zc-0.5*dx3, a,b,eps, B0mag,r0,gamma_bg,kD,kQ, zref, z0, Ax,Ay_m,Az);

        f(m,k,j,i) = (Az_p - Az_m)/dx2 - (Ay_p - Ay_m)/dx3;
      });
  }
  {
    auto f = b0.x2f; const int Nk=f.extent_int(1), Nj=f.extent_int(2), Ni=f.extent_int(3);
    Kokkos::parallel_for("faces_x2_mdr",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,0,0,0},{nmb,Nk,Nj,Ni}),
      KOKKOS_LAMBDA(int m,int k,int j,int i){
        const auto sz = dsize(m);
        const Real x1min=sz.x1min, x1max=sz.x1max, dx1=sz.dx1;
        const Real x2min=sz.x2min, x2max=sz.x2max, dx2=sz.dx2;
        const Real x3min=sz.x3min, x3max=sz.x3max, dx3=sz.dx3;

        const Real xc = CellCenterX(i - is, nx1, x1min, x1max);
        const Real yf = LeftEdgeX(j - js, nx2, x2min, x2max);
        const Real zc = CellCenterX(k - ks, nx3, x3min, x3max);

        Real Ax,Ay,Az, Ax_p,Ax_m, Az_p,Az_m;
        A_cart(xc, yf, zc+0.5*dx3, a,b,eps, B0mag,r0,gamma_bg,kD,kQ, zref, z0, Ax_p,Ay,Az);
        A_cart(xc, yf, zc-0.5*dx3, a,b,eps, B0mag,r0,gamma_bg,kD,kQ, zref, z0, Ax_m,Ay,Az);
        A_cart(xc+0.5*dx1, yf, zc, a,b,eps, B0mag,r0,gamma_bg,kD,kQ, zref, z0, Ax,Ay,Az_p);
        A_cart(xc-0.5*dx1, yf, zc, a,b,eps, B0mag,r0,gamma_bg,kD,kQ, zref, z0, Ax,Ay,Az_m);

        f(m,k,j,i) = (Ax_p - Ax_m)/dx3 - (Az_p - Az_m)/dx1;
      });
  }
  {
    auto f = b0.x3f; const int Nk=f.extent_int(1), Nj=f.extent_int(2), Ni=f.extent_int(3);
    Kokkos::parallel_for("faces_x3_mdr",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,0,0,0},{nmb,Nk,Nj,Ni}),
      KOKKOS_LAMBDA(int m,int k,int j,int i){
        const auto sz = dsize(m);
        const Real x1min=sz.x1min, x1max=sz.x1max, dx1=sz.dx1;
        const Real x2min=sz.x2min, x2max=sz.x2max, dx2=sz.dx2;
        const Real x3min=sz.x3min, x3max=sz.x3max, dx3=sz.dx3;

        const Real xc = CellCenterX(i - is, nx1, x1min, x1max);
        const Real yc = CellCenterX(j - js, nx2, x2min, x2max);
        const Real zf = LeftEdgeX(k - ks, nx3, x3min, x3max);

        Real Ax,Ay,Az, Ay_p,Ay_m, Ax_p,Ax_m;
        A_cart(xc+0.5*dx1, yc, zf, a,b,eps, B0mag,r0,gamma_bg,kD,kQ, zref, z0, Ax,Ay_p,Az);
        A_cart(xc-0.5*dx1, yc, zf, a,b,eps, B0mag,r0,gamma_bg,kD,kQ, zref, z0, Ax,Ay_m,Az);
        A_cart(xc, yc+0.5*dx2, zf, a,b,eps, B0mag,r0,gamma_bg,kD,kQ, zref, z0, Ax_p,Ay,Az);
        A_cart(xc, yc-0.5*dx2, zf, a,b,eps, B0mag,r0,gamma_bg,kD,kQ, zref, z0, Ax_m,Ay,Az);

        f(m,k,j,i) = (Ay_p - Ay_m)/dx1 - (Ax_p - Ax_m)/dx2;
      });
  }
  Kokkos::fence();

  // --------------------- Conserved variables (ρ, m, E) ----------------------
  const Real vfac = static_cast<Real>(signz) / sqrt(rho0_base);

  // ρ and momenta (Elsässer-like: v ∝ B − B0)
  {
    auto u = u0;
    auto f1 = b0.x1f; auto f2 = b0.x2f; auto f3 = b0.x3f;
    const int Nk=u.extent_int(2), Nj=u.extent_int(3), Ni=u.extent_int(4);

    Kokkos::parallel_for("init_u0",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,0,0,0},{nmb,Nk,Nj,Ni}),
      KOKKOS_LAMBDA(int m,int k,int j,int i){
        // face-averaged B at cell center (edge clamped)
        const int iR = (i+1 < f1.extent_int(3)) ? i+1 : i;
        const int jR = (j+1 < f2.extent_int(2)) ? j+1 : j;
        const int kR = (k+1 < f3.extent_int(1)) ? k+1 : k;
        const Real Bcx = 0.5*(f1(m,k,j,i) + f1(m,k,j,iR));
        const Real Bcy = 0.5*(f2(m,k,j,i) + f2(m,k,jR,i));
        const Real Bcz = 0.5*(f3(m,k,j,i) + f3(m,kR,j,i));

        // background B0 at cell center
        const auto sz = dsize(m);
        const Real x1min=sz.x1min, x1max=sz.x1max;
        const Real x2min=sz.x2min, x2max=sz.x2max;
        const Real x3min=sz.x3min, x3max=sz.x3max;
        const Real xc = CellCenterX(i - is, nx1, x1min, x1max);
        const Real yc = CellCenterX(j - js, nx2, x2min, x2max);
        const Real zc = CellCenterX(k - ks, nx3, x3min, x3max);

        Real B0x,B0y,B0z; B_superradial(xc,yc,zc, B0mag,r0,gamma_bg,kD,kQ, z0, B0x,B0y,B0z);

        const Real vx = vfac * (Bcx - B0x);
        const Real vy = vfac * (Bcy - B0y);
        const Real vz = vfac * (Bcz - B0z);

        const Real rho = StratDensityProfile(zc, base_z, length_unit, solar_radius,
                                             g_over_rspT, rho0_base);
        u(m,IDN,k,j,i) = rho;
        u(m,IM1,k,j,i) = rho*vx;
        u(m,IM2,k,j,i) = rho*vy;
        u(m,IM3,k,j,i) = rho*vz;
      });
  }

  // Energy: E = p/(γ-1) + ½ρv² + ½|B|² (B via face-averaging)
  {
    auto u = u0; auto f1 = b0.x1f; auto f2 = b0.x2f; auto f3 = b0.x3f;
    const int Nk=u.extent_int(2), Nj=u.extent_int(3), Ni=u.extent_int(4);

    Kokkos::parallel_for("init_E",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,0,0,0},{nmb,Nk,Nj,Ni}),
      KOKKOS_LAMBDA(int m,int k,int j,int i){
        const Real rho = u(m,IDN,k,j,i);
        const Real vx = u(m,IM1,k,j,i)/rho;
        const Real vy = u(m,IM2,k,j,i)/rho;
        const Real vz = u(m,IM3,k,j,i)/rho;
        const Real v2 = vx*vx + vy*vy + vz*vz;

        const int iR = (i+1 < f1.extent_int(3)) ? i+1 : i;
        const int jR = (j+1 < f2.extent_int(2)) ? j+1 : j;
        const int kR = (k+1 < f3.extent_int(1)) ? k+1 : k;
        const Real Bcx = 0.5*(f1(m,k,j,i) + f1(m,k,j,iR));
        const Real Bcy = 0.5*(f2(m,k,j,i) + f2(m,k,jR,i));
        const Real Bcz = 0.5*(f3(m,k,j,i) + f3(m,kR,j,i));
        const Real emag = 0.5*(Bcx*Bcx + Bcy*Bcy + Bcz*Bcz);
        const Real pressure = StratPressureFromDensity(rho, rsp, temperature);
        const Real eint = (gm1_h > 0.0) ? pressure / gm1_h : 0.0;

        u(m,IEN,k,j,i) = eint + 0.5*rho*v2 + emag;
      });
  }

  Kokkos::fence();
}

void Switchback3DStratifiedBCs(Mesh *pm) {
  auto pmbp = pm->pmb_pack;
  if (pmbp->pmhd == nullptr) {
    return;
  }

  auto &indcs = pm->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int nmb = pmbp->nmb_thispack;
  int &ks = indcs.ks;
  int &ke = indcs.ke;
  auto &mb_bcs = pmbp->pmb->mb_bcs;
  auto &size = pmbp->pmb->mb_size;

  auto &u0 = pmbp->pmhd->u0;
  auto &b0 = pmbp->pmhd->b0;

  const Real base_z = strat_params.base_z;
  const Real length_unit = strat_params.length_unit;
  const Real Rsun = strat_params.Rsun;
  const Real g_over_rspT = strat_params.g_over_rspT;
  const Real rho0_base = strat_params.rho0;
  const Real rsp = strat_params.rsp;
  const Real temperature = strat_params.temperature;
  const Real gm1 = strat_params.gm1;

  // Hydrodynamic variables: density and momenta
  par_for("switchback_bc_hydro", DevExeSpace(), 0,(nmb-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int j, int i) {
    const bool inner_user = (mb_bcs.d_view(m, BoundaryFace::inner_x3) == BoundaryFlag::user);
    const bool outer_user = (mb_bcs.d_view(m, BoundaryFace::outer_x3) == BoundaryFlag::user);

    const auto sz = size.d_view(m);
    const Real x3min = sz.x3min;
    const Real x3max = sz.x3max;
    const int nx3 = indcs.nx3;

    if (inner_user) {
      const Real dens_in = u0(m,IDN,ks,j,i);
      const Real inv_dens_in = (dens_in > 0.0) ? 1.0/dens_in : 0.0;
      const Real vx = u0(m,IM1,ks,j,i) * inv_dens_in;
      const Real vy = u0(m,IM2,ks,j,i) * inv_dens_in;
      const Real vz = u0(m,IM3,ks,j,i) * inv_dens_in;
      for (int k=1; k<=ng; ++k) {
        const int kg = ks - k;
        const Real zc = CellCenterX(kg - ks, nx3, x3min, x3max);
        const Real rho = StratDensityProfile(zc, base_z, length_unit, Rsun,
                                             g_over_rspT, rho0_base);
        u0(m,IDN,kg,j,i) = rho;
        u0(m,IM1,kg,j,i) = rho*vx;
        u0(m,IM2,kg,j,i) = rho*vy;
        u0(m,IM3,kg,j,i) = rho*vz;
      }
    }

    if (outer_user) {
      const Real dens_out = u0(m,IDN,ke,j,i);
      const Real inv_dens_out = (dens_out > 0.0) ? 1.0/dens_out : 0.0;
      const Real vx = u0(m,IM1,ke,j,i) * inv_dens_out;
      const Real vy = u0(m,IM2,ke,j,i) * inv_dens_out;
      const Real vz = u0(m,IM3,ke,j,i) * inv_dens_out;
      for (int k=1; k<=ng; ++k) {
        const int kg = ke + k;
        const Real zc = CellCenterX(kg - ks, nx3, x3min, x3max);
        const Real rho = StratDensityProfile(zc, base_z, length_unit, Rsun,
                                             g_over_rspT, rho0_base);
        u0(m,IDN,kg,j,i) = rho;
        u0(m,IM1,kg,j,i) = rho*vx;
        u0(m,IM2,kg,j,i) = rho*vy;
        u0(m,IM3,kg,j,i) = rho*vz;
      }
    }
  });

  // Magnetic fields: zero-gradient copy with full face coverage
  if (indcs.nx3 > 1) {
    auto bx = b0.x1f;
    const int n1f_x = bx.extent_int(3);
    const int n2f_x = bx.extent_int(2);
    // inner boundary
    Kokkos::parallel_for("switchback_bc_bx_inner",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,0,0,0},{nmb,ng,n2f_x,n1f_x}),
      KOKKOS_LAMBDA(int m,int kk,int j,int i){
        if (mb_bcs.d_view(m, BoundaryFace::inner_x3) != BoundaryFlag::user) return;
        const int kg = ks - 1 - kk;
        bx(m,kg,j,i) = bx(m,ks,j,i);
      });
    // outer boundary
    Kokkos::parallel_for("switchback_bc_bx_outer",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,0,0,0},{nmb,ng,n2f_x,n1f_x}),
      KOKKOS_LAMBDA(int m,int kk,int j,int i){
        if (mb_bcs.d_view(m, BoundaryFace::outer_x3) != BoundaryFlag::user) return;
        const int kg = ke + 1 + kk;
        bx(m,kg,j,i) = bx(m,ke,j,i);
      });

    auto by = b0.x2f;
    const int n1f_y = by.extent_int(3);
    const int n2f_y = by.extent_int(2);
    Kokkos::parallel_for("switchback_bc_by_inner",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,0,0,0},{nmb,ng,n2f_y,n1f_y}),
      KOKKOS_LAMBDA(int m,int kk,int j,int i){
        if (mb_bcs.d_view(m, BoundaryFace::inner_x3) != BoundaryFlag::user) return;
        const int kg = ks - 1 - kk;
        by(m,kg,j,i) = by(m,ks,j,i);
      });
    Kokkos::parallel_for("switchback_bc_by_outer",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,0,0,0},{nmb,ng,n2f_y,n1f_y}),
      KOKKOS_LAMBDA(int m,int kk,int j,int i){
        if (mb_bcs.d_view(m, BoundaryFace::outer_x3) != BoundaryFlag::user) return;
        const int kg = ke + 1 + kk;
        by(m,kg,j,i) = by(m,ke,j,i);
      });

    auto bz = b0.x3f;
    const int n1f_z = bz.extent_int(3);
    const int n2f_z = bz.extent_int(2);
    Kokkos::parallel_for("switchback_bc_bz_inner",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,0,0,0},{nmb,ng,n2f_z,n1f_z}),
      KOKKOS_LAMBDA(int m,int kk,int j,int i){
        if (mb_bcs.d_view(m, BoundaryFace::inner_x3) != BoundaryFlag::user) return;
        const int kg = ks - 1 - kk;
        bz(m,kg,j,i) = bz(m,ks,j,i);
      });
    Kokkos::parallel_for("switchback_bc_bz_outer",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,0,0,0},{nmb,ng,n2f_z,n1f_z}),
      KOKKOS_LAMBDA(int m,int kk,int j,int i){
        if (mb_bcs.d_view(m, BoundaryFace::outer_x3) != BoundaryFlag::user) return;
        const int kg = ke + 1 + kk;
        bz(m,kg,j,i) = bz(m,ke+1,j,i);
      });
  }

  // Recompute total energy in ghost zones to maintain consistency
  par_for("switchback_bc_energy", DevExeSpace(), 0,(nmb-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int j, int i) {
    const bool inner_user = (mb_bcs.d_view(m, BoundaryFace::inner_x3) == BoundaryFlag::user);
    const bool outer_user = (mb_bcs.d_view(m, BoundaryFace::outer_x3) == BoundaryFlag::user);

    if (inner_user) {
      for (int k=1; k<=ng; ++k) {
        const int kg = ks - k;
        const Real rho = u0(m,IDN,kg,j,i);
        const Real inv_rho = (rho > 0.0) ? 1.0/rho : 0.0;
        const Real vx = u0(m,IM1,kg,j,i) * inv_rho;
        const Real vy = u0(m,IM2,kg,j,i) * inv_rho;
        const Real vz = u0(m,IM3,kg,j,i) * inv_rho;
        const Real v2 = vx*vx + vy*vy + vz*vz;

        const int iR = (i+1 < b0.x1f.extent_int(3)) ? i+1 : i;
        const int jR = (j+1 < b0.x2f.extent_int(2)) ? j+1 : j;
        const int kR = (kg+1 < b0.x3f.extent_int(1)) ? kg+1 : kg;
        const Real Bcx = 0.5*(b0.x1f(m,kg,j,i) + b0.x1f(m,kg,j,iR));
        const Real Bcy = 0.5*(b0.x2f(m,kg,j,i) + b0.x2f(m,kg,jR,i));
        const Real Bcz = 0.5*(b0.x3f(m,kg,j,i) + b0.x3f(m,kR,j,i));
        const Real emag = 0.5*(Bcx*Bcx + Bcy*Bcy + Bcz*Bcz);

        const Real pressure = StratPressureFromDensity(rho, rsp, temperature);
        const Real eint = (gm1 > 0.0) ? pressure / gm1 : 0.0;
        u0(m,IEN,kg,j,i) = eint + 0.5*rho*v2 + emag;
      }
    }

    if (outer_user) {
      for (int k=1; k<=ng; ++k) {
        const int kg = ke + k;
        const Real rho = u0(m,IDN,kg,j,i);
        const Real inv_rho = (rho > 0.0) ? 1.0/rho : 0.0;
        const Real vx = u0(m,IM1,kg,j,i) * inv_rho;
        const Real vy = u0(m,IM2,kg,j,i) * inv_rho;
        const Real vz = u0(m,IM3,kg,j,i) * inv_rho;
        const Real v2 = vx*vx + vy*vy + vz*vz;

        const int iR = (i+1 < b0.x1f.extent_int(3)) ? i+1 : i;
        const int jR = (j+1 < b0.x2f.extent_int(2)) ? j+1 : j;
        const int kR = (kg+1 < b0.x3f.extent_int(1)) ? kg+1 : kg;
        const Real Bcx = 0.5*(b0.x1f(m,kg,j,i) + b0.x1f(m,kg,j,iR));
        const Real Bcy = 0.5*(b0.x2f(m,kg,j,i) + b0.x2f(m,kg,jR,i));
        const Real Bcz = 0.5*(b0.x3f(m,kg,j,i) + b0.x3f(m,kR,j,i));
        const Real emag = 0.5*(Bcx*Bcx + Bcy*Bcy + Bcz*Bcz);

        const Real pressure = StratPressureFromDensity(rho, rsp, temperature);
        const Real eint = (gm1 > 0.0) ? pressure / gm1 : 0.0;
        u0(m,IEN,kg,j,i) = eint + 0.5*rho*v2 + emag;
      }
    }
  });
}

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  Switchback3D(pin, restart);
}
