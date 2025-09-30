// switchback3d.cpp — divergence-free cylindrical switchback + Alfvénic z- velocity
#include <cmath>
#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "pgen/pgen.hpp"

KOKKOS_INLINE_FUNCTION
static Real sgn(const Real z) { return (z > 0.0 ? 1.0 : (z < 0.0 ? -1.0 : 0.0)); }


KOKKOS_INLINE_FUNCTION
Real G_fun(Real z, Real a){ Real za=z/a, ez=exp(-(za*za)); return (z*z/(a*a))*ez; }

KOKKOS_INLINE_FUNCTION
Real Gp_fun(Real z, Real a){ Real za=z/a, ez=exp(-(za*za)); return (2.0*z/(a*a))*ez*(1.0 - (z*z)/(a*a)); }

KOKKOS_INLINE_FUNCTION
Real S_fun(Real r, Real b){ Real b2=b*b, r2=r*r; Real denom=b2+r2; Real denom3=denom*denom*denom; return (denom3>0.0)? (4.0*b2*r2/denom3) : 0.0; }

KOKKOS_INLINE_FUNCTION
Real F_fun(Real r, Real b){ Real b2=b*b, r2=r*r; Real denom=b2+r2; Real denom2=denom*denom; return (denom2>0.0)? ((r2*r)/denom2) : 0.0; }

// Stable 1 - Br^2 - Bz^2: (1 - Bz)*(1 + Bz) - Br^2
KOKKOS_INLINE_FUNCTION
void Br_Bz_Bphi(Real r, Real z, Real a, Real b, Real eps,
                Real &Br, Real &Bz, Real &Bphi) {
  const Real G  = G_fun(z,a);
  const Real Gp = Gp_fun(z,a);
  const Real S  = S_fun(r,b);
  const Real F  = F_fun(r,b);

  Br = eps * F * Gp;
  Bz = 1.0 - eps * S * G;

  Real rad = (1.0 - Bz)*(1.0 + Bz) - Br*Br;  // better-conditioned than 1 - Br^2 - Bz^2
  if (rad < 0.0) rad = 0.0;
  Bphi = (z > 0 ? 1.0 : (z < 0 ? -1.0 : 0.0)) * std::sqrt(rad);
}

// 16-pt Gauss–Legendre on [0,1], weights sum to 1
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

// A_phi(r,z) and A_z(r,z) = -∫_0^r B_phi(ρ,z) dρ  (GL8 quad)
KOKKOS_INLINE_FUNCTION
Real Aphi_fun(Real r, Real z, Real a, Real b, Real eps){
  // Aphi = -eps F(r) G(z) + r/2
  return -eps * F_fun(r,b) * G_fun(z,a) + 0.5*r;
}

KOKKOS_INLINE_FUNCTION
Real Az_fun(Real r, Real z, Real a, Real b, Real eps){
  if (r <= 1e-14) return 0.0;
  Real acc = 0.0;
  for (int q=0; q<16; ++q){
    const Real rho = r * gl16_x(q);             // map [0,1] → [0,r]
    Real Br,Bz,Bphi; Br_Bz_Bphi(rho,z,a,b,eps,Br,Bz,Bphi);
    acc += gl16_w(q) * Bphi;
  }
  return -r * acc;                               // ∫_0^r ≈ r Σ w_i f(r x_i)
}

// Cartesian A at arbitrary (x,y,z)
KOKKOS_INLINE_FUNCTION
void A_cart(Real x, Real y, Real z, Real a, Real b, Real eps, Real &Ax, Real &Ay, Real &Az){
  Real r = sqrt(x*x + y*y);
  Real inv_r = (r > 1e-14)? 1.0/r : 0.0;
  Real Aphi = Aphi_fun(r,z,a,b,eps);
  Az        = Az_fun(r,z,a,b,eps);
  // A_r = 0
  // A = Aphi * e_phi + Az * e_z
  Ax = Aphi * (-y * inv_r);
  Ay = Aphi * ( x * inv_r);
  if (r <= 1e-14){ Ax = 0.0; Ay = 0.0; } // well-defined limit
}

void ProblemGenerator::Switchback3D(ParameterInput *pin, const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pmhd == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Switchback3D requires <mhd> block in input file." << std::endl;
    exit(EXIT_FAILURE);
  }

  // Parameters
  const Real rho0  = pin->GetOrAddReal   ("problem","rho0",  1.0);
  const Real p0    = pin->GetOrAddReal   ("problem","p0",    0.1);
  const Real a     = pin->GetOrAddReal   ("problem","a",     0.25);
  const Real b     = pin->GetOrAddReal   ("problem","b",     0.25);
  const Real eps   = pin->GetOrAddReal   ("problem","eps",   0.5);
  const int  signz = pin->GetOrAddInteger("problem","sign_z",-1); // -1 => z^-
  EOS_Data &eos = pmbp->pmhd->peos->eos_data;
  const Real gm1 = eos.gamma - 1.0;

  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;

  auto &u0  = pmbp->pmhd->u0;     // conserved
  auto &b0  = pmbp->pmhd->b0;     // face-centered B
  auto &siz = pmbp->pmb->mb_size; // geometry

  par_for("pgen_faces_curlA", DevExeSpace(), 0,(pmbp->nmb_thispack-1), ks,ke, js,je, is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
  auto &sz = siz.d_view(m);
  const Real x1min=sz.x1min, x1max=sz.x1max, dx1=sz.dx1;
  const Real x2min=sz.x2min, x2max=sz.x2max, dx2=sz.dx2;
  const Real x3min=sz.x3min, x3max=sz.x3max, dx3=sz.dx3;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;

  // centers & faces
  const Real xc = CellCenterX(i-is, nx1, x1min, x1max);
  const Real yc = CellCenterX(j-js, nx2, x2min, x2max);
  const Real zc = CellCenterX(k-ks, nx3, x3min, x3max);

  const Real xf   = LeftEdgeX(i-is,   nx1, x1min, x1max);
  const Real xf_p = LeftEdgeX(i+1-is, nx1, x1min, x1max);
  const Real yf   = LeftEdgeX(j-js,   nx2, x2min, x2max);
  const Real yf_p = LeftEdgeX(j+1-js, nx2, x2min, x2max);
  const Real zf   = LeftEdgeX(k-ks,   nx3, x3min, x3max);
  const Real zf_p = LeftEdgeX(k+1-ks, nx3, x3min, x3max);

  // ---- Bx at x-faces: (xf, yc, zc)
  {
    // Bx = d/dy Az - d/dz Ay
    Real Ax,Ay,Az, tmp;
    Az = 0.0; Ay = 0.0; Ax = 0.0;
    // Az(y±)
    Real Az_p, Az_m;
    A_cart(xf, yc+0.5*dx2, zc, a,b,eps, Ax,Ay,Az_p);
    A_cart(xf, yc-0.5*dx2, zc, a,b,eps, Ax,Ay,Az_m);
    const Real dAz_dy = (Az_p - Az_m)/dx2;
    // Ay(z±)
    Real Ay_p, Ay_m;
    A_cart(xf, yc, zc+0.5*dx3, a,b,eps, Ax,Ay_p,Az);
    A_cart(xf, yc, zc-0.5*dx3, a,b,eps, Ax,Ay_m,Az);
    const Real dAy_dz = (Ay_p - Ay_m)/dx3;

    b0.x1f(m,k,j,i) = dAz_dy - dAy_dz;
    if (i==ie){
      // also fill i+1 face
      A_cart(xf_p, yc+0.5*dx2, zc, a,b,eps, Ax,Ay,Az_p);
      A_cart(xf_p, yc-0.5*dx2, zc, a,b,eps, Ax,Ay,Az_m);
      const Real dAz_dy2 = (Az_p - Az_m)/dx2;
      A_cart(xf_p, yc, zc+0.5*dx3, a,b,eps, Ax,Ay_p,Az);
      A_cart(xf_p, yc, zc-0.5*dx3, a,b,eps, Ax,Ay_m,Az);
      const Real dAy_dz2 = (Ay_p - Ay_m)/dx3;
      b0.x1f(m,k,j,i+1) = dAz_dy2 - dAy_dz2;
    }
  }

  // ---- By at y-faces: (xc, yf, zc)
  {
    // By = d/dz Ax - d/dx Az
    Real Ax,Ay,Az;
    Real Ax_p, Ax_m, Az_p, Az_m;
    A_cart(xc, yf, zc+0.5*dx3, a,b,eps, Ax_p,Ay,Az);
    A_cart(xc, yf, zc-0.5*dx3, a,b,eps, Ax_m,Ay,Az);
    const Real dAx_dz = (Ax_p - Ax_m)/dx3;

    A_cart(xc+0.5*dx1, yf, zc, a,b,eps, Ax,Ay,Az_p);
    A_cart(xc-0.5*dx1, yf, zc, a,b,eps, Ax,Ay,Az_m);
    const Real dAz_dx = (Az_p - Az_m)/dx1;

    b0.x2f(m,k,j,i) = dAx_dz - dAz_dx;
    if (j==je){
      // also fill j+1 face
      A_cart(xc, yf_p, zc+0.5*dx3, a,b,eps, Ax_p,Ay,Az);
      A_cart(xc, yf_p, zc-0.5*dx3, a,b,eps, Ax_m,Ay,Az);
      const Real dAx_dz2 = (Ax_p - Ax_m)/dx3;

      A_cart(xc+0.5*dx1, yf_p, zc, a,b,eps, Ax,Ay,Az_p);
      A_cart(xc-0.5*dx1, yf_p, zc, a,b,eps, Ax,Ay,Az_m);
      const Real dAz_dx2 = (Az_p - Az_m)/dx1;

      b0.x2f(m,k,j+1,i) = dAx_dz2 - dAz_dx2;
    }
  }

  // ---- Bz at z-faces: (xc, yc, zf)
  {
    // Bz = d/dx Ay - d/dy Ax
    Real Ax,Ay,Az;
    Real Ay_p, Ay_m, Ax_p, Ax_m;
    A_cart(xc+0.5*dx1, yc, zf, a,b,eps, Ax,Ay_p,Az);
    A_cart(xc-0.5*dx1, yc, zf, a,b,eps, Ax,Ay_m,Az);
    const Real dAy_dx = (Ay_p - Ay_m)/dx1;

    A_cart(xc, yc+0.5*dx2, zf, a,b,eps, Ax_p,Ay,Az);
    A_cart(xc, yc-0.5*dx2, zf, a,b,eps, Ax_m,Ay,Az);
    const Real dAx_dy = (Ax_p - Ax_m)/dx2;

    b0.x3f(m,k,j,i) = dAy_dx - dAx_dy;
    if (k==ke){
      // also fill k+1 face
      A_cart(xc+0.5*dx1, yc, zf_p, a,b,eps, Ax,Ay_p,Az);
      A_cart(xc-0.5*dx1, yc, zf_p, a,b,eps, Ax,Ay_m,Az);
      const Real dAy_dx2 = (Ay_p - Ay_m)/dx1;

      A_cart(xc, yc+0.5*dx2, zf_p, a,b,eps, Ax_p,Ay,Az);
      A_cart(xc, yc-0.5*dx2, zf_p, a,b,eps, Ax_m,Ay,Az);
      const Real dAx_dy2 = (Ax_p - Ax_m)/dx2;

      b0.x3f(m,k+1,j,i) = dAy_dx2 - dAx_dy2;
    }
  }
  });

  // ---------------------------------------------
  // 2) Conserved vars with z- Elsässer velocity
  //     v = signz * (B_c - e_z)/sqrt(rho0),  μ0=1
  //     (B_c from face-averaged CT fields)
  // ---------------------------------------------
  // v = signz * (B_c - e_z)/sqrt(rho0)
  const Real vfac = static_cast<Real>(signz) / std::sqrt(rho0);
  par_for("pgen_sbu0_from_faces", DevExeSpace(), 0,(pmbp->nmb_thispack-1), ks,ke, js,je, is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real Bcx = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1));
    const Real Bcy = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i));
    const Real Bcz = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i));

    const Real vx = vfac * ( Bcx      );
    const Real vy = vfac * ( Bcy      );
    const Real vz = vfac * ( Bcz - 1.0);

    u0(m,IDN,k,j,i) = rho0;
    u0(m,IM1,k,j,i) = rho0 * vx;
    u0(m,IM2,k,j,i) = rho0 * vy;
    u0(m,IM3,k,j,i) = rho0 * vz;
  });

  // 3) Total energy = p/(γ-1) + ½ρv² + ½|B|² (B from face-averages)
  par_for("pgen_sbE", DevExeSpace(), 0,(pmbp->nmb_thispack-1), ks,ke, js,je, is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real egas = p0 / gm1;
    const Real vx = u0(m,IM1,k,j,i)/u0(m,IDN,k,j,i);
    const Real vy = u0(m,IM2,k,j,i)/u0(m,IDN,k,j,i);
    const Real vz = u0(m,IM3,k,j,i)/u0(m,IDN,k,j,i);
    const Real v2 = vx*vx + vy*vy + vz*vz;

    const Real Bcx = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1));
    const Real Bcy = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i));
    const Real Bcz = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i));
    const Real emag = 0.5*(Bcx*Bcx + Bcy*Bcy + Bcz*Bcz);

    u0(m,IEN,k,j,i) = egas + 0.5*u0(m,IDN,k,j,i)*v2 + emag;
  });
}

// If the build expects a user-problem entry point, forward it.
void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  Switchback3D(pin, restart);
}

