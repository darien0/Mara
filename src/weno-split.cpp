

#include <iostream>
#include <cstring>
#include <algorithm>
#include "eulers.hpp"
#include "weno-split.hpp"
#include "riemann_hll.hpp"
#include "matrix.h"


typedef WenoSplit Deriv;

/*------------------------------------------------------------------------------
 *
 * Private inline functions
 *
 */
static inline double weno5_fiph_apos(const double *f);
static inline double weno5_fiph_aneg(const double *f);


std::valarray<double> Deriv::dUdt(const std::valarray<double> &Uin)
{
  this->prepare_integration();

  Uglb.resize(stride[0]);
  Pglb.resize(stride[0]);
  Lglb.resize(stride[0]);

  Fiph.resize(stride[0]*(ND>=1));
  Giph.resize(stride[0]*(ND>=2));
  Hiph.resize(stride[0]*(ND>=3));

  Uglb = Uin;
  Pglb = Mara->PrimitiveArray;
  ConsToPrim(Uglb, Pglb);

  switch (ND) {
  case 1: drive_sweeps_1d(); break;
  case 2: drive_sweeps_2d(); break;
  case 3: drive_sweeps_3d(); break;
  }

  return Lglb;
}
void Deriv::intercell_flux_sweep(const double *U, const double *P,
                                 const double *F, const double *A,
                                 double *Fiph, int dim)
{
  const int Nx = Mara->domain->get_N(dim);
  const int Ng = Mara->domain->get_Ng();

  // Local memory requirements for WENO flux calculation
  // ---------------------------------------------------------------------------
  double *lam = new double[NQ];       // Characteristic eigenvalues
  double *Piph = new double[NQ];      // Primitive variables at i+1/2 (averaged)
  double *Uiph = new double[NQ];      // Conserved " " (taken from prim)
  double *Liph = new double[NQ*NQ];   // Left eigenvectors at i+1/2
  double *Riph = new double[NQ*NQ];   // Right eigenvectors at i+1/2
  double *fp = new double[6*NQ];      // Characteristic split fluxes on local stencil
  double *fm = new double[6*NQ];      // " " Left going
  double *Fp = new double[NQ];        // Component-wise split fluxes on local cell
  double *Fm = new double[NQ];        // " " Left going

  double *f = new double[NQ];         // WENO characteristic flux
  double *fpT = new double[NQ*6];     // Transposed split flux to [wave, zone]
  double *fmT = new double[NQ*6];     // " " Left going
  // ---------------------------------------------------------------------------
  const double Ptest[5] = { 1,1,0,0,0 };

  for (int i=Ng-1; i<Nx+Ng; ++i) {

    for (int q=0; q<NQ; ++q) {
      Piph[q] = 0  *  Ptest[q]    +    1  *  0.5*(P[i*NQ + q] + P[(i+1)*NQ + q]);
    }

    Mara->fluid->PrimToCons(Piph, Uiph);
    Mara->fluid->Eigensystem(Uiph, Piph, Liph, Riph, lam, dim);

    // Select the maximum wavespeed on the local stencil
    // -------------------------------------------------------------------------
    const double ml = *std::max_element(A+i-2, A+i+4);

    for (int j=0; j<6; ++j) {
      for (int q=0; q<NQ; ++q) {

        // Local Lax-Friedrichs flux splitting
        // ---------------------------------------------------------------------
        const int m = (i+j-2)*NQ + q;
        Fp[q] = 0.5*(F[m] + ml*U[m]);
        Fm[q] = 0.5*(F[m] - ml*U[m]);
      }
      matrix_vector_product(Liph, Fp, fp+j*NQ, NQ, NQ);
      matrix_vector_product(Liph, Fm, fm+j*NQ, NQ, NQ);
    }

    for (int q=0; q<NQ; ++q) {
      for (int j=0; j<6; ++j) {
        fpT[q*6 + j] = fp[j*NQ + q];
	fmT[q*6 + j] = fm[j*NQ + q];
      }
    }

    for (int q=0; q<NQ; ++q) {
      const int m = q*6 + 2;
      f[q] = weno5_fiph_apos(fpT+m) + weno5_fiph_aneg(fmT+m);
    }

    matrix_vector_product(Riph, f, Fiph+i*NQ, NQ, NQ);
  }

  // Clean up local memory requirements for WENO flux calculation
  // ---------------------------------------------------------------------------
  delete [] lam;
  delete [] Piph;
  delete [] Uiph;
  delete [] Liph;
  delete [] Riph;
  delete [] fp;
  delete [] fm;
  delete [] Fp;
  delete [] Fm;

  delete [] f;
  delete [] fpT;
  delete [] fmT;
  // ---------------------------------------------------------------------------
}


void Deriv::drive_single_sweep(const double *Ug, const double *Pg,
                               double *Fiph_g, int dim)
// Ug   .... Pointer to start of conserved variables for this sweep
// Pg   .... "                 " primitive "                      "
// Fiph .... "                 " intercell flux in the dim-direction
// dim  .... Direction along which to take the sweep (x=1, y=2, z=3)
// -----------------------------------------------------------------------------
{
  const int N = Mara->domain->aug_shape()[dim-1];
  const int S = stride[dim];

  double *U = (double*) malloc(N*NQ*sizeof(double)); // Conserved
  double *P = (double*) malloc(N*NQ*sizeof(double)); // Primitive
  double *F = (double*) malloc(N*NQ*sizeof(double)); // Fluxes in along dim-axis
  double *A = (double*) malloc(N*   sizeof(double)); // Max wavespeed for dim

  for (int i=0; i<N; ++i) {
    // In this loop, i is the zone index, not the memory offset
    // -------------------------------------------------------------------------

    memcpy(U+i*NQ, Ug+i*S, NQ*sizeof(double));
    memcpy(P+i*NQ, Pg+i*S, NQ*sizeof(double));

    double ap, am;
    Mara->fluid->FluxAndEigenvalues(U+i*NQ, P+i*NQ, F+i*NQ, &ap, &am, dim);

    A[i] = (fabs(ap)>fabs(am)) ? fabs(ap) : fabs(am);
    if (MaxLambda < A[i]) MaxLambda = A[i];
  }
  double *Fiph_l = (double*) malloc(N*NQ*sizeof(double));
  intercell_flux_sweep(U, P, F, A, Fiph_l, dim);

  // Here we unload the local intercell fluxes, Fiph_l, into the global array,
  // Fiph_g.
  // ---------------------------------------------------------------------------
  for (int i=0; i<N; ++i) {
    memcpy(Fiph_g+i*S, Fiph_l+i*NQ, NQ*sizeof(double));
  }
  free(U); free(P); free(F); free(A); free(Fiph_l);
}


void Deriv::drive_sweeps_1d()
{
  const int Sx = stride[1];

  drive_single_sweep(&Uglb[0], &Pglb[0], &Fiph[0], 1);

  for (int i=Sx; i<stride[0]; ++i) {
    Lglb[i] = -(Fiph[i]-Fiph[i-Sx])/dx;
  }
}
void Deriv::drive_sweeps_2d()
{
  const int Nx = Mara->domain->get_N(1);
  const int Ny = Mara->domain->get_N(2);
  const int Ng = Mara->domain->get_Ng();
  const int Sx = stride[1];
  const int Sy = stride[2];

  for (int i=0; i<Nx+2*Ng; ++i) {
    drive_single_sweep(&Uglb[i*Sx], &Pglb[i*Sx], &Giph[i*Sx], 2);
  }
  for (int j=0; j<Ny+2*Ng; ++j) {
    drive_single_sweep(&Uglb[j*Sy], &Pglb[j*Sy], &Fiph[j*Sy], 1);
  }

  Mara->fluid->ConstrainedTransport2d(&Fiph[0], &Giph[0], stride);

  for (int i=Sx; i<stride[0]; ++i) {
    Lglb[i] = -(Fiph[i]-Fiph[i-Sx])/dx - (Giph[i]-Giph[i-Sy])/dy;
  }
}
void Deriv::drive_sweeps_3d()
{
  /*
    const int Nx = Mara->domain->get_N(1);
    const int Ny = Mara->domain->get_N(2);
    const int Nz = Mara->domain->get_N(3);
    const int Ng = Mara->domain->get_Ng();
    const int Sx = stride[1];
    const int Sy = stride[2];
    const int Sz = stride[3];

    // Not written yet
    // ---------------

    Mara->fluid->ConstrainedTransport3d(&Fiph[0], &Giph[0], &Hiph[0], stride);

    for (int i=Sx; i<stride[0]; ++i) {
    Lglb[i] = -(Fiph[i]-Fiph[i-Sx])/dx - (Giph[i]-Giph[i-Sy])/dy - (Hiph[i]-Hiph[i-Sz])/dz;
    }
  */
}





double weno5_fiph_apos(const double *f)
{
  const double eps = 1e-16;
  const double d[3] = { 0.3, 0.6, 0.1 };
  const double fiph[3] = {
    (  1.0/3.0)*f[ 0] + (  5.0/6.0)*f[ 1] + ( -1.0/6.0)*f[2],
    ( -1.0/6.0)*f[-1] + (  5.0/6.0)*f[ 0] + (  1.0/3.0)*f[1],
    (  1.0/3.0)*f[-2] + ( -7.0/6.0)*f[-1] + ( 11.0/6.0)*f[0]
  };
  const double B[3] = {
    (13.0/12.0)*pow(  f[ 0] - 2*f[ 1] +   f[ 2], 2.0) +
    ( 1.0/ 4.0)*pow(3*f[ 0] - 4*f[ 1] +   f[ 2], 2.0),

    (13.0/12.0)*pow(  f[-1] - 2*f[ 0] +   f[ 1], 2.0) +
    ( 1.0/ 4.0)*pow(  f[-1] - 0*f[ 0] -   f[ 1], 2.0),

    (13.0/12.0)*pow(  f[-2] - 2*f[-1] +   f[ 0], 2.0) +
    ( 1.0/ 4.0)*pow(  f[-2] - 4*f[-1] + 3*f[ 0], 2.0)
  };
  const double wgt[3] = {
    d[0] / pow(eps + B[0], 2.0),
    d[1] / pow(eps + B[1], 2.0),
    d[2] / pow(eps + B[2], 2.0)
  };
  const double wgt_ttl = wgt[0] + wgt[1] + wgt[2];
  return (wgt[0]*fiph[0] + wgt[1]*fiph[1] + wgt[2]*fiph[2]) / wgt_ttl;
}

double weno5_fiph_aneg(const double *f)
{
  const double eps = 1e-16;
  const double d[3] = { 0.1, 0.6, 0.3 };
  const double fiph[3] = {
    ( 11.0/6.0)*f[ 1] + ( -7.0/6.0)*f[ 2] + (  1.0/3.0)*f[3],
    (  1.0/3.0)*f[ 0] + (  5.0/6.0)*f[ 1] + ( -1.0/6.0)*f[2],
    ( -1.0/6.0)*f[-1] + (  5.0/6.0)*f[ 0] + (  1.0/3.0)*f[1],
  };
  const double B[3] = {
    (13.0/12.0)*pow(  f[ 1] - 2*f[ 2] +   f[ 3], 2.0) +
    ( 1.0/ 4.0)*pow(3*f[ 1] - 4*f[ 2] +   f[ 3], 2.0),

    (13.0/12.0)*pow(  f[ 0] - 2*f[ 1] +   f[ 2], 2.0) +
    ( 1.0/ 4.0)*pow(  f[ 0] - 0*f[ 1] -   f[ 2], 2.0),

    (13.0/12.0)*pow(  f[-1] - 2*f[ 0] +   f[ 1], 2.0) +
    ( 1.0/ 4.0)*pow(  f[-1] - 4*f[ 0] + 3*f[ 1], 2.0)
  };
  const double wgt[3] = {
    d[0] / pow(eps + B[0], 2.0),
    d[1] / pow(eps + B[1], 2.0),
    d[2] / pow(eps + B[2], 2.0)
  };
  const double wgt_ttl = wgt[0] + wgt[1] + wgt[2];
  return (wgt[0]*fiph[0] + wgt[1]*fiph[1] + wgt[2]*fiph[2]) / wgt_ttl;
}
