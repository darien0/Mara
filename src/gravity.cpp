#include "gravity.hpp"
#include "valman.hpp"
#include "eulers.hpp"
#include "config.h"
#include <mpi.h>
#include <complex>


#define SCALED_NOT 0
#define SCALED_YES 1

#define PERMUTE_NONE 0
#define FFT_FWD (-1)
#define FFT_REV (+1)


#if (__MARA_USE_FFTW && __MARA_USE_MPI)

extern "C" {
#include "fft_3d.h"
}

typedef std::complex<double> Complex;

static const Complex I = Complex(0,1);

enum { ddd, tau, Sx, Sy, Sz, Bx, By, Bz }; // Conserved
enum { rho, pre, vx, vy, vz };             // Primitive

static struct fft_plan_3d *call_fft_plan_3d(int *nbuf);
static double k_at(int i, int j, int k, double *khat);

std::valarray<double> GravSourceTerms::AddSources(std::valarray<double> &P)
{
  const int Nq = Mara->domain->get_Nq();
  const int Ng = Mara->domain->get_Ng();
  std::valarray<double> S(P.size());
  const int Nx = HydroModule::Mara->domain->GetLocalShape()[0];
  const int Ny = HydroModule::Mara->domain->GetLocalShape()[1];
  const int Nz = HydroModule::Mara->domain->GetLocalShape()[2];

  
  std::valarray<double> dens(Nx*Ny*Nz);
  std::valarray<double> f_guardless(Nx*Ny*Nz*3);
  std::valarray<double> f_guarded((Nx+2*Ng)*(Ny+2*Ng)*(Nz+2*Ng)*3);


// Two types of strides
// S stride has ghost zones
  int sx = (Ny+2*Ng)*(Nz+2*Ng); 
  int sy = (Nz+2*Ng);
  int sz = 1;
// T stride has no ghost zones
  int tx = Ny*Nz;
  int ty = Nz;
  int tz = 1;

  for (int i=0; i<Nx; i++){
   for (int j=0; j<Ny; j++){
    for (int k=0; k<Nz; k++){
     dens[i*tx + j*ty + k*tz] = P[((i+Ng)*sx+(j+Ng)*sy+(k+Ng)*sz)*Nq+rho];
     if (dens[i*tx+j*ty+k*tz] < 0.0){printf("Density went negative at %d,%d,%d\n", i,j,k);}
    }}}

  //GET GRAD OF PHI FROM DENS
  f_guardless = GetGravity(dens);

  for (int i=0; i<(Nx); i++){
   for (int j=0; j<(Ny); j++){
    for (int k=0; k<(Nz); k++){
     for (int dim=0; dim<3; dim++){
      f_guarded[((i+Ng)*sx+(j+Ng)*sy+(k+Ng)*sz)*3+dim] = 
                                           f_guardless[(i*tx+j*ty+k*tz)*3+dim];
//      printf("Here: %e\n",f_guarded[((i+Ng)*sx+(j+Ng)*sy+(k+Ng)*sz)*3+dim]);
    }}}}

  for (size_t i=0; i<P.size()/Nq; ++i) {

    std::slice m(i*Nq, Nq, 1);
    std::slice n(i*3, 3, 1);
    std::valarray<double> P0 = P[m];
    std::valarray<double> F0 = f_guarded[n];
    std::valarray<double> S0(Nq);

     //Calculate the Energy
  //  printf("Here: %e\n", P0[rho]*(P0[vx]*F0[0]+ P0[vy]*F0[1] + P0[vz]*F0[2]));
    S0[tau] = -P0[rho]*(P0[vx]*F0[0]+ P0[vy]*F0[1] + P0[vz]*F0[2]);
    S0[Sx]  = -P0[rho]*F0[0]; 
    S0[Sy]  = -P0[rho]*F0[1];
    S0[Sz]  = -P0[rho]*F0[2];
    S0[rho] = 0;

    S[m] = S0;
  }

  return S;
}








//This function takes a gaurdless array of the density, and returns a 
//gaurdless negative gradient of the potential
std::valarray<double> GravSourceTerms::GetGravity(std::valarray<double> &dens)
{
  const int Nx = HydroModule::Mara->domain->GetLocalShape()[0];
  const int Ny = HydroModule::Mara->domain->GetLocalShape()[1];
  const int Nz = HydroModule::Mara->domain->GetLocalShape()[2];

// T stride has no ghost zones
  int tx = Ny*Nz;
  int ty = Nz;
  int tz = 1;

  double kvec[3];

  int nbuf;
  struct fft_plan_3d *plan = call_fft_plan_3d(&nbuf);
  std::valarray<Complex> DensK(Nx*Ny*Nz);
  std::valarray<Complex> Dens(Nx*Ny*Nz);
  std::valarray<Complex> FKx(Nx*Ny*Nz);
  std::valarray<Complex> FKy(Nx*Ny*Nz);
  std::valarray<Complex> FKz(Nx*Ny*Nz);
  std::valarray<Complex> Fx(Nx*Ny*Nz);
  std::valarray<Complex> Fy(Nx*Ny*Nz);
  std::valarray<Complex> Fz(Nx*Ny*Nz);
  std::valarray<double> f(Nx*Ny*Nz*3);

  for (int i=0; i<Nx; i++){
   for (int j=0; j<Ny; j++){
    for (int k=0; k<Nz; k++){
     Dens[i*tx + j*ty + k*tz] = dens[i*tx + j*ty + k*tz];
    }}} 
  
  //preform fft
  fft_3d((FFT_DATA*)&Dens[0], (FFT_DATA*)&DensK[0], FFT_FWD, plan);

  for (int i=0; i<Nx; i++){
   for (int j=0; j<Ny; j++){
    for (int k=0; k<Nz; k++){
     double kmag = k_at(i,j,k,kvec);
     if (kmag < 1e-13){kmag = 1;}
     double ksqr = kmag*kmag;
     ////maybe some normalization here ?
FKx[i*tx+j*ty+k*tz] = -I*kvec[0]*DensK[i*tx+j*ty+k*tz]/ksqr * 4.0*M_PI*1.0;//1=G 
FKy[i*tx+j*ty+k*tz] = -I*kvec[1]*DensK[i*tx+j*ty+k*tz]/ksqr * 4.0*M_PI*1.0; 
FKz[i*tx+j*ty+k*tz] = -I*kvec[2]*DensK[i*tx+j*ty+k*tz]/ksqr * 4.0*M_PI*1.0; 

    }}}

  fft_3d( (FFT_DATA*)&FKx[0],(FFT_DATA*)&Fx[0], FFT_REV, plan); 
  fft_3d( (FFT_DATA*)&FKy[0],(FFT_DATA*)&Fy[0], FFT_REV, plan);
  fft_3d( (FFT_DATA*)&FKz[0],(FFT_DATA*)&Fz[0], FFT_REV, plan); 
  fft_3d_destroy_plan(plan);

//Reverse fft
  for (int i=0; i<Nx; i++){
   for (int j=0; j<Ny; j++){
    for (int k=0; k<Nz; k++){
  f[(i*tx+j*ty+k*tz)*3+0] = Fx[i*tx+j*ty+k*tz].real(); 
  //printf("Small? %e\n",Fx[i*tx+j*ty+k*tz].real());
  f[(i*tx+j*ty+k*tz)*3+1] = Fy[i*tx+j*ty+k*tz].real(); 
  f[(i*tx+j*ty+k*tz)*3+2] = Fz[i*tx+j*ty+k*tz].real();   
    }}}


return f;

}











 ////////////////////////////////////////////////////////
////////////////////FOURIER STUFF/////////////////////////
 ////////////////////////////////////////////////////////






double k_at(int i, int j, int k, double *kvec)
// -----------------------------------------------------------------------------
// Here, we populate the wave vectors on the Fourier lattice. The convention
// used by FFTW is the same as that used by numpy, described at the link
// below. For N odd, the (positive) Nyquist frequency is placed in the middle
// bin.
//
// http://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.fftfreq.html
// -----------------------------------------------------------------------------
{
  i += HydroModule::Mara->domain->GetGlobalStart()[0];
  j += HydroModule::Mara->domain->GetGlobalStart()[1];
  k += HydroModule::Mara->domain->GetGlobalStart()[2];

  const int Nx = HydroModule::Mara->domain->GetGlobalShape()[0];
  const int Ny = HydroModule::Mara->domain->GetGlobalShape()[1];
  const int Nz = HydroModule::Mara->domain->GetGlobalShape()[2];

  kvec[0] = (Nx % 2 == 0) ?
    ((i<  Nx   /2) ? i : i-Nx):  // N even
    ((i<=(Nx-1)/2) ? i : i-Nx);  // N odd

  kvec[1] = (Ny % 2 == 0) ?
    ((j<  Ny   /2) ? j : j-Ny):
    ((j<=(Ny-1)/2) ? j : j-Ny);

  kvec[2] = (Nz % 2 == 0) ?
    ((k<  Nz   /2) ? k : k-Nz):
    ((k<=(Nz-1)/2) ? k : k-Nz);

  return sqrt(kvec[0]*kvec[0] + kvec[1]*kvec[1] + kvec[2]*kvec[2]);
}


struct fft_plan_3d *call_fft_plan_3d(int *nbuf)
{
  const int i0 = HydroModule::Mara->domain->GetGlobalStart()[0];
  const int i1 = HydroModule::Mara->domain->GetLocalShape()[0] + i0-1;

  const int j0 = HydroModule::Mara->domain->GetGlobalStart()[1];
  const int j1 = HydroModule::Mara->domain->GetLocalShape()[1] + j0-1;

  const int k0 = HydroModule::Mara->domain->GetGlobalStart()[2];
  const int k1 = HydroModule::Mara->domain->GetLocalShape()[2] + k0-1;

  const int Nx = HydroModule::Mara->domain->GetGlobalShape()[0];
  const int Ny = HydroModule::Mara->domain->GetGlobalShape()[1];
  const int Nz = HydroModule::Mara->domain->GetGlobalShape()[2];

  return fft_3d_create_plan(MPI_COMM_WORLD,
                            Nz, Ny, Nx,
                            k0,k1, j0,j1, i0,i1,
                            k0,k1, j0,j1, i0,i1,
                            SCALED_YES, PERMUTE_NONE, nbuf);
}
#endif // (__MARA_USE_FFTW && __MARA_USE_MPI)
