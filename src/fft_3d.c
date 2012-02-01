/* parallel FFT functions - 1998, 1999

   Steve Plimpton, MS 1111, Dept 9221, Sandia National Labs
   (505) 845-7873
   sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

   See the README file in the top-level directory of the distribution.
*/


#include "config.h"
#if (__MARA_USE_FFTW)

#include "stdio.h"
#include "mpi.h"

#include "pack_3d.h"
#include "remap_3d.h"
#include "fft_3d.h"

#define MIN(A,B) ((A) < (B)) ? (A) : (B)
#define MAX(A,B) ((A) > (B)) ? (A) : (B)

/* ------------------------------------------------------------------- */
/* Data layout for 3d FFTs:

   data set of Nfast x Nmid x Nslow elements is owned by P procs
   on input, each proc owns a subsection of the elements
   on output, each proc will own a (possibly different) subsection
   my subsection must not overlap with any other proc's subsection,
     i.e. the union of all proc's input (or output) subsections must
     exactly tile the global Nfast x Nmid x Nslow data set
   when called from C, all subsection indices are 
     C-style from 0 to N-1 where N = Nfast or Nmid or Nslow
   when called from F77, all subsection indices are 
     F77-style from 1 to N where N = Nfast or Nmid or Nslow
   a proc can own 0 elements on input or output
     by specifying hi index < lo index
   on both input and output, data is stored contiguously on a processor
     with a fast-varying, mid-varying, and slow-varying index
*/
/* ------------------------------------------------------------------- */

/* ------------------------------------------------------------------- */
/* Perform 3d FFT */

/* Arguments:

   in           starting address of input data on this proc
   out          starting address of where output data for this proc
                  will be placed (can be same as in)
   flag         1 for forward FFT, -1 for inverse FFT
   plan         plan returned by previous call to fft_3d_create_plan
*/

void fft_3d(FFT_DATA *in, FFT_DATA *out, int flag, struct fft_plan_3d *plan)

{
  int i,total,length,/*offset,*/num;
  double norm;
  FFT_DATA *data,*copy;

/* system specific constants */

#ifdef FFT_DEC
  char c = 'C';
  char f = 'F';
  char b = 'B';
  int one = 1;
#endif
#ifdef FFT_T3E
  int isys = 0;
  double scalef = 1.0;
#endif

/* pre-remap to prepare for 1st FFTs if needed
   copy = loc for remap result */

  if (plan->pre_plan) {
    if (plan->pre_target == 0)
      copy = out;
    else
      copy = plan->copy;
    remap_3d((double *) in, (double *) copy, (double *) plan->scratch,
	     plan->pre_plan);
    data = copy;
  }
  else
    data = in;

/* 1d FFTs along fast axis */

  total = plan->total1;
  length = plan->length1;

#ifdef FFT_SGI
  for (offset = 0; offset < total; offset += length)
    FFT_1D(flag,length,&data[offset],1,plan->coeff1);
#endif
#ifdef FFT_INTEL
  for (offset = 0; offset < total; offset += length)
    FFT_1D(&data[offset],&length,&flag,plan->coeff1);
#endif
#ifdef FFT_DEC
  if (flag == -1)
    for (offset = 0; offset < total; offset += length)
      FFT_1D(&c,&c,&f,&data[offset],&data[offset],&length,&one);
  else
    for (offset = 0; offset < total; offset += length)
      FFT_1D(&c,&c,&b,&data[offset],&data[offset],&length,&one);
#endif
#ifdef FFT_T3E
  for (offset = 0; offset < total; offset += length)
    FFT_1D(&flag,&length,&scalef,&data[offset],&data[offset],plan->coeff1,
	   plan->work1,&isys);
#endif
#ifdef FFT_FFTW
  if (flag == -1)
    fftw(plan->plan_fast_forward,total/length,data,1,length,NULL,0,0);
  else
    fftw(plan->plan_fast_backward,total/length,data,1,length,NULL,0,0);
#endif

/* 1st mid-remap to prepare for 2nd FFTs
   copy = loc for remap result */

  if (plan->mid1_target == 0)
    copy = out;
  else
    copy = plan->copy;
  remap_3d((double *) data, (double *) copy, (double *) plan->scratch,
	   plan->mid1_plan);
  data = copy;

/* 1d FFTs along mid axis */

  total = plan->total2;
  length = plan->length2;

#ifdef FFT_SGI
  for (offset = 0; offset < total; offset += length)
    FFT_1D(flag,length,&data[offset],1,plan->coeff2);
#endif
#ifdef FFT_INTEL
  for (offset = 0; offset < total; offset += length)
    FFT_1D(&data[offset],&length,&flag,plan->coeff2);
#endif
#ifdef FFT_DEC
  if (flag == -1)
    for (offset = 0; offset < total; offset += length)
      FFT_1D(&c,&c,&f,&data[offset],&data[offset],&length,&one);
  else
    for (offset = 0; offset < total; offset += length)
      FFT_1D(&c,&c,&b,&data[offset],&data[offset],&length,&one);
#endif
#ifdef FFT_T3E
  for (offset = 0; offset < total; offset += length)
    FFT_1D(&flag,&length,&scalef,&data[offset],&data[offset],plan->coeff2,
	   plan->work2,&isys);
#endif
#ifdef FFT_FFTW
  if (flag == -1)
    fftw(plan->plan_mid_forward,total/length,data,1,length,NULL,0,0);
  else
    fftw(plan->plan_mid_backward,total/length,data,1,length,NULL,0,0);
#endif

/* 2nd mid-remap to prepare for 3rd FFTs
   copy = loc for remap result */

  if (plan->mid2_target == 0)
    copy = out;
  else
    copy = plan->copy;
  remap_3d((double *) data, (double *) copy, (double *) plan->scratch,
	   plan->mid2_plan);
  data = copy;

/* 1d FFTs along slow axis */

  total = plan->total3;
  length = plan->length3;

#ifdef FFT_SGI
  for (offset = 0; offset < total; offset += length)
    FFT_1D(flag,length,&data[offset],1,plan->coeff3);
#endif
#ifdef FFT_INTEL
  for (offset = 0; offset < total; offset += length)
    FFT_1D(&data[offset],&length,&flag,plan->coeff3);
#endif
#ifdef FFT_DEC
  if (flag == -1)
    for (offset = 0; offset < total; offset += length)
      FFT_1D(&c,&c,&f,&data[offset],&data[offset],&length,&one);
  else
    for (offset = 0; offset < total; offset += length)
      FFT_1D(&c,&c,&b,&data[offset],&data[offset],&length,&one);
#endif
#ifdef FFT_T3E
  for (offset = 0; offset < total; offset += length)
    FFT_1D(&flag,&length,&scalef,&data[offset],&data[offset],plan->coeff3,
	   plan->work3,&isys);
#endif
#ifdef FFT_FFTW
  if (flag == -1)
    fftw(plan->plan_slow_forward,total/length,data,1,length,NULL,0,0);
  else
    fftw(plan->plan_slow_backward,total/length,data,1,length,NULL,0,0);
#endif

/* post-remap to put data in output format if needed
   destination is always out */

  if (plan->post_plan)
    remap_3d((double *) data, (double *) out, (double *) plan->scratch,
	     plan->post_plan);

/* scaling if required */

#ifndef FFT_T3E
  if (flag == 1 && plan->scaled) {
    norm = plan->norm;
    num = plan->normnum;
    for (i = 0; i < num; i++) {
      out[i].re *= norm;
      out[i].im *= norm;
    }
  }
#endif

#ifdef FFT_T3E
  if (flag == 1 && plan->scaled) {
    norm = plan->norm;
    num = plan->normnum;
    for (i = 0; i < num; i++)
      out[i] *= (norm,norm);
  }
#endif

}

/* ------------------------------------------------------------------- */
/* Create plan for performing a 3d FFT */

/* Arguments:

   comm                 MPI communicator for the P procs which own the data
   nfast,nmid,nslow     size of global 3d matrix
   in_ilo,in_ihi        input bounds of data I own in fast index
   in_jlo,in_jhi        input bounds of data I own in mid index
   in_klo,in_khi        input bounds of data I own in slow index
   out_ilo,out_ihi      output bounds of data I own in fast index
   out_jlo,out_jhi      output bounds of data I own in mid index
   out_klo,out_khi      output bounds of data I own in slow index
   scaled               0 = no scaling of result, 1 = scaling
   permute              permutation in storage order of indices on output
                          0 = no permutation
			  1 = permute once = mid->fast, slow->mid, fast->slow
			  2 = permute twice = slow->fast, fast->mid, mid->slow
   nbuf                 returns size of internal storage buffers used by FFT
*/

struct fft_plan_3d *fft_3d_create_plan(
       MPI_Comm comm, int nfast, int nmid, int nslow,
       int in_ilo, int in_ihi, int in_jlo, int in_jhi,
       int in_klo, int in_khi,
       int out_ilo, int out_ihi, int out_jlo, int out_jhi,
       int out_klo, int out_khi,
       int scaled, int permute, int *nbuf)

{
  struct fft_plan_3d *plan;
  int me,nprocs;
  int /*i,num,*/flag,remapflag;//,fftflag;
  int first_ilo,first_ihi,first_jlo,first_jhi,first_klo,first_khi;
  int second_ilo,second_ihi,second_jlo,second_jhi,second_klo,second_khi;
  int third_ilo,third_ihi,third_jlo,third_jhi,third_klo,third_khi;
  int out_size,first_size,second_size,third_size,copy_size,scratch_size;
  int np1,np2,ip1,ip2;
  //  int list[50];

/* system specific variables */

#ifdef FFT_INTEL
  FFT_DATA dummy;
#endif
#ifdef FFT_T3E
  FFT_DATA dummy[5];
  int isign,isys;
  double scalef;
#endif

/* query MPI info */

  MPI_Comm_rank(comm,&me);
  MPI_Comm_size(comm,&nprocs);

/* compute division of procs in 2 dimensions not on-processor */

  bifactor(nprocs,&np1,&np2);
  ip1 = me % np1;
  ip2 = me/np1;

/* allocate memory for plan data struct */

  plan = (struct fft_plan_3d *) malloc(sizeof(struct fft_plan_3d));
  if (plan == NULL) return NULL;

/* remap from initial distribution to layout needed for 1st set of 1d FFTs
   not needed if all procs own entire fast axis initially
   first indices = distribution after 1st set of FFTs */

  if (in_ilo == 0 && in_ihi == nfast-1)
    flag = 0;
  else
    flag = 1;

  MPI_Allreduce(&flag,&remapflag,1,MPI_INT,MPI_MAX,comm);

  if (remapflag == 0) {
    first_ilo = in_ilo;
    first_ihi = in_ihi;
    first_jlo = in_jlo;
    first_jhi = in_jhi;
    first_klo = in_klo;
    first_khi = in_khi;
    plan->pre_plan = NULL;
  }
  else {
    first_ilo = 0;
    first_ihi = nfast - 1;
    first_jlo = ip1*nmid/np1;
    first_jhi = (ip1+1)*nmid/np1 - 1;
    first_klo = ip2*nslow/np2;
    first_khi = (ip2+1)*nslow/np2 - 1;
    plan->pre_plan =
      remap_3d_create_plan(comm,in_ilo,in_ihi,in_jlo,in_jhi,in_klo,in_khi,
			   first_ilo,first_ihi,first_jlo,first_jhi,
			   first_klo,first_khi,
			   FFT_PRECISION,0,0,2);
    if (plan->pre_plan == NULL) return NULL;
  }

/* 1d FFTs along fast axis */

  plan->length1 = nfast;
  plan->total1 = nfast * (first_jhi-first_jlo+1) * (first_khi-first_klo+1);

/* remap from 1st to 2nd FFT
   choose which axis is split over np1 vs np2 to minimize communication
   second indices = distribution after 2nd set of FFTs */

  second_ilo = ip1*nfast/np1;
  second_ihi = (ip1+1)*nfast/np1 - 1;
  second_jlo = 0;
  second_jhi = nmid - 1;
  second_klo = ip2*nslow/np2;
  second_khi = (ip2+1)*nslow/np2 - 1;
  plan->mid1_plan =
      remap_3d_create_plan(comm,
			   first_ilo,first_ihi,first_jlo,first_jhi,
			   first_klo,first_khi,
			   second_ilo,second_ihi,second_jlo,second_jhi,
			   second_klo,second_khi,
			   FFT_PRECISION,1,0,2);
  if (plan->mid1_plan == NULL) return NULL;

/* 1d FFTs along mid axis */

  plan->length2 = nmid;
  plan->total2 = (second_ihi-second_ilo+1) * nmid * (second_khi-second_klo+1);

/* remap from 2nd to 3rd FFT
   if final distribution is permute=2 with all procs owning entire slow axis
     then this remapping goes directly to final distribution
   third indices = distribution after 3rd set of FFTs */

  if (permute == 2 && out_klo == 0 && out_khi == nslow-1)
    flag = 0;
  else
    flag = 1;

  MPI_Allreduce(&flag,&remapflag,1,MPI_INT,MPI_MAX,comm);

  if (remapflag == 0) {
    third_ilo = out_ilo;
    third_ihi = out_ihi;
    third_jlo = out_jlo;
    third_jhi = out_jhi;
    third_klo = out_klo;
    third_khi = out_khi;
  }
  else {
    third_ilo = ip1*nfast/np1;
    third_ihi = (ip1+1)*nfast/np1 - 1;
    third_jlo = ip2*nmid/np2;
    third_jhi = (ip2+1)*nmid/np2 - 1;
    third_klo = 0;
    third_khi = nslow - 1;
  }
  
  plan->mid2_plan =
    remap_3d_create_plan(comm,
			 second_jlo,second_jhi,second_klo,second_khi,
			 second_ilo,second_ihi,
			 third_jlo,third_jhi,third_klo,third_khi,
			 third_ilo,third_ihi,
			 FFT_PRECISION,1,0,2);
  if (plan->mid2_plan == NULL) return NULL;

/* 1d FFTs along slow axis */

  plan->length3 = nslow;
  plan->total3 = (third_ihi-third_ilo+1) * (third_jhi-third_jlo+1) * nslow;

/* remap from 3rd FFT to final distribution
   not needed if permute = 2 and third indices = out indices on all procs */

  if (permute == 2 &&
      out_ilo == third_ilo && out_ihi == third_ihi &&
      out_jlo == third_jlo && out_jhi == third_jhi &&
      out_klo == third_klo && out_khi == third_khi)
    flag = 0;
  else
    flag = 1;

  MPI_Allreduce(&flag,&remapflag,1,MPI_INT,MPI_MAX,comm);

  if (remapflag == 0)
    plan->post_plan = NULL;
  else {
    plan->post_plan =
      remap_3d_create_plan(comm,
			   third_klo,third_khi,third_ilo,third_ihi,
			   third_jlo,third_jhi,
			   out_klo,out_khi,out_ilo,out_ihi,
			   out_jlo,out_jhi,
			   FFT_PRECISION,(permute+1)%3,0,2);
    if (plan->post_plan == NULL) return NULL;
  }

/* configure plan memory pointers and allocate work space
   out_size = amount of memory given to FFT by user
   first/second/third_size = amount of memory needed after pre,mid1,mid2 remaps
   copy_size = amount needed internally for extra copy of data
   scratch_size = amount needed internally for remap scratch space
   for each remap:
     use out space for result if big enough, else require copy buffer
     accumulate largest required remap scratch space */

  out_size = (out_ihi-out_ilo+1) * (out_jhi-out_jlo+1) * (out_khi-out_klo+1);
  first_size = (first_ihi-first_ilo+1) * (first_jhi-first_jlo+1) * 
    (first_khi-first_klo+1);
  second_size = (second_ihi-second_ilo+1) * (second_jhi-second_jlo+1) * 
    (second_khi-second_klo+1);
  third_size = (third_ihi-third_ilo+1) * (third_jhi-third_jlo+1) * 
    (third_khi-third_klo+1);

  copy_size = 0;
  scratch_size = 0;

  if (plan->pre_plan) {
    if (first_size <= out_size)
      plan->pre_target = 0;
    else {
      plan->pre_target = 1;
      copy_size = MAX(copy_size,first_size);
    }
    scratch_size = MAX(scratch_size,first_size);
  }

  if (plan->mid1_plan) {
    if (second_size <= out_size)
      plan->mid1_target = 0;
    else {
      plan->mid1_target = 1;
      copy_size = MAX(copy_size,second_size);
    }
    scratch_size = MAX(scratch_size,second_size);
  }

  if (plan->mid2_plan) {
    if (third_size <= out_size)
      plan->mid2_target = 0;
    else {
      plan->mid2_target = 1;
      copy_size = MAX(copy_size,third_size);
    }
    scratch_size = MAX(scratch_size,third_size);
  }

  if (plan->post_plan)
    scratch_size = MAX(scratch_size,out_size);

  *nbuf = copy_size + scratch_size;

  if (copy_size) {
    plan->copy = (FFT_DATA *) malloc(copy_size*sizeof(FFT_DATA));
    if (plan->copy == NULL) return NULL;
  }
  else plan->copy = NULL;

  if (scratch_size) {
    plan->scratch = (FFT_DATA *) malloc(scratch_size*sizeof(FFT_DATA));
    if (plan->scratch == NULL) return NULL;
  }
  else plan->scratch = NULL;

/* system specific pre-computation of 1d FFT coeffs 
   and scaling normalization */

#ifdef FFT_SGI

  plan->coeff1 = (FFT_DATA *) malloc((nfast+15)*sizeof(FFT_DATA));
  plan->coeff2 = (FFT_DATA *) malloc((nmid+15)*sizeof(FFT_DATA));
  plan->coeff3 = (FFT_DATA *) malloc((nslow+15)*sizeof(FFT_DATA));

  if (plan->coeff1 == NULL || plan->coeff2 == NULL ||
      plan->coeff3 == NULL) return NULL;

  FFT_1D_INIT(nfast,plan->coeff1);
  FFT_1D_INIT(nmid,plan->coeff2);
  FFT_1D_INIT(nslow,plan->coeff3);

  if (scaled == 0) 
    plan->scaled = 0;
  else {
    plan->scaled = 1;
    plan->norm = 1.0/(nfast*nmid*nslow);
    plan->normnum = (out_ihi-out_ilo+1) * (out_jhi-out_jlo+1) *
      (out_khi-out_klo+1);
  }

#endif

#ifdef FFT_INTEL

  flag = 0;

  num = 0;
  factor(nfast,&num,list);
  for (i = 0; i < num; i++)
    if (list[i] != 2 && list[i] != 3 && list[i] != 5) flag = 1;
  num = 0;
  factor(nmid,&num,list);
  for (i = 0; i < num; i++)
    if (list[i] != 2 && list[i] != 3 && list[i] != 5) flag = 1;
  num = 0;
  factor(nslow,&num,list);
  for (i = 0; i < num; i++)
    if (list[i] != 2 && list[i] != 3 && list[i] != 5) flag = 1;

  MPI_Allreduce(&flag,&fftflag,1,MPI_INT,MPI_MAX,comm);
  if (fftflag) {
    if (me == 0) printf("ERROR: FFTs are not power of 2,3,5\n");
    return NULL;
  }

  plan->coeff1 = (FFT_DATA *) malloc((3*nfast/2+1)*sizeof(FFT_DATA));
  plan->coeff2 = (FFT_DATA *) malloc((3*nmid/2+1)*sizeof(FFT_DATA));
  plan->coeff3 = (FFT_DATA *) malloc((3*nslow/2+1)*sizeof(FFT_DATA));

  if (plan->coeff1 == NULL || plan->coeff2 == NULL || 
      plan->coeff3 == NULL) return NULL;

  flag = 0;
  FFT_1D_INIT(&dummy,&nfast,&flag,plan->coeff1);
  FFT_1D_INIT(&dummy,&nmid,&flag,plan->coeff2);
  FFT_1D_INIT(&dummy,&nslow,&flag,plan->coeff3);

  if (scaled == 0) {
    plan->scaled = 1;
    plan->norm = nfast*nmid*nslow;
    plan->normnum = (out_ihi-out_ilo+1) * (out_jhi-out_jlo+1) *
      (out_khi-out_klo+1);
  }
  else
    plan->scaled = 0;

#endif

#ifdef FFT_DEC

  if (scaled == 0) {
    plan->scaled = 1;
    plan->norm = nfast*nmid*nslow;
    plan->normnum = (out_ihi-out_ilo+1) * (out_jhi-out_jlo+1) *
      (out_khi-out_klo+1);
  }
  else
    plan->scaled = 0;

#endif

#ifdef FFT_T3E

  plan->coeff1 = (double *) malloc((12*nfast)*sizeof(double));
  plan->coeff2 = (double *) malloc((12*nmid)*sizeof(double));
  plan->coeff3 = (double *) malloc((12*nslow)*sizeof(double));

  if (plan->coeff1 == NULL || plan->coeff2 == NULL || 
      plan->coeff3 == NULL) return NULL;

  plan->work1 = (double *) malloc((8*nfast)*sizeof(double));
  plan->work2 = (double *) malloc((8*nmid)*sizeof(double));
  plan->work3 = (double *) malloc((8*nslow)*sizeof(double));

  if (plan->work1 == NULL || plan->work2 == NULL || 
      plan->work3 == NULL) return NULL;

  isign = 0;
  scalef = 1.0;
  isys = 0;

  FFT_1D_INIT(&isign,&nfast,&scalef,dummy,dummy,plan->coeff1,dummy,&isys);
  FFT_1D_INIT(&isign,&nmid,&scalef,dummy,dummy,plan->coeff2,dummy,&isys);
  FFT_1D_INIT(&isign,&nslow,&scalef,dummy,dummy,plan->coeff3,dummy,&isys);

  if (scaled == 0) 
    plan->scaled = 0;
  else {
    plan->scaled = 1;
    plan->norm = 1.0/(nfast*nmid*nslow);
    plan->normnum = (out_ihi-out_ilo+1) * (out_jhi-out_jlo+1) *
      (out_khi-out_klo+1);
  }

#endif

#ifdef FFT_FFTW

  plan->plan_fast_forward = 
    fftw_create_plan(nfast,FFTW_FORWARD,FFTW_ESTIMATE | FFTW_IN_PLACE);
  plan->plan_fast_backward = 
    fftw_create_plan(nfast,FFTW_BACKWARD,FFTW_ESTIMATE | FFTW_IN_PLACE);

  if (nmid == nfast) {
    plan->plan_mid_forward = plan->plan_fast_forward;
    plan->plan_mid_backward = plan->plan_fast_backward;
  }
  else {
    plan->plan_mid_forward = 
      fftw_create_plan(nmid,FFTW_FORWARD,FFTW_ESTIMATE | FFTW_IN_PLACE);
    plan->plan_mid_backward = 
      fftw_create_plan(nmid,FFTW_BACKWARD,FFTW_ESTIMATE | FFTW_IN_PLACE);
  }

  if (nslow == nfast) {
    plan->plan_slow_forward = plan->plan_fast_forward;
    plan->plan_slow_backward = plan->plan_fast_backward;
  }
  else if (nslow == nmid) {
    plan->plan_slow_forward = plan->plan_mid_forward;
    plan->plan_slow_backward = plan->plan_mid_backward;
  }
  else {
    plan->plan_slow_forward = 
      fftw_create_plan(nslow,FFTW_FORWARD,FFTW_ESTIMATE | FFTW_IN_PLACE);
    plan->plan_slow_backward = 
      fftw_create_plan(nslow,FFTW_BACKWARD,FFTW_ESTIMATE | FFTW_IN_PLACE);
  }

  if (scaled == 0)
    plan->scaled = 0;
  else {
    plan->scaled = 1;
    plan->norm = 1.0/(nfast*nmid*nslow);
    plan->normnum = (out_ihi-out_ilo+1) * (out_jhi-out_jlo+1) *
      (out_khi-out_klo+1);
  }

#endif

  return plan;
}

/* ------------------------------------------------------------------- */
/* Destroy a 3d fft plan */

void fft_3d_destroy_plan(struct fft_plan_3d *plan)

{
  if (plan->pre_plan) remap_3d_destroy_plan(plan->pre_plan);
  if (plan->mid1_plan) remap_3d_destroy_plan(plan->mid1_plan);
  if (plan->mid2_plan) remap_3d_destroy_plan(plan->mid2_plan);
  if (plan->post_plan) remap_3d_destroy_plan(plan->post_plan);

  if (plan->copy) free(plan->copy);
  if (plan->scratch) free(plan->scratch);

#ifdef FFT_SGI
  free(plan->coeff1);
  free(plan->coeff2);
  free(plan->coeff3);
#endif
#ifdef FFT_INTEL
  free(plan->coeff1);
  free(plan->coeff2);
  free(plan->coeff3);
#endif
#ifdef FFT_T3E
  free(plan->coeff1);
  free(plan->coeff2);
  free(plan->coeff3);
  free(plan->work1);
  free(plan->work2);
  free(plan->work3);
#endif
#ifdef FFT_FFTW
  if (plan->plan_slow_forward != plan->plan_mid_forward &&
      plan->plan_slow_forward != plan->plan_fast_forward) {
    fftw_destroy_plan(plan->plan_slow_forward);
    fftw_destroy_plan(plan->plan_slow_backward);
  }
  if (plan->plan_mid_forward != plan->plan_fast_forward) {
    fftw_destroy_plan(plan->plan_mid_forward);
    fftw_destroy_plan(plan->plan_mid_backward);
  }
  fftw_destroy_plan(plan->plan_fast_forward);
  fftw_destroy_plan(plan->plan_fast_backward);
#endif

  free(plan);
}


#else
void __fft_3d_stub() { }
#endif // __MARA_USE_FFTW
