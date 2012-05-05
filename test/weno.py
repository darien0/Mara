#!/usr/bin/env python

import numpy as np

rho = 0
pre = 1
vx = 2
vy = 3
vz = 4

nrg = 1
px = 2
py = 3
pz = 4

Gamma = 1.4


def sound_speed(P):
    return (Gamma * P[pre]/P[rho])**0.5

def flux(P):
    F = np.zeros(5)
    U = prim_to_cons(P)
    F[rho]  =  U[rho] * P[vx];
    F[nrg]  = (U[nrg] + P[pre])*P[vx]
    F[px]   =  U[px]  * P[vx] + P[pre]
    F[py]   =  U[py]  * P[vx]
    F[pz]   =  U[pz]  * P[vx]
    return F

def cons_to_prim(U):
    P = np.zeros(5)
    gm1 = Gamma - 1.0
    P[rho] = U[rho]
    P[pre] =(U[nrg] - 0.5*(U[px]*U[px] + U[py]*U[py] + U[pz]*U[pz])/U[rho])*gm1
    P[vx ] = U[px ] / U[rho]
    P[vy ] = U[py ] / U[rho]
    P[vz ] = U[pz ] / U[rho]
    return P

def prim_to_cons(P):
    U = np.zeros(5)
    gm1 = Gamma - 1.0
    U[rho] = P[rho]
    U[px]  = P[rho] * P[vx]
    U[py]  = P[rho] * P[vy]
    U[pz]  = P[rho] * P[vz]
    U[nrg] = P[rho] * 0.5*(P[vx]*P[vx] + P[vy]*P[vy] + P[vz]*P[vz]) + P[pre]/gm1
    return U

def max_wavespeed(P, take_abs=True):
    cs = sound_speed(P)
    ap, am = P[vx] + cs, P[vx] - cs
    if take_abs:
        return max(abs(ap), abs(am))
    else:
        return ap, am
    
def left_right_eigenvectors(P):
    U = prim_to_cons(P)
    gm = Gamma
    gm1 = gm - 1.0
    u = P[vx]
    v = P[vy]
    w = P[vz]
    V2 = u*u + v*v + w*w
    a = (gm * P[pre] / P[rho])**0.5
    H = (U[nrg] + P[pre]) / P[rho]

    # Toro Equation 3.82
    # --------------------------------------------------------------------------
    RR = [[       1,      1,      0,      0,     1   ],
          [     u-a,      u,      0,      0,     u+a ],
          [       v,      v,      1,      0,     v   ],
          [       w,      w,      0,      1,     w   ],
          [ H - u*a, 0.5*V2,      v,      w, H + u*a ]]
    
    # Toro Equation 3.83 up to (gam - 1) / (2*a^2)
    # --------------------------------------------------------------------------
    LL = [[    H + (a/gm1)*(u-a),  -(u+a/gm1),        -v,        -w,  1 ],
          [ -2*H + (4/gm1)*(a*a),         2*u,       2*v,       2*w, -2 ],
          [         -2*v*a*a/gm1,           0, 2*a*a/gm1,         0,  0 ],
          [         -2*w*a*a/gm1,           0,         0, 2*a*a/gm1,  0 ],
          [    H - (a/gm1)*(u+a),  -(u-a/gm1),        -v,        -w,  1 ]]

    norm = gm1 / (2*a*a)
    return np.matrix(LL)*norm, np.matrix(RR)


def weno5(v, c, d):
    eps = 1e-16
    B = [(13.0/12.0)*(  v[ 2] - 2*v[ 3] +   v[ 4])**2 +
         ( 1.0/ 4.0)*(3*v[ 2] - 4*v[ 3] +   v[ 4])**2,

         (13.0/12.0)*(  v[ 1] - 2*v[ 2] +   v[ 3])**2 +
         ( 1.0/ 4.0)*(  v[ 1] - 0*v[ 2] -   v[ 3])**2,
         
         (13.0/12.0)*(  v[ 0] - 2*v[ 1] +   v[ 2])**2 +
         ( 1.0/ 4.0)*(  v[ 0] - 4*v[ 1] + 3*v[ 2])**2]

    vs = [c[0][0]*v[ 2] + c[0][1]*v[ 3] + c[0][2]*v[4],
          c[1][0]*v[ 1] + c[1][1]*v[ 2] + c[1][2]*v[3],
          c[2][0]*v[ 0] + c[2][1]*v[ 1] + c[2][2]*v[2]]

    w = [d[0] / (eps + B[0])**2,
         d[1] / (eps + B[1])**2,
         d[2] / (eps + B[2])**2]
    
    wtot = w[0] + w[1] + w[2]
    return (w[0]*vs[0] + w[1]*vs[1] + w[2]*vs[2])/wtot



CeesC2L = [ [11./6., -7./6.,  1./3. ],
            [ 1./3.,  5./6., -1./6. ],
            [-1./6.,  5./6.,  1./3. ] ]
CeesC2R = [ [ 1./3.,  5./6., -1./6. ],
            [-1./6.,  5./6.,  1./3. ],
            [ 1./3., -7./6., 11./6. ] ]
DeesC2L = [ 0.1, 0.6, 0.3 ]
DeesC2R = [ 0.3, 0.6, 0.1 ]



# A 4-letter variable means a domain-global array. 1-letter variables are
# 6-components lists of vectors. 2 letter variables, like LL and RR are
# matrices. The indices 0 ... 5 inclusively label the zones surrounding the
# i+1/2 interface.

def get_weno_flux(Cons, Prim, Flux, Mlam, i):
    Ptest = [1,1,0,0,0]
    LL, RR = left_right_eigenvectors(0.5*(Prim[i] + Prim[i+1]))

    P = [Prim[i-2+j] for j in range(6)]
    U = [Cons[i-2+j] for j in range(6)]
    F = [Flux[i-2+j] for j in range(6)]
    A = [Mlam[i-2+j] for j in range(6)]

    ml = max(A)

    Fp = [np.matrix(0.5*(F[j] + ml*U[j])).T for j in range(6)]
    Fm = [np.matrix(0.5*(F[j] - ml*U[j])).T for j in range(6)]

    fp = [np.array(LL*Fp[j])[:,0] for j in range(6)]
    fm = [np.array(LL*Fm[j])[:,0] for j in range(6)]

    f = np.matrix(weno5(fp[0:5], CeesC2R, DeesC2R) +
                  weno5(fm[1:6], CeesC2L, DeesC2L)).T

    return np.array(RR*f)[:,0]


def get_hll_flux(Cons, Prim, Flux, Mlam, i):
    U, P, F = Cons, Prim, Flux

    epl, eml = max_wavespeed(Prim[i  ], take_abs=False)
    epr, emr = max_wavespeed(Prim[i+1], take_abs=False)

    ap = max(epl, epr, 0.0)
    am = min(eml, emr, 0.0)
    return (ap*F[i] - am*F[i+1] + ap*am*(U[i+1] - U[i])) / (ap - am)


def set_periodic_bc(A, Ng):
    Nx = A.shape[0] - 2*Ng
    A[:Ng] = A[Nx-1:Nx+Ng-1]
    A[-Ng:] = A[Ng+1:2*Ng+1]


def set_outflow_bc(A, Ng):
    Nx = A.shape[0] - 2*Ng
    A[:Ng] = A[Ng+1]
    A[-Ng:] = A[-(Ng+1)]



set_bc = set_outflow_bc
get_flux = get_weno_flux


def dUdt(Cons, Ng, dx):
    set_bc(Cons, Ng)

    Prim = np.array([cons_to_prim(U) for U in Cons])
    Flux = np.array([flux(P) for P in Prim])
    Mlam = np.array([max_wavespeed(P) for P in Prim])

    Nx_tot = Cons.shape[0]
    L = np.zeros_like(Cons)
    F_hat = np.zeros_like(Cons)

    for i in range(2,Nx_tot-3):
        F_hat[i] = get_flux(Cons, Prim, Flux, Mlam, i)

    for i in range(1,Nx_tot):
        L[i] = -(F_hat[i] - F_hat[i-1]) / dx

    return L


def test_c2p():
    P = [1.0, 5.0, 0.2, 0.5, 0.4]
    print "%s ?= %s" % (P, cons_to_prim(prim_to_cons(P)))


def test_eigenvectors():
    P = [1.0, 5.0, 0.4, 0.2, 0.8]
    LL, RR = left_right_eigenvectors(P)
    print "0 ?= ", LL - np.linalg.inv(RR)


def setup_1d_problem():
    Nx = 32
    Ng = 3
    CFL = 0.6

    Prim = np.zeros((Nx + 2*Ng, 5))
    x, dx = np.linspace(0.0, 1.0, Nx, retstep=True)

    if False:
        # Initial conditions for density wave
        Prim[Ng:-Ng,rho] = 1.0 + 3.2e-1 * np.sin(2*np.pi*x)
        Prim[Ng:-Ng,pre] = 1.0
        Prim[Ng:-Ng,vx] = 1.0
    else:
        # Initial conditions for shocktube
        Prim[Ng:-Ng,pre] = np.where(x < 0.5, 1.0, 0.125)
        Prim[Ng:-Ng,rho] = np.where(x < 0.5, 1.0, 0.1)

    set_bc(Prim, Ng)
    Cons = np.array([prim_to_cons(P) for P in Prim])

    dt = CFL * dx / np.array([max_wavespeed(P) for P in Prim]).max()
    t = 0.0
    tmax = 0.1

    while t < tmax:
        L1 = dt * dUdt(Cons, Ng, dx)
        L2 = dt * dUdt(Cons + 0.5*L1, Ng, dx)
        L3 = dt * dUdt(Cons + 0.5*L2, Ng, dx)
        L4 = dt * dUdt(Cons + 1.0*L3, Ng, dx)

        Cons += (1.0/6.0) * (L1 + 2.0*L2 + 2.0*L3 + L4)
        t += dt

        print "we did it! t=%3.2f" % t


    from matplotlib import pyplot as plt

    Prim = np.array([cons_to_prim(U) for U in Cons])
    plt.plot(Prim[Ng:-Ng,rho], "-o", label=r"$\rho$")
    plt.plot(Prim[Ng:-Ng,pre], "-x", label=r"$p$")
    plt.plot(Prim[Ng:-Ng,vx], "-o", label=r"$v_x$")
    plt.plot(Prim[Ng:-Ng,vy], "-x", label=r"$v_y$")
    plt.plot(Prim[Ng:-Ng,vz], "-o", label=r"$v_z$")
    plt.legend()
    plt.show()


def main():
    setup_1d_problem()
    #test_c2p()
    #test_eigenvectors()


if __name__ == "__main__":
    main()



