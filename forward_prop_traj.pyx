#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: infertypes=True
#cython: initializedcheck=False
#cython: cdivision=True
#distutils: language = c++
#distutils: libraries = ['stdc++']
#distutils: extra_compile_args = -Wno-unused-function -Wno-unneeded-internal-declaration

import numpy as np
cimport numpy as np
from libc.math cimport sin, cos, sqrt
# from scipy.optimize import minimize
import fast_rollout
import cython

# from scipy.optimize.slsqp import _minimize_slsqp

from scipy.optimize._slsqp import slsqp
from numpy import (zeros, array, linalg, append, asfarray, concatenate, finfo,
                   vstack, exp, inf, isfinite, atleast_1d)
# from scipy.optimize.optimize import OptimizeResult, _check_unknown_options


DTYPE = np.double
ctypedef np.double_t DTYPE_t
# F_DTYPE = np.float
F_DTYPE = float
ctypedef np.float_t F_DTYPE_t
# I_DTYPE = np.int
I_DTYPE = int
ctypedef np.int_t I_DTYPE_t
_epsilon = np.sqrt(finfo(float).eps)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple prop_traj(DTYPE_t[:] start_point, DTYPE_t[:] end_point, DTYPE_t[:] start_con, Py_ssize_t N,
       object LinBounds, DTYPE_t[:, :] obs_np, float ts=.1, float targ_tol=.1,
       Py_ssize_t maxiter=100, ftol=1.0E-6, DTYPE_t epsilon=_epsilon, object jac_func=None):
    
    ##################################################################################
    # cdef double tmp = 0.
    cdef DTYPE_t[:] cur_point = np.empty(start_point.shape[0], dtype=DTYPE)
    cdef DTYPE_t[:] cur_con = np.empty(start_con.shape[0], dtype=DTYPE)
    cdef DTYPE_t[:] x = np.empty(start_con.shape[0], dtype=DTYPE)
    
    cdef DTYPE_t[:, :] J = np.zeros((3, 2), dtype=DTYPE)
    
    cdef DTYPE_t[:, :] poses = np.zeros((N, start_point.shape[0]), dtype=DTYPE)
    cdef DTYPE_t[:, :] veles = np.zeros((N, start_con.shape[0]), dtype=DTYPE)

    # init
    cur_point[:] = start_point
    cur_con[:] = start_con
    
    cdef object res
    
    cdef Py_ssize_t _x, _y, i, __i
    cdef Py_ssize_t x_max = J.shape[0]
    cdef Py_ssize_t y_max = J.shape[1]
    
    cdef object bounds = new_bounds_to_old(
        LinBounds.lb, 
        LinBounds.ub, 
        cur_con.shape[0]
    )
    
    ###############################################################################
    # func_ = fast_rollout.rollout
    # args=(cur_point, end_point, obs_np)
    # ################
    # fprime = jac_func

    # Transform x0 into an array.
    x0 = asfarray(cur_con).flatten()

    # Set the parameters that SLSQP will need
    # meq, mieq: number of equality and inequality constraints
    meq = 0
    mieq = 0
    # m = The total number of constraints
    m = meq + mieq
    # la = The number of constraints, or 1 if there are no constraints
    la = array([1, m]).max()
    # n = The number of independent variables
    n = len(x0)

    # Define the workspaces for SLSQP
    _n1 = n + 1
    mineq = m - meq + _n1 + _n1
    len_w = (3*_n1+m)*(_n1+1)+(_n1-meq+1)*(mineq+2) + 2*mineq+(_n1+mineq)*(_n1-meq) \
            + 2*meq + _n1 + ((n+1)*n)//2 + 2*m + 3*n + 3*_n1 + 1
    len_jw = mineq
    w = zeros(len_w)
    jw = zeros(len_jw)

    # Decompose bounds into xl and xu
    if bounds is None or len(bounds) == 0:
        xl = np.empty(n, dtype=float)
        xu = np.empty(n, dtype=float)
        xl.fill(np.nan)
        xu.fill(np.nan)
    else:
        bnds = array(bounds, float)
        if bnds.shape[0] != n:
            raise IndexError('SLSQP Error: the length of bounds is not '
                             'compatible with that of x0.')

        with np.errstate(invalid='ignore'):
            bnderr = bnds[:, 0] > bnds[:, 1]

        if bnderr.any():
            raise ValueError('SLSQP Error: lb > ub in bounds')
        xl, xu = bnds[:, 0], bnds[:, 1]

        # Mark infinite bounds with nans; the Fortran code understands this
        infbnd = ~isfinite(bnds)
        xl[infbnd[:, 0]] = np.nan
        xu[infbnd[:, 1]] = np.nan

    # Clip initial guess to bounds (SLSQP may fail with bounds-infeasible
    # initial point)
    have_bound = np.isfinite(xl)
    x0[have_bound] = np.clip(x0[have_bound], xl[have_bound], np.inf)
    have_bound = np.isfinite(xu)
    x0[have_bound] = np.clip(x0[have_bound], -np.inf, xu[have_bound])
    # assign the bound-safe x0 back to x
    for _x in range(x0.shape[0]):
        x[_x] = x0[_x]
    cur_con = x
    ###############################################################################
    # Init varaibles for internal states of slsqp
    # Initialize the iteration counter and the mode value
    cdef np.ndarray[I_DTYPE_t, ndim=0] mode = array(1, I_DTYPE)
    cdef np.ndarray[F_DTYPE_t, ndim=0] acc = array(ftol, F_DTYPE)
    cdef np.ndarray[I_DTYPE_t, ndim=0] majiter = array(maxiter, I_DTYPE)
    cdef int majiter_prev = 0
    # Initialize internal SLSQP state variables
    cdef np.ndarray[F_DTYPE_t, ndim=0] alpha = array(0, F_DTYPE)
    cdef np.ndarray[F_DTYPE_t, ndim=1] f0 = array([0], F_DTYPE)
    cdef np.ndarray[F_DTYPE_t, ndim=0] gs = array(0, F_DTYPE)
    cdef np.ndarray[F_DTYPE_t, ndim=0] h1 = array(0, F_DTYPE)
    cdef np.ndarray[F_DTYPE_t, ndim=0] h2 = array(0, F_DTYPE)
    cdef np.ndarray[F_DTYPE_t, ndim=0] h3 = array(0, F_DTYPE)
    cdef np.ndarray[F_DTYPE_t, ndim=0] h4 = array(0, F_DTYPE)
    cdef np.ndarray[F_DTYPE_t, ndim=0] t = array(0, F_DTYPE)
    cdef np.ndarray[F_DTYPE_t, ndim=0] t0 = array(0, F_DTYPE)
    cdef np.ndarray[F_DTYPE_t, ndim=0] tol = array(0, F_DTYPE)
    cdef np.ndarray[I_DTYPE_t, ndim=0] iexact = array(0, I_DTYPE)
    cdef np.ndarray[I_DTYPE_t, ndim=0] incons = array(0, I_DTYPE)
    cdef np.ndarray[I_DTYPE_t, ndim=0] ireset = array(0, I_DTYPE)
    cdef np.ndarray[I_DTYPE_t, ndim=0] itermx = array(0, I_DTYPE)
    cdef np.ndarray[I_DTYPE_t, ndim=0] line = array(0, I_DTYPE)
    cdef np.ndarray[I_DTYPE_t, ndim=0] n1 = array(0, I_DTYPE)
    cdef np.ndarray[I_DTYPE_t, ndim=0] n2 = array(0, I_DTYPE)
    cdef np.ndarray[I_DTYPE_t, ndim=0] n3 = array(0, I_DTYPE)
    # Compute the constraints
    cdef np.ndarray[F_DTYPE_t, ndim=1] c_eq = zeros(0)
    cdef np.ndarray[F_DTYPE_t, ndim=1] c_ieq = zeros(0)
    # Now combine c_eq and c_ieq into a single matrix
    cdef np.ndarray[F_DTYPE_t, ndim=1] c = concatenate((c_eq, c_ieq))
    # Compute the normals of the constraints
    cdef np.ndarray[F_DTYPE_t, ndim=2] a_eq = zeros((meq, n))
    cdef np.ndarray[F_DTYPE_t, ndim=2] a_ieq = zeros((mieq, n))
    # Now combine a_eq and a_ieq into a single a matrix
    cdef np.ndarray[F_DTYPE_t, ndim=2] a
    if m == 0:  # no constraints
        a = zeros((la, n))
    else:
        a = vstack((a_eq, a_ieq))
    a = concatenate((a, zeros([la, 1])), 1)
    ###############################################################################
    cdef Py_ssize_t _len_x0 = len(x0)
    cdef np.ndarray[DTYPE_t, ndim=1] dx
    cdef DTYPE_t[:] jac
    # cdef np.ndarray jac
    # cdef np.ndarray g
    cdef DTYPE_t fx

    for i in range(N):
#         res = minimize(fast_rollout.rollout,
#                     cur_con,
#                     method='slsqp',
#                     args=(cur_point, end_point, obs_np),
#                     bounds=LinBounds,
#                     options={'ftol':0.1}
#                     )

        # x = asfarray(cur_con).flatten()
        x[:] = cur_con
        ###############################################################################
        # # Wrap func
        # args=(cur_point, end_point, obs_np)
        # feval, func = wrap_function(func_, args)
        # # Wrap fprime, if provided, or approx_jacobian if not
        # if fprime:
        #     geval, fprime = wrap_function(fprime, args)
        # else:
        #     geval, fprime = wrap_function(approx_jacobian, (func, epsilon))

        ####
        # moved initialisation of interla states to be outside of the loop
        ####
        # reset variables
        mode[...] = 0
        acc[...] = ftol
        majiter[...] = maxiter
        majiter_prev = 0
        # Initialize internal SLSQP state variables
        alpha[...] = 0.0
        f0[...] = 0.0
        gs[...] = 0.0
        h1[...] = 0.0
        h2[...] = 0.0
        h3[...] = 0.0
        h4[...] = 0.0
        t[...] = 0.0
        t0[...] = 0.0
        tol[...] = 0.0
        iexact[...] = 0
        incons[...] = 0
        ireset[...] = 0
        itermx[...] = 0
        line[...] = 0
        n1[...] = 0
        n2[...] = 0
        n3[...] = 0
        
        while 1:
            if mode == 0 or mode == 1:  # objective and constraint evaluation required
                # Compute objective function
                # fx = func(x)
                fx = fast_rollout.rollout(x, cur_point, end_point, obs_np)
                # try:
                #     fx = float(np.asarray(fx))
                # except (TypeError, ValueError):
                #     raise ValueError("Objective function must return a scalar")
            if mode == 0 or mode == -1:  # gradient evaluation required
                # g = append(fprime(x), 0.0)
                #####################
                # _x0 = asfarray(x).copy()
                # f0 = atleast_1d(func(*((_x0,))))
                f0 = atleast_1d(fast_rollout.rollout(x, cur_point, end_point, obs_np))
                # len(f0) is obviously 1 as it's a scalar...
                # jac = zeros([len(_x0), len(f0)])
                jac = zeros(_len_x0 + 1)
                dx = zeros(_len_x0)
                for __i in range(_len_x0):
                    dx[__i] = epsilon
                    jac[__i] = (fast_rollout.rollout(x + dx, cur_point, end_point, obs_np) - f0) / epsilon
                    dx[__i] = 0.0
                #### jac = zeros([_len_x0, 1])
                #### dx = zeros(_len_x0)
                #### for __i in range(_len_x0):
                ####     dx[__i] = epsilon
                ####     jac[__i] = (fast_rollout.rollout(x + dx, cur_point, end_point, obs_np) - f0) / epsilon
                ####     dx[__i] = 0.0
                #### g = append(jac.transpose(), 0.0) 
                #####################
            # print(np.asarray(x), g)

            # Call SLSQP
            slsqp(m, meq, np.asarray(x), xl, xu, fx, c, 
                # g, 
                jac, 
                a, acc, majiter, mode, w, jw,
                alpha, f0, gs, h1, h2, h3, h4, t, t0, tol,
                iexact, incons, ireset, itermx, line, 
                n1, n2, n3)
            # If exit mode is not -1 or 1, slsqp has completed
            if abs(mode) != 1:
                break
    #         majiter_prev = int(majiter)
        cur_con[:] = x
        # print(np.asarray(x))
        # return x
        ###############################################################################


#         res = _minimize_slsqp(
#             fast_rollout.rollout,
#             x0=cur_con,
#             args=(cur_point, end_point, obs_np),
#             jac=None,
#             bounds=bounds,
# #             constraints=[],
# #             callback=None,
#             ftol=0.1,
#         )

        with nogil:
        # if 1:

            poses[i, :] = cur_point
            veles[i, :] = cur_con

            # fast set jacobian matrix
            J[0, 0] = ts * cos(cur_point[2])
            J[1, 0] = ts * sin(cur_point[2])
            J[2, 1] = ts #* 1. 

            for _x in range(x_max):
                # tmp = 0.
                for _y in range(y_max):#2
                    # tmp += J[x, y] * cur_con[y]
                    cur_point[_x] += J[_x, _y] * cur_con[_y]
                # cur_point[x] += tmp * ts

            if sqrt((cur_point[0] - end_point[0])**2 + (cur_point[1] - end_point[1])**2) < targ_tol:
                break
    # print(i)
    return np.asarray(poses)[:i], np.asarray(veles)[:i]
    

cdef list new_bounds_to_old(lb, ub, int n):
    """Convert the new bounds representation to the old one.

    The new representation is a tuple (lb, ub) and the old one is a list
    containing n tuples, i-th containing lower and upper bound on a i-th
    variable.
    """
    lb = np.asarray(lb)
    ub = np.asarray(ub)
    if lb.ndim == 0:
        lb = np.resize(lb, n)
    if ub.ndim == 0:
        ub = np.resize(ub, n)

    lb = [x if x > -np.inf else None for x in lb]
    ub = [x if x < np.inf else None for x in ub]

    return list(zip(lb, ub))


cdef approx_jacobian(x, func, epsilon):
    """
    Approximate the Jacobian matrix of a callable function.
    """
    x0 = asfarray(x)
    f0 = atleast_1d(func(*((x0,))))
    jac = zeros([len(x0), len(f0)])
    dx = zeros(len(x0))
    for i in range(len(x0)):
        dx[i] = epsilon
        jac[i] = (func(*((x0+dx,))) - f0)/epsilon
        dx[i] = 0.0
    return jac.transpose()
