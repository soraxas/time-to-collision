

import numpy as np
cimport numpy as np
from libc.math cimport sin, cos, sqrt
# from scipy.optimize import minimize
import fast_rollout
import cython

# from scipy.optimize.slsqp import _minimize_slsqp


cpdef new_bounds_to_old(lb, ub, int n):
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









import numpy as np
from scipy.optimize._slsqp import slsqp
from numpy import (zeros, array, linalg, append, asfarray, concatenate, finfo,
                   vstack, exp, inf, isfinite, atleast_1d)
from scipy.optimize.optimize import wrap_function, OptimizeResult, _check_unknown_options



_epsilon = np.sqrt(finfo(float).eps)











cdef approx_jacobian(x, func, epsilon):
    """
    Approximate the Jacobian matrix of a callable function.

    Parameters
    ----------
    x : array_like
        The state vector at which to compute the Jacobian matrix.
    func : callable f(x,*args)
        The vector-valued function.
    epsilon : float
        The perturbation used to determine the partial derivatives.
    args : sequence
        Additional arguments passed to func.

    Returns
    -------
    An array of dimensions ``(lenf, lenx)`` where ``lenf`` is the length
    of the outputs of `func`, and ``lenx`` is the number of elements in
    `x`.

    Notes
    -----
    The approximation is done using forward differences.

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







cpdef _minimize_slsqp(func, x0, args=(), jac=None, bounds=None,
#                     constraints=(),
                    maxiter=100, ftol=1.0E-6,
                    eps=_epsilon, callback=None):
    """
    Minimize a scalar function of one or more variables using Sequential
    Least SQuares Programming (SLSQP).

    Options
    -------
    ftol : float
        Precision goal for the value of f in the stopping criterion.
    eps : float
        Step size used for numerical approximation of the Jacobian.
    disp : bool
        Set to True to print convergence messages. If False,
        `verbosity` is ignored and set to 0.
    maxiter : int
        Maximum number of iterations.

    """
    fprime = jac
    iter = maxiter
    acc = ftol
    epsilon = eps

    # Transform x0 into an array.
    x = asfarray(x0).flatten()

    # Set the parameters that SLSQP will need
    # meq, mieq: number of equality and inequality constraints
    meq = 0
    mieq = 0
    # m = The total number of constraints
    m = meq + mieq
    # la = The number of constraints, or 1 if there are no constraints
    la = array([1, m]).max()
    # n = The number of independent variables
    n = len(x)

    # Define the workspaces for SLSQP
    n1 = n + 1
    mineq = m - meq + n1 + n1
    len_w = (3*n1+m)*(n1+1)+(n1-meq+1)*(mineq+2) + 2*mineq+(n1+mineq)*(n1-meq) \
            + 2*meq + n1 + ((n+1)*n)//2 + 2*m + 3*n + 3*n1 + 1
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
    x[have_bound] = np.clip(x[have_bound], xl[have_bound], np.inf)
    have_bound = np.isfinite(xu)
    x[have_bound] = np.clip(x[have_bound], -np.inf, xu[have_bound])
    
    #####################################
    
    # Wrap func
    feval, func = wrap_function(func, args)

    # Wrap fprime, if provided, or approx_jacobian if not
    if fprime:
        geval, fprime = wrap_function(fprime, args)
    else:
        geval, fprime = wrap_function(approx_jacobian, (func, epsilon))
        
        

    # Initialize the iteration counter and the mode value
    mode = array(0, int)
    acc = array(acc, float)
    majiter = array(iter, int)
    majiter_prev = 0

    # Initialize internal SLSQP state variables
    alpha = array(0, float)
    f0 = array(0, float)
    gs = array(0, float)
    h1 = array(0, float)
    h2 = array(0, float)
    h3 = array(0, float)
    h4 = array(0, float)
    t = array(0, float)
    t0 = array(0, float)
    tol = array(0, float)
    iexact = array(0, int)
    incons = array(0, int)
    ireset = array(0, int)
    itermx = array(0, int)
    line = array(0, int)
    n1 = array(0, int)
    n2 = array(0, int)
    n3 = array(0, int)
    
    # Compute the constraints
    c_eq = zeros(0)
    c_ieq = zeros(0)
    # Now combine c_eq and c_ieq into a single matrix
    c = concatenate((c_eq, c_ieq))
    
    # Compute the normals of the constraints
    a_eq = zeros((meq, n))
    a_ieq = zeros((mieq, n))
    # Now combine a_eq and a_ieq into a single a matrix
    if m == 0:  # no constraints
        a = zeros((la, n))
    else:
        a = vstack((a_eq, a_ieq))
    a = concatenate((a, zeros([la, 1])), 1)

    while 1:

        if mode == 0 or mode == 1:  # objective and constraint evaluation required

            # Compute objective function
            fx = func(x)
#             try:
#                 fx = float(np.asarray(fx))
#             except (TypeError, ValueError):
#                 raise ValueError("Objective function must return a scalar")

        if mode == 0 or mode == -1:  # gradient evaluation required

            # Compute the derivatives of the objective function
            # For some reason SLSQP wants g dimensioned to n+1
            g = append(fprime(x), 0.0)

        # Call SLSQP
        slsqp(m, meq, x, xl, xu, fx, c, g, a, acc, majiter, mode, w, jw,
              alpha, f0, gs, h1, h2, h3, h4, t, t0, tol,
              iexact, incons, ireset, itermx, line, 
              n1, n2, n3)
        
        # If exit mode is not -1 or 1, slsqp has completed
        if abs(mode) != 1:
            break

#         majiter_prev = int(majiter)
    
    return x

#     return OptimizeResult(x=x, fun=fx, jac=g[:-1], nit=int(majiter),
#                           nfev=feval[0], njev=geval[0], status=int(mode),
#                           message=exit_modes[int(mode)], success=(mode == 0))









@cython.boundscheck(False)
@cython.wraparound(False)
cpdef prop_traj(double [:] start_point, double [:] end_point, double [:] start_con, int N,
       object LinBounds, double [:, :] obs_np, float ts=.1, float targ_tol=.1):
    
#     pos_list=[]
#     vel_list=[]

#     cur_point=np.asarray(start_point).copy()
#     cur_con=np.asarray(start_con).copy()
    
#     for i in range(0,n):
#         pos_list.append(copy.copy(cur_point))
#         res=minimize(fast_rollout.rollout,
#                      cur_con,
#                      method='slsqp',
#                      args=(cur_point, end_point, obs_np),
#                      bounds=LinBounds,
#                      options={'ftol':0.1}
#                     )
#     #     res=minimize(rollout,cur_con,method='slsqp',args=(cur_point),bounds=LinBounds,options={'ftol':0.1})
#         cur_con=copy.copy(res.x)
#         cur_con=res.x
#         cur_point=integrate_next_step(cur_point,cur_con)
#         vel_list.append(cur_con)
#         if(np.linalg.norm(cur_point[:2]-end_point)<0.1):
#             break
#     return np.array(pos_list), np.array(vel_list)
    
    ##################################################################################
    
    # cdef double tmp = 0.
    cdef double [:] cur_point = np.empty(start_point.shape[0], dtype=np.double)
    cdef double [:] cur_con = np.empty(start_con.shape[0], dtype=np.double)
    cdef double [:] x = np.empty(start_con.shape[0], dtype=np.double)
    
    cdef double [:, :] J = np.zeros((3, 2), dtype=np.double)
    
    cdef double [:, :] poses = np.zeros((N, start_point.shape[0]), dtype=np.double)
    cdef double [:, :] veles = np.zeros((N, start_con.shape[0]), dtype=np.double)

    # init
    cur_point[:] = start_point
    cur_con[:] = start_con
    
    cdef object res
    
    cdef Py_ssize_t _x, _y
    cdef Py_ssize_t x_max = J.shape[0]
    cdef Py_ssize_t y_max = J.shape[1]
    
    cdef object bounds = new_bounds_to_old(
        LinBounds.lb, 
        LinBounds.ub, 
        cur_con.shape[0]
    )
    
    ###############################################################################
    
# cpdef _minimize_slsqp(func, x0, args=(), jac=None, bounds=None,
# #                     constraints=(),
#                     maxiter=100, , callback=None):

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
    jac = None
    maxiter = 100
    ftol=1.0E-6
    eps=_epsilon
    ################
    ftol=.1
    func_ = fast_rollout.rollout
    args=(cur_point, end_point, obs_np)
    ################
    fprime = jac
    iter = maxiter
    acc = ftol
    epsilon = eps

    # Transform x0 into an array.
#     x = asfarray(x0).flatten()
#     x = cur_con
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
    n1 = n + 1
    mineq = m - meq + n1 + n1
    len_w = (3*n1+m)*(n1+1)+(n1-meq+1)*(mineq+2) + 2*mineq+(n1+mineq)*(n1-meq) \
            + 2*meq + n1 + ((n+1)*n)//2 + 2*m + 3*n + 3*n1 + 1
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

    print(x0)
    print(x)
    print(np.asarray(x0))
    print(np.asarray(x))
    # assign the bound-safe x0 back to x
    for _x in range(x0.shape[0]):
        x[_x] = x0[_x]
    
#     #####################################
    
#     # Wrap func
#     feval, func = wrap_function(func, args)

#     # Wrap fprime, if provided, or approx_jacobian if not
#     if fprime:
#         geval, fprime = wrap_function(fprime, args)
#     else:
#         geval, fprime = wrap_function(approx_jacobian, (func, epsilon))
        
        

#     # Initialize the iteration counter and the mode value
#     mode = array(0, int)
#     acc = array(acc, float)
#     majiter = array(iter, int)
#     majiter_prev = 0

#     # Initialize internal SLSQP state variables
#     alpha = array(0, float)
#     f0 = array(0, float)
#     gs = array(0, float)
#     h1 = array(0, float)
#     h2 = array(0, float)
#     h3 = array(0, float)
#     h4 = array(0, float)
#     t = array(0, float)
#     t0 = array(0, float)
#     tol = array(0, float)
#     iexact = array(0, int)
#     incons = array(0, int)
#     ireset = array(0, int)
#     itermx = array(0, int)
#     line = array(0, int)
#     n1 = array(0, int)
#     n2 = array(0, int)
#     n3 = array(0, int)
    
#     # Compute the constraints
#     c_eq = zeros(0)
#     c_ieq = zeros(0)
#     # Now combine c_eq and c_ieq into a single matrix
#     c = concatenate((c_eq, c_ieq))
    
#     # Compute the normals of the constraints
#     a_eq = zeros((meq, n))
#     a_ieq = zeros((mieq, n))
#     # Now combine a_eq and a_ieq into a single a matrix
#     if m == 0:  # no constraints
#         a = zeros((la, n))
#     else:
#         a = vstack((a_eq, a_ieq))
#     a = concatenate((a, zeros([la, 1])), 1)
    
#     ###############################################################################

#     while 1:

#         if mode == 0 or mode == 1:  # objective and constraint evaluation required

#             # Compute objective function
#             fx = func(x)
#             try:
#                 fx = float(np.asarray(fx))
#             except (TypeError, ValueError):
#                 raise ValueError("Objective function must return a scalar")

#         if mode == 0 or mode == -1:  # gradient evaluation required
#             g = append(fprime(x), 0.0)

#         # Call SLSQP
#         slsqp(m, meq, np.asarray(x), xl, xu, fx, c, g, a, acc, majiter, mode, w, jw,
#               alpha, f0, gs, h1, h2, h3, h4, t, t0, tol,
#               iexact, incons, ireset, itermx, line, 
#               n1, n2, n3)
        
#         # If exit mode is not -1 or 1, slsqp has completed
#         if abs(mode) != 1:
#             break

# #         majiter_prev = int(majiter)
    
#     return x

    
    

    for i in range(N):
        
#         res = minimize(fast_rollout.rollout,
#                     cur_con,
#                     method='slsqp',
#                     args=(cur_point, end_point, obs_np),
#                     bounds=LinBounds,
#                     options={'ftol':0.1}
#                     )
        # print(i, '-------')

        x = asfarray(cur_con).flatten()

        ###############################################################################
        # Wrap func
        args=(cur_point, end_point, obs_np)
        feval, func = wrap_function(func_, args)
        # Wrap fprime, if provided, or approx_jacobian if not
        if fprime:
            geval, fprime = wrap_function(fprime, args)
        else:
            geval, fprime = wrap_function(approx_jacobian, (func, epsilon))
        # Initialize the iteration counter and the mode value
        mode = array(0, int)
        acc = array(acc, float)
        majiter = array(iter, int)
        majiter_prev = 0
        # Initialize internal SLSQP state variables
        alpha = array(0, float)
        f0 = array(0, float)
        gs = array(0, float)
        h1 = array(0, float)
        h2 = array(0, float)
        h3 = array(0, float)
        h4 = array(0, float)
        t = array(0, float)
        t0 = array(0, float)
        tol = array(0, float)
        iexact = array(0, int)
        incons = array(0, int)
        ireset = array(0, int)
        itermx = array(0, int)
        line = array(0, int)
        n1 = array(0, int)
        n2 = array(0, int)
        n3 = array(0, int)
        # Compute the constraints
        c_eq = zeros(0)
        c_ieq = zeros(0)
        # Now combine c_eq and c_ieq into a single matrix
        c = concatenate((c_eq, c_ieq))
        # Compute the normals of the constraints
        a_eq = zeros((meq, n))
        a_ieq = zeros((mieq, n))
        # Now combine a_eq and a_ieq into a single a matrix
        if m == 0:  # no constraints
            a = zeros((la, n))
        else:
            a = vstack((a_eq, a_ieq))
        a = concatenate((a, zeros([la, 1])), 1)
        while 1:
            if mode == 0 or mode == 1:  # objective and constraint evaluation required
                # Compute objective function
                # fx = func(x)
                fx = fast_rollout.rollout(x, cur_point, end_point, obs_np)
                try:
                    fx = float(np.asarray(fx))
                except (TypeError, ValueError):
                    raise ValueError("Objective function must return a scalar")
            if mode == 0 or mode == -1:  # gradient evaluation required
                # g = append(fprime(x), 0.0)
                #####################
                _x0 = asfarray(x).copy()
                # f0 = atleast_1d(func(*((_x0,))))
                f0 = atleast_1d(fast_rollout.rollout(_x0, cur_point, end_point, obs_np))
                jac = zeros([len(_x0), len(f0)])
                dx = zeros(len(_x0))
                for __i in range(len(_x0)):
                    dx[__i] = epsilon
                    jac[__i] = (fast_rollout.rollout(_x0 + dx, cur_point, end_point, obs_np) - f0)/epsilon
                    dx[__i] = 0.0

                g = append(jac.transpose(), 0.0) 
                #####################
            # print(np.asarray(x), g)

            # Call SLSQP
            slsqp(m, meq, np.asarray(x), xl, xu, fx, c, g, a, acc, majiter, mode, w, jw,
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

#         cur_con = res.x.copy()
        # cur_con = res
    
        # with nogil:
        if 1:
            # print('VVV')
            # print(np.asarray(cur_point))
            # print(np.asarray(cur_con))

            poses[i, :] = cur_point
            veles[i, :] = cur_con

            # print(np.asarray(poses[i, :]))

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

            # print('VVV')
            # print(np.asarray(cur_point))
            # print(np.asarray(cur_con))
            if sqrt((cur_point[0] - end_point[0])**2 + (cur_point[1] - end_point[1])**2) < targ_tol:
                break
    # print(i)
    return np.asarray(poses)[:i], np.asarray(veles)[:i]
    