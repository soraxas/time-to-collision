
import numpy as np
cimport numpy as np
from libc.math cimport sin, cos, sqrt
# from scipy.optimize import minimize
import fast_rollout
import cython

from scipy.optimize.slsqp import _minimize_slsqp


cpdef new_bounds_to_old(lb, ub, int n):
    """
    Convert the new bounds representation to the old one.
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


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef prop_traj(double [:] start_point, double [:] end_point, double [:] start_con, int n,
       object LinBounds, double [:, :] obs_np, float ts=.1, float tol=.1):
    
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
    
    cdef double [:, :] J = np.zeros((3, 2), dtype=np.double)
    
    cdef double [:, :] poses = np.zeros((n, start_point.shape[0]), dtype=np.double)
    cdef double [:, :] veles = np.zeros((n, start_con.shape[0]), dtype=np.double)

    # init
    cur_point[:] = start_point
    cur_con[:] = start_con
    
    cdef object res
    
    cdef Py_ssize_t x, y
    cdef Py_ssize_t x_max = J.shape[0]
    cdef Py_ssize_t y_max = J.shape[1]
    
    cdef object bounds = new_bounds_to_old(
        LinBounds.lb, 
        LinBounds.ub, 
        cur_con.shape[0]
    )
    

    for i in range(n):
        
#         res = minimize(fast_rollout.rollout,
#                     cur_con,
#                     method='slsqp',
#                     args=(cur_point, end_point, obs_np),
#                     bounds=LinBounds,
#                     options={'ftol':0.1}
#                     )
    
        res = _minimize_slsqp(
            fast_rollout.rollout,
            x0=cur_con,
            args=(cur_point, end_point, obs_np),
            jac=None,
            bounds=bounds,
            constraints=[],
            callback=None,
            ftol=0.1,
        )

        cur_con = res.x.copy()
    
        with nogil:

            poses[i, :] = cur_point
            veles[i, :] = cur_con

            # fast set jacobian matrix
            J[0, 0] = ts * cos(cur_point[2])
            J[1, 0] = ts * sin(cur_point[2])
            J[2, 1] = ts #* 1. 

            for x in range(x_max):
                # tmp = 0.
                for y in range(y_max):#2
                    # tmp += J[x, y] * cur_con[y]
                    cur_point[x] += J[x, y] * cur_con[y]
                # cur_point[x] += tmp * ts


            if sqrt((cur_point[0] - end_point[0])**2 + (cur_point[1] - end_point[1])**2) < tol:
                break

    return np.asarray(poses)[:i], np.asarray(veles)[:i]
    