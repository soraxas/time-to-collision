#!/home/soraxas/.pyenv/versions/res37/bin/python

import fast_rollout

import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint
import matplotlib.pyplot as pl
import copy

# from sklearn.metrics.pairwise import euclidean_distances


def J_vehicle(x_c):
    result = np.zeros((3, 2), dtype=float)
    result[0, 0] = np.cos(x_c[2])
    result[1, 0] = np.sin(x_c[2])
    result[2, 1] = 1.0
    return result


def rollout(x_dot, cur_point, ts=0.1, th=50):
    x_c = cur_point.copy()
    # print(x_dot)
    # x_dot_r = x_dot.reshape(-1)
    # print(x_dot_r)
    cost_v = 0.0
    for i in range(1, th + 1):
        jac_v = J_vehicle(x_c)
        nextstep = np.matmul(jac_v, x_dot) * ts
        x_c = x_c + nextstep
        norm_dist = np.linalg.norm(x_c[:2] - end_point)
        obs_ratio = 15.0 / (ts * i * np.min(np.linalg.norm(x_c[:2] - obs_np, axis=1)))
        # print(x_c[:2] - obs_np)
        # print((np.linalg.norm(x_c[:2] - obs_np, axis=1)))
        # print(np.min(np.linalg.norm(x_c[:2] - obs_np, axis=1)))
        if norm_dist < 0.1:
            return cost_v
        cost_v = cost_v + (norm_dist) + (obs_ratio)
    return cost_v


def f(q):
    print(q)
    return q * q


np.random.seed(0)
random_x = np.random.uniform(21, 37, 50).reshape((-1, 1))
random_y = np.random.uniform(21, 37, 50).reshape((-1, 1))
obs_np = np.concatenate([random_x, random_y], axis=1)

control = np.random.rand(2)
start_point = np.array([38.0, 38.0, -2.0])

control = np.array([0.0, 0.0])
control = np.random.rand(2)
end_point = np.array([20.0, 20.0])

# print(J_vehicle(start_point))


def out(a):
    print("VVV")
    print(a)
    print("-")


print("===")
print(J_vehicle(start_point))
print("===")
print(fast_rollout.jacobian_vehicle(start_point))
print("===")
print(fast_rollout.rollout.__doc__)
for i in range(10):
    control = np.random.rand(2)
    start_point = np.random.rand(3)
    print(rollout(control, start_point))
    print(
        fast_rollout.rollout(
            control,
            init_x=start_point,
            targ_x=end_point,
            n=50,
            obs=obs_np,
            x_len=3,
        )
    )

# import sys
# sys.exit()


# np.random.seed(0)
# random_x = np.random.uniform(21, 37, 50).reshape((-1, 1))
# random_y = np.random.uniform(21, 37, 50).reshape((-1, 1))
# obs_np = np.concatenate([random_x, random_y], axis=1)


# print(rollout(control, start_point))

single_con = np.array([1.5, 0.0])
ubounds = np.array([1.5, 1.0])
lbounds = np.array([0.0, -1.0])
ubounds = ubounds.reshape(-1)
lbounds = lbounds.reshape(-1)
LinBounds = Bounds(lbounds, ubounds)


def integrate_next_step(x_0, x_dot, ts_c=0.1):
    x_c = copy.copy(x_0)
    jac_v = J_vehicle(x_c)
    x_c += np.matmul(jac_v, x_dot) * ts_c
    return x_c


import time


def a(start_point, end_point, cur_con):
    cur_point = copy.copy(start_point)
    pos_list = []
    vel_list = []
    for i in range(0, 500):
        pos_list.append(copy.copy(cur_point))
        res = minimize(
            fast_rollout.rollout,
            cur_con,
            method="slsqp",
            args=(cur_point, end_point, obs_np),
            bounds=LinBounds,
            options={"ftol": 0.1},
        )
        #     res=minimize(rollout,cur_con,method='slsqp',args=(cur_point),bounds=LinBounds,options={'ftol':0.1})
        cur_con = copy.copy(res.x)
        cur_con = res.x
        next_point = integrate_next_step(cur_point, cur_con)
        cur_point = next_point
        vel_list.append(cur_con)
        if np.linalg.norm(cur_point[:2] - end_point) < 0.1:
            break
    print(i)
    return np.array(pos_list), np.array(vel_list)


start_point = np.array([38.0, 38.0, -2.0])
end_point = np.array([20.0, 20.0])
cur_control = np.array([1.5, 0.0])
cur_con = cur_control

# start_point=np.random.uniform(21,37,3)
# start_point[-1] = -2.
# end_point=np.random.uniform(21,37,2)

print(start_point, end_point)
start = time.time()
a(start_point, end_point, cur_control)
print(time.time() - start)

import forward_prop_traj

start = time.time()
r = forward_prop_traj.prop_traj(
    start_point=start_point,
    end_point=end_point,
    start_con=cur_con,
    N=500,
    LinBounds=LinBounds,
    obs_np=obs_np,
    ftol=0.1,
)
print((time.time() - start) * 1000)
print(r[0][0:2], r[0][-2:])
print(r[1][0:2], r[1][-2:])
# print('=')
# print(r)
# print(a(start_point, end_point, cur_control))
# print(np.asarray(r))
