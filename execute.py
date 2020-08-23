#!/home/soraxas/.pyenv/versions/res37/bin/python

import fast_rollout

import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint
import matplotlib.pyplot as pl
import copy
from sklearn.metrics.pairwise import euclidean_distances


def J_vehicle(x_c):
    result = np.zeros((3, 2), dtype=np.float)
    result[0, 0] = np.cos(x_c[2])
    result[1, 0] = np.sin(x_c[2])
    result[2, 1] = 1.
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
        obs_ratio = 15.0 / (
            ts * i * np.min(np.linalg.norm(x_c[:2] - obs_np, axis=1))
        )
        # print(x_c[:2] - obs_np)
        # print((np.linalg.norm(x_c[:2] - obs_np, axis=1)))
        # print(np.min(np.linalg.norm(x_c[:2] - obs_np, axis=1)))
        if norm_dist < 0.1:
            return cost_v
        cost_v = cost_v + (norm_dist) + (obs_ratio)
    return cost_v

def f(q):
    print(q)
    return q*q

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
    print('VVV')
    print(a)
    print('-')

print('===')
print(J_vehicle(start_point))
print('===')
print(fast_rollout.jacobian_vehicle(start_point))
print('===')
print(fast_rollout.rollout.__doc__)
for i in range(10):
    control = np.random.rand(2)
    start_point = np.random.rand(3)
    print(rollout(control, start_point))
    print(fast_rollout.rollout(
        control, 
        init_x=start_point,
        targ_x=end_point,
        n=50,
        obs=obs_np,
        x_len=3,
    ))

import sys
sys.exit()


np.random.seed(0)
random_x = np.random.uniform(21, 37, 50).reshape((-1, 1))
random_y = np.random.uniform(21, 37, 50).reshape((-1, 1))
obs_np = np.concatenate([random_x, random_y], axis=1)


print(rollout(control, start_point))

