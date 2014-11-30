import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("--interactive", action="store_true")
# parser.add_argument("--position_only", action="store_true")
# args = parser.parse_args()

import scipy.io
import numpy as np

from lfd.registration import tps, solver, settings
from lfd.transfer import settings as transfer_settings

matfile = scipy.io.loadmat('test_tps_prob.mat')

# Scene correspondences
x_na = matfile['X'].T
y_ng = matfile['Y'].T

# Need to add 3rd dimension to work with lfd code. Can't just be zeros because
# that would result in singular matrix.
x_na3 = np.c_[x_na, np.random.normal(0, 0.001, [20,1])]
y_ng3 = np.c_[y_ng, np.random.normal(0, 0.001, [20,1))]
n, d = x_na3.shape

# parameters to tune
alpha = 2; # penalty coefficient for the TPS transformation
beta = 1; # penalty coeffient for trajectory points deviating from the transformed demonstration trajectory

### Do initial tps registration here. It will return correspondences,
### null-space matrix N, and maybe weights on the points (o/w use np.ones(n))
### Also, might return bend coeffs and rot coeffs.

# Could use GpuSolverFactory or Auto, but only Cpu works on cfinn's computer
# right now. Also, typically use_cache is true for f_solver_factory, but
# you probably need to set a cachedir appropriately since we're not in lfd/
f_solver_factory = solver.CpuTpsSolverFactory(use_cache=False)
g_solver_factory = solver.CpuTpsSolverFactory(use_cache=False)
f, g, corr = tps.tps_rpm_bij(x_na3, y_ng3, f_solver_factory, g_solver_factory,
                             settings.N_ITER, settings.EM_ITER,
                             settings.REG[0], settings.REG[1],
                             settings.RAD[0], settings.RAD[1],
                             settings.ROT_REG, settings.OUTLIER_PRIOR,
                             settings.OURLIER_FRAC)

bend_coefs = np.ones(d) * f.bend_coef if np.isscalar(f.bend_coef) else f.bend_coef
rot_coefs = np.ones(d) * f.rot_coef if np.isscalar(f.rot_coef) else f.rot_coef
if f.wt_n is None:
  wt_n = np.ones(n)
else:
  wt_n = f.wt_n

if wt_n.ndim == 1:
  wt_n = wt_n[:, None]
if wt_n.shape[1] == 1:
  wt_n = np.tile(wt_n, (1,d))

N = f.N
init_z = f.z

# TODO - Initialize with transfered demonstration trajectory.
init_traj = matfile['demo_traj'].T
num_steps = init_traj.shape[0]

start_fixed = True

request = {
    "basic_info" : {
        "n_steps" : n_steps,
        "m_ext" : n,
        "n_ext" : d,
        "manip" : manip_name,
         "start_fixed" : start_fixed
    },
    "traj_costs" : [
    {
        "type" : "joint_vel",
        "params": {"coeffs" : [transfer_settings.GAMMA/(n_steps-1)]}
    },
    ],
    "tps_costs" : [
    {
        "type" : "joint_vel",
        "params": {"coeffs" : [transfer_settings.GAMMA/(n_steps-1)]}
    },
    {
        "type" : "tps",
        "name" : "tps",
        "params" : {"x_na" : [row.tolist() for row in f.x_na],
                    "y_ng" : [row.tolist() for row in f.y_ng],
                    "bend_coefs" : bend_coefs.tolist(),
                    "rot_coefs" : rot_coefs.tolist(),
                    "wt_n" : [row.tolist() for row in wt_n],
                    "N" : [row.tolist() for row in N],
                    "alpha" : alpha,
        }
    }
    ],
    "traj_constraints" : [
    ],
    "tps_constraints" : [
    ],
    "init_info" : {
        "type":"given_traj",
        "data":[x.tolist() for x in init_traj],
        "data_ext":[row.tolist() for row in init_z]
    }
}

if use_collision_cost:
    request["traj_costs"].append(
        {
            "type" : "collision",
            "params" : {
              "continuous" : True,
              "coeffs" : [1000],  # penalty coefficients. list of length one is automatically expanded to a list of length n_timesteps
              "dist_pen" : [0.025]  # robot-obstacle distance that penalty kicks in. expands to length n_timesteps
            }
        })

if joint_vel_limits is not None:
    request["traj_constraints"].append(
          {
            "type" : "joint_vel_limits",
            "params": {"vals" : joint_vel_limits,
                       "first_step" : 0,
                       "last_step" : n_steps-1
                      }
          })


paren = @(x, varargin) x(varargin{:});
curly = @(x, varargin) x{varargin{:}};

% TPS
N = null([X_s; ones(1,n)]);
tps_dim = (n-(d+1))*d + d*d + d; % dim(A) + dim(B) + dim(c)

% parameters to tune
lambda = .1; % TPS regularization coefficient
alpha = 2; % penalty coefficient for the TPS transformation
beta = 1; % penalty coeffient for trajectory points deviating from the transformed demonstration trajectory

% The robot trajectory is a state_dim x T matrix
% the rows are the degrees of freedom of the robot: x,y,angle.
% the columns are timesteps 1 to T.
T = size(demo_traj, 2);
X_g = [demo_traj; zeros(1, T)];
[state_dim, ~] = size(X_g);
state_dim_T = state_dim * T;

dsafe = 0.05; % Safety margin
obstacles = test_obs;%{orientedBoxToPolygon([0 0 1 1 0])};
car_length = .4;
car_width = .2;

% Function that maps state vector to polygon (which is used for collision
% checking)
make_robot_poly = @(x) orientedBoxToPolygon([x(1), x(2), car_length, car_width, rad2deg(x(3))]);


% The variable being optimized is the flattened concatenation of the
% trajectory and the TPS transformation parameters.
% x = [traj(:); A(:); B(:); c(:)]
[~, iwarp, ~, Q, B, C] = compute_warp(X_s, X_s_new);
x_target = [warp_pts(demo_traj, iwarp); zeros(1, T)];
x0 = [x_target(:); Q(:); B(:); C];

% Functions for extracting trajectory and TPS transformation parameters
% from the variable being optimized
% sequence of states
getTraj = @(x) reshape(x(1:state_dim_T), state_dim, T);
% sequence of states' points. these points are 2D points on the geometry of the robot for every state
getTrajPts = @(x) paren(getTraj(x), 1:2, :);
% A B c describes the TPS transformation
getA = @(x) N*reshape(x(state_dim_T+1:state_dim_T+(n-(d+1))*d), (n-(d+1)), d);
getB = @(x) reshape(x(state_dim_T+(n-(d+1))*d+1:end-d), d, d);
getc = @(x) x(end-d+1:end);

% create Q matrix for the sum-of-squared-displacements cost
% \sum_t || \theta_{t+1} - \theta_t ||^2
%   + alpha*(||X_s_new - f(X_s)||^2 + \lambda * regularizer(f))
%   + beta*\sum_t ||points(\theta_t) - f(points(X_g_t))||^2
% where \theta is the state vector (x, y, angle)
% and points maps the state vector into points (which are on the
% geometry of the robot at that state)
square = @(x) x.^2;
quad_obj = @(x) sum(sum(square(paren(getTraj(x),:,2:T) - paren(getTraj(x),:,1:T-1)))) + ...
    alpha*(sum(sum(square(X_s_new' - K*getA(x) - X_s'*getB(x) - ones(n,1)*getc(x).'))) + lambda * trace(getA(x).'*K*getA(x))) + ...
    beta*(sum(sum(square(getTrajPts(x) - warp_pts(getTrajPts(X_g), make_warp(getA(x), getB(x), getc(x), X_s))))));

[Q, q] = calc_quad_expr(quad_obj, [state_dim_T + tps_dim,1]);
f = @(x) 0;

% The constraint function g does all the work of computing signed distances
% and their gradients
g = @(x) g_collisions(x, dsafe, [state_dim,T], make_robot_poly, obstacles);
h = @(x) 0;

% create linear inequality constraints to enforce that the
% displacement is smaller than .2 (componentwise) over each timestep
f_ineqs = { ...
    @(x) paren(getTraj(x),:,2:T) - paren(getTraj(x),:,1:T-1) - .2*ones(state_dim,T-1), ...
    @(x) paren(getTraj(x),:,1:T-1) - paren(getTraj(x),:,2:T) - .2*ones(state_dim,T-1) ...
    };
[A_ineq, neg_b_ineq] = calc_lin_expr(f_ineqs, [state_dim_T + tps_dim,1]);
b_ineq = -neg_b_ineq;

% create linear equality constraints to fix the beginning and end of the
% trajectory to the beginning and end of the warped demonstrated trajectory
f_eqs = { ...
    @(x) paren(getTrajPts(x),:,1) - paren(warp_pts(getTrajPts(X_g), make_warp(getA(x), getB(x), getc(x), X_s)),:,1), ...
    @(x) paren(getTrajPts(x),:,T) - paren(warp_pts(getTrajPts(X_g), make_warp(getA(x), getB(x), getc(x), X_s)),:,T), ...
    @(x) ones(1,n)*getA(x) ...
    };
[A_eq, neg_b_eq] = calc_lin_expr(f_eqs, [state_dim_T + tps_dim,1]);
b_eq = -neg_b_eq;

orig_fig = figure();
figure(orig_fig);
hold on;
plot_traj(make_robot_poly, {}, X_g, false);
h1 = scatter(X_s(1,:), X_s(2,:), 'red');
h2 = scatter(X_g(1,:), X_g(2,:), 'cyan', 'x');
draw_grid_orig([-1 1], [1 -1], 5);
legend([h1 h2], 'x_i^{(S)}', 'x_t^{(G)}');

warp_fig = figure();
figure(warp_fig);

cfg = struct();
cfg.callback = @(x,~) plot_traj_warp(make_warp(getA(x), getB(x), getc(x), X_s), X_s_new, X_s, getTraj(x), X_g, getTrajPts, @(X) plot_traj(make_robot_poly, obstacles, X, false));
cfg.initial_trust_box_size=.1;
cfg.g_use_numerical = false;
cfg.min_approx_improve = 1e-2;

x = penalty_sqp(x0, Q, q, f, A_ineq, b_ineq, A_eq, b_eq, g, h, cfg);
%   minimize (1/2) x'*Q*x + x'*q + f(x)
%   subject to
%       A_ineq*x <= b_ineq
%       A_eq*x == b_eq
%       g(x) <= 0
%       h(x) == 0

warp = make_warp(getA(x), getB(x), getc(x), X_s);
plot_traj_warp(warp, X_s_new, X_s, getTraj(x), X_g, getTrajPts, @(X) plot_traj(make_robot_poly, obstacles, X, false));

[warped_pts, warp_fn, err] = compute_warp([X_s getTrajPts(X_g)], [X_s_new, getTrajPts(x)]);
