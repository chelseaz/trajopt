import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--interactive", action="store_true")
parser.add_argument("--position_only", action="store_true")
args = parser.parse_args()

import openravepy
import trajoptpy
import json
import numpy as np
import trajoptpy.kin_utils as ku
from rapprentice import tps, registration

env = openravepy.Environment()
env.StopSimulation()
env.Load("robots/pr2-beta-static.zae")
env.Load("../data/table.xml")

trajoptpy.SetInteractive(args.interactive) # pause every iteration, until you press 'p'. Press escape to disable further plotting

robot = env.GetRobots()[0]
joint_start = [-1.832, -0.332, -1.011, -1.437, -1.1  , -2.106,  3.074]
robot.SetDOFValues(joint_start, robot.GetManipulator('rightarm').GetArmIndices())

quat_target = [1,0,0,0] # wxyz
xyz_target = [6.51073449e-01,  -1.87673551e-01, 4.91061915e-01]
hmat_target = openravepy.matrixFromPose( np.r_[quat_target, xyz_target] )

# BEGIN ik
manip = robot.GetManipulator("rightarm")
init_joint_target = ku.ik_for_link(hmat_target, manip, "r_gripper_tool_frame",
    filter_options = openravepy.IkFilterOptions.CheckEnvCollisions)
# END ik

# X_s = np.load('./x_nd.npy')
# K = tps.tps_kernel_matrix(X_s)
# X_s_new = np.load('./xtarg_nd.npy')
# lambd = 0.01

# X_s = np.array([[1,1,0], [1,0,0], [0,0,0], [0,1,0], [0.5,0.5,0]])
# X_s = np.r_[X_s, X_s+np.array([0,0,1])]
# K = tps.tps_kernel_matrix(X_s)
# X_s_new = X_s
# lambd = 0.01

import scipy.io
tps_data = scipy.io.loadmat('/home/alex/rll/matlab/lfd-tps-trajopt/validation/tps_data.mat')
X_s = tps_data['X_s']
K = tps_data['K']
X_s_new = tps_data['X_s_new']
lambd = tps_data['lambda'][0,0]

f = registration.fit_ThinPlateSpline(X_s, X_s_new, bend_coef = lambd, wt_n=None, rot_coef = np.r_[0,0,0])
A_r = f.w_ng
B_r = f.lin_ag
c_r = f.trans_g

(n, d) = X_s.shape
tps_dim = (n-(d+1))*d + d*d + d;

def nullspace(A, atol=1e-13, rtol=0):
    A = np.atleast_2d(A)
    u, s, vh = np.linalg.svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns
N = nullspace(np.r_[X_s.T, np.ones((1,n))]);

# import scipy.io
# f = registration.fit_ThinPlateSpline(X_s, X_s_new, bend_coef = lambd, wt_n=None, rot_coef = np.r_[0,0,0])
# tps_data = {}
# tps_data['X_s'] = X_s
# tps_data['K'] = tps.tps_kernel_matrix(X_s)
# tps_data['N'] = N
# tps_data['X_s_new'] = X_s_new
# tps_data['lambda'] = lambd
# tps_data['A'] = f.w_ng
# tps_data['B'] = f.lin_ag
# tps_data['c'] = f.trans_g
# scipy.io.savemat('/home/alex/rll/matlab/lfd-tps-trajopt/validation/tps_data.mat', mdict=tps_data)


# TODO: remove truncation of x after testing
#x_nd = x_nd[0:5, 0:3];
#xtarg_nd = xtarg_nd[0:5, 0:3];


request = {
  "basic_info" : {
    "n_steps" : 10,
    "n_ext" : tps_dim,
    "manip" : "rightarm", # see below for valid values
    "start_fixed" : True # i.e., DOF values at first timestep are fixed based on current robot state
  },
  "costs" : [
  {
    "type" : "joint_vel", # joint-space velocity cost
    "params": {"coeffs" : [1]} # a list of length one is automatically expanded to a list of length n_dofs
  },
  {
    "type" : "collision",
    "name" :"cont_coll", # shorten name so printed table will be prettier
    "params" : {
      "continuous" : True,
      "coeffs" : [20], # penalty coefficients. list of length one is automatically expanded to a list of length n_timesteps
      "dist_pen" : [0.025] # robot-obstacle distance that penalty kicks in. expands to length n_timesteps
    }
  },
  {
    "type" : "tps_cost_cnt",
    "name" : "tps_cost_cnt",
    "params" : {
      "lambda" : lambd,
      "alpha" : 1,
      "beta" : 1
    }
  }
  ],
  "constraints" : [
  # BEGIN pose_constraint
  {
    "type" : "pose", 
    "params" : {"xyz" : xyz_target, 
                "wxyz" : quat_target, 
                "link": "r_gripper_tool_frame",
                "timestep" : 9
                }
                 
  }
  # END pose_constraint
  ],
  # BEGIN init
  "init_info" : {
      "type" : "straight_line", # straight line in joint space.
      "endpoint" : init_joint_target.tolist() # need to convert numpy array to list
  }
  # END init
}

A_right = np.zeros((n-(d+1), d))
B = np.eye(d)
c = np.zeros((d, 1))
request["init_info"]["data_ext"] = np.r_[A_right.flatten(), B.flatten(), c.flatten()].tolist()

request["costs"][2]["X_s"] = [row.tolist() for row in X_s.T]
request["costs"][2]["X_s_new"] = [row.tolist() for row in X_s_new.T]
request["costs"][2]["X_g"] = [row.tolist() for row in X_s.T]  ## TODO: correct this
request["costs"][2]["K"] = [row.tolist() for row in K]

if args.position_only: request["constraints"][0]["params"]["rot_coeffs"] = [0,0,0]

s = json.dumps(request) # convert dictionary into json-formatted string
prob = trajoptpy.ConstructProblem(s, env) # create object that stores optimization problem
result = trajoptpy.OptimizeProblem(prob) # do optimization
print result

ext = result.GetExt()
A = N.dot(ext[0:(n-(d+1))*d].reshape(n-(d+1), d))
B = ext[(n-(d+1))*d:(n-(d+1))*d+d*d].reshape(d,d)
c = ext[(n-(d+1))*d+d*d:(n-(d+1))*d+d*d+d].reshape(d,)

from trajoptpy.check_traj import traj_is_safe
prob.SetRobotActiveDOFs() # set robot DOFs to DOFs in optimization problem
assert traj_is_safe(result.GetTraj(), robot) # Check that trajectory is collision free

# Now we'll check to see that the final constraint was satisfied
robot.SetActiveDOFValues(result.GetTraj()[-1])
posevec = openravepy.poseFromMatrix(robot.GetLink("r_gripper_tool_frame").GetTransform())
quat, xyz = posevec[0:4], posevec[4:7]

quat *= np.sign(quat.dot(quat_target))
if args.position_only:
    assert (quat - quat_target).max() > 1e-3
else:
    assert (quat - quat_target).max() < 1e-3

