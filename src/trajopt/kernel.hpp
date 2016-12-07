#pragma once

namespace trajopt {

static VectorXd null_vector;

double rbf_kernel(double x, double y);
VectorXd kernel_timesteps(int n_steps);
MatrixXd kernel_matrix(int dofs, const VectorXd& timesteps);
MatrixXd compute_trajectory(int N, int D, const MatrixXd& K, const VectorXd& a);
VectorXd coefs_for_trajectory(const MatrixXd& K, const MatrixXd& traj);  // inverse of compute_trajectory
}