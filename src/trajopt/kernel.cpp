#include <cmath>
#include <Eigen/Dense>
#include <iostream>

using namespace Eigen;

namespace trajopt {

double rbf_kernel(double x, double y) {
  double gamma = 1;
  return exp(-gamma * pow(x-y, 2));
}

VectorXd kernel_timesteps(int n_steps) {
  // TODO: is this the right scale for timesteps?
  return VectorXd::LinSpaced(n_steps, 0.0, 1.0);
}

MatrixXd kernel_matrix(int dofs, const VectorXd& timesteps) {
  int D = dofs;  // number of dofs
  int N = timesteps.rows();  // number of timesteps
  MatrixXd K(D*N, D*N);  // kernel matrix

  // precompute kernel matrix
  for (int i=0; i < N; ++i) {
    for (int j=0; j < N; ++j) {
      double k = rbf_kernel(timesteps[i], timesteps[j]);
      // assign DxD block of kernel matrix to kernel value at timesteps i,j
      // K.block(i*D,j*D,D,D) = MatrixXd::Constant(D,D,k);

      // assign DxD block of kernel matrix to kI
      // where k = kernel value at timesteps i,j
      K.block(i*D,j*D,D,D) = k * MatrixXd::Identity(D, D);
    }
  }

  return K;
}

MatrixXd compute_trajectory(int N, int D, const MatrixXd& K, const VectorXd& a) {
  MatrixXd all_xi = K * a;
  all_xi.resize(D, N);  // resize preserves column order
  std::cout << "trajectory is\n" << all_xi << "\n\n";
  return all_xi.transpose();  // N x D
}

VectorXd coefs_for_trajectory(const MatrixXd& K, const MatrixXd& traj) {
  MatrixXd traj_flat = traj.transpose();  // D x N
  traj_flat.resize(K.rows(), 1);  // DN x 1
  return K.ldlt().solve(traj_flat);
}
}