#include <cmath>
#include <Eigen/Core>

using namespace Eigen;

namespace trajopt {
  
double rbf_kernel(double x, double y) {
  double gamma = 1;
  return exp(-gamma * pow(x-y, 2));
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

}