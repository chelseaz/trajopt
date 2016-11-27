#pragma once

namespace trajopt {

double rbf_kernel(double x, double y);
MatrixXd kernel_matrix(int dofs, const VectorXd& timesteps);

}