#include <Eigen/Core>
#include <cmath>
#include "sco/expr_ops.hpp"
#include "sco/modeling_utils.hpp"
#include "trajopt/trajectory_costs.hpp"
#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace sco;
using namespace Eigen;

namespace {


static MatrixXd diffAxis0(const MatrixXd& in) {
  return in.middleRows(1, in.rows()-1) - in.middleRows(0, in.rows()-1);
}

static double rbf_kernel(const double x, const double y) {
  double gamma = 1;
  return exp(-gamma * pow(x-y, 2));
}


}

namespace trajopt {



//////////// Quadratic cost functions /////////////////

JointPosCost::JointPosCost(const VarVector& vars, const VectorXd& vals, const VectorXd& coeffs) :
    Cost("JointPos"), vars_(vars), vals_(vals), coeffs_(coeffs) {
    for (int i=0; i < vars.size(); ++i) {
      if (coeffs[i] > 0) {
        AffExpr diff = exprSub(AffExpr(vars[i]), AffExpr(vals[i]));
        exprInc(expr_, exprMult(exprSquare(diff), coeffs[i]));
      }
    }
}
double JointPosCost::value(const vector<double>& xvec) {
  VectorXd dofs = getVec(xvec, vars_);
  return ((dofs - vals_).array().square() * coeffs_.array()).sum();
}
ConvexObjectivePtr JointPosCost::convex(const vector<double>& x, Model* model) {
  ConvexObjectivePtr out(new ConvexObjective(model));
  out->addQuadExpr(expr_);
  return out;
}


JointVelCost::JointVelCost(const VarArray& vars, const VectorXd& coeffs) :
    Cost("JointVel"), vars_(vars), coeffs_(coeffs) {
  for (int i=0; i < vars.rows()-1; ++i) {
    for (int j=0; j < vars.cols(); ++j) {
      AffExpr vel;
      exprInc(vel, exprMult(vars(i,j), -1));
      exprInc(vel, exprMult(vars(i+1,j), 1));
      exprInc(expr_, exprMult(exprSquare(vel),coeffs_[j]));
    }
  }
}
double JointVelCost::value(const vector<double>& xvec) {
  MatrixXd traj = getTraj(xvec, vars_);
  return (diffAxis0(traj).array().square().matrix() * coeffs_.asDiagonal()).sum();
}
ConvexObjectivePtr JointVelCost::convex(const vector<double>& x, Model* model) {
  ConvexObjectivePtr out(new ConvexObjective(model));
  out->addQuadExpr(expr_);
  return out;
}


JointAccCost::JointAccCost(const VarArray& vars, const VectorXd& coeffs) :
    Cost("JointAcc"), vars_(vars), coeffs_(coeffs) {
  for (int i=0; i < vars.rows()-2; ++i) {
    for (int j=0; j < vars.cols(); ++j) {
      AffExpr acc;
      exprInc(acc, exprMult(vars(i,j), -1));
      exprInc(acc, exprMult(vars(i+1,j), 2));
      exprInc(acc, exprMult(vars(i+2,j), -1));
      exprInc(expr_, exprMult(exprSquare(acc), coeffs_[j]));
    }
  }
}
double JointAccCost::value(const vector<double>& xvec) {
  MatrixXd traj = getTraj(xvec, vars_);
  return (diffAxis0(diffAxis0(traj)).array().square().matrix() * coeffs_.asDiagonal()).sum();
}
ConvexObjectivePtr JointAccCost::convex(const vector<double>& x, Model* model) {
  ConvexObjectivePtr out(new ConvexObjective(model));
  out->addQuadExpr(expr_);
  return out;
}


HilbertNormCost::HilbertNormCost(const VarVector& vars, const VectorXd& timesteps, const int n_dofs) :
    Cost("HilbertNorm"), vars_(vars), timesteps_(timesteps) {
  int D = n_dofs;  // number of dofs
  int N = timesteps_.rows();  // number of timesteps
  K_.resize(D*N, D*N);  // kernel matrix

  // precompute kernel matrix
  for (int i=0; i < N; ++i) {
    for (int j=0; j < N; ++j) {
      double k = rbf_kernel(timesteps_[i], timesteps_[j]);
      // assign DxD block of kernel matrix to kernel value at timesteps i,j
      // K_.block(i*D,j*D,D,D) = MatrixXd::Constant(D,D,k);

      // assign DxD block of kernel matrix to kI
      // where k = kernel value at timesteps i,j
      K_.block(i*D,j*D,D,D) = k * MatrixXd::Identity(D, D);
    }
  }
  
  // shouldn't need to do the following; RBF kernel is psd  
  // // Need to ensure K is PSD, so add \lambda_{min} times identity
  // // TODO: convert cout to LOG_DEBUG
  // SelfAdjointEigenSolver<MatrixXd> eigensolver(K_);
  // if (eigensolver.info() == Success) {
  //   VectorXd eigenvalues = eigensolver.eigenvalues();
  //   cout << "The eigenvalues of K are:\n" << eigenvalues << endl;
  //   double lambda_min = eigenvalues.minCoeff();
  //   if (lambda_min < 0) {
  //     cout << "Adding " << lambda_min << " * I to make K psd" << endl;
  //     K_ += lambda_min * MatrixXd::Identity(D*N, D*N);
  //   }
  // }

  for (int i=0; i < D*N; ++i) {
    for (int j=0; j < D*N; ++j) {
      exprInc(expr_, exprMult(exprMult(AffExpr(vars[i]), AffExpr(vars[j])), K_(i,j)));        
    }
  }
  cleanupQuad(expr_);
}
double HilbertNormCost::value(const vector<double>& xvec) {
  VectorXd a = getVec(xvec, vars_);
  return (a.transpose() * K_ * a)(0,0);
}
ConvexObjectivePtr HilbertNormCost::convex(const vector<double>& x, Model* model) {
  ConvexObjectivePtr out(new ConvexObjective(model));
  out->addQuadExpr(expr_);
  return out;
}

}
