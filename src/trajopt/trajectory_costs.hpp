#pragma once

/**
Simple quadratic costs on trajectory
*/

#include "macros.h"
#include "sco/modeling.hpp"
#include "trajopt/common.hpp"

namespace trajopt {

class TRAJOPT_API JointPosCost : public Cost {
public:
  JointPosCost(const VarVector& vars, const VectorXd& vals, const VectorXd& coeffs);
  virtual ConvexObjectivePtr convex(const vector<double>& x, Model* model);
  virtual double value(const vector<double>&);
private:
  VarVector vars_;
  VectorXd vals_, coeffs_;
  QuadExpr expr_;
};

class TRAJOPT_API JointVelCost : public Cost {
public:
  JointVelCost(const VarArray& traj, const VectorXd& coeffs);
  virtual ConvexObjectivePtr convex(const vector<double>& x, Model* model);
  virtual double value(const vector<double>&);
private:
  VarArray vars_;
  VectorXd coeffs_;
  QuadExpr expr_;
};

class TRAJOPT_API JointAccCost : public Cost {
public:
  JointAccCost(const VarArray& traj, const VectorXd& coeffs);
  virtual ConvexObjectivePtr convex(const vector<double>& x, Model* model);
  virtual double value(const vector<double>&);
private:
  VarArray vars_;
  VectorXd coeffs_;
  QuadExpr expr_;
};

class TRAJOPT_API OldTpsCost : public Cost {
public:
  OldTpsCost(const VarArray& traj_vars, const VarArray& tps_vars, double lambda, double alpha, double beta, const MatrixXd& X_s_new, const MatrixXd& X_s, const MatrixXd& K, const MatrixXd& X_g);
  virtual ConvexObjectivePtr convex(const vector<double>& x, Model* model);
  virtual double value(const vector<double>&);
  static AffArray multiply(MatrixXd A, VarArray B);
  static AffArray multiply(MatrixXd A, AffArray B);
  inline MatrixXd getN() {
    return N_;
  }
private:
  VarArray traj_vars_;
  VarArray tps_vars_;
  double lambda_;
  double alpha_;
  double beta_;
  MatrixXd X_s_new_;
  MatrixXd X_s_;
  MatrixXd K_;
  MatrixXd X_g_;
  MatrixXd N_;
  QuadExpr expr_;
};

class TRAJOPT_API TpsCost : public Cost {
  /**
   * solve equality-constrained qp
   * min tr(x'Hx) + sum(f'x)
   * s.t. Ax = 0
   */
public:
  TpsCost(const VarArray& traj_vars, const VarArray& tps_vars, const MatrixXd& H, const MatrixXd& f, const MatrixXd& A);
  virtual ConvexObjectivePtr convex(const vector<double>& x, Model* model);
  virtual double value(const vector<double>&);
private:
  VarArray traj_vars_;
  VarArray tps_vars_;
  MatrixXd H_;
  MatrixXd f_;
  MatrixXd A_;
  MatrixXd N_;
  MatrixXd NHN_;
  MatrixXd fN_;
  QuadExpr expr_;
};

}

