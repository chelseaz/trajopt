#pragma once

#include "macros.h"
#include "sco/modeling.hpp"
#include "trajopt/common.hpp"

namespace trajopt {

class Transformation {
  public:
    virtual MatrixXd transform_points(const MatrixXd& x_ma) = 0;
    virtual vector<MatrixXd> compute_jacobian(const MatrixXd& x_ma) = 0;
    vector<Matrix3d> transform_bases(const MatrixXd& x_ma, const vector<Matrix3d>& rot_mad);
    vector<OR::Transform> transform_hmats(const vector<OR::Transform>& hmat_mAD);
    MatrixXd compute_numerical_jacobian(const MatrixXd& x_d, double epsilon = 0.0001);
};

class ThinPlateSpline : public Transformation {
  MatrixXd x_na_;    // (n,d), centers of basis functions
  MatrixXd w_ng_;    // (n,d)
  MatrixXd lin_ag_;  // (d,d),_transpose of linear part, so you take x_na.dot(lin_ag)
  VectorXd trans_g_; // (d), translation part
  int n_;
  int d_;
public:
  ThinPlateSpline(double d = 3); // identity transformation
  ThinPlateSpline(const MatrixXd& x_na);
  ThinPlateSpline(const MatrixXd& theta, const MatrixXd& x_na); // theta is (n+d+1, d)
  void setTheta(const MatrixXd& theta);
  MatrixXd transform_points(const MatrixXd& x_ma);
  vector<MatrixXd> compute_jacobian(const MatrixXd& x_ma);
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

struct TpsCartPoseErrCalculator : public VectorOfVector {
  MatrixXd x_na_;
  OR::Transform src_pose_;
  ConfigurationPtr manip_;
  OR::KinBody::LinkPtr link_;
  int n_dof_;
  int n_;
  int d_;
  TpsCartPoseErrCalculator(const MatrixXd& x_na, const OR::Transform& src_pose, ConfigurationPtr manip, OR::KinBody::LinkPtr link) :
    x_na_(x_na),
    src_pose_(src_pose),
    manip_(manip),
    link_(link),
    n_dof_(manip->GetDOF()),
    n_(x_na.rows()),
    d_(x_na.cols()) {}
  VectorXd operator()(const VectorXd& dof_theta_vals) const;
};

}
