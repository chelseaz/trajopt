#pragma once

#include "macros.h"
#include "sco/modeling.hpp"
#include "trajopt/common.hpp"

namespace trajopt {

class Transformation {
  public:
    virtual MatrixXd transform_points(const MatrixXd& x_ma) = 0;
    virtual vector<Matrix3d> compute_jacobian(const MatrixXd& x_ma) = 0;
    vector<Matrix3d> transform_bases(const MatrixXd& x_ma, const vector<Matrix3d>& rot_mad);
    vector<OR::Transform> transform_hmats(const vector<OR::Transform>& hmat_mAD);
    MatrixXd compute_numerical_jacobian(const MatrixXd& x_d, double epsilon = 0.0001);
};

class ThinPlateSpline : public Transformation {
public: //TODO should be private
  MatrixXd x_na_;    // (n,d), centers of basis functions
  MatrixXd w_ng_;    // (n,d)
  MatrixXd lin_ag_;  // (d,d),_transpose of linear part, so you take x_na.dot(lin_ag)
  VectorXd trans_g_; // (d), translation part
  int n_;
  int d_;
public:
  ThinPlateSpline(double d = 3); // identity transformation
  ThinPlateSpline(const MatrixXd& x_na);
  ThinPlateSpline(const MatrixXd& theta, const MatrixXd& x_na); // theta has dimension (n+d+1, d)
  void setTheta(const MatrixXd& theta);
  MatrixXd transform_points(const MatrixXd& x_ma);
  vector<Matrix3d> compute_jacobian(const MatrixXd& x_ma);
};

using namespace Eigen;
MatrixXd TRAJOPT_API tps_fit3(const MatrixX3d& x_na, const MatrixX3d& y_ng, double bend_coef, const Vector3d& rot_coef, const VectorXd& wt_n);
MatrixXd TRAJOPT_API tps_rpm_bij_corr_iter_part(const MatrixX3d& x_nd, const MatrixX3d& y_md, const Vector3d& trans_g, int n_iter, const VectorXd& regs, const VectorXd& rads, const Vector3d& rot_reg);

class TRAJOPT_API TpsCost : public Cost {
  /**
   * solve equality-constrained qp
   * min tr(x'Hx) + 2 tr(f'x)
   * s.t. Ax = 0
   */
public:
  TpsCost(const VarArray& tps_vars, const MatrixX3d& x_na, const MatrixX3d& y_ng, const Vector3d& bend_coefs, const Vector3d& rot_coefs, const MatrixX3d& wt_n, const MatrixXd& N, double alpha);
  virtual ConvexObjectivePtr convex(const vector<double>& x, Model* model);
  virtual double value(const vector<double>&);
  ThinPlateSpline getThinPlateSpline(const MatrixXd& z_vals);

  //private: // TODO should be private
  VarArray tps_vars_;
  MatrixX3d x_na_;
  MatrixX3d y_ng_;
  Vector3d bend_coefs_;
  Vector3d rot_coefs_;
  MatrixX3d wt_n_;
  MatrixXd N_;
  double alpha_;

private:
  vector<MatrixXd> NHNs;
  vector<VectorXd> fNs;

  QuadExpr expr_;
};

struct TpsCostPlotter : public Plotter {
  boost::shared_ptr<TpsCost> m_tps_cost; //actually points to a TpsCost
  TpsCostPlotter(boost::shared_ptr<TpsCost> tps_cost) : m_tps_cost(tps_cost) {}
  void Plot(const DblVec& x, OR::EnvironmentBase& env, std::vector<OR::GraphHandlePtr>& handles);
};

struct TpsPtsErrCalculator : public VectorOfVector {
  boost::shared_ptr<TpsCost> tps_cost_;
  MatrixX3d src_xyzs_;
  MatrixX3d targ_xyzs_;
  VectorXd max_abs_err_;
  TpsPtsErrCalculator(boost::shared_ptr<TpsCost> tps_cost, const MatrixX3d& src_xyzs, const MatrixX3d& targ_xyzs, const VectorXd& max_abs_err) :
    tps_cost_(tps_cost),
    src_xyzs_(src_xyzs),
    targ_xyzs_(targ_xyzs),
    max_abs_err_(max_abs_err) {}
  VectorXd operator()(const VectorXd& z_vals) const;
};

struct TpsPtsErrJacCalculator : MatrixOfVector {
  boost::shared_ptr<TpsPtsErrCalculator> m_calc;
  MatrixX3d src_xyzs_;
  MatrixXd jac_;
  TpsPtsErrJacCalculator(VectorOfVectorPtr calc, const MatrixX3d& src_xyzs);
  MatrixXd operator()(const VectorXd& z_vals) const;
};

#if 0
struct TpsCorrErrCalculator : public VectorOfVector {
public:
  TpsCorrErrCalculator(const MatrixXd& x_na, const MatrixXd& N, const MatrixXd& y_ng, double alpha);
  VectorXd operator()(const VectorXd& theta_vals) const;
  MatrixXd x_na_;
  MatrixXd N_;
  MatrixXd y_ng_;
  double alpha_;
};
#endif

struct TpsCartPoseErrCalculator : public VectorOfVector {
  MatrixXd x_na_;
  MatrixXd N_;
  OR::Transform src_pose_;
  ConfigurationPtr manip_;
  OR::KinBody::LinkPtr link_;
  int n_dof_;
  int n_;
  int d_;
  TpsCartPoseErrCalculator(const MatrixXd& x_na, const MatrixXd& N, const OR::Transform& src_pose, ConfigurationPtr manip, OR::KinBody::LinkPtr link);
  VectorXd operator()(const VectorXd& dof_theta_vals) const;
  VectorXd extractDofVals(const VectorXd& dof_theta_vals) const;
  MatrixXd extractThetaVals(const VectorXd& dof_theta_vals) const;
};

struct TpsCartPoseErrorPlotter : public Plotter {
  boost::shared_ptr<void> m_calc; //actually points to a TpsCartPoseErrCalculator
  VarVector m_vars;
  TpsCartPoseErrorPlotter(boost::shared_ptr<void> calc, const VarVector& vars) : m_calc(calc), m_vars(vars) {}
  void Plot(const DblVec& x, OR::EnvironmentBase& env, std::vector<OR::GraphHandlePtr>& handles);
};

struct TpsRelPtsErrCalculator : public VectorOfVector {
  boost::shared_ptr<TpsCost> tps_cost_;
  MatrixX3d src_xyzs_;
  MatrixX3d rel_xyzs_;
  ConfigurationPtr manip_;
  OR::KinBody::LinkPtr link_;
  TpsRelPtsErrCalculator(boost::shared_ptr<TpsCost> tps_cost, const MatrixX3d& src_xyzs, const MatrixX3d& rel_xyzs, ConfigurationPtr manip, OR::KinBody::LinkPtr link) :
    tps_cost_(tps_cost),
    src_xyzs_(src_xyzs),
    rel_xyzs_(rel_xyzs),
    manip_(manip),
    link_(link) {}
  VectorXd operator()(const VectorXd& dof_z_vals) const;
};

struct TpsRelPtsErrJacCalculator : MatrixOfVector {
  boost::shared_ptr<TpsRelPtsErrCalculator> m_calc;
  MatrixX3d src_xyzs_;
  MatrixX3d rel_xyzs_;
  ConfigurationPtr manip_;
  OR::KinBody::LinkPtr link_;
  MatrixXd tps_jac_;
  TpsRelPtsErrJacCalculator(VectorOfVectorPtr calc, const MatrixX3d& src_xyzs, const MatrixX3d& rel_xyzs, ConfigurationPtr manip, OR::KinBody::LinkPtr link);
  MatrixXd operator()(const VectorXd& dof_z_vals) const;
};

struct TpsRelPtsErrorPlotter : public Plotter {
  boost::shared_ptr<TpsRelPtsErrCalculator> m_calc;
  VarVector m_vars;
  TpsRelPtsErrorPlotter(VectorOfVectorPtr calc, const VarVector& vars) : m_calc(boost::dynamic_pointer_cast<TpsRelPtsErrCalculator>(calc)), m_vars(vars) {}
  void Plot(const DblVec& x, OR::EnvironmentBase& env, std::vector<OR::GraphHandlePtr>& handles);
};

struct TpsJacOrthErrCalculator : public VectorOfVector {
  boost::shared_ptr<TpsCost> tps_cost_;
  MatrixX3d pts_;
  TpsJacOrthErrCalculator(boost::shared_ptr<TpsCost> tps_cost, const MatrixX3d& pts) :
  tps_cost_(tps_cost),
  pts_(pts) {}
  VectorXd operator()(const VectorXd& z_vals) const;
};

}
