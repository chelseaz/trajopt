#pragma once

#include "sco/modeling.hpp"
#include "sco/modeling_utils.hpp"
#include "sco/sco_fwd.hpp"
#include <Eigen/Core>
#include "trajopt/common.hpp"
#include <openrave/openrave.h>
namespace trajopt {

using namespace sco;
typedef BasicArray<Var> VarArray;

#if 0
void makeTrajVariablesAndBounds(int n_steps, const RobotAndDOF& manip, OptProb& prob_out, VarArray& vars_out);

class FKFunc {
public:
  virtual OpenRAVE::Transform operator()(const VectorXd& x) const = 0;
  virtual ~FKFunc() {}
};

class FKPositionJacobian {
public:
  virtual Eigen::MatrixXd operator()(const VectorXd& x) const = 0;
  virtual ~FKPositionJacobian() {}
};
#endif


struct CartPoseErrCalculator : public VectorOfVector {
  OR::Transform pose_inv_;
  ConfigurationPtr manip_;
  OR::KinBody::LinkPtr link_;
  CartPoseErrCalculator(const OR::Transform& pose, ConfigurationPtr manip, OR::KinBody::LinkPtr link) :
    pose_inv_(pose.inverse()),
    manip_(manip),
    link_(link) {}
  VectorXd operator()(const VectorXd& dof_vals) const;
};

struct RelPtsErrCalculator : public VectorOfVector {
  Eigen::MatrixX3d xyzs_;
  Eigen::MatrixX3d rel_xyzs_;
  ConfigurationPtr manip_;
  OR::KinBody::LinkPtr link_;
  RelPtsErrCalculator(const Eigen::MatrixX3d& xyzs, const Eigen::MatrixX3d& rel_xyzs, ConfigurationPtr manip, OR::KinBody::LinkPtr link) :
    xyzs_(xyzs),
    rel_xyzs_(rel_xyzs),
    manip_(manip),
    link_(link) {}
  VectorXd operator()(const VectorXd& dof_vals) const;
};

struct RelPtsPenaltyCalculator : public VectorOfVector {
  Eigen::MatrixX3d lambdas_;
  Eigen::MatrixX3d rel_xyzs_;
  ConfigurationPtr manip_;
  OR::KinBody::LinkPtr link_;
  RelPtsPenaltyCalculator(const Eigen::MatrixX3d& lambdas, const Eigen::MatrixX3d& rel_xyzs, ConfigurationPtr manip, OR::KinBody::LinkPtr link) :
    lambdas_(lambdas),
    rel_xyzs_(rel_xyzs),
    manip_(manip),
    link_(link) {}
  VectorXd operator()(const VectorXd& dof_vals) const;
};


struct CartPoseErrorPlotter : public Plotter {
  boost::shared_ptr<void> m_calc; //actually points to a CartPoseErrCalculator = CartPoseCost::f_
  VarVector m_vars;
  CartPoseErrorPlotter(boost::shared_ptr<void> calc, const VarVector& vars) : m_calc(calc), m_vars(vars) {}
  void Plot(const DblVec& x, OR::EnvironmentBase& env, std::vector<OR::GraphHandlePtr>& handles);
};

struct RelPtsErrorPlotter : public Plotter {
  boost::shared_ptr<void> m_calc;
  VarVector m_vars;
  RelPtsErrorPlotter(boost::shared_ptr<void> calc, const VarVector& vars) : m_calc(calc), m_vars(vars) {}
  void Plot(const DblVec& x, OR::EnvironmentBase& env, std::vector<OR::GraphHandlePtr>& handles);
};

struct CartVelJacCalculator : MatrixOfVector {
  ConfigurationPtr manip_;
  KinBody::LinkPtr link_;
  double limit_;
  CartVelJacCalculator(ConfigurationPtr manip, KinBody::LinkPtr link, double limit) :
    manip_(manip), link_(link), limit_(limit) {}

  MatrixXd operator()(const VectorXd& dof_vals) const;
};

struct CartVelCalculator : VectorOfVector {
  ConfigurationPtr manip_;
  KinBody::LinkPtr link_;
  double limit_;
  CartVelCalculator(ConfigurationPtr manip, KinBody::LinkPtr link, double limit) :
    manip_(manip), link_(link), limit_(limit) {}

  VectorXd operator()(const VectorXd& dof_vals) const;
};

#if 0
class CartPoseCost : public CostFromErrFunc {
public:
  CartPoseCost(const VarVector& vars, const OR::Transform& pose, RobotAndDOFPtr manip, KinBody::LinkPtr link, const VectorXd& coeffs);
};

class CartPoseConstraint : public ConstraintFromFunc {
public:
  CartPoseConstraint(const VarVector& vars, const OR::Transform& pose, RobotAndDOFPtr manip, KinBody::LinkPtr link, const VectorXd& coeffs);
};

class CartVelConstraint : public ConstraintFromFunc {
public:
  CartVelConstraint(const VarVector& step0vars, const VarVector& step1vars, RobotAndDOFPtr manip, KinBody::LinkPtr link, double distlimit);
};
#endif



}
