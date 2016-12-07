#pragma once
#include "trajopt/common.hpp"
namespace trajopt {

struct TRAJOPT_API TrajPlotter {
  OpenRAVE::EnvironmentBasePtr m_env;
  ConfigurationPtr m_config;
  VarArray m_trajvars;
  VarArray m_extvars;
  MatrixXd m_kernel_matrix;
  vector<PlotterPtr> m_plotters;
  std::set<KinBody::LinkPtr> m_links; // links for which we'll plot the trajectory
  int m_decimation;

  TrajPlotter(OR::EnvironmentBasePtr env, ConfigurationPtr config, const VarArray& trajvars, const VarArray& extvars, const MatrixXd& kernel_matrix);
  void Add(const vector<CostPtr>& costs);
  void Add(const vector<ConstraintPtr>& constraints);
  void Add(const vector<PlotterPtr>& plotters);
  void Add(PlotterPtr plotter);
  void AddLink(OpenRAVE::KinBody::LinkPtr link);
  void OptimizerCallback(OptProb*, DblVec& x);
  void SetDecimation(int dec) {m_decimation=dec;}
  bool UsingKernel() {return m_kernel_matrix.size() > 0;}

};
typedef boost::shared_ptr<TrajPlotter> TrajPlotterPtr;

}
