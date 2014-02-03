#include <Eigen/Core>
#include <iostream>
#include <set>
#include <sstream>
#include "sco/expr_ops.hpp"
#include "sco/modeling_utils.hpp"
#include "trajopt/tps_costs.hpp"
#include "utils/eigen_conversions.hpp"


using namespace std;
using namespace sco;
using namespace Eigen;

#define pM(a) std::cout << #a << ".shape = (" << ((a).rows()) << ", " << ((a).cols()) << ")" << std::endl

void printNpArray(const MatrixXd& m) {
  cout << "np.array([";
  for (int i = 0; i < m.rows(); i++) {
    cout << "[";
    for (int j = 0; j < m.cols(); j++) {
      cout << m(i,j) << ",";
    }
    cout << "],";
  }
  cout << "])";
}

namespace {

inline Vector3d rotVec(const OpenRAVE::Vector& q) {
  return Vector3d(q[1], q[2], q[3]);
}

}

namespace trajopt {

void python_check_transform_hmats(ThinPlateSpline& f, const OR::Transform& src_pose, const OR::Transform& warped_src_pose) {
  // python code to verify transform_hmats
  cout << "f = ThinPlateSpline()" << endl;
  cout << "f.lin_ag=";
  printNpArray(f.lin_ag_);
  cout << endl;
  cout << "f.trans_g=";
  printNpArray(f.trans_g_);
  cout << endl;
  cout << "f.trans_g = f.trans_g[:,0]" << endl;
  cout << "f.w_ng=";
  printNpArray(f.w_ng_);
  cout << endl;
  cout << "f.x_na=";
  printNpArray(f.x_na_);
  cout << endl;

  cout << "src_pose = np.eye(4)" << endl;
  cout << "src_pose[:3,3] = ";
  printNpArray(toVector3d(src_pose.trans));
  cout << ".T" << endl;
  cout << "src_pose[:3,:3] = ";
  printNpArray(toRot(src_pose.rot));
  cout << endl;

  cout << "warped_src_pose = np.eye(4)" << endl;
  cout << "warped_src_pose[:3,3] = ";
  printNpArray(toVector3d(warped_src_pose.trans));
  cout << ".T" << endl;
  cout << "warped_src_pose[:3,:3] = ";
  printNpArray(toRot(warped_src_pose.rot));
  cout << endl;

  cout << "print f.transform_hmats(np.array([src_pose]))" << endl;
  cout << "print warped_src_pose" << endl;
}

MatrixXd cdist(const MatrixXd& A, const MatrixXd& B) {
  int m_A = A.rows();
  int n = A.cols();
  int m_B = B.rows();
  assert(n == B.cols());
  MatrixXd dist(m_A, m_B);
  for (int i = 0; i < m_A; i++) {
    dist.row(i) = (B.rowwise() - A.row(i)).rowwise().norm();
  }
  return dist;
}

MatrixXd tps_apply_kernel(const MatrixXd& distmat, int dim) {
  if (dim == 2) {
    MatrixXd distmat_nonzero = distmat + 1e-20*MatrixXd::Ones(distmat.rows(), distmat.cols());
    return 4.0 * distmat.array().square() * distmat_nonzero.array().log();
  } else if (dim == 3) {
    return -distmat;
  } else {
    throw std::runtime_error("tps_apply_kernel is only defined for dim 2 and 3");
  }
}

MatrixXd tps_kernel_matrix2(const MatrixXd& x_na, const MatrixXd& y_ma) {
  int dim = x_na.cols();
  MatrixXd distmat = cdist(x_na, y_ma);
  return tps_apply_kernel(distmat, dim);
}

MatrixXd tps_eval(const MatrixXd& x_ma, const MatrixXd& lin_ag, const VectorXd& trans_g, const MatrixXd& w_ng, const MatrixXd& x_na) {
  MatrixXd K_mn = tps_kernel_matrix2(x_ma, x_na);
  return ((MatrixXd)(K_mn * w_ng + x_ma * lin_ag)).rowwise() + trans_g.transpose();
}

// maybe there is better way to do this
MatrixXd nan2zero(const MatrixXd& m) {
  MatrixXd out = m;
  for (int i = 0; i < out.rows(); i++) {
    for (int j = 0; j < out.cols(); j++) {
      if (isnan((double)out(i,j))) {
        out(i,j) = 0.0;
      }
    }
  }
  return out;
}

vector<Matrix3d> tps_grad(const MatrixXd& x_ma, const MatrixXd& lin_ag, const VectorXd& trans_g, const MatrixXd& w_ng, const MatrixXd& x_na) {
  int n = x_na.rows();
  int d = x_na.cols();
  int m = x_ma.rows();
  assert(3 == x_ma.cols());

  MatrixXd dist_mn = cdist(x_ma, x_na);

  vector<Matrix3d> grad_mga(m);

  MatrixXd lin_ga = lin_ag.transpose();
  for (int a = 0; a < d; a++) {
    MatrixXd diffa_mn = x_ma.col(a).replicate(1,n).rowwise() - x_na.col(a).transpose();
    assert(m == diffa_mn.rows());
    assert(n == diffa_mn.cols());
    MatrixXd tmp = nan2zero(diffa_mn.cwiseQuotient(dist_mn)) * w_ng;
    assert(tmp.rows() == m);
    assert(tmp.cols() == d);
    for (int i = 0; i < m; i++) {
      grad_mga[i].col(a) = lin_ga.col(a) - tmp.row(i).transpose();
    }
  }
  return grad_mga;
}

Matrix3d orthogonalize3_cross(const Matrix3d& mat) {
  Vector3d x_n3 = mat.col(0);
  Vector3d z_n3 = mat.col(2);

  Vector3d znew_n3 = z_n3.normalized();
  Vector3d ynew_n3 = znew_n3.cross(x_n3).normalized();
  Vector3d xnew_n3 = ynew_n3.cross(znew_n3).normalized();

  Matrix3d new_mat;
  new_mat.col(0) = xnew_n3;
  new_mat.col(1) = ynew_n3;
  new_mat.col(2) = znew_n3;

  return new_mat;
}

vector<Matrix3d> Transformation::transform_bases(const MatrixXd& x_ma, const vector<Matrix3d>& rot_mad) {
  int m = x_ma.rows();
  int d = x_ma.cols();
  assert(m == rot_mad.size());

  vector<Matrix3d> grad_mga = compute_jacobian(x_ma);
  assert(m == grad_mga.size());

  vector<Matrix3d> newrot_mgd(m);
  for (int i = 0; i < m; i++) {
    newrot_mgd[i] = orthogonalize3_cross(grad_mga[i] * rot_mad[i]);
  }
  return newrot_mgd;
}

vector<OR::Transform> Transformation::transform_hmats(const vector<OR::Transform>& hmat_mAD) {
  int m = hmat_mAD.size();
  MatrixXd hmat_mAD_trans(m,3);
  vector<Matrix3d> hmat_mAD_rot(m);
  for (int i = 0; i < m; i++) {
    hmat_mAD_trans.row(i) = toVector3d(hmat_mAD[i].trans);
    hmat_mAD_rot[i] = toRot(hmat_mAD[i].rot);
  }
  MatrixXd hmat_mGD_trans = transform_points(hmat_mAD_trans);
  vector<Matrix3d> hmat_mGD_rot = transform_bases(hmat_mAD_trans, hmat_mAD_rot);
  assert(hmat_mGD_trans.rows() == m);
  assert(hmat_mGD_trans.cols() == 3);
  assert(hmat_mGD_rot.size() == m);
  vector<OR::Transform> hmat_mGD(m);
  for (int i = 0; i < m; i++) {
    hmat_mGD[i] = toRaveTransform(hmat_mGD_rot[i], hmat_mGD_trans.row(i));
  }
  assert(hmat_mAD.size() == hmat_mGD.size());
  return hmat_mGD;
}

MatrixXd Transformation::compute_numerical_jacobian(const MatrixXd& x_d, double epsilon) {
  MatrixXd x0 = x_d;
  MatrixXd f0 = transform_points(x0);
  MatrixXd jac(x0.rows(), f0.rows());
  VectorXd dx = VectorXd::Zero(x0.rows());
  for (int i = 0; i < x0.rows(); i++) {
    dx(i) = epsilon;
    jac.row(i) = transform_points(x0.colwise()+dx) / epsilon;
    dx(i) = 0;
  }
  return jac.transpose();
}


ThinPlateSpline::ThinPlateSpline(double d) {
  n_ = 0;
  d_ = d;
  x_na_ = MatrixXd::Zero(0,d);
  lin_ag_ = MatrixXd::Identity(d,d);
  trans_g_ = VectorXd::Zero(d);
  w_ng_ = MatrixXd::Zero(0,d);
}

ThinPlateSpline::ThinPlateSpline(const MatrixXd& x_na) {
  n_ = x_na.rows();
  d_ = x_na.cols();
  x_na_ = x_na;
}

ThinPlateSpline::ThinPlateSpline(const MatrixXd& theta, const MatrixXd& x_na) {
  n_ = x_na.rows();
  d_ = x_na.cols();
  setTheta(theta);
  x_na_ = x_na;
}

void ThinPlateSpline::setTheta(const MatrixXd& theta) {
  assert(n_+d_+1 == theta.rows());
  assert(d_ == theta.cols());
  trans_g_ = theta.row(0);
  lin_ag_ = theta.middleRows(1,d_);
  w_ng_ = theta.bottomRows(n_);
}

MatrixXd ThinPlateSpline::transform_points(const MatrixXd& x_ma) {
  MatrixXd y_ng = tps_eval(x_ma, lin_ag_, trans_g_, w_ng_, x_na_);
  assert(x_ma.rows() == y_ng.rows());
  assert(x_ma.cols() == y_ng.cols());
  return y_ng;
}

vector<Matrix3d> ThinPlateSpline::compute_jacobian(const MatrixXd& x_ma) {
  int m = x_ma.rows();
  int d = x_ma.cols();
  assert(d==3);
  vector<Matrix3d> grad_mga = tps_grad(x_ma, lin_ag_, trans_g_, w_ng_, x_na_);
  assert(m == grad_mga.size());
  return grad_mga;
}


TpsCost::TpsCost(const VarArray& traj_vars, const VarArray& tps_vars, const MatrixXd& H, const MatrixXd& f, const MatrixXd& A) :
    Cost("Tps"), traj_vars_(traj_vars), tps_vars_(tps_vars), H_(H), f_(f), A_(A) {
  /**
   * solve equality-constrained qp
   * min tr(x'Hx) + sum(f'x)
   * s.t. Ax = 0
   *
   * Let x = Nz
   * then the problem becomes the unconstrained minimization z'NHNz + f'Nz
   */
  int n = tps_vars.rows();
  int dim = tps_vars.cols();
  assert(dim == 3);
  assert(tps_vars.cols() == dim);
  assert(H.rows() == n+dim+1);
  assert(H.cols() == n+dim+1);
  assert(f.rows() == n+dim+1);
  assert(f.cols() == dim);
  assert(A.cols() == n+dim+1);
  int n_cnts = A.rows();

  JacobiSVD<MatrixXd> svd(A, ComputeFullV);
  VectorXd singular_values = svd.singularValues();
  MatrixXd V = svd.matrixV();
  int nullity = n;
  N_ = V.block(0, V.cols()-nullity, V.rows(), nullity);

  NHN_ = N_.transpose()*H*N_;

  QuadExpr exprTrzNHNz;
  exprTrzNHNz.coeffs.reserve(dim * NHN_.rows()*(NHN_.cols()+1)/2);
  exprTrzNHNz.vars1.reserve(dim * NHN_.rows()*(NHN_.cols()+1)/2);
  exprTrzNHNz.vars2.reserve(dim * NHN_.rows()*(NHN_.cols()+1)/2);
  for (int d = 0; d < dim; d++) {
    for (int i = 0; i < NHN_.rows(); i++) {
      for (int j = i; j < NHN_.cols(); j++) {
        if (i == j) {
          exprTrzNHNz.coeffs.push_back(NHN_(i,j));
        } else {
          exprTrzNHNz.coeffs.push_back(NHN_(i,j)+NHN_(j,i));
        }
        exprTrzNHNz.vars1.push_back(tps_vars_(i,d));
        exprTrzNHNz.vars2.push_back(tps_vars_(j,d));
      }
    }
  }

  fN_ = f.transpose() * N_;

  AffExpr exprSumfNz;
  exprSumfNz.coeffs.reserve(dim * fN_.rows()*fN_.cols());
  exprSumfNz.vars.reserve(dim * fN_.rows()*fN_.cols());
  for (int d = 0; d < dim; d++) {
    for (int i = 0; i < fN_.rows(); i++) {
      for (int j = 0; j < fN_.cols(); j++) {
        exprSumfNz.coeffs.push_back(fN_(i,j));
        exprSumfNz.vars.push_back(tps_vars_(j,d));
      }
    }
  }

  expr_ = exprTrzNHNz;
  exprInc(expr_, exprSumfNz);
}

double TpsCost::value(const vector<double>& xvec) {
  MatrixXd z = getTraj(xvec, tps_vars_);
  double ret = (z.transpose() * NHN_ * z).trace() + (fN_ * z).sum();
  cout << "value check " << ret << " " << expr_.value(xvec) << endl;
  return ret;
}

ConvexObjectivePtr TpsCost::convex(const vector<double>& x, Model* model) {
  ConvexObjectivePtr out(new ConvexObjective(model));
  out->addQuadExpr(expr_);
  return out;
}

TpsCartPoseErrCalculator::TpsCartPoseErrCalculator(const MatrixXd& x_na, const MatrixXd& A, const OR::Transform& src_pose, ConfigurationPtr manip, OR::KinBody::LinkPtr link) :
  x_na_(x_na),
  src_pose_(src_pose),
  manip_(manip),
  link_(link),
  n_dof_(manip->GetDOF()),
  n_(x_na.rows()),
  d_(x_na.cols())
{
  JacobiSVD<MatrixXd> svd(A, ComputeFullV);
  VectorXd singular_values = svd.singularValues();
  MatrixXd V = svd.matrixV();
  int nullity = n_;
  N_ = V.block(0, V.cols()-nullity, V.rows(), nullity); // N_ has dimension (n+d+1,n)
}

VectorXd TpsCartPoseErrCalculator::operator()(const VectorXd& dof_theta_vals) const {
  VectorXd dof_vals = dof_theta_vals.topRows(n_dof_);
  VectorXd theta_vals = dof_theta_vals.bottomRows(dof_theta_vals.size() - n_dof_);
  assert(dof_theta_vals.size() - n_dof_ == n_*d_);
  MatrixXd theta = N_ * Map<const MatrixXd>(theta_vals.data(), n_, d_);

  manip_->SetDOFValues(toDblVec(dof_vals));
  OR::Transform targ_pose = link_->GetTransform();

  ThinPlateSpline f(theta, x_na_);
  OR::Transform warped_src_pose = f.transform_hmats(vector<OR::Transform>(1, src_pose_))[0];

  OR::Transform pose_err = warped_src_pose.inverse() * targ_pose;
  VectorXd err = concat(rotVec(pose_err.rot), toVector3d(pose_err.trans));
  return err;
}

}
