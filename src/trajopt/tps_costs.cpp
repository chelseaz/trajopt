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

namespace {

inline Vector3d rotVec(const OpenRAVE::Vector& q) {
  return Vector3d(q[1], q[2], q[3]);
}

}

namespace trajopt {

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
  return ((MatrixXd)(K_mn * w_ng + x_ma * lin_ag)).rowwise() + trans_g;
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

vector<MatrixXd> tps_grad(const MatrixXd& x_ma, const MatrixXd& lin_ag, const VectorXd& trans_g, const MatrixXd& w_ng, const MatrixXd& x_na) {
  int n = x_na.rows();
  int d = x_na.cols();
  int m = x_ma.rows();
  assert(3 == x_ma.cols());

  MatrixXd dist_mn = cdist(x_ma, x_na);

  vector<MatrixXd> grad_mga(d);

  MatrixXd lin_ga = lin_ag.transpose();
  for (int a = 0; a < d; a++) {
    MatrixXd diffa_mn = x_ma.col(a).replicate(1,n).rowwise() - x_na.col(a);
    assert(m == diffa_mn.rows());
    assert(n == diffa_mn.cols());
    grad_mga[a] = -((nan2zero(diffa_mn.cwiseQuotient(dist_mn)) * w_ng).rowwise() - lin_ga.col(a)); //TODO check this does the right thing
  }
  return grad_mga;
}

vector<Matrix3d> Transformation::transform_bases(const MatrixXd& x_ma, const vector<Matrix3d>& rot_mad) {
  //TODO
/*
     def transform_bases(self, x_ma, rot_mad, orthogonalize=True, orth_method = "cross"):
        """
        orthogonalize: none, svd, qr
        """

        grad_mga = self.compute_jacobian(x_ma)
        newrot_mgd = np.array([grad_ga.dot(rot_ad) for (grad_ga, rot_ad) in zip(grad_mga, rot_mad)])


        if orthogonalize:
            if orth_method == "qr":
                newrot_mgd =  orthogonalize3_qr(newrot_mgd)
            elif orth_method == "svd":
                newrot_mgd = orthogonalize3_svd(newrot_mgd)
            elif orth_method == "cross":
                newrot_mgd = orthogonalize3_cross(newrot_mgd)
            else: raise Exception("unknown orthogonalization method %s"%orthogonalize)
        return newrot_mgd
*/
}

vector<OR::Transform> Transformation::transform_hmats(const vector<OR::Transform>& hmat_mAD) {
  //TODO
/*

    def transform_hmats(self, hmat_mAD):
        """
        Transform (D+1) x (D+1) homogenius matrices
        """
        hmat_mGD = np.empty_like(hmat_mAD)
        hmat_mGD[:,:3,3] = self.transform_points(hmat_mAD[:,:3,3])
        hmat_mGD[:,:3,:3] = self.transform_bases(hmat_mAD[:,:3,3], hmat_mAD[:,:3,:3])
        hmat_mGD[:,3,:] = np.array([0,0,0,1])
        return hmat_mGD
*/
}

MatrixXd Transformation::compute_numerical_jacobian(const MatrixXd& x_d, double epsilon) {
  //TODO
/*
def compute_numerical_jacobian(self, x_d, epsilon=0.0001):
    "numerical jacobian"
    x0 = np.asfarray(x_d)
    f0 = self.transform_points(x0)
    jac = np.zeros(len(x0), len(f0))
    dx = np.zeros(len(x0))
    for i in range(len(x0)):
        dx[i] = epsilon
        jac[i] = (self.transform_points(x0+dx) - f0) / epsilon
        dx[i] = 0.
    return jac.transpose()
*/
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
  assert((n_+d_+1) == theta.rows());
  assert(d_ == theta.cols());
  trans_g_ = theta.topRows(1);
  lin_ag_ = theta.middleRows(1,d_);
  w_ng_ = theta.bottomRows(n_);
}

MatrixXd ThinPlateSpline::transform_points(const MatrixXd& x_ma) {
  MatrixXd y_ng = tps_eval(x_ma, lin_ag_, trans_g_, w_ng_, x_na_);
  assert(x_ma.rows() == y_ng.rows());
  assert(x_ma.cols() == y_ng.cols());
  return y_ng;
}

vector<MatrixXd> ThinPlateSpline::compute_jacobian(const MatrixXd& x_ma) {
  vector<MatrixXd> grad_mga = tps_grad(x_ma, lin_ag_, trans_g_, w_ng_, x_na_);
  assert(x_ma.cols() == grad_mga.size());
  assert(grad_mga > 0 ? x_ma.rows() == grad_mga[0].rows() : true);
  assert(grad_mga > 0 ? x_na_.rows() == grad_mga[0].cols() : true);
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
  int m_vars = tps_vars.rows();
  int dim = tps_vars.cols();
  assert(dim == 3);
  assert(tps_vars.cols() == 1);
  assert(H.rows() == m_vars);
  assert(H.cols() == m_vars);
  assert(f.rows() == m_vars);
  assert(f.cols() == dim);
  assert(A.cols() == m_vars);
  int n_cnts = A.rows();

  JacobiSVD<MatrixXd> svd(A.transpose(), ComputeFullV);
  VectorXd singular_values = svd.singularValues();
  MatrixXd V = svd.matrixV();
  int nullity = n_cnts - (dim+1);
  N_ = V.block(0, V.cols()-nullity, V.rows(), nullity);

  NHN_ = N_.transpose()*H*N_;

  QuadExpr exprTrzNHNz;
  exprTrzNHNz.coeffs.reserve(NHN_.rows()*NHN_.cols());
  exprTrzNHNz.vars1.reserve(NHN_.rows()*NHN_.cols());
  exprTrzNHNz.vars2.reserve(NHN_.rows()*NHN_.cols());
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
  exprSumfNz.coeffs.reserve(m_vars);
  exprSumfNz.vars.reserve(m_vars);
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

VectorXd TpsCartPoseErrCalculator::operator()(const VectorXd& dof_theta_vals) const {
  VectorXd dof_vals = dof_theta_vals.topRows(n_dof_);
  VectorXd theta_vals = dof_theta_vals.bottomRows(dof_theta_vals.size() - n_dof_);
  const MatrixXd theta = Map<const MatrixXd>(theta_vals.data(), n_, 1+d_+n_); //TODO check matrix gotten in right order

  manip_->SetDOFValues(toDblVec(dof_vals));
  OR::Transform targ_pose = link_->GetTransform();

  ThinPlateSpline f(theta, x_na_);
  OR::Transform warped_src_pose = f.transform_hmats(vector<OR::Transform>(1, src_pose_))[0];

  OR::Transform pose_err = warped_src_pose.inverse() * targ_pose;
  VectorXd err = concat(rotVec(pose_err.rot), toVector3d(pose_err.trans));
  return err;
}

}
