#include <Eigen/Core>
#include <iostream>
#include <set>
#include <sstream>
#include "sco/expr_ops.hpp"
#include "sco/modeling_utils.hpp"
#include "trajopt/tps_costs.hpp"
#include "trajopt/rave_utils.hpp"
#include "utils/eigen_conversions.hpp"
#include <boost/foreach.hpp>

using namespace std;
using namespace sco;
using namespace Eigen;
using namespace util;

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

MatrixXd cdist(const MatrixX3d& A, const MatrixX3d& B) {
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

MatrixXd tps_kernel_matrix2(const MatrixX3d& x_na, const MatrixX3d& y_ma) {
  int dim = x_na.cols();
  MatrixXd distmat = cdist(x_na, y_ma);
  return tps_apply_kernel(distmat, dim);
}

MatrixXd tps_eval(const MatrixXd& x_ma, const MatrixXd& lin_ag, const VectorXd& trans_g, const MatrixXd& w_ng, const MatrixXd& x_na) {
  MatrixXd K_mn = tps_kernel_matrix2(x_ma, x_na);
  return ((MatrixXd)(K_mn * w_ng + x_ma * lin_ag)).rowwise() + trans_g.transpose();
}

MatrixX3d solve_eqp1(const MatrixXd& H, const MatrixX3d& f, const Matrix4Xd& A) {
  /*
   * solve equality-constrained qp
   * min tr(x'Hx) + sum(f'x)
   * s.t. Ax = 0
   */
  int n_vars = H.rows();
  assert(H.cols() == n_vars);
  assert(f.rows() == n_vars);
  assert(A.cols() == n_vars);
  int n_cnts = A.rows();

  JacobiSVD<MatrixXd> svd(A.transpose(), ComputeFullU);
  MatrixXd U = svd.matrixU();
  MatrixXd N = U.rightCols(U.cols() - n_cnts);
  // columns of N span the null space

  // x = Nz
  // then problem becomes unconstrained minimization .5*z'NHNz + z'Nf
  // NHNz + Nf = 0
  MatrixXd NHN = N.transpose()*H*N;
  MatrixX3d b = -N.transpose()*f;
  MatrixX3d z = NHN.llt().solve(b);

  MatrixX3d x = N * z;

  return x;
}

MatrixXd tps_fit3(const MatrixX3d& x_na, const MatrixX3d& y_ng, double bend_coef, const Vector3d& rot_coef, const VectorXd& wt_n) {
    int n = x_na.rows();
    int d = x_na.cols();

    MatrixXd K_nn = tps_kernel_matrix2(x_na, x_na);
    MatrixXd Q(n,1+d+n);
    Q.leftCols(1) = MatrixXd::Ones(n,1);
    Q.middleCols(1,d) = x_na;
    Q.rightCols(n) = K_nn;
    MatrixXd WQ = Q.cwiseProduct(wt_n.rowwise().replicate(Q.cols()));
    MatrixXd QWQ = Q.transpose() * WQ;
    MatrixXd& H = QWQ;
    H.block(d+1,d+1,n,n) += bend_coef * K_nn;
    H.block(1,1,d,d) += rot_coef.asDiagonal();

    MatrixX3d f = -WQ.transpose() * y_ng;
    f.middleRows(1,d) -= (Matrix3d)rot_coef.asDiagonal();

    Matrix4Xd A(4,1+d+n);
    A.leftCols(1+d) = MatrixXd::Zero(1+d,1+d);
    A.block(0,1+d,1,n) = MatrixXd::Ones(1,n);
    A.bottomRightCorner(d,n) = x_na.transpose();

    MatrixX3d Theta = solve_eqp1(H,f,A);

    return Theta;
}

MatrixXd balance_matrix3(const MatrixXd& prob_nm, int max_iter, double p, double outlierfrac) {
  int n = prob_nm.rows();
  int m = prob_nm.cols();
  MatrixXd prob_NM(n+1, m+1);
  prob_NM.topLeftCorner(n,m) = prob_nm;
  prob_NM.col(m) = p*VectorXd::Ones(n+1);
  prob_NM.row(n) = p*VectorXd::Ones(m+1);
  prob_NM(n,m) = p*sqrt(n*m);

  VectorXd a_N = VectorXd::Ones(n+1);
  a_N(n) = m*outlierfrac;
  VectorXd b_M = VectorXd::Ones(m+1);
  b_M(m) = n*outlierfrac;

  VectorXd r_N = VectorXd::Ones(n+1);
  VectorXd c_M;
  for (int i = 0; i < max_iter; i++) {
    c_M = b_M.cwiseQuotient(prob_NM.transpose() * r_N);
    r_N = a_N.cwiseQuotient(prob_NM * c_M);
  }

  prob_NM = prob_NM.array() * r_N.rowwise().replicate(m+1).array() * c_M.rowwise().replicate(n+1).transpose().array();

  return prob_NM.block(0,0,n,m);
}

MatrixXd tps_rpm_bij_corr_iter_part(const MatrixX3d& x_nd, const MatrixX3d& y_md, const Vector3d& trans_g, int n_iter, const VectorXd& regs, const VectorXd& rads, const Vector3d& rot_reg) {
  int n = x_nd.rows();
  int m = y_md.rows();

  ThinPlateSpline f(x_nd), g(y_md);
  f.trans_g_ = trans_g;
  g.trans_g_ = -trans_g;

  MatrixX3d xwarped_nd, ywarped_md;
  MatrixXd fwddist_nm, invdist_nm;
  MatrixXd prob_nm, corr_nm;
  VectorXd wt_n, wt_m;
  MatrixX3d xtarg_nd, ytarg_md;
  for (int i = 0; i < n_iter; i++) {
    xwarped_nd = f.transform_points(x_nd);
    ywarped_md = g.transform_points(y_md);

    fwddist_nm = cdist(xwarped_nd, y_md);
    invdist_nm = cdist(x_nd, ywarped_md);

    double r = rads[i];
    prob_nm = (-(fwddist_nm + invdist_nm) / (2*r)).array().exp();
    corr_nm = balance_matrix3(prob_nm, 10, .1, 1e-2);
    corr_nm.array() += 1e-9;

    wt_n = corr_nm.rowwise().sum();
    wt_m = corr_nm.colwise().sum();

    MatrixXd tmp = wt_n.rowwise().replicate(m);
    xtarg_nd = corr_nm.cwiseQuotient(wt_n.rowwise().replicate(m)) * y_md;
    ytarg_md = corr_nm.transpose().cwiseQuotient(wt_m.rowwise().replicate(n)) * x_nd;

    f.setTheta(tps_fit3(x_nd, xtarg_nd, regs[i], rot_reg, wt_n));
    g.setTheta(tps_fit3(y_md, ytarg_md, regs[i], rot_reg, wt_m));
  }
  return corr_nm;
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
  lin_ag_ = MatrixXd::Identity(d_,d_);
  trans_g_ = VectorXd::Zero(d_);
  w_ng_ = MatrixXd::Zero(n_,d_);
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


TpsCost::TpsCost(const VarArray& tps_vars, const MatrixX3d& x_na, const MatrixX3d& y_ng, const Vector3d& bend_coefs, const Vector3d& rot_coefs, const MatrixX3d& wt_n, const MatrixXd& N, double alpha) :
    Cost("Tps"), tps_vars_(tps_vars), x_na_(x_na), y_ng_(y_ng), bend_coefs_(bend_coefs), rot_coefs_(rot_coefs), wt_n_(wt_n), N_(N), alpha_(alpha) {
  /**
   * solve equality-constrained qp
   * min tr(x'Hx) + 2 tr(f'x)
   * s.t. Ax = 0
   *
   * Let x = Nz
   * then the problem becomes the unconstrained minimization tr(z'N'HNz) + 2 tr(f'Nz)
   */
  int n = tps_vars.rows();
  int dim = tps_vars.cols();
  assert(dim == 3);
  assert(tps_vars.cols() == dim);
  assert(x_na.rows() == n);
  assert(x_na.cols() == dim);
  assert(N.rows() == n+dim+1);
  assert(N.cols() == n);
  assert(y_ng.rows() == n);
  assert(y_ng.cols() == dim);
  assert(wt_n.rows() == n);
  assert(wt_n.cols() == dim);

  MatrixXd K_nn = tps_kernel_matrix2(x_na_, x_na_);
  MatrixXd Q(n,1+dim+n);
  Q.leftCols(1) = MatrixXd::Ones(n,1);
  Q.middleCols(1,dim) = x_na_;
  Q.rightCols(n) = K_nn;

  QuadExpr exprTrzNHNz;
  exprTrzNHNz.coeffs.reserve(dim * (1+dim+n)*(1+dim+n + 1)/2);
  exprTrzNHNz.vars1.reserve(dim * (1+dim+n)*(1+dim+n + 1)/2);
  exprTrzNHNz.vars2.reserve(dim * (1+dim+n)*(1+dim+n + 1)/2);

  AffExpr expr2TrfNz;
  expr2TrfNz.coeffs.reserve(dim*(1+dim+n));
  expr2TrfNz.vars.reserve(dim*(1+dim+n));

  for (int d = 0; d < dim; d++) {
    VectorXd wt_n_d = wt_n_.col(d);
    MatrixXd WQ = Q.cwiseProduct(wt_n_d.rowwise().replicate(Q.cols()));
    MatrixXd QWQ = Q.transpose() * WQ;
    MatrixXd H = QWQ;
    H.block(dim+1,dim+1,n,n) += bend_coefs_(d) * K_nn;
    H.block(1,1,dim,dim) += rot_coefs_.asDiagonal();

    VectorXd f = -WQ.transpose() * y_ng_.col(d);
    f(1+d) -= rot_coefs_(d);

    MatrixXd NHN = N_.transpose()*H*N_;

    for (int i = 0; i < NHN.rows(); i++) {
      for (int j = i; j < NHN.cols(); j++) {
        if (i == j) {
          exprTrzNHNz.coeffs.push_back(NHN(i,j));
        } else {
          exprTrzNHNz.coeffs.push_back(NHN(i,j)+NHN(j,i));
        }
        exprTrzNHNz.vars1.push_back(tps_vars_(i,d));
        exprTrzNHNz.vars2.push_back(tps_vars_(j,d));
      }
    }
    NHNs.push_back(NHN);

    VectorXd fN = f.transpose() * N_;

    for (int j = 0; j < fN.size(); j++) {
      expr2TrfNz.coeffs.push_back(2.0*fN(j));
      expr2TrfNz.vars.push_back(tps_vars_(j,d));
    }
    fNs.push_back(fN);
  }

  expr_ = exprTrzNHNz;
  exprInc(expr_, expr2TrfNz);
  MatrixXd w_sqrt = wt_n_.array().sqrt();
  double const_term = y_ng_.cwiseProduct(w_sqrt).squaredNorm() + rot_coefs_.sum();
  exprInc(expr_, const_term);
  exprScale(expr_, alpha);
}

double TpsCost::value(const vector<double>& xvec) {
  MatrixXd z = getTraj(xvec, tps_vars_);
  MatrixXd w_sqrt = wt_n_.array().sqrt();

  double obj_value = 0, quad_obj_value = 0, lin_obj_value = 0;
  for (int d = 0; d < NHNs.size(); d++) {
    quad_obj_value += (z.col(d).transpose() * NHNs[d] * z.col(d)).trace();
  }
  for (int d = 0; d < fNs.size(); d++) {
    lin_obj_value += 2.0*(fNs[d].dot(z.col(d)));
  }
  obj_value = alpha_ * (quad_obj_value + lin_obj_value + y_ng_.cwiseProduct(w_sqrt).squaredNorm() + rot_coefs_.sum());

//  MatrixXd theta = N_ * z;
//  ThinPlateSpline f(theta, x_na_);
//  MatrixXd f_x_na = f.transform_points(x_na_);
//  MatrixXd K = tps_kernel_matrix2(f.x_na_, f.x_na_);
//
//  int n = tps_vars_.rows();
//  int dim = tps_vars_.cols();
//  MatrixXd Q(n,1+dim+n);
//  Q.leftCols(1) = MatrixXd::Ones(n,1);
//  Q.middleCols(1,dim) = x_na_;
//  Q.rightCols(n) = K;
//
//  double quad_explicit = (f_x_na - y_ng_).cwiseProduct(w_sqrt).squaredNorm() + 2.0 * (y_ng_.cwiseProduct(wt_n_).transpose() * Q * theta).trace() - y_ng_.cwiseProduct(w_sqrt).squaredNorm()
//      + (bend_coefs_.asDiagonal() * f.w_ng_.transpose()*K*f.w_ng_).trace()
//      + (f.lin_ag_.transpose() * rot_coefs_.asDiagonal() * f.lin_ag_).trace();
//  double lin_explicit = -2.0 * (y_ng_.cwiseProduct(wt_n_).transpose() * Q * theta).trace() - 2.0*(rot_coefs_.asDiagonal() * f.lin_ag_).trace();
//
//  cout << "quad value check " << quad_explicit << " " << quad_obj_value << endl;
//  cout << "lin value check " << lin_explicit << " " << lin_obj_value << endl;
//
//  double obj_explicit = alpha_ * ((f_x_na - y_ng_).cwiseProduct(w_sqrt).squaredNorm()
//      + (bend_coefs_.asDiagonal() * f.w_ng_.transpose()*K*f.w_ng_).trace()
//      + ((f.lin_ag_ - Matrix3d::Identity()).transpose() * rot_coefs_.asDiagonal() * (f.lin_ag_ - Matrix3d::Identity())).trace());
//
//  double obj_expr = expr_.value(xvec);
//  cout << "value check " << obj_explicit << " " << obj_value << " " << obj_expr << endl;

  return obj_value;
}

ConvexObjectivePtr TpsCost::convex(const vector<double>& x, Model* model) {
  ConvexObjectivePtr out(new ConvexObjective(model));
  out->addQuadExpr(expr_);
  return out;
}

void TpsCostPlotter::Plot(const DblVec& x, OR::EnvironmentBase& env, std::vector<OR::GraphHandlePtr>& handles) {
  MatrixXd theta = m_tps_cost->N_ * getTraj(x, m_tps_cost->tps_vars_);

  ThinPlateSpline f(theta, m_tps_cost->x_na_);
  MatrixXd f_x_na = f.transform_points(m_tps_cost->x_na_);
  PlotPointCloud(env, m_tps_cost->x_na_, 5, handles, OR::Vector(1,0,0,1));
  PlotPointCloud(env, f_x_na, 5, handles, OR::Vector(0,1,0,1));
  PlotPointCloud(env, m_tps_cost->y_ng_, 5, handles, OR::Vector(0,0,1,1));

  //TODO put grid drawing in rave_utils by passing a boost function pointer f.transform_points
  Vector3d mins = m_tps_cost->x_na_.colwise().minCoeff();
  Vector3d maxs = m_tps_cost->x_na_.colwise().maxCoeff();
  Vector3d means = 0.5*(maxs+mins);
  mins = means - 2*(maxs-means);
  maxs = means + 2*(maxs-means);
  float xres = 0.1;
  float yres = 0.1;
  float zres = 0.04;

  int nfine = 30;
  vector<float> xcoarse(1,mins(0));
  while (xcoarse[xcoarse.size()-1] < maxs(0)) {
    xcoarse.push_back(xcoarse[xcoarse.size()-1] + xres);
  }
  maxs(0) = xcoarse.back();
  vector<float> ycoarse(1,mins(1));
  while (ycoarse[ycoarse.size()-1] < maxs(1)) {
    ycoarse.push_back(ycoarse[ycoarse.size()-1] + yres);
  }
  maxs(1) = ycoarse.back();
  vector<float> zcoarse(1,mins(2));
  while (zcoarse[zcoarse.size()-1] < maxs(2)) {
     zcoarse.push_back(zcoarse[zcoarse.size()-1] + zres);
  }
  maxs(2) = zcoarse.back();

  VectorXd xfine(nfine), yfine(nfine), zfine(nfine);
  for (int i = 0; i < nfine; i++) {
    xfine(i) = mins(0) + float(i) * (maxs(0)-mins(0))/float(nfine-1);
    yfine(i) = mins(1) + float(i) * (maxs(1)-mins(1))/float(nfine-1);
    zfine(i) = mins(2) + float(i) * (maxs(2)-mins(2))/float(nfine-1);
  }

  vector<MatrixXd> lines;
  BOOST_FOREACH(float& x, xcoarse) {
    BOOST_FOREACH(float& y, ycoarse) {
      MatrixXd xyz(nfine, 3);
      xyz.col(0) = x * VectorXd::Ones(nfine);
      xyz.col(1) = y * VectorXd::Ones(nfine);
      xyz.col(2) = zfine;
      lines.push_back(f.transform_points(xyz));
    }
  }

  BOOST_FOREACH(float& y, ycoarse) {
    BOOST_FOREACH(float& z, zcoarse) {
      MatrixXd xyz(nfine, 3);
      xyz.col(0) = xfine;
      xyz.col(1) = y * VectorXd::Ones(nfine);
      xyz.col(2) = z * VectorXd::Ones(nfine);;
      lines.push_back(f.transform_points(xyz));
    }
  }

  BOOST_FOREACH(float& z, zcoarse) {
    BOOST_FOREACH(float& x, xcoarse) {
      MatrixXd xyz(nfine, 3);
      xyz.col(0) = x * VectorXd::Ones(nfine);
      xyz.col(1) = yfine;
      xyz.col(2) = z * VectorXd::Ones(nfine);
      lines.push_back(f.transform_points(xyz));
    }
  }

  for (int l = 0; l < lines.size(); l++) {
    int numPoints = lines[l].rows();
    float ppoints[3*numPoints];
    for (int i = 0; i < numPoints; i++) {
      for (int j = 0; j < 3; j++) {
        ppoints[3*i + j] = lines[l](i,j);
      }
    }
    handles.push_back(env.drawlinestrip(ppoints, numPoints, 3*sizeof(float)/sizeof(char), 2, OR::Vector(0,1,1,1)));
  }
}

TpsCorrErrCalculator::TpsCorrErrCalculator(const MatrixXd& x_na, const MatrixXd& N, const MatrixXd& y_ng, double alpha) :
  x_na_(x_na),
  N_(N),
  y_ng_(y_ng),
  alpha_(alpha)
{}

VectorXd TpsCorrErrCalculator::operator()(const VectorXd& theta_vals) const {
  int n = x_na_.rows();
  int d = x_na_.cols();
  MatrixXd theta = N_ * Map<const MatrixXd>(theta_vals.data(), n, d);
  ThinPlateSpline f(theta, x_na_);
  MatrixXd f_x_na = f.transform_points(x_na_);
  VectorXd err = (alpha_) * (f_x_na - y_ng_).norm() * VectorXd::Ones(1); // TODO include regularizer
  cout << "err " << err << endl;
  return err;
}

TpsCartPoseErrCalculator::TpsCartPoseErrCalculator(const MatrixXd& x_na, const MatrixXd& N, const OR::Transform& src_pose, ConfigurationPtr manip, OR::KinBody::LinkPtr link) :
  x_na_(x_na),
  N_(N),
  src_pose_(src_pose),
  manip_(manip),
  link_(link),
  n_dof_(manip->GetDOF()),
  n_(x_na.rows()),
  d_(x_na.cols())
{}

VectorXd TpsCartPoseErrCalculator::operator()(const VectorXd& dof_theta_vals) const {
  VectorXd dof_vals = extractDofVals(dof_theta_vals);
  MatrixXd theta = extractThetaVals(dof_theta_vals);

  manip_->SetDOFValues(toDblVec(dof_vals));
  OR::Transform cur_pose = link_->GetTransform();

  ThinPlateSpline f(theta, x_na_);
  OR::Transform warped_src_pose = f.transform_hmats(vector<OR::Transform>(1, src_pose_))[0];

  OR::Transform pose_err = warped_src_pose.inverse() * cur_pose;
  VectorXd err = concat(rotVec(pose_err.rot), toVector3d(pose_err.trans));
  return err;
}

VectorXd TpsCartPoseErrCalculator::extractDofVals(const VectorXd& dof_theta_vals) const {
  return dof_theta_vals.topRows(n_dof_);
}

MatrixXd TpsCartPoseErrCalculator::extractThetaVals(const VectorXd& dof_theta_vals) const {
  VectorXd theta_vals = dof_theta_vals.bottomRows(dof_theta_vals.size() - n_dof_);
  assert(dof_theta_vals.size() - n_dof_ == n_*d_);
  return N_ * Map<const MatrixXd>(theta_vals.data(), n_, d_);
}

void TpsCartPoseErrorPlotter::Plot(const DblVec& x, OR::EnvironmentBase& env, std::vector<OR::GraphHandlePtr>& handles) {
  TpsCartPoseErrCalculator* calc = static_cast<TpsCartPoseErrCalculator*>(m_calc.get());
  VectorXd dof_theta_vals = toVectorXd(getDblVec(x, m_vars));
  VectorXd dof_vals = calc->extractDofVals(dof_theta_vals);
  MatrixXd theta = calc->extractThetaVals(dof_theta_vals);
  calc->manip_->SetDOFValues(toDblVec(dof_vals));
  ThinPlateSpline f(theta, calc->x_na_);
  OR::Transform target = f.transform_hmats(vector<OR::Transform>(1, calc->src_pose_))[0];
  OR::Transform cur = calc->link_->GetTransform();
  PlotAxes(env, cur, .05,  handles);
  PlotAxes(env, target, .05,  handles);
  handles.push_back(env.drawarrow(cur.trans, target.trans, .005, OR::Vector(1,0,1,1)));
  MatrixXd x_na = calc->x_na_;
  MatrixXd f_x_na = f.transform_points(x_na);
  PlotPointCloud(env, x_na, 5, handles, OR::Vector(1,0,0,1));
  PlotPointCloud(env, f_x_na, 5, handles, OR::Vector(0,1,0,1));
}

}
