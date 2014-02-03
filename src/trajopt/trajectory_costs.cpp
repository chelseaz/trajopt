#include <Eigen/Core>
#include <iostream>
#include <set>
#include <sstream>
#include "sco/expr_ops.hpp"
#include "sco/modeling_utils.hpp"
#include "trajopt/trajectory_costs.hpp"
#include "utils/eigen_conversions.hpp"


using namespace std;
using namespace sco;
using namespace Eigen;

namespace {


static MatrixXd diffAxis0(const MatrixXd& in) {
  return in.middleRows(1, in.rows()-1) - in.middleRows(0, in.rows()-1);
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

AffArray OldTpsCost::multiply(MatrixXd A, VarArray B) {
	AffArray C(A.rows(), B.cols());
	for (int i = 0; i < A.rows(); i++) {
		for (int j = 0; j < B.cols(); j++) {
			AffExpr e;
			for (int k = 0; k < A.cols(); k++) {
				exprInc(e, exprMult(B(k,j), A(i,k)));
			}
			C(i,j) = e;
		}
	}
	return C;
}

AffArray OldTpsCost::multiply(MatrixXd A, AffArray B) {
	AffArray C(A.rows(), B.cols());
	for (int i = 0; i < A.rows(); i++) {
		for (int j = 0; j < B.cols(); j++) {
			AffExpr e;
			for (int k = 0; k < A.cols(); k++) {
				exprInc(e, exprMult(B(k,j), A(i,k)));
			}
			C(i,j) = e;
		}
	}
	return C;
}

OldTpsCost::OldTpsCost(const VarArray& traj_vars, const VarArray& tps_vars, double lambda, double alpha, double beta,
		const MatrixXd& X_s_new, const MatrixXd& X_s, const MatrixXd& K, const MatrixXd& X_g) :
    Cost("OldTps"), traj_vars_(traj_vars), tps_vars_(tps_vars), lambda_(lambda), alpha_(alpha), beta_(beta), X_s_new_(X_s_new), X_s_(X_s), K_(K), X_g_(X_g) {

	int n_steps = traj_vars.rows();
	int n_dof = traj_vars.cols();
	int d = X_s.rows();
	int n = X_s.cols();

	MatrixXd X_s_h(d+1,n);
	X_s_h << X_s, MatrixXd::Ones(1, n);
	JacobiSVD<MatrixXd> svd(X_s_h, ComputeFullV);
	VectorXd singular_values = svd.singularValues();
	MatrixXd V = svd.matrixV();
	int nullity = n - (d+1);
	N_ = V.block(0, V.cols()-nullity, V.rows(), nullity);

  VarArray A_right = tps_vars_.block(0, 0, (n-(d+1))*d, 1);
	A_right.resize(n-(d+1), d);
	AffArray A = multiply(N_, A_right);

  VarArray B = tps_vars_.block((n-(d+1))*d, 0, d*d, 1);
	B.resize(d,d);

	VarArray c = tps_vars_.block(n*d-d, 0, d, 1);

	AffArray KA = multiply(K_, A);
	//AffArray XsB = multiply((MatrixXd) X_s_.transpose(), B);

	cout << "KA.size() " << KA.rows() << " " << KA.cols() << endl;

  MatrixXd M(n, n);
  M.leftCols(n-(d+1)) = K*N_;
  M.middleCols(n-(d+1), d) = X_s.transpose();
  M.rightCols(1) = MatrixXd::Ones(n, 1);

  QuadExpr exprNorm;
  exprNorm.affexpr.constant = X_s_new.array().square().sum();
  MatrixXd CtM = -2 * X_s_new * M;
  CtM = Map<MatrixXd>(CtM.data(), 1, n*d);

  MatrixXd Mblock = MatrixXd::Zero(n, n);
  for (int i = 0; i < n; ++i) {
    Mblock += M.row(i).transpose() * M.row(i);
  }

  exprNorm.affexpr.coeffs.reserve(CtM.cols());
  for (int i = 0; i < CtM.cols(); ++i) {
    exprNorm.affexpr.coeffs.push_back(CtM(0, i));
  }

  // Note: tps_vars_ is [A B c]', row-wise
  vector<Var> tps_vars_colwise;
  tps_vars_colwise.reserve(d*n);
  for (int j = 0; j < d; ++j) {
    for (int i = 0; i < n; ++i) {
      tps_vars_colwise.push_back(tps_vars_(i*d + j, 0));
    }
  }
  exprNorm.affexpr.vars = tps_vars_colwise;

  exprNorm.coeffs.reserve(n*n*d);
  exprNorm.vars1.reserve(n*n*d);
  exprNorm.vars2.reserve(n*n*d);
  for (int dim = 0; dim < d; ++dim) {
    for (int i = 0; i < n; ++i) {
      for (int j = i; j < n; ++j) {
        int ii = dim*n + i;
        int jj = dim*n + j;
        Var var1 = tps_vars_colwise[ii];
        Var var2 = tps_vars_colwise[jj];
        if (i != j) {
          exprNorm.coeffs.push_back(2*Mblock(i, j));
        } else {
          exprNorm.coeffs.push_back(Mblock(i, j));
        }
        exprNorm.vars1.push_back(var1);
        exprNorm.vars2.push_back(var2);
      }
    }
  }
  cout << "Constructing AKA" << endl;

  MatrixXd NKN = N_.transpose() * K_ * N_;
  QuadExpr exprTrace;
  exprTrace.coeffs.reserve(d*NKN.rows()*NKN.cols());
  exprTrace.vars1.reserve(d*NKN.rows()*NKN.cols());
  exprTrace.vars2.reserve(d*NKN.rows()*NKN.cols());

  for (int dim = 0; dim < d; ++dim) {
      for (int i = 0; i < NKN.rows(); ++i) {
        for (int j = i; j < NKN.cols(); ++j) {
          Var var1 = A_right(i, dim);
          Var var2 = A_right(j, dim);
          if (i != j) {
            exprTrace.coeffs.push_back(2*NKN(i, j));
          } else {
            exprTrace.coeffs.push_back(NKN(i, j));
          }
          exprTrace.vars1.push_back(var1);
          exprTrace.vars2.push_back(var2);
        }
      }
  }

  cout << "Postprocessing AKA" << endl;
  expr_ = exprNorm;
  exprScale(exprTrace, lambda);
  exprInc(expr_, exprTrace);
  exprScale(expr_, alpha);

  cout << "Done constructing fast AKA" << endl;

/*
  //alpha*(sum(sum(square(X_s_new' - K*getA(x) - X_s'*getB(x) - ones(n,1)*getc(x).'))) + lambda * trace(getA(x).'*K*getA(x))) + ...
  beta*(sum(sum(square(getTrajPts(x) - warp_pts(getTrajPts(X_g), make_warp(getA(x), getB(x), getc(x), X_s))))));
 */
}

double OldTpsCost::value(const vector<double>& xvec) {
  MatrixXd tps = getTraj(xvec, tps_vars_);
  int d = X_s_.rows();
  int n = X_s_.cols();

  MatrixXd A_right = tps.topRows((n-(d+1))*d);
  A_right.resize(n-(d+1), d);
  MatrixXd A = N_ * A_right;

  MatrixXd B = tps.middleRows((n-(d+1))*d, d*d);
  B.resize(d,d);

  MatrixXd c = tps.bottomRows(d);

  double ret = alpha_*(((MatrixXd) (X_s_new_.transpose() - K_*A - X_s_.transpose()*B - MatrixXd::Ones(n, 1)*c.transpose())).array().square().sum() + lambda_ * (A.transpose() * K_ * A).trace());
	cout << "value check " << ret << " " << expr_.value(xvec) << endl;

	return ret;
}

ConvexObjectivePtr OldTpsCost::convex(const vector<double>& x, Model* model) {
  ConvexObjectivePtr out(new ConvexObjective(model));
  out->addQuadExpr(expr_);
  return out;
}

}
