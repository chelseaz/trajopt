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

AffArray TpsCost::multiply(MatrixXd A, VarArray B) {
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

AffArray TpsCost::multiply(MatrixXd A, AffArray B) {
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

TpsCost::TpsCost(const VarArray& traj_vars, const VarArray& tps_vars, double lambda, double alpha, double beta,
		const MatrixXd& X_s_new, const MatrixXd& X_s, const MatrixXd& K, const MatrixXd& X_g) :
    Cost("Tps"), traj_vars_(traj_vars), tps_vars_(tps_vars), lambda_(lambda), alpha_(alpha), beta_(beta), X_s_new_(X_s_new), X_s_(X_s), K_(K), X_g_(X_g) {

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

  expr_.affexpr.constant = X_s_new.array().square().sum();
  MatrixXd CtM = -2 * X_s_new * M;
  CtM = Map<MatrixXd>(CtM.data(), 1, n*d);

  MatrixXd Mblock = MatrixXd::Zero(n, n);
  for (int i = 0; i < n; ++i) {
    Mblock += M.row(i).transpose() * M.row(i);
  }

  expr_.affexpr.coeffs.reserve(CtM.cols());
  for (int i = 0; i < CtM.cols(); ++i) {
    expr_.affexpr.coeffs.push_back(CtM(0, i));
  }

  // Note: tps_vars_ is [A B c]', row-wise
  vector<Var> tps_vars_colwise;
  tps_vars_colwise.reserve(d*n);
  for (int j = 0; j < d; ++j) {
    for (int i = 0; i < n; ++i) {
      tps_vars_colwise.push_back(tps_vars_(i*d + j, 0));
    }
  }
  expr_.affexpr.vars = tps_vars_colwise;

  expr_.coeffs.reserve(n*n*d);
  expr_.vars1.reserve(n*n*d);
  expr_.vars2.reserve(n*n*d);
  for (int dim = 0; dim < d; ++dim) {
    for (int i = 0; i < n; ++i) {
      for (int j = i; j < n; ++j) {
        int ii = dim*n + i;
        int jj = dim*n + j;
        Var var1 = tps_vars_colwise[ii];
        Var var2 = tps_vars_colwise[jj];
        if (i != j) {
          expr_.coeffs.push_back(2*Mblock(i, j));
        } else {
          expr_.coeffs.push_back(Mblock(i, j));
        }
        expr_.vars1.push_back(var1);
        expr_.vars2.push_back(var2);
      }
    }
  }
  cout << "Constructing AKA" << endl;

	QuadExpr tr_AKA; // lambda * trace(A'*K*A)
	for (int j = 0; j < A.cols(); ++j) {
	    for (int i = 0; i < A.rows(); ++i) {
			exprInc(tr_AKA, exprMult(A(i, j), KA(i, j)));
		}
	}
	cout << "Postprocessing AKA" << endl;
	exprScale(tr_AKA, lambda);
	exprInc(expr_, tr_AKA);
	exprScale(expr_, alpha);
	cout << "Finished tps cost construction" << endl;
/*
  //alpha*(sum(sum(square(X_s_new' - K*getA(x) - X_s'*getB(x) - ones(n,1)*getc(x).'))) + lambda * trace(getA(x).'*K*getA(x))) + ...
  beta*(sum(sum(square(getTrajPts(x) - warp_pts(getTrajPts(X_g), make_warp(getA(x), getB(x), getc(x), X_s))))));
 */
}

double TpsCost::value(const vector<double>& xvec) {
  MatrixXd tps = getTraj(xvec, tps_vars_);
  int d = X_s_.rows();
  int n = X_s_.cols();

  MatrixXd A_right = tps.topRows((n-(d+1))*d);
  A_right.resize(n-(d+1), d);
  MatrixXd A = N_ * A_right;

  MatrixXd B = tps.middleRows((n-(d+1))*d, d*d);
  B.resize(d,d);

  MatrixXd c = tps.bottomRows(d);


  double ret = alpha_*((MatrixXd) (X_s_new_.transpose() - K_*A - X_s_.transpose()*B - MatrixXd::Ones(n, 1)*c.transpose())).array().square().sum() + lambda_ * (A.transpose() * K_ * A).trace();
	cout << "value check " << ret << " " << expr_.value(xvec) << endl;

	return ret;

  //return (diffAxis0(tps).array().square().matrix() * coeffs_.asDiagonal()).sum();
}

ConvexObjectivePtr TpsCost::convex(const vector<double>& x, Model* model) {
  ConvexObjectivePtr out(new ConvexObjective(model));
  out->addQuadExpr(expr_);
  return out;
}

}
