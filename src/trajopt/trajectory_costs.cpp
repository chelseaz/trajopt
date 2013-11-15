#include <Eigen/Core>
#include "sco/expr_ops.hpp"
#include "sco/expr_ops.hpp"
#include "sco/modeling_utils.hpp"
#include "trajopt/trajectory_costs.hpp"


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

AffArray multiply(MatrixXd A, VarArray B) {
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
AffArray multiply(MatrixXd A, AffArray B) {
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
	X_s_h << X_s, VectorXd::Ones(n);
	cout << "X_s_h " << endl << X_s_h << endl;
	JacobiSVD<MatrixXd> svd(X_s_h, ComputeFullV);
	VectorXd singular_values = svd.singularValues();
	MatrixXd V = svd.matrixV();
	int nullity = 0;
	for (int j=0; j<singular_values.size(); j++) {
		if (singular_values(j) == 0) nullity++;
	}
	MatrixXd N = V.block(0, V.cols()-nullity, V.rows(), nullity);
	cout << "N " << endl << N << endl;

	VarArray A_right = tps_vars_.topRows((n-(d+1))*d);
	A_right.resize(n-(d+1), d);
	AffArray A = multiply(N, A_right);

	VarArray B = tps_vars_.middleRows((n-(d+1))*d, d*d);
	B.resize(d,d);

	VarArray c = tps_vars_.bottomRows(d);

/*
 *
  alpha*(sum(sum(square(X_s_new' - K*getA(x) - X_s'*getB(x) - ones(n,1)*getc(x).'))) + lambda * trace(getA(x).'*K*getA(x))) + ...
  beta*(sum(sum(square(getTrajPts(x) - warp_pts(getTrajPts(X_g), make_warp(getA(x), getB(x), getc(x), X_s))))));
 */
}

double TpsCost::value(const vector<double>& xvec) {
  MatrixXd traj = getTraj(xvec, vars_);
  return (diffAxis0(traj).array().square().matrix() * coeffs_.asDiagonal()).sum();
}
ConvexObjectivePtr TpsCost::convex(const vector<double>& x, Model* model) {
  ConvexObjectivePtr out(new ConvexObjective(model));
  out->addQuadExpr(expr_);
  return out;
}

}
