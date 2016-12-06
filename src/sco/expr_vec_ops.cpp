#include "sco/expr_ops.hpp"
#include "expr_vec_ops.hpp"
namespace sco {

AffExpr varDot(const VectorXd& x, const VarVector& v) {

  AffExpr out;
  out.constant = 0;
  out.vars = v;
  out.coeffs = vector<double>(x.data(), x.data()+x.size());
  return out;
}

AffExpr exprDot(const VectorXd& x, const AffExprVector& v) {
  // probably not the most efficient implementation
  AffExpr out;
  for (int i=0; i < x.size(); ++i) {
    exprInc(out, exprMult(v[i], x[i]));
  }
  return out;
}

}
