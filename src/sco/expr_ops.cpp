#include "expr_ops.hpp"
#include <cmath>
#include <iostream>
#include <map>
#include <sstream>

static inline double sq(double x) {return x*x;}

namespace sco {

QuadExpr exprMult(AffExpr a, const Var& b) {
  QuadExpr q;
  q.affexpr.constant = 0;
  q.affexpr.coeffs.push_back(a.constant);
  q.affexpr.vars.push_back(b);

  q.coeffs = a.coeffs;
  q.vars1 = a.vars;
  q.vars2 = std::vector<Var>(a.vars.size(), b);
  return q;
}

QuadExpr exprMult(AffExpr a, AffExpr b) {
  QuadExpr q;
  q.affexpr = exprMult(b, a.constant);
  for (int i = 0; i < a.vars.size(); ++i) {
	  exprInc(q, exprMult(exprMult(b, a.vars[i]), a.coeffs[i]));
  }
  return q;
}

QuadExpr exprSquare(const Var& a) {
  QuadExpr out;
  out.coeffs.push_back(1);
  out.vars1.push_back(a);
  out.vars2.push_back(a);
  return out;
}

QuadExpr exprSquare(const AffExpr& affexpr) {
  QuadExpr out;
  size_t naff = affexpr.coeffs.size();
  size_t nquad = (naff*(naff+1))/2;

  out.affexpr.constant = sq(affexpr.constant);

  out.affexpr.vars = affexpr.vars;
  out.affexpr.coeffs.resize(naff);
  for (size_t i=0; i < naff; ++i) out.affexpr.coeffs[i] = 2*affexpr.constant*affexpr.coeffs[i];

  out.coeffs.reserve(nquad);
  out.vars1.reserve(nquad);
  out.vars2.reserve(nquad);
  for (size_t i=0; i < naff; ++i) {
    out.vars1.push_back(affexpr.vars[i]);
    out.vars2.push_back(affexpr.vars[i]);
    out.coeffs.push_back(sq(affexpr.coeffs[i]));
    for (size_t j=i+1; j < naff; ++j) {
      out.vars1.push_back(affexpr.vars[i]);
      out.vars2.push_back(affexpr.vars[j]);
      out.coeffs.push_back(2 * affexpr.coeffs[i] * affexpr.coeffs[j]);
    }
  }
  return out;
}


AffExpr cleanupAff(const AffExpr& a) {
  AffExpr out;
  for (size_t i=0; i < a.size(); ++i) {
    if (fabs(a.coeffs[i]) > 1e-7) {
      out.coeffs.push_back(a.coeffs[i]);
      out.vars.push_back(a.vars[i]);
    }
  }
  out.constant = a.constant;
  return out;
}

QuadExpr cleanupQuad(const QuadExpr& q) {
  QuadExpr out;
  out.affexpr = cleanupAff(q.affexpr);
  for (size_t i=0; i < q.size(); ++i) {
    if (fabs(q.coeffs[i]) > 1e-8) {
      out.coeffs.push_back(q.coeffs[i]);
      out.vars1.push_back(q.vars1[i]);
      out.vars2.push_back(q.vars2[i]);
    }
  }
  return out;
}

AffExpr combineRepeatedTermsAff(const AffExpr& a) {
  AffExpr out;
  std::map<int, int> varIndexToTermIndex;
  for (size_t i = 0; i < a.size(); ++i) {
    if (varIndexToTermIndex.find(a.vars[i].var_rep->index) == varIndexToTermIndex.end()) {
      varIndexToTermIndex[a.vars[i].var_rep->index] = out.coeffs.size();
      out.vars.push_back(a.vars[i]);
      out.coeffs.push_back(a.coeffs[i]);
    } else {
      out.coeffs[varIndexToTermIndex[a.vars[i].var_rep->index]] += a.coeffs[i];
    }
  }

  out.constant = a.constant;
  return out;
}

QuadExpr combineRepeatedTermsQuad(const QuadExpr& q) {
  QuadExpr out;
  out.affexpr = combineRepeatedTermsAff(q.affexpr);

  /*
  std::map<string, int> varStrToTermIndex;
  for (size_t i = 0; i < q.size(); ++i) {
    Var var1 = q.vars1[i];
    Var var2 = q.vars2[i];
    string varsStr;
    std::stringstream varsSS;
    if (var1.var_rep->index <= var2.var_rep->index) {
      varsSS << var1.var_rep->index << "_" << var2.var_rep->index;
    } else {
      varsSS << var2.var_rep->index << "_" << var1.var_rep->index;
    }
    varsStr = varsSS.str();
    if (varStrToTermIndex.find(varsStr) == varStrToTermIndex.end()) {
      varStrToTermIndex[varsStr] = out.coeffs.size();
      out.vars1.push_back(var1);
      out.vars2.push_back(var2);
      out.coeffs.push_back(q.coeffs[i]);
    } else {
      out.coeffs[varStrToTermIndex[varsStr]] += q.coeffs[i];
    }
  }
  */

  std::map<long, int> indPairToTermIndex;
  for (size_t i = 0; i < q.size(); ++i) {
    int ind1 = q.vars1[i].var_rep->index;
    int ind2 = q.vars2[i].var_rep->index;
    long ind_pair;
    if (ind1 <= ind2) {
      ind_pair = (ind1+ind2)*(ind1+ind2+1)/2 + ind2;
    } else {
      ind_pair = (ind1+ind2)*(ind1+ind2+1)/2 + ind1;
    }
    // map.insert() only inserts if ind_pair is not already in the map
    std::pair<std::map<long,int>::iterator, bool> it_inserted =
        indPairToTermIndex.insert(std::pair<long,int>(ind_pair, out.coeffs.size()));
    if (it_inserted.second) { // ind_pair was not in the map and ind_pair is actually inserted
      out.vars1.push_back(q.vars1[i]);
      out.vars2.push_back(q.vars2[i]);
      out.coeffs.push_back(q.coeffs[i]);
    } else {
      out.coeffs[it_inserted.first->second] += q.coeffs[i];
    }
  }


  return out;
}

///////////////////////////////////////////////////////////////


}
