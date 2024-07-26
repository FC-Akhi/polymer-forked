//===- ConvertFromOpenScop.h ------------------------------------*- C++ -*-===//
//
// This file implements the interfaces for converting OpenScop representation to
// MLIR modules.
//
//===----------------------------------------------------------------------===//

#include <fstream>
#include <iostream>
#include <string>



#include "cloog/cloog.h"
#include "cloog/pprint.h"

#include "osl/osl.h"
#include "pluto/internal/pluto.h"
#include "pluto/osl_pluto.h"
#include "pluto/pluto.h"
extern "C" {
#include "pluto/internal/ast_transform.h"
}

#include "polymer/Support/OslScop.h"
#include "polymer/Support/OslScopStmtOpSet.h"
#include "polymer/Support/OslSymbolTable.h"
#include "polymer/Support/ScopStmt.h"
#include "polymer/Support/Utils.h"
#include "polymer/Target/OpenScop.h"

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Translation.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/SourceMgr.h"
#include <llvm/Support/raw_ostream.h>
#include <sstream>


#include "polymer/Support/nlohmann/json.hpp"

using ordered_json = nlohmann::ordered_json;
using json = nlohmann::json;

using namespace polymer;
using namespace mlir;



typedef llvm::StringMap<mlir::Operation *> StmtOpMap;
typedef llvm::StringMap<mlir::Value> NameValueMap;
typedef llvm::StringMap<std::string> IterScatNameMap;



/// My snippet
// Create a JSON global object
ordered_json trace;

int counter = 0;
int i = 1;
std::string boundStr;


/// F: Function to convert mlir::Type to std::string
std::string typeToString(mlir::Type type) {
  
  std::string str;
  llvm::raw_string_ostream os(str);
  
  type.print(os);
  return os.str();

}




/// F: Function to convert mlir::Location to std::string
std::string locationToString(mlir::Location loc) {
  
  std::string locStr;
  llvm::raw_string_ostream locStream(locStr);
  
  loc.print(locStream);
  
  return locStream.str();

}




/// F: Function to print mlir::Value to string
std::string valueToString(mlir::Value value) {
  
  std::string str;
  llvm::raw_string_ostream os(str);
  
  value.print(os);
  
  return os.str();

}




/// F: Function to print mlir::Operation to string
std::string operationToString(mlir::Operation* op) {

  std::string str;
  llvm::raw_string_ostream os(str);

  op->print(os);

  return os.str();

}






/// @brief Snitch to convert clast_expr to string
/// @param expr 
/// @return 
std::string clast_expr_to_string(clast_expr *expr) {
    // Implementation depends on the structure of clast_expr
    // This is a simple placeholder
    if (!expr) return "null";
    // For example purposes, let's just return the address
    return "clast_expr at " + std::to_string(reinterpret_cast<uintptr_t>(expr));
}




// static void pprint_sum(struct cloogoptions *opt,
// 			FILE *dst, struct clast_reduction *r);

// static void pprint_minmax_f(struct cloogoptions *info,
// 			FILE *dst, struct clast_reduction *r);
// static void pprint_minmax_c(struct cloogoptions *info,
// 			FILE *dst, struct clast_reduction *r);

static void pprint_expr(struct cloogoptions *i, FILE *dst, struct clast_expr *e);
static void pprint_name(FILE *dst, struct clast_name *n);
static void pprint_term(struct cloogoptions *i, FILE *dst, struct clast_term *t);
static void pprint_reduction(struct cloogoptions *i, FILE *dst, struct clast_reduction *r);
static void pprint_binary(struct cloogoptions *i, FILE *dst, struct clast_binary *b);




void pprint_expr(struct cloogoptions *i, FILE *dst, struct clast_expr *e)
{
    if (!e)
	return;
    switch (e->type) {
    case clast_expr_name:
	pprint_name(dst, (struct clast_name*) e);
	break;
    case clast_expr_term:
	pprint_term(i, dst, (struct clast_term*) e);
	break;
    case clast_expr_red:
	pprint_reduction(i, dst, (struct clast_reduction*) e);
	break;
    case clast_expr_bin:
	pprint_binary(i, dst, (struct clast_binary*) e);
	break;
    default:
	assert(0);
    }
}


void pprint_sum(struct cloogoptions *opt, FILE *dst, struct clast_reduction *r)
{
    int i;
    struct clast_term *t;

    assert(r->n >= 1);
    assert(r->elts[0]->type == clast_expr_term);
    t = (struct clast_term *) r->elts[0];
    pprint_term(opt, dst, t);

    for (i = 1; i < r->n; ++i) {
	assert(r->elts[i]->type == clast_expr_term);
	t = (struct clast_term *) r->elts[i];
	if (cloog_int_is_pos(t->val))
	    fprintf(dst, "+");
	pprint_term(opt, dst, t);
    }
}


void pprint_minmax_f(struct cloogoptions *info, FILE *dst, struct clast_reduction *r)
{
    int i;
    if (r->n == 0)
	return;
    fprintf(dst, r->type == clast_red_max ? "MAX(" : "MIN(");
    pprint_expr(info, dst, r->elts[0]);
    for (i = 1; i < r->n; ++i) {
	fprintf(dst, ",");
	pprint_expr(info, dst, r->elts[i]);
    }
    fprintf(dst, ")");
}

void pprint_minmax_c(struct cloogoptions *info, FILE *dst, struct clast_reduction *r)
{
    int i;
    for (i = 1; i < r->n; ++i)
	fprintf(dst, r->type == clast_red_max ? "max(" : "min(");
    if (r->n > 0)
	pprint_expr(info, dst, r->elts[0]);
    for (i = 1; i < r->n; ++i) {
	fprintf(dst, ",");
	pprint_expr(info, dst, r->elts[i]);
	fprintf(dst, ")");
    }
}


void pprint_name(FILE *dst, struct clast_name *n)
{
    fprintf(dst, "%s", n->name);
}


/**
 * This function returns a string containing the printing of a value (possibly
 * an iterator or a parameter with its coefficient or a constant).
 * - val is the coefficient or constant value,
 * - name is a string containing the name of the iterator or of the parameter,
 */
void pprint_term(struct cloogoptions *i, FILE *dst, struct clast_term *t)
{
    if (t->var) {
	int group = t->var->type == clast_expr_red &&
		    ((struct clast_reduction*) t->var)->n > 1;
	if (cloog_int_is_one(t->val))
	    ;
	else if (cloog_int_is_neg_one(t->val))
	    fprintf(dst, "-");
        else {
	    cloog_int_print(dst, t->val);
	    fprintf(dst, "*");
	}
	if (group)
	    fprintf(dst, "(");
	pprint_expr(i, dst, t->var);
	if (group)
	    fprintf(dst, ")");
    } else
	cloog_int_print(dst, t->val);
}


void pprint_reduction(struct cloogoptions *i, FILE *dst, struct clast_reduction *r)
{
    switch (r->type) {
      case clast_red_sum:
        pprint_sum(i, dst, r);
        break;
      case clast_red_min:
      case clast_red_max:
        if (r->n == 1) {
            pprint_expr(i, dst, r->elts[0]);
            break;
        }

      if (i->language == CLOOG_LANGUAGE_FORTRAN)
          pprint_minmax_f(i, dst, r);
      else
          pprint_minmax_c(i, dst, r);
      break;
        default:
      assert(0);
    }
}



void pprint_binary(struct cloogoptions *i, FILE *dst, struct clast_binary *b)
{
    const char *s1 = NULL, *s2 = NULL, *s3 = NULL;
    int group = b->LHS->type == clast_expr_red &&
		((struct clast_reduction*) b->LHS)->n > 1;
    if (i->language == CLOOG_LANGUAGE_FORTRAN) {
	switch (b->type) {
	case clast_bin_fdiv:
	    s1 = "FLOOR(REAL(", s2 = ")/REAL(", s3 = "))";
	    break;
	case clast_bin_cdiv:
	    s1 = "CEILING(REAL(", s2 = ")/REAL(", s3 = "))";
	    break;
	case clast_bin_div:
	    if (group)
		s1 = "(", s2 = ")/", s3 = "";
	    else
		s1 = "", s2 = "/", s3 = "";
	    break;
	case clast_bin_mod:
	    s1 = "MOD(", s2 = ", ", s3 = ")";
	    break;
	}
    } else {
	switch (b->type) {
	case clast_bin_fdiv:
	    s1 = "floord(", s2 = ",", s3 = ")";
	    break;
	case clast_bin_cdiv:
	    s1 = "ceild(", s2 = ",", s3 = ")";
	    break;
	case clast_bin_div:
	    if (group)
		s1 = "(", s2 = ")/", s3 = "";
	    else
		s1 = "", s2 = "/", s3 = "";
	    break;
	case clast_bin_mod:
	    if (group)
		s1 = "(", s2 = ")%", s3 = "";
	    else
		s1 = "", s2 = "%", s3 = "";
	    break;
	}
    }
    fprintf(dst, "%s", s1);
    pprint_expr(i, dst, b->LHS);
    fprintf(dst, "%s", s2);
    cloog_int_print(dst, b->RHS);
    fprintf(dst, "%s", s3);
}






/// @brief AffineExprBuilder class 
namespace {

  typedef llvm::StringMap<mlir::Value> SymbolTable;

  /// Build AffineExpr from a clast_expr.
  /// TODO: manage the priviledge.
  class AffineExprBuilder {
  
  public:
    AffineExprBuilder(MLIRContext *context, OslSymbolTable *symTable,
                      SymbolTable *symbolTable, OslScop *scop,
                      CloogOptions *options)
        : b(context), context(context), scop(scop), symTable(symTable),
          symbolTable(symbolTable), options(options) {
      reset();
    }

    LogicalResult process(clast_expr *expr,
                          llvm::SmallVectorImpl<AffineExpr> &affExprs);

    void reset();

    LogicalResult process(clast_name *expr, llvm::SmallVectorImpl<AffineExpr> &affExprs);
    
    LogicalResult process(clast_term *expr, llvm::SmallVectorImpl<AffineExpr> &affExprs);
    
    LogicalResult process(clast_binary *expr, llvm::SmallVectorImpl<AffineExpr> &affExprs);
    
    LogicalResult process(clast_reduction *expr, llvm::SmallVectorImpl<AffineExpr> &affExprs);

    LogicalResult processSumReduction(clast_reduction *expr, llvm::SmallVectorImpl<AffineExpr> &affExprs);
    
    LogicalResult processMinOrMaxReduction(clast_reduction *expr, llvm::SmallVectorImpl<AffineExpr> &affExprs);

    /// OpBuilder used to create AffineExpr.
    OpBuilder b;
    
    /// The MLIR context
    MLIRContext *context;
    
    /// The OslScop of the whole program.
    OslScop *scop;
    
    /// TODO: keep only one of them
    OslSymbolTable *symTable;
    
    SymbolTable *symbolTable;
    
    ///
    CloogOptions *options;

    llvm::StringMap<unsigned> symbolNames;
    llvm::StringMap<unsigned> dimNames;
    llvm::DenseMap<Value, std::string> valueMap;
  
  
  };


} // namespace




void AffineExprBuilder::reset() {

  symbolNames.clear();
  dimNames.clear();

}




/// Get the int64_t representation of a cloog_int_t.
static LogicalResult getI64(cloog_int_t num, int64_t *res) {
  
  // TODO: is there a better way to work around this file-based interface?
  // First, we read the cloog integer into a char buffer.
  char buf[100]; // Should be sufficient for int64_t in string.
  
  FILE *bufFile = fmemopen(reinterpret_cast<void *>(buf), 32, "w");
  
  cloog_int_print(bufFile, num);
  
  fclose(bufFile); // Should close the file or the buf won't be flushed.

  // Then we parse the string as int64_t.
  *res = strtoll(buf, NULL, 10);

  // TODO: error handling.
  return success();

}





LogicalResult AffineExprBuilder::process(clast_expr *expr, llvm::SmallVectorImpl<AffineExpr> &affExprs) {

  switch (expr->type) {

  case clast_expr_name:

    std::printf("[process(clast_expr *expr..)]clast_expr_name: step: %d\n", i);
    std::printf("\n");
    i++;

    if (failed(process(reinterpret_cast<clast_name *>(expr), affExprs)))
      return failure();
    break;
  
  
  case clast_expr_term:


    std::printf("[process(clast_expr *expr..)]clast_expr_term: step: %d\n", i);
    std::printf("\n");
    i++;

    if (failed(process(reinterpret_cast<clast_term *>(expr), affExprs)))
      return failure();
    break;
  
  
  // case clast_expr_bin:
  //   if (failed(process(reinterpret_cast<clast_binary *>(expr), affExprs)))
  //     return failure();
  //   break;
  
  
  case clast_expr_red:

    std::printf("[process(clast_expr *expr..)]clast_expr_red: step: %d\n", i);
    std::printf("\n");
    i++;

    if (failed(process(reinterpret_cast<clast_reduction *>(expr), affExprs)))
      return failure();
    break;
  
  
  }
  
  
  return success();

}




/// Find the name in the scop to determine the type (dim or symbol). The
/// position is decided by the size of dimNames/symbolNames.
/// TODO: handle the dim case.
LogicalResult AffineExprBuilder::process(clast_name *expr, llvm::SmallVectorImpl<AffineExpr> &affExprs) {
  
  /// @F: My snitch=============================================================================================
  char file_contents[1000];

  /// Dump the clast_term clast_expr	expr in a file for expression(eg:32*t1 not 32*t1+31)
  FILE *process_clast_name_output = fopen("output-files/1.polymer-commit-bda08-forOp/9.clast_name.txt", "w+");
  
  pprint_name(process_clast_name_output, expr);
  
  fclose(process_clast_name_output);

  /// Read the expression(eg:t1, t2, variables) from the file
  process_clast_name_output = fopen("output-files/1.polymer-commit-bda08-forOp/9.clast_name.txt", "r");

  /// Read the contents one by one and store in a string variable for expression(eg:t1 or t2 or any variable in loop bound)
  std::string exprStr;
  while (fgets(file_contents, sizeof(file_contents), process_clast_name_output)) {
    exprStr += file_contents;
  }
  fclose(process_clast_name_output);

  
  trace["processStmt(clast_for *forStmt)"][std::to_string(counter)][boundStr]["getAffineLoopBound()"]["processClastLoopBound()"]["process(clast_reduction *expr)"]["process(clast_expr *expr)"]["process(clast_term *expr)"]["process(clast_expr *expr)"]["process(clast_name *expr)"]["expr"] = exprStr;
  trace["processStmt(clast_for *forStmt)"][std::to_string(counter)][boundStr]["getAffineLoopBound()"]["processClastLoopBound()"]["process(clast_reduction *expr)"]["process(clast_expr *expr)"]["process(clast_term *expr)"]["process(clast_expr *expr)"]["process(clast_name *expr)"]["expr->name"] = std::string(expr->name);
  ///=============================================================================================


  /// Check if the Name is a Symbol
  if (scop->isSymbol(expr->name)) {


    /// Start processing the Symbol
    /// This Check if the Symbol is Already in symbolNames map. That means it has been proceesed before.
    /// symbolNames is a StringMap type declared in AffineExprBuilder class
    if (symbolNames.find(expr->name) != symbolNames.end())

      /// push the affine expression of symbol from symbolNames map to small vector affExprs
      affExprs.push_back(b.getAffineSymbolExpr(symbolNames[expr->name]));
    
    else {
      
      
      affExprs.push_back(b.getAffineSymbolExpr(symbolNames.size()));
      size_t numSymbols = symbolNames.size();

      symbolNames[expr->name] = numSymbols;

      Value v = symbolTable->lookup(expr->name);
      valueMap[v] = expr->name;
    }

  } 
  

  /// Here symbolTable is of SymbolTable type which is declared from AffinExprBuilder
  else if (mlir::Value iv = symbolTable->lookup(expr->name)) {
    

    if (dimNames.find(expr->name) != dimNames.end())
  
      affExprs.push_back(b.getAffineDimExpr(dimNames[expr->name]));
  
    else {
  
      affExprs.push_back(b.getAffineDimExpr(dimNames.size()));
      size_t numDims = dimNames.size();
      dimNames[expr->name] = numDims;
      valueMap[iv] = expr->name;


  
    }
  
  } else {
  
    return failure();
  
  }

  return success();

}



LogicalResult AffineExprBuilder::process(clast_term *expr, llvm::SmallVectorImpl<AffineExpr> &affExprs) {
  
  // First get the I64 representation of a cloog int.
  int64_t constant;


  /// This extracts the 64-bit integer value from the clast_term's val field using the getI64 function. 
  /// If this extraction fails, the function returns a failure.
  if (failed(getI64(expr->val, &constant)))
    return failure();


  /// @F: My snitch=============================================================================================
  char file_contents[1000];

  /// Dump the clast_term clast_expr	expr in a file for expression(eg:32*t1 not 32*t1+31)
  FILE *process_clast_term_output = fopen("output-files/1.polymer-commit-bda08-forOp/7.clast_term.txt", "w+");
  
  pprint_term(options, process_clast_term_output, expr);
  
  fclose(process_clast_term_output);

  /// Read the expression(eg:32*t1 not 32*t1+31) from the file
  process_clast_term_output = fopen("output-files/1.polymer-commit-bda08-forOp/7.clast_term.txt", "r");

  /// Read the contents one by one and store in a string variable for expression(eg:32*t1 not 32*t1+31)
  std::string exprStr;
  while (fgets(file_contents, sizeof(file_contents), process_clast_term_output)) {
    exprStr += file_contents;
  }
  fclose(process_clast_term_output);


  std::cout << "[process(clast_term *expr...)]exprStr: " << exprStr << "\n";
  std::cout << "[process(clast_term *expr...)]constant: " << constant << "\n";
  
  trace["processStmt(clast_for *forStmt)"][std::to_string(counter)][boundStr]["getAffineLoopBound()"]["processClastLoopBound()"]["process(clast_reduction *expr)"]["process(clast_expr *expr)"]["process(clast_term *expr)"]["expr"] = exprStr;
  trace["processStmt(clast_for *forStmt)"][std::to_string(counter)][boundStr]["getAffineLoopBound()"]["processClastLoopBound()"]["process(clast_reduction *expr)"]["process(clast_expr *expr)"]["process(clast_term *expr)"]["expr->val"] = constant;
  /// ===============================================================================================================





  /// Next create a constant AffineExpr.
  /// This creates a constant affine expression using the extracted 64-bit integer value. 
  /// The getAffineConstantExpr method of the OpBuilder (b) is used to create this constant expression.
  AffineExpr affExpr = b.getAffineConstantExpr(constant);



  /// This checks if the clast_term has a variable part (var). 
  /// If var is not NULL, it means this term is of the form var * val. We should create the
  /// expr that denotes var and multiplies it with the AffineExpr for val.
  if (expr->var) {

    /// @F: My snitch=============================================================================================
    process_clast_term_output = fopen("output-files/1.polymer-commit-bda08-forOp/7.clast_term.txt", "w+");
  
    pprint_expr(options, process_clast_term_output, expr->var);
  
    fclose(process_clast_term_output);

    /// Read the expression(eg:32*t1 not 32*t1+31) from the file
    process_clast_term_output = fopen("output-files/1.polymer-commit-bda08-forOp/7.clast_term.txt", "r");

    /// Read the contents one by one and store in a string variable for expression(eg:32*t1 not 32*t1+31)
    std::string exprVarStr;
    while (fgets(file_contents, sizeof(file_contents), process_clast_term_output)) {
      exprVarStr += file_contents;
    }
    fclose(process_clast_term_output);

    trace["processStmt(clast_for *forStmt)"][std::to_string(counter)][boundStr]["getAffineLoopBound()"]["processClastLoopBound()"]["process(clast_reduction *expr)"]["process(clast_expr *expr)"]["process(clast_term *expr)"]["expr->var"] = exprVarStr;
    /// =========================================================================================================


    /// A new SmallVector of AffineExpr is created to hold the affine expressions corresponding to the variable part.
    SmallVector<AffineExpr, 1> varAffExprs;


    /// The process function is called recursively to process the variable part of the term. 
    /// If this processing fails, the function returns a failure.
    /// The process function should produce exactly one AffineExpr for the variable part.
    if (failed(process(expr->var, varAffExprs)))

      return failure();


    /// An assertion checks that the process call for the variable part produced exactly one AffineExpr.
    assert(varAffExprs.size() == 1 && "There should be a single expression that stands for the var expr.");


    /// The constant affine expression (affExpr) is then multiplied by the affine expression for the variable part (varAffExprs[0]).
    affExpr = affExpr * varAffExprs[0];
  
  }


  /// The resulting affine expression, which may now represent var * val or just val 
  /// if there was no variable part, is added to the affExprs vector.
  affExprs.push_back(affExpr);
  
  // Print the created AffineExpr
  raw_ostream &os = llvm::outs();
  
  os << "[process(clast_term *expr...)]Resulting Affine Expression at the end of process(clast_term): ";
  affExpr.print(os);
  os << "\n";
  

  return success();

}




LogicalResult AffineExprBuilder::process(clast_binary *expr, llvm::SmallVectorImpl<AffineExpr> &affExprs) {
  
  
  // Handle the LHS expression.
  SmallVector<AffineExpr, 1> lhsAffExprs;
  if (failed(process(expr->LHS, lhsAffExprs)))
    return failure();
  assert(lhsAffExprs.size() == 1 &&
         "There should be a single LHS affine expr.");

  // Handle the RHS expression, which is an integer constant.
  int64_t rhs;
  if (failed(getI64(expr->RHS, &rhs)))
    return failure();
  AffineExpr rhsAffExpr = b.getAffineConstantExpr(rhs);

  AffineExpr affExpr;

  switch (expr->type) {
  case clast_bin_fdiv:
    affExpr = lhsAffExprs[0].floorDiv(rhsAffExpr);
    break;
  case clast_bin_cdiv:
  case clast_bin_div:
    affExpr = lhsAffExprs[0].ceilDiv(rhsAffExpr);
    break;
  case clast_bin_mod:
    affExpr = lhsAffExprs[0] % rhsAffExpr;
    break;
  }

  affExprs.push_back(affExpr);

  return success();

}





LogicalResult AffineExprBuilder::process(clast_reduction *expr, llvm::SmallVectorImpl<AffineExpr> &affExprs) {
  
  /// @F: My snitch=============================================================================================
  char file_contents[1000];

  /// Dump the clast_reduction expr in a file for expression(eg:32*t1 or 32*t1+31)
  FILE *process_clast_reduction_output = fopen("output-files/1.polymer-commit-bda08-forOp/8.clast_reduction.txt", "w");
  
  pprint_reduction(options, process_clast_reduction_output, expr);
  
  fclose(process_clast_reduction_output);

  /// Read the expression(eg:32*t1 or 32*t1+31) from the file
  process_clast_reduction_output = fopen("output-files/1.polymer-commit-bda08-forOp/8.clast_reduction.txt", "r");

  /// Read the contents one by one and store in a string variable for expression(eg:32*t1 or 32*t1+31)
  std::string clast_reduction_exprStr;
  while (fgets(file_contents, sizeof(file_contents), process_clast_reduction_output)) {
    clast_reduction_exprStr += file_contents;
  }
  fclose(process_clast_reduction_output);

  trace["processStmt(clast_for *forStmt)"][std::to_string(counter)][boundStr]["getAffineLoopBound()"]["processClastLoopBound()"]["process(clast_reduction *expr)"]["expr"] = clast_reduction_exprStr;
  
  /// ==================================================================================================================

  std::printf("expr->n: %d\n", expr->n);

  if (expr->n == 1) {

    /// @F: My snitch=============================================================================================
    std::printf("[process(clast_reduction *expr...) inside first if]expr->n == 1\n");
    process_clast_reduction_output = fopen("output-files/1.polymer-commit-bda08-forOp/8.clast_reduction.txt", "w");
    pprint_expr(options, process_clast_reduction_output, expr->elts[0]);
    fclose(process_clast_reduction_output);

    /// Read the contents inside clast_reduction expr->elts[0] from the file
    process_clast_reduction_output = fopen("output-files/1.polymer-commit-bda08-forOp/8.clast_reduction.txt", "r");

    /// Read the contents inside clast_reduction expr->elts[0] one by one character
    std::string clast_reduction_expr_elts;
    while (fgets(file_contents, sizeof(file_contents), process_clast_reduction_output)) {
      clast_reduction_expr_elts += file_contents;
    }
    fclose(process_clast_reduction_output);

    trace["processStmt(clast_for *forStmt)"][std::to_string(counter)][boundStr]["getAffineLoopBound()"]["processClastLoopBound()"]["process(clast_reduction *expr)"]["expr->n"] = std::to_string(expr->n);
    trace["processStmt(clast_for *forStmt)"][std::to_string(counter)][boundStr]["getAffineLoopBound()"]["processClastLoopBound()"]["process(clast_reduction *expr)"]["expr->elts[0]"] = clast_reduction_expr_elts;
    /// ==================================================================================================================


    if (failed(process(expr->elts[0], affExprs))) {

      std::printf("[process(clast_reduction *expr...) inside first if and then nested if]expr->elts[0], affExprs\n");
      return failure();

    } 

    std::printf("[process(clast_reduction *expr...) inside first if and after nested if]expr->elts[0], affExprs\n");
    return success();
  
  }

  switch (expr->type) {
  case clast_red_sum:

    std::printf("[process(clast_reduction *expr...) inside first switch]clast_red_sum\n");

    if (failed(processSumReduction(expr, affExprs)))
      return failure();
    break;
  case clast_red_min:
  case clast_red_max:

    std::printf("[process(clast_reduction *expr...) inside first switch]clast_red_min/max\n");

    if (failed(processMinOrMaxReduction(expr, affExprs)))
      return failure();
    break;
  }

  return success();

}





LogicalResult AffineExprBuilder::processSumReduction(clast_reduction *expr, llvm::SmallVectorImpl<AffineExpr> &affExprs) {

  assert(expr->n >= 1 && "Number of reduction elements should be non-zero.");
  assert(expr->elts[0]->type == clast_expr_term &&
         "The first element should be a term.");

  // Build the reduction expression.
  unsigned numAffExprs = affExprs.size();
  if (failed(process(expr->elts[0], affExprs)))
    return failure();
  assert(numAffExprs + 1 == affExprs.size() &&
         "A single affine expr should be appended after processing an expr in "
         "reduction.");

  for (int i = 1; i < expr->n; ++i) {

    // std::printf("[processSumReduction_for_loop]: I AM HIT=======\n");

    assert(expr->elts[i]->type == clast_expr_term &&
           "Each element in the reduction list should be a term.");

    clast_term *term = reinterpret_cast<clast_term *>(expr->elts[i]);
    SmallVector<AffineExpr, 1> currExprs;
    if (failed(process(term, currExprs)))
      return failure();
    assert(currExprs.size() == 1 &&
           "There should be one affine expr corresponds to a single term.");

    // TODO: deal with negative terms.
    // numAffExprs is the index for the current affExpr, i.e., the newly
    // appended one from processing expr->elts[0].
    affExprs[numAffExprs] = affExprs[numAffExprs] + currExprs[0];
  }

  return success();
}





LogicalResult AffineExprBuilder::processMinOrMaxReduction(clast_reduction *expr, llvm::SmallVectorImpl<AffineExpr> &affExprs) {

  if (failed(process(expr->elts[0], affExprs)))
    return failure();

  for (int i = 1; i < expr->n; i++) {
    if (failed(process(expr->elts[i], affExprs)))
      return failure();
  }

  return success();
}






/// Builds the mapping from the iterator names in a statement to their
/// corresponding names in <scatnames>, based on the matrix provided by the
/// scattering relation.
static void buildIterToScatNameMap(IterScatNameMap &iterToScatName,
                                   osl_statement_p stmt,
                                   osl_generic_p scatnames) {
  // Get the body from the statement.
  osl_body_p body = osl_statement_get_body(stmt);
  assert(body != nullptr && "The body of the statement should not be NULL.");
  assert(body->expression != nullptr &&
         "The body expression should not be NULL.");
  assert(body->iterators != nullptr &&
         "The body iterators should not be NULL.");

  // Get iterator names.
  unsigned numIterNames = osl_strings_size(body->iterators);
  llvm::SmallVector<std::string, 8> iterNames(numIterNames);
  for (unsigned i = 0; i < numIterNames; i++)
    iterNames[i] = body->iterators->string[i];

  // Split the scatnames into a list of strings.
  osl_strings_p scatNamesData =
      reinterpret_cast<osl_scatnames_p>(scatnames->data)->names;
  unsigned numScatNames = osl_strings_size(scatNamesData);

  llvm::SmallVector<std::string, 8> scatNames(numScatNames);
  for (unsigned i = 0; i < numScatNames; i++)
    scatNames[i] = scatNamesData->string[i];

  // Get the scattering relation.
  osl_relation_p scats = stmt->scattering;
  assert(scats != nullptr && "scattering in the statement should not be NULL.");
  assert(scats->nb_input_dims == static_cast<int>(iterNames.size()) &&
         "# input dims should equal to # iter names.");
  assert(scats->nb_output_dims <= static_cast<int>(scatNames.size()) &&
         "# output dims should be less than or equal to # scat names.");

  // Build the mapping.
  for (int i = 0; i < scats->nb_output_dims; i++)
    for (int j = 0; j < scats->nb_input_dims; j++)
      if (scats->m[i][j + scats->nb_output_dims + 1].dp)
        iterToScatName[iterNames[j]] = scatNames[i];
}










/// @brief IterScatNameMapper class
namespace {

  /// Build mapping between the iter names in the original code to the scatname in
  /// the OpenScop.
  class IterScatNameMapper {
    public:

      IterScatNameMapper(OslScop *scop) : scop(scop) {}

      void visitStmtList(clast_stmt *s);

      IterScatNameMap getIterScatNameMap() { return iterScatNameMap; };

    private:

      void visit(clast_for *forStmt);
      void visit(clast_user_stmt *userStmt);

      OslScop *scop;

      IterScatNameMap iterScatNameMap;
      
  };

} // namespace




void IterScatNameMapper::visitStmtList(clast_stmt *s) {
  for (; s; s = s->next) {
    if (CLAST_STMT_IS_A(s, stmt_user)) {
      visit(reinterpret_cast<clast_user_stmt *>(s));
    } else if (CLAST_STMT_IS_A(s, stmt_for)) {
      visit(reinterpret_cast<clast_for *>(s));
    } 
    
  }
}





void IterScatNameMapper::visit(clast_for *forStmt) {
  visitStmtList(forStmt->body);
}




void IterScatNameMapper::visit(clast_user_stmt *userStmt) {
  osl_statement_p stmt;
  if (failed(scop->getStatement(userStmt->statement->number - 1, &stmt)))
    return assert(false);

  osl_body_p body = osl_statement_get_body(stmt);
  assert(body != NULL && "The body of the statement should not be NULL.");
  assert(body->expression != NULL && "The body expression should not be NULL.");
  assert(body->iterators != NULL && "The body iterators should not be NULL.");

  // Map iterator names in the current statement to the values in <scatnames>.
  osl_generic_p scatnames = scop->getExtension("scatnames");
  assert(scatnames && "There should be a <scatnames> in the scop.");
  buildIterToScatNameMap(iterScatNameMap, stmt, scatnames);
}











/// @brief: Below this point all things belongs to Importer  
/// @brief Importer class
namespace {

  /// Import MLIR code from the clast AST.
  class Importer {

    public:
    
      Importer(MLIRContext *context, ModuleOp module, OslSymbolTable *symTable, OslScop *scop, CloogOptions *options);

      LogicalResult processStmtList(clast_stmt *s);

      mlir::Operation *getFunc() { 
        
        return func; 
      
      }

    private:
      
      /// Number of internal functions created.
      int64_t numInternalFunctions = 0;

      /// The current builder, pointing at where the next Instruction should be generated.
      OpBuilder b;

      /// The current context.
      MLIRContext *context;
      
      /// The current module being created.
      ModuleOp module;
      
      /// The main function.
      FuncOp func;
      
      /// The OpenScop object pointer.
      OslScop *scop;
      
      /// The symbol table for labels in the OpenScop input (to be deprecated).
      OslSymbolTable *symTable;
      
      /// The symbol table that will be built on the fly.
      SymbolTable symbolTable;

      /// Map from symbol names to block arguments.
      llvm::DenseMap<llvm::StringRef, BlockArgument> symNameToArg;
      
      /// Map from callee names to callee operation.
      llvm::StringMap<Operation *> calleeMap;

      // Map from an not yet initialized symbol to the Values that depend on it.
      llvm::StringMap<llvm::SetVector<mlir::Value>> symbolToDeps;
      
      // Map from a value to all the symbols it depends on.
      llvm::DenseMap<mlir::Value, llvm::SetVector<llvm::StringRef>> valueToDepSymbols;

      IterScatNameMap iterScatNameMap;

      llvm::StringMap<clast_stmt *> lhsToAss;

      CloogOptions *options;


      void initializeSymbolTable();
      void initializeFuncOpInterface();
      void initializeSymbol(mlir::Value val);

      LogicalResult processStmt(clast_root *rootStmt);
      LogicalResult processStmt(clast_for *forStmt);

      LogicalResult processStmt(clast_user_stmt *userStmt);
      LogicalResult processStmt(clast_assignment *ass);

      std::string getSourceFuncName() const;
      mlir::FuncOp getSourceFuncOp();

      LogicalResult getAffineLoopBound(clast_expr *expr, llvm::SmallVectorImpl<mlir::Value> &operands, AffineMap &affMap, bool isUpper = false);
      
      
      void getAffineExprForLoopIterator(clast_stmt *subst, llvm::SmallVectorImpl<mlir::Value> &operands, AffineMap &affMap);


      void getInductionVars(clast_user_stmt *userStmt, osl_body_p body, SmallVectorImpl<mlir::Value> &inductionVars);

      LogicalResult parseUserStmtBody(llvm::StringRef body, std::string &calleeName, llvm::SmallVectorImpl<std::string> &args);

      bool isMemrefArg(llvm::StringRef argName);

      /// Functions are always inserted before the module terminator.
      Block::iterator getFuncInsertPt() {

        return std::prev(module.getBody()->end());
      
      }

      /// A helper to create a callee.
      void createCalleeAndCallerArgs(llvm::StringRef calleeName,
                                    llvm::ArrayRef<std::string> args,
                                    mlir::FuncOp &callee,
                                    SmallVectorImpl<mlir::Value> &callerArgs);

      
  };

} // namespace





Importer::Importer(MLIRContext *context, ModuleOp module, OslSymbolTable *symTable, OslScop *scop, CloogOptions *options)
                  : b(context), context(context), module(module), scop(scop), symTable(symTable), options(options) {

  b.setInsertionPointToStart(module.getBody());

}








static mlir::Value findBlockArg(mlir::Value v) {

  mlir::Value r = v;

  while (r != nullptr) {

    if (r.isa<BlockArgument>())

      break;

    mlir::Operation *defOp = r.getDefiningOp();

    if (!defOp || defOp->getNumOperands() != 1)

      return nullptr;

    if (!isa<mlir::arith::IndexCastOp>(defOp))

      return nullptr;

    r = defOp->getOperand(0);

  }

  return r;

}





static std::unique_ptr<OslScop> readOpenScop(llvm::MemoryBufferRef buf) {
  
  // Read OpenScop by OSL API.
  // TODO: is there a better way to get the FILE pointer from
  // MemoryBufferRef?
  FILE *inputFile = fmemopen(reinterpret_cast<void *>(const_cast<char *>(buf.getBufferStart())), buf.getBufferSize(), "r");

  auto scop = std::make_unique<OslScop>(osl_scop_read(inputFile));
  
  fclose(inputFile);

  return scop;

}






bool Importer::isMemrefArg(llvm::StringRef argName) {

  // TODO: should find a better way to do this, e.g., using the old symbol table.
  return argName.size() >= 2 && argName[0] == 'A';

}





LogicalResult Importer::parseUserStmtBody(llvm::StringRef body, std::string &calleeName, llvm::SmallVectorImpl<std::string> &args) {

  unsigned bodyLen = body.size();
  unsigned pos = 0;

  // Read until the left bracket for the function name.
  for (; pos < bodyLen && body[pos] != '('; pos++)
  
    calleeName.push_back(body[pos]);
  
  pos++; // Consume the left bracket.

  // Read argument names.
  while (pos < bodyLen) {
  
    std::string arg;
  
    for (; pos < bodyLen && body[pos] != ',' && body[pos] != ')'; pos++) {
  
      if (body[pos] != ' ') // Ignore whitespaces
  
        arg.push_back(body[pos]);
    }

    if (!arg.empty())
  
      args.push_back(arg);
  
    // Consume either ',' or ')'.
    pos++;
  
  }

  return success();

}



void Importer::createCalleeAndCallerArgs(llvm::StringRef calleeName, llvm::ArrayRef<std::string> args, mlir::FuncOp &callee, SmallVectorImpl<mlir::Value> &callerArgs) {
  
  // TODO: avoid duplicated callee creation
  // Cache the current insertion point before changing it for the new callee
  // function.
  auto currBlock = b.getBlock();
  
  auto currPt = b.getInsertionPoint();

  // Create the callee.
  // First, we create the callee function type.
  unsigned numArgs = args.size();

  llvm::SmallVector<mlir::Type, 8> calleeArgTypes;

  for (unsigned i = 0; i < numArgs; i++) {
  
    if (isMemrefArg(args[i])) {
  
      // Memref. A memref name and its number of dimensions.
      auto memName = args[i];
      auto memShape = std::vector<int64_t>(std::stoi(args[i + 1]), -1);
      MemRefType memType = MemRefType::get(memShape, b.getF32Type());
      calleeArgTypes.push_back(memType);
      i++;
    
    } else {
    
      // Loop IV.
      calleeArgTypes.push_back(b.getIndexType());
    
    }
  
  }

  auto calleeType = b.getFunctionType(calleeArgTypes, llvm::None);
  
  // TODO: should we set insertion point for the callee before the main
  // function?
  b.setInsertionPoint(module.getBody(), getFuncInsertPt());
  
  callee = b.create<FuncOp>(UnknownLoc::get(context), calleeName, calleeType);
  
  calleeMap[calleeName] = callee;

  // Create the caller.
  b.setInsertionPoint(currBlock, currPt);

  // Initialise all the caller arguments. The first argument should be the
  // memory object, which is set to be a BlockArgument.
  auto &entryBlock = *func.getBlocks().begin();

  for (unsigned i = 0; i < numArgs; i++) {
  
    if (isMemrefArg(args[i])) {
  
      // TODO: refactorize this.
      auto memShape = std::vector<int64_t>(std::stoi(args[i + 1]), -1);
  
      MemRefType memType = MemRefType::get(memShape, b.getF32Type());

      // TODO: refactorize these two lines into a single API.
      Value memref = symTable->getValue(args[i]);
  
      if (!memref) {
  
        memref = entryBlock.addArgument(memType, b.getUnknownLoc());
        symTable->setValue(args[i], memref, OslSymbolTable::Memref);
  
      }
  
      callerArgs.push_back(memref);
  
      i++;
  
    } else if (auto val = symTable->getValue(args[i])) {
  
      // The rest of the arguments are access indices. They could be the loop
      // IVs or the parameters. Loop IV
      callerArgs.push_back(val);
  
      // Symbol.
      // TODO: manage sym name by the symTable.
  
    } else if (symNameToArg.find(args[i]) != symNameToArg.end()) {
    
      callerArgs.push_back(symNameToArg.lookup(args[i]));
      // TODO: what if an index is a constant?
    
    } else if (iterScatNameMap.find(args[i]) != iterScatNameMap.end()) {
    
      auto newArgName = iterScatNameMap[args[i]];
    
      if (auto iv = symTable->getValue(newArgName)) {
    
        callerArgs.push_back(iv);
    
        // We should set the symbol table for args[i], otherwise we cannot
        // build a correct mapping from the original symbol table (only
        // args[i] exists in it).
        symTable->setValue(args[i], iv, OslSymbolTable::LoopIV);
    
      } else {
    
        llvm::errs() << "Cannot find the scatname " << newArgName
                     << " as a valid loop IV.\n";
        return;
    
      }
    
    } else { // TODO: error handling
    
      llvm::errs() << "Cannot find " << args[i]
                   << " as a loop IV name or a symbole name. Please check if "
                      "the statement body uses the same iterator name as the "
                      "one in <scatnames>.\n";
      return;
    
    }
  
  }

}




void Importer::getAffineExprForLoopIterator( clast_stmt *subst, llvm::SmallVectorImpl<mlir::Value> &operands, AffineMap &affMap) {

  assert(CLAST_STMT_IS_A(subst, stmt_ass) && "Should use clast assignment here.");

  clast_assignment *substAss = reinterpret_cast<clast_assignment *>(subst);

  AffineExprBuilder builder(context, symTable, &symbolTable, scop, options);
  
  SmallVector<AffineExpr, 1> affExprs;
  
  assert(succeeded(builder.process(substAss->RHS, affExprs)));

  // Insert dim operands.
  for (llvm::StringRef dimName : builder.dimNames.keys()) {
  
    mlir::Value iv = symbolTable[dimName];
  
    assert(iv != nullptr);
  
    operands.push_back(iv);
  
  }
  
  // Symbol operands
  for (llvm::StringRef symName : builder.symbolNames.keys()) {
  
    mlir::Value operand = symbolTable[symName];
  
    assert(operand != nullptr);
  
    operands.push_back(operand);
  
  }

  // Create the AffineMap for loop bound.
  affMap = AffineMap::get(builder.dimNames.size(), builder.symbolNames.size(), affExprs, context);


}




void Importer::getInductionVars(clast_user_stmt *userStmt, osl_body_p body, SmallVectorImpl<mlir::Value> &inductionVars) {

  char *expr = osl_util_identifier_substitution(body->expression->string[0], body->iterators->string);

  // dbgs() << "Getting induction vars from: " << (*body->expression->string[0])
  //        << '\n' << (*expr) << '\n';
  char *tmp = expr;
  
  clast_stmt *subst;

  /* Print the body expression, substituting the @...@ markers. */
  while (*expr) {
  
    if (*expr == '@') {
  
      int iterator;
      expr += sscanf(expr, "@%d", &iterator) + 2; /* 2 for the @s */
      subst = userStmt->substitutions;
  
      for (int i = 0; i < iterator; i++)
        subst = subst->next;

      SmallVector<mlir::Value, 8> substOperands;
  
      AffineMap substMap;
  
      getAffineExprForLoopIterator(subst, substOperands, substMap);

      mlir::Operation *op;
  
      if (substMap.isSingleConstant())
  
        op = b.create<mlir::arith::ConstantOp>(b.getUnknownLoc(), b.getIndexType(),
                                              b.getIntegerAttr(b.getIndexType(),
                                              substMap.getSingleConstantResult()));
      else
        
        op = b.create<mlir::AffineApplyOp>(b.getUnknownLoc(), substMap, substOperands);

      inductionVars.push_back(op->getResult(0));

    } else {

      expr++;
    
    }
  
  }

  free(tmp);

}









LogicalResult Importer::processStmt(clast_assignment *ass) {

  printf("inside assignement\n");

  SmallVector<mlir::Value, 8> substOperands;

  AffineMap substMap;

  getAffineExprForLoopIterator((clast_stmt *)ass, substOperands, substMap);

  mlir::Operation *op;


  if (substMap.isSingleConstant()) {

    op = b.create<mlir::arith::ConstantOp>(b.getUnknownLoc(), b.getIndexType(), b.getIntegerAttr(b.getIndexType(), substMap.getSingleConstantResult()));

  } 
  
  else if (substMap.getNumResults() == 1) {

    op = b.create<mlir::AffineApplyOp>(b.getUnknownLoc(), substMap,
                                       substOperands);
  } 
  
  else {

    assert(ass->RHS->type == clast_expr_red);

    clast_reduction *red = reinterpret_cast<clast_reduction *>(ass->RHS);

    assert(red->type != clast_red_sum);

    if (red->type == clast_red_max)

      op = b.create<mlir::AffineMaxOp>(b.getUnknownLoc(), substMap, substOperands);

    else
      
      op = b.create<mlir::AffineMinOp>(b.getUnknownLoc(), substMap, substOperands);
  
  }

  assert(op->getNumResults() == 1);
  
  symbolTable[ass->LHS] = op->getResult(0);
  
  lhsToAss[ass->LHS] = (clast_stmt *)ass;
  
  return success();

}






/// Create a custom call operation for each user statement. A user statement
/// should be in the format of <stmt-id>`(`<ssa-id>`)`, in which a SSA ID can be
/// a memref, a loop IV, or a symbol parameter (defined as a block argument). We
/// will also generate the declaration of the function to be called, which has
/// an empty body, in order to make the compiler happy.
LogicalResult Importer::processStmt(clast_user_stmt *userStmt) {
  
  printf("inside userstmt\n");

  OslScop::ScopStmtMap *scopStmtMap = scop->getScopStmtMap();
  OslScop::ValueTable *valueTable = scop->getValueTable();

  osl_statement_p stmt;
  
  assert(succeeded(scop->getStatement(userStmt->statement->number - 1, &stmt)));

  osl_body_p body = osl_statement_get_body(stmt);
  
  assert(body != NULL && "The body of the statement should not be NULL.");
  assert(body->expression != NULL && "The body expression should not be NULL.");
  assert(body->iterators != NULL && "The body iterators should not be NULL.");

  // Map iterator names in the current statement to the values in <scatnames>.
  osl_generic_p scatnames = scop->getExtension("scatnames");
  
  assert(scatnames && "There should be a <scatnames> in the scop.");

  SmallVector<mlir::Value, 8> inductionVars;
  
  getInductionVars(userStmt, body, inductionVars);

  // Parse the statement body.
  llvm::SmallVector<std::string, 8> args;
  std::string calleeName;
  
  if (failed(parseUserStmtBody(body->expression->string[0], calleeName, args)))
  
    return failure();

  // Create the callee and the caller args.
  FuncOp callee;
  
  llvm::SmallVector<mlir::Value, 8> callerArgs;

  Location loc = b.getUnknownLoc();

  // If the calleeName can be found in the scopStmtMap, i.e., we have the
  // definition of the callee already, we will generate the caller based on that
  // interface.
  if (scopStmtMap->find(calleeName) != scopStmtMap->end()) {
    const auto &it = scopStmtMap->find(calleeName);
    callee = it->second.getCallee();

    assert(callee.getName() == calleeName && "Callee names should match.");
    // Note that caller is in the original function.
    mlir::func::CallOp origCaller = it->second.getCaller();
    loc = origCaller.getLoc();
    unsigned currInductionVar = 0;

    for (mlir::Value arg : origCaller.getOperands()) {
      std::string argSymbol = valueTable->lookup(arg);
      if (argSymbol.empty()) {
        mlir::Value blockArg = findBlockArg(arg);
        argSymbol = valueTable->lookup(blockArg);
        assert(!argSymbol.empty());
      }

      // Type casting
      if (mlir::Value val = this->symbolTable.lookup(argSymbol)) {
        if (arg.getType() != val.getType()) {
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPointAfterValue(val);
          mlir::Operation *castOp = b.create<mlir::arith::IndexCastOp>(
              b.getUnknownLoc(), arg.getType(), val);
          callerArgs.push_back(castOp->getResult(0));
        } else {
          callerArgs.push_back(val);
        }
        continue;
      }

      // Special handling for the memory allocation case.
      mlir::Operation *defOp = arg.getDefiningOp();
      if (defOp && isa<memref::AllocaOp>(defOp)) {
        // If this memory has been allocated, need to check its owner.
        if (mlir::Value val = this->symbolTable.lookup(argSymbol)) {
          DominanceInfo dom(func);
          if (dom.dominates(val.getParentBlock(), b.getBlock())) {
            callerArgs.push_back(val);
            continue;
          }
        }

        // Otherwise, create a new alloca op.
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(b.getBlock());
        mlir::Operation *newDefOp = b.clone(*defOp);

        this->symbolTable[argSymbol] = newDefOp->getResult(0);

        callerArgs.push_back(newDefOp->getResult(0));
      } else if (scop->isDimSymbol(argSymbol)) {
        // dbgs() << "currInductionVar: " << currInductionVar << '\n';
        // dbgs() << "inductionVars: \n";
        // interleave(inductionVars, dbgs(), "\n");
        // dbgs() << '\n';
        callerArgs.push_back(inductionVars[currInductionVar++]);
      } else if (mlir::Value val = this->symbolTable.lookup(argSymbol)) {
        callerArgs.push_back(val);
      } else {
        std::string scatName = iterScatNameMap.lookup(argSymbol);
        // Dealing with loop IV.
        if (!scatName.empty()) {
          argSymbol = scatName;
          if (mlir::Value val = this->symbolTable.lookup(argSymbol))
            callerArgs.push_back(val);
          else // no IV symbol means this statement is called errside the loop.
            callerArgs.push_back(this->symbolTable.lookup("zero"));
        } else {
          llvm::errs() << "Missing symbol: " << arg << "\n";
          assert(false && "Cannot insert the correct number of caller args. "
                          "Some symbols must be missing in the symbol table.");
        }
      }
    }
  } else {
    createCalleeAndCallerArgs(calleeName, args, callee, callerArgs);
  }

  // Finally create the CallOp.
  b.create<mlir::func::CallOp>(loc, callee, callerArgs);

  return success();

}




/// We treat the provided the clast_expr as a loop bound. If it is a min/max
/// reduction, we will expand that into multiple expressions.
static LogicalResult processClastLoopBound(clast_expr *expr,
                                           AffineExprBuilder &builder,
                                           SmallVectorImpl<AffineExpr> &exprs,
                                           CloogOptions *options) {


  SmallVector<clast_expr *, 1> expandedExprs;

  if (expr->type == clast_expr_red) {

    clast_reduction *red = reinterpret_cast<clast_reduction *>(expr);

    if (red->type == clast_red_max || red->type == clast_red_min) {

      for (int i = 0; i < red->n; i++) {

        expandedExprs.push_back(red->elts[i]);

      }
    
    }
  
  }

  
  if (expandedExprs.empty()) {// no expansion, just put the original input in.
  
    expandedExprs.push_back(expr);

  }


  
  /// Iterate over items in expandedExprs(it is expanded because of min and max reductions, If there is no min/max, then there is single item(lB/uB))
  for (clast_expr *e : expandedExprs) {


    /// Passing individual lB or uB and initially black exprs SV. This SV will get filled up by AffineExprs inside process()
    std::printf("=====================================================================\n");
    if (failed(builder.process(e, exprs)))
  
      return failure();

  }

  return success();
}




LogicalResult Importer::getAffineLoopBound(clast_expr *expr,
                                           llvm::SmallVectorImpl<mlir::Value> &operands,
                                           AffineMap &affMap, bool isUpper) {



  /// F: My snitch
  /// Prepare to store data for this loop statement
  ordered_json loopBoundInfos;
  char file_contents[1000];

  /// Dump the clast expr in a file for bound
  FILE *clast_for_output = fopen("output-files/1.polymer-commit-bda08-forOp/5.clast_for.txt", "w+");
  
  clast_pprint_expr(options, clast_for_output, expr);
  
  fclose(clast_for_output);

  /// Read the bound from the file
  clast_for_output = fopen("output-files/1.polymer-commit-bda08-forOp/5.clast_for.txt", "r");

  /// Read the contents one by one and store in a string variable for bound
  /// As boundStr is globally declared to hold boundary values of loop for JSON, we need to clear previous data
  boundStr.clear();

  while (fgets(file_contents, sizeof(file_contents), clast_for_output)) {
    boundStr += file_contents;
  }
  fclose(clast_for_output);
  
  /// Loop bound
  boundStr = "Loop Bound "+ boundStr;

  trace["processStmt(clast_for *forStmt)"][std::to_string(counter)][boundStr]["getAffineLoopBound()"]["Some Inputs"] = "forStmt->LB, lbOperands, lbMap or forStmt->UB, ubOperands, ubMap";
  

  /// An AffineExprBuilder instance is created to help build affine expressions.
  AffineExprBuilder builder(context, symTable, &symbolTable, scop, options);


  /// A vector boundExprs is initialized to store the resulting affine expressions.
  SmallVector<AffineExpr, 4> boundExprs;
  
  /// The processClastLoopBound function is called to process the clast expression and convert it into one or more affine expressions.
  /// Here expr is lower or upper bound
  /// boundExprs is empty small vector
  if (failed(processClastLoopBound(expr, builder, boundExprs, options)))
  
    return failure();



  /// If looking at the upper bound, we should add 1 to all of them.
  if (isUpper)
  
    for (auto &expr : boundExprs)
      /// On the fly it adds one with upper bound after converting '1' to affine expression
      /// And its storing the updated upper bound to boundExprs
      expr = expr + b.getAffineConstantExpr(1);



  // Print the created AffineExpr type SV
  // Use a string stream to capture the output from raw_ostream
  std::string stringStream;
  llvm::raw_string_ostream os1(stringStream);
  
  os1 << "Affine Expression after processClastLoopBound(): ";  
  
  for (AffineExpr e : boundExprs) {
    e.print(os1);
    // os1 << "\n";
  }


  // Flush the contents to the string stream
  os1.flush();


  /// F: Dump the boundexprs after processClastLoopBound
  trace["processStmt(clast_for *forStmt)"][std::to_string(counter)][boundStr]["getAffineLoopBound()"]["after processClastLoopBound()"]["boundExprs"] = stringStream;


  // Insert dim operands.
  /// These are the names of the loop induction variables (i.e., the dimension variables) in the affine loops.
  unsigned numDims = builder.dimNames.size(); 

  /// symbolNames: These are symbols that are typically constants or parameters that are not directly related 
  /// to the loop induction variables but might be used in the bounds of the loops.
  unsigned numSymbols = builder.symbolNames.size();


  /// @F: My Snitch
  for (const auto &entry : builder.dimNames) {
    trace["processStmt(clast_for *forStmt)"][std::to_string(counter)][boundStr]["getAffineLoopBound()"]["builder.dimNames"][entry.first().str()] = entry.second;
  }

  for (const auto &entry : builder.symbolNames) {
    trace["processStmt(clast_for *forStmt)"][std::to_string(counter)][boundStr]["getAffineLoopBound()"]["builder.symbolNames"][entry.first().str()] = entry.second;
  }

  trace["processStmt(clast_for *forStmt)"][std::to_string(counter)][boundStr]["getAffineLoopBound()"]["builder.dimNames.size"] = numDims;
  trace["processStmt(clast_for *forStmt)"][std::to_string(counter)][boundStr]["getAffineLoopBound()"]["builder.symbolNames.size"] = numSymbols;




  /// 
  operands.resize(numDims + numSymbols);

  for (const auto &it : builder.dimNames) {
  
    if (auto iv = symbolTable[it.first()]) {
  
      operands[it.second] = iv;
  
    } else {
  
      llvm::errs() << "Dim " << it.first()
                   << " cannot be recognized as a value.\n";
  
      return failure();
  
    }
  
  }



  // Create or get BlockArgument for the symbols. We assume all symbols come
  // from the BlockArgument of the generated function.
  for (const auto &it : builder.symbolNames) {
  
    mlir::Value operand = symbolTable[it.first()];
  
    assert(operand != nullptr);
  
    operands[it.second + numDims] = operand;
  
  }

  // Create the AffineMap for loop bound.
  /// Example: affine.for %arg3 = affine_map<(d0) -> (d0 * 32)>(%arg0) to affine_map<(d0) -> (d0 * 32 + 32)>(%arg0) {
  /// This below line is responsible for affine_map ir gen. It is crucial for creating affine maps that define loop bounds in MLIR.
  affMap = AffineMap::get(numDims, numSymbols, boundExprs, context);

  return success();


}






/// Generate the AffineForOp from a clast_for statement. First we create
/// AffineMaps for the lower and upper bounds. Then we decide the step if
/// there is any. And finally, we create the AffineForOp instance and generate
/// its body.
/// @brief : translates CLooG AST (i.e. clast_for) representations of loops into MLIR AffineForOp operations. The function Importer::processStmt is 
/// responsible for processing a CLooG for-loop statement (i.e. clast_for) and converting it into an MLIR affine for-loop.
/// @param forStmt : clast_for *forStmt: A pointer to a CLooG AST for-loop statement
/// @return : 
LogicalResult Importer::processStmt(clast_for *forStmt) {


  
  
  
  /// Get affine loop bounds
  AffineMap lbMap, ubMap;
  llvm::SmallVector<mlir::Value, 8> lbOperands, ubOperands;


  /// Ensures that the loop has both a lower bound (LB) and an upper bound (UB). If either bound is missing, it asserts with an error message
  assert((forStmt->LB && forStmt->UB) && "Unbounded loops are not allowed.");


  /// TODO: simplify these sanity checks.
  assert(!(forStmt->LB->type == clast_expr_red &&
           reinterpret_cast<clast_reduction *>(forStmt->LB)->type == clast_red_min) &&
         "If the lower bound is a reduced result, it should not use min for reduction.");

  assert(!(forStmt->UB->type == clast_expr_red &&
           reinterpret_cast<clast_reduction *>(forStmt->UB)->type == clast_red_max) &&
         "If the upper bound is a reduced result, it should not use max for reduction.");
    






  

  if (failed(getAffineLoopBound(forStmt->LB, lbOperands, lbMap)) ||
      failed(getAffineLoopBound(forStmt->UB, ubOperands, ubMap, /*isUpper=*/true))) {
    return failure();
  }



  /// Initializes the loop stride to 1. If the stride is greater than 1, 
  /// it converts the stride to an int64_t value. If the conversion fails, it returns a failure
  int64_t stride = 1;

  if (cloog_int_gt_si(forStmt->stride, 1)) {

    if (failed(getI64(forStmt->stride, &stride)))

      return failure();

  }

  /// Log the lbOperands
  for (auto &operand : lbOperands) {
    
    trace["processStmt(clast_for *forStmt)"][std::to_string(counter)]["After getAffineLoopBound() operation"]["Operands"]["lbOperands"].push_back(valueToString(operand));
    
  }
  /// Log the ubOperands
  for (auto &operand : ubOperands) {
  
    trace["processStmt(clast_for *forStmt)"][std::to_string(counter)]["After getAffineLoopBound() operation"]["Operands"]["ubOperands"].push_back(valueToString(operand));
  
  }


  /// Create the for operation.
  /// Creates an MLIR affine for-loop operation (AffineForOp) using the lower bound operands and map, upper bound operands and map, and the stride. 
  /// The loop's location is set to an unknown location in the context
  mlir::AffineForOp forOp = b.create<mlir::AffineForOp>(UnknownLoc::get(context), lbOperands, lbMap, ubOperands, ubMap, stride);

  // Update the loop IV mapping.
  auto &entryBlock = *forOp.getLoopBody().getBlocks().begin();

  /// @F: My snitch
  trace["processStmt(clast_for *forStmt)"][std::to_string(counter)]["forOp after IR gen"] = operationToString(forOp);


  // TODO: confirm is there a case that forOp has multiple operands.
  assert(entryBlock.getNumArguments() == 1 && "affine.for should only have one block argument (iv).");



  symTable->setValue(forStmt->iterator, entryBlock.getArgument(0), OslSymbolTable::LoopIV);

  // Symbol table is mutable.
  // TODO: is there a better way to improve this? Not very safe.
  mlir::Value symValue = symbolTable[forStmt->iterator];

  symbolTable[forStmt->iterator] = entryBlock.getArgument(0);




  // ******************Create the loop body****************************
  b.setInsertionPointToStart(&entryBlock);
  
  entryBlock.walk([&](mlir::AffineYieldOp op) { b.setInsertionPoint(op); });
  
  /// This helps to create each for body
  assert(processStmtList(forStmt->body).succeeded());
  
  b.setInsertionPointAfter(forOp);

  // Restore the symbol value.
  symbolTable[forStmt->iterator] = symValue;

  // TODO: affine.parallel currently has more restrictions on what it can cover.
  // So we don't create a parallel op at this stage.
  if (forStmt->parallel)

    forOp->setAttr("scop.parallelizable", b.getUnitAttr());




  // Finally, we will move this affine.for op into a FuncOp if it uses values
  // defined by affine.min/max as loop bound operands.
  auto isMinMaxDefined = [](mlir::Value operand) {

    return isa_and_nonnull<mlir::AffineMaxOp, mlir::AffineMinOp>(

        operand.getDefiningOp());

  };


  

  if (std::none_of(lbOperands.begin(), lbOperands.end(), isMinMaxDefined) &&
      std::none_of(ubOperands.begin(), ubOperands.end(), isMinMaxDefined))

    return success();




  // Extract forOp out of the current block into a function.
  Block *prevBlock = forOp->getBlock();
  Block *currBlock = prevBlock->splitBlock(forOp);
  Block *nextBlock = currBlock->splitBlock(forOp->getNextNode());

  llvm::SetVector<mlir::Value> args;
  inferBlockArgs(currBlock, args);



  // Create the function body
  mlir::FunctionType funcTy = b.getFunctionType(TypeRange(args.getArrayRef()), llvm::None);

  b.setInsertionPoint(&*getFuncInsertPt());
  
  mlir::FuncOp func = b.create<mlir::FuncOp>(forOp->getLoc(), std::string("T") + std::to_string(numInternalFunctions), funcTy);

  numInternalFunctions++;
  
  Block *newEntry = func.addEntryBlock();
  
  BlockAndValueMapping vMap;
  
  vMap.map(args, func.getArguments());
  
  b.setInsertionPointToStart(newEntry);
  
  b.clone(*forOp.getOperation(), vMap);
  
  b.create<mlir::func::ReturnOp>(func.getLoc(), llvm::None);



  // Create function call.
  b.setInsertionPointAfter(forOp);
  
  b.create<mlir::func::CallOp>(forOp.getLoc(), func, ValueRange(args.getArrayRef()));




  // Clean up
  forOp.erase();
  
  b.setInsertionPointToEnd(prevBlock);
  
  for (Operation &op : *currBlock)
  
    b.clone(op);
  
  for (Operation &op : *nextBlock)
  
    b.clone(op);
  
  currBlock->erase();
  nextBlock->erase();



  // Set the insertion point right before the terminator.
  b.setInsertionPoint(&*std::prev(prevBlock->end()));



  return success();


}






void Importer::initializeSymbol(mlir::Value val) {

  assert(val != nullptr);

  /// Pointer points to valueTable where we have mapping of mlir value to symolic name 
  OslScop::ValueTable *oslValueTable = scop->getValueTable();


  /// Obtains the first block in the function's body, which serves as the entry block.
  auto &entryBlock = *func.getBody().begin();


  /// Save and restore the insertion point of the OpBuilder
  OpBuilder::InsertionGuard guard(b);

  /// Look up for mlir value in oslValueTable and collect the corresponding symbolic name of that mlir::Value
  ///Example: If val is %0 = memref.alloc() : memref<64x64xf32>, symbol would be A1.
  std::string symbol = oslValueTable->lookup(val);



  assert(!symbol.empty() && "val to initialize should have a corresponding "
                            "symbol in the original code.");



  /// If val is a block argument, there's nothing more to initialize, so the function returns early.
  if (mlir::BlockArgument arg = val.dyn_cast<mlir::BlockArgument>())
    return;


  
  /// Retrieves the operation that defines val.
  /// Example: If val is %0 = memref.alloc() : memref<64x64xf32>, defOp is the memref.alloc operation.
  mlir::Operation *defOp = val.getDefiningOp();

  /// F: My snitch
  std::string defOpStr = operationToString(defOp);
  trace["processStmt(clast_root *rootStmt)"]["initializeSymbolTable()"]["initializeSymbol(mlir::Value val)"]["defOp"].push_back(defOpStr);


  
  /// If the operation is a memory allocation with no operands
  if (isa<memref::AllocaOp>(defOp) && defOp->getNumOperands() == 0) {

    

    /// Set the insertion point to the start of the entry block
    b.setInsertionPointToStart(&entryBlock);

    /// Clone the operation and add its result to the symbolTable with the symbolic name.
    symbolTable[symbol] = b.clone(*defOp)->getResult(0);

    /// return early as the symbol is now initialized
    return;

  }




  // This indicates whether we have set an insertion point.
  bool hasInsertionPoint = false;

  /// Retrieve the parentOp of %0 = memref.alloc() : memref<64x64xf32> and others
  mlir::Operation *parentOp = defOp->getParentOp();


  /// F: My snitch
  std::string parentOpStr = operationToString(parentOp);
  trace["processStmt(clast_root *rootStmt)"]["initializeSymbolTable()"]["initializeSymbol(mlir::Value val)"]["parentOp"].push_back(parentOpStr);


  /// The parentOp is not AffineForOp for matmul so SKIPPING FOR NOW
  if (mlir::AffineForOp forOp = dyn_cast<mlir::AffineForOp>(parentOp)) {

    
    mlir::Value srcIV = forOp.getInductionVar();

    
    std::string ivName = oslValueTable->lookup(srcIV);
    
    
    mlir::Value dstIV = symbolTable[ivName];


   
    if (dstIV == nullptr) {

      symbolToDeps[ivName].insert(val);
      valueToDepSymbols[val].insert(ivName);

    } else {

    
      mlir::Block *blockToInsert = dstIV.cast<mlir::BlockArgument>().getOwner();

      hasInsertionPoint = true;

      b.setInsertionPointToStart(blockToInsert);
    }
  
  } 
  
  

  /// parentOp is indeed FuncOp for matmul.mlir eg
  else if (mlir::FuncOp funOp = dyn_cast<mlir::FuncOp>(parentOp)) {

    

    /// Insert at the beginning of this function.
    hasInsertionPoint = true;

    /// Just setting insertion point at the start of entryBlock
    b.setInsertionPointToStart(&entryBlock);


  } 
  
  /// Handle Unsupported Parent Operation
  else {

    assert(false);

  }



  /// ================SKIPPING BELOW PART FOR NOW======================
  /// Take a small vector for storing new operands
  /// Here, it is used to store up to 8 operands
  SmallVector<mlir::Value, 8> newOperands;
  /// Next, we check whether all operands are in the symbol table
  /// Loop through each operand of the defining operation (defOp)
  for (mlir::Value operand : defOp->getOperands()) {


    std::string operandSymbol = oslValueTable->lookup(operand);

    ///  If the operand symbol is empty, check if the operand is defined by a constant operation
    if (operandSymbol.empty()) {

      /// operand.getDefiningOp() retrieves the operation that defines the operand
      mlir::Operation *operandDefOp = operand.getDefiningOp();

      /// checks if this defining operation is a constant operation
      if (operandDefOp && isa<mlir::arith::ConstantOp>(operandDefOp)) {
        
        /// If true, clone the constant operation and add its result to newOperands
        newOperands.push_back(b.clone(*operandDefOp)->getResult(0));

        continue;

      }

    }

    /// The assert ensures that the operand symbol exists in the symbolTable
    assert(!operandSymbol.empty() && "operand should be in the original symbol table.");

    /// symbolTable[operandSymbol] retrieves the new operand value
    mlir::Value newOperand = symbolTable[operandSymbol];
    
    // If the symbol is not yet initialized, we update the two dependence
    // tables. Note that here we implicitly assume that the operand symbol
    // should exist.
    assert(newOperand != nullptr);
    
    newOperands.push_back(newOperand);
  
  }

  // The operands are not sufficient, should wait.
  /// Ensure that the number of new operands matches the number of operands of defOp
  if (newOperands.size() < defOp->getNumOperands()) {
    
    return;

  }


  // Finally do the initialization.
  if (!hasInsertionPoint) {
    
    return;
  
  }






  BlockAndValueMapping vMap;
  
  /// Create a mapping from the original operands to the new operands
  for (unsigned i = 0; i < newOperands.size(); i++) {

    vMap.map(defOp->getOperand(i), newOperands[i]);

  }

  /// ==================SKIPPING TILL ABOVE PART FOR NOW except BlockAndValueMapping vMap;===================

  

  /// This is the one of the best culprit i have ever seen
  /// The clone function creates a duplicate of an existing operation and inserts it into the IR.
  /// For more fucking explanation about the culprit check: About clone() : https://mlir.llvm.org/doxygen/Builders_8cpp_source.html#l00555
  /// About insert() : https://mlir.llvm.org/doxygen/Builders_8cpp_source.html#l00428
  mlir::Operation *newOp = b.clone(*defOp, vMap);

  /// F: My snitch
  trace["processStmt(clast_root *rootStmt)"]["initializeSymbolTable()"]["initializeSymbol(mlir::Value val)"]["func"].push_back(operationToString(func));

  
  assert(newOp != nullptr);
  
  assert(newOp->getNumResults() == 1 && "Should only have one result.");

  symbolTable[symbol] = newOp->getResult(0);


  ///
  /// Key Points:
  /// Order and Instances: The order in which the symbolTable entries appear might not match the original oslValueTable 
  /// due to the cloning process. Each new operation (newOp) is a distinct clone and not the same instance as the original.
  /// Unique Instances: Cloning ensures that each newOp is a unique instance, even if it has the same structure as the original. 
  /// This is crucial for maintaining the integrity of the IR, avoiding potential conflicts or unintended references to original operations.
  /// Logging and Tracking: When logging or tracking these operations, they appear as distinct but structurally identical operations. 
  /// The discrepancies in the symbol table arise because these are newly created instances with their unique identities.
  /// Conclusion:
  /// The perceived discrepancies are due to the nature of the cloning process, which creates new, distinct instances of operations 
  /// that are structurally identical to the original ones. This ensures that the IR maintains separate, independent operations, 
  /// which is critical for accurate transformations and optimizations.


}




void Importer::initializeSymbolTable() {

  /// This is SymbolTable type from OslScop. Retrieve SymbolTable from scop
  /// using SymbolTable = llvm::StringMap<mlir::Value>;
  OslScop::SymbolTable *oslSymbolTable = scop->getSymbolTable();

  /// F: Iterate over the oslSymbolTable
  for (const auto& entry : *oslSymbolTable) {
    
    
    std::string key = entry.getKey().str();
    mlir::Value value = entry.second;

    std::string valueStr = valueToString(value);

    /// Insert into JSON object, using an array to store multiple values
    trace["processStmt(clast_root *rootStmt)"]["initializeSymbolTable()"]["oslSymbolTable"][key].push_back(valueStr);
  }


  OpBuilder::InsertionGuard guard(b);

  auto &entryBlock = *func.getBody().begin();

  /// Insertion point setting at the start of entryBlock of function op
  b.setInsertionPointToStart(&entryBlock);



  /// Creates a new constant operation that produces the value 0 of type index. This operation is added to the symbol table with the symbolic name "zero".
  symbolTable["zero"] = b.create<mlir::arith::ConstantOp>(b.getUnknownLoc(), b.getIndexType(), b.getIntegerAttr(b.getIndexType(), 0));

  
  // Check that the symbol was added
  if (symbolTable.find("zero") != symbolTable.end()) {

    std::cout << "Symbol 'zero' was successfully added to the symbolTable." << std::endl;
  
  } else {
    std::cerr << "Failed to add 'zero' to the symbolTable." << std::endl;
  }

 

  
  for (const auto &it : *oslSymbolTable)

    initializeSymbol(it.second);


  /// F: Iterate over the symbolTable
  for (const auto& entry : symbolTable) {
    
    
    std::string key = entry.getKey().str();
    mlir::Value value = entry.second;

    std::string valueStr = valueToString(value);

    /// Insert into JSON object, using an array to store multiple values
    trace["processStmt(clast_root *rootStmt)"]["initializeSymbolTable()"]["symbolTable"][key].push_back(valueStr);
  }

 


}



/// If there is anything in the comment, we will use it as a function name.
/// Otherwise, we return an empty string.
std::string Importer::getSourceFuncName() const {

  osl_generic_p comment = scop->getExtension("comment");

  

  if (comment) {

    char *commentStr = reinterpret_cast<osl_comment_p>(comment->data)->comment;

    return std::string(commentStr);

  }

  return std::string("");

}





mlir::FuncOp Importer::getSourceFuncOp() {

  /// Get function name. E.g: matmul
  std::string sourceFuncName = getSourceFuncName();
  

  ///Look up the operation (in this case, a function) by its name within the module.
  /// The function name is stored in the variable 'sourceFuncName', which was obtained earlier.
  /// The 'module' is a container for a collection of operations, typically representing
  /// the entire code being compiled or transformed.
  /// 'lookupSymbol' is a method provided by the 'SymbolTable' trait, which allows us to
  /// search for operations by their names within the module.
  /// If the function with the name 'sourceFuncName' is found, 'lookupSymbol' returns a pointer
  /// to that operation. If it is not found, it returns a nullptr.
  /// This line essentially retrieves the function operation from the module by its name.
  /// Link: Trait vs Interface: https://mlir.llvm.org/getting_started/Faq/#what-is-the-difference-between-traits-and-interfaces
  /// The current module created publically inside Importer class.
  /// ModuleOp module;
  mlir::Operation *sourceFuncOp = module.lookupSymbol(sourceFuncName);




  /// F: My Json dumping
  trace["processStmt(clast_root *rootStmt)"]["initializeFuncOpInterface()"]["getSourceFuncOp()"]["sourceFuncName"] = sourceFuncName;

  trace["processStmt(clast_root *rootStmt)"]["initializeFuncOpInterface()"]["getSourceFuncOp()"]["sourceFuncOp"] = operationToString(sourceFuncOp);

  

  /// The assert statements ensure that the symbol was found and is of the correct type (FuncOp).
  assert(sourceFuncOp != nullptr && "sourceFuncName cannot be found in the module");
  
  assert(isa<mlir::FuncOp>(sourceFuncOp) && "Found sourceFuncOp should be of type mlir::FuncOp.");

  /// Casting to mlir FuncOp specifically
  return cast<mlir::FuncOp>(sourceFuncOp);

}






/// Initialize FuncOpInterface
void Importer::initializeFuncOpInterface() {

  /// Retrieve mapping of mlir value to symbolic names out of valueTable
  /// And this method getValueTable() is written under OslScop class in this format 
  /// `using ValueTable = llvm::DenseMap<mlir::Value, std::string>;` here ValueTable is acting as alias 
  /// for the type `llvm::DenseMap<mlir::Value, std::string>`
  OslScop::ValueTable *oslValueTable = scop->getValueTable();
  
  

  /// F: Iterate over the oslValueTable
  for (const auto& entry : *oslValueTable) {
    mlir::Value key = entry.first;
    std::string value = entry.second;

    std::string keyStr = valueToString(key);

    /// Insert into JSON object, using an array to store multiple values
    trace["processStmt(clast_root *rootStmt)"]["initializeFuncOpInterface()"]["oslValueTable"][keyStr].push_back(value);
  }



  /// Retrieves the original function operation that needs to be transformed.
  mlir::FuncOp sourceFuncOp = getSourceFuncOp();



  
  /// Sets the insertion point in the module where the new function operation will be inserted.
  b.setInsertionPoint(module.getBody(), getFuncInsertPt());




  /// New function operation name determination
  /// The default function name is main.
  std::string funcName("main");
  /// Get the function name.
  std::string sourceFuncName = getSourceFuncName();
  /// if sourceFuncName is not empty than "sourceFuncName.empty()" will return False and "!" will make condition true
  if (!sourceFuncName.empty()) 

    /// funcName will have E.g: matmul_opt. matmul function name we already got from getSourceFuncName() method call
    /// But, if there is no function name then by default it will be main
    funcName = std::string(formatv("{0}_opt", sourceFuncName));
  



  /// @brief: The 'create' method of OpBuilder class constructs a new Instance of FuncOp class
  /// Link for Opbuilder create method: https://github.com/llvm/llvm-project/blob/22dfa1aa2c6b4026b4a5d1f594197ee22af3136d/mlir/lib/IR/Builders.cpp#L469
  /// @param: 
  /// - sourceFuncOp.getLoc(): provides the location information from the original source code where the function is defined
  ///                          It's useful for debugging and diagnostics, as it helps trace back to the original source code.
  /// - funcName: This is the name to be assigned to the new function in output MLIR. 
  ///             It can be a default name like "main" or derived from the original function name with some modification (e.g., appending "_opt").
  /// - sourceFuncOp.getType(): This defines the type signature of the function, including its list of function inputs and outputs types. 
  ///                           The new function operation will have the same type signature as the original source function.
  ///                           FunctionType is one of the Type which holds this signature
  ///
  /// The result is stored in the 'func' member variable of the Importer class, which holds newly created instance of FuncOp class
  ///
  /// Again though here they used variable name as 'func', in MLIR it is the name of dialect under which we have FuncOp class(C++) or we can say Ops(MLIR).
  ///  
  /// *Note:* When you create an operation using the builder b, it is automatically inserted into the IR at the specified insertion point. 
  func = b.create<FuncOp>(sourceFuncOp.getLoc(), funcName, sourceFuncOp.getType());


  std::string sourceFuncOpLoc = locationToString(sourceFuncOp.getLoc());
  std::string sourceFuncOpType = typeToString(sourceFuncOp.getType());
  

  trace["processStmt(clast_root *rootStmt)"]["initializeFuncOpInterface()"]["FuncOp class or Ops: object instantiation needs input"]["sourceFuncOp.getLoc()"] = sourceFuncOpLoc;
  trace["processStmt(clast_root *rootStmt)"]["initializeFuncOpInterface()"]["FuncOp class or Ops: object instantiation needs input"]["function name"].push_back(funcName);
  trace["processStmt(clast_root *rootStmt)"]["initializeFuncOpInterface()"]["FuncOp class or Ops: object instantiation needs input"]["sourceFuncOp.getType()"].push_back(sourceFuncOpType);


  

  /// addEntryBlock(): Method that creates an entry block for the function Operation
  /// entryBlock: reference to the newly created block
  auto& entryBlock = *func.addEntryBlock();


  /// This sets the insertion point within the function Ops body at beginning location where the new entry block will be inserted.
  b.setInsertionPointToStart(&entryBlock);


  /// ReturnOp is another Op under 'func' dialect.
  /// Once it is created and inserted into the IR, there is no immediate need to manipulate or reference it further in the current context.
  /// If you needed to manipulate the ReturnOp after creating it (e.g., setting attributes, connecting it to other operations), 
  /// you would assign it to a variable like the way we did before for FunOp
  b.create<mlir::func::ReturnOp>(UnknownLoc::get(context));
  


  /// Resets the insertion point to the start of the entry block for further operations insertion.
  b.setInsertionPointToStart(&entryBlock);




  /// To understand how entryBlock could get number of function arguments see below few lines
  /// auto funcType = FunctionType::get(context, {IntegerType::get(context, 32), FloatType::get(context)}, {FloatType::get(context)});
  /// auto funcOp = FuncOp::create(Location::unknown(), "example", funcType);
  /// auto &entryBlock = *funcOp.addEntryBlock();
  /// This adds an entry block to the function.
  /// The entry block gets arguments based on the function's signature.

  /// This part handles function parameters
  for (unsigned i = 0; i < entryBlock.getNumArguments(); i++) {
    
    /// sourceFuncOp.getArgument(i) retrieves the i-th argument of the sourceFuncOp.
    /// oslValueTable->lookup(...) looks up the symbolic name for the given MLIR value (argument).
    /// The symbolic name is stored in argSymbol. Check in Json under oslValueTable
    std::string argSymbol = oslValueTable->lookup(sourceFuncOp.getArgument(i));

    /// Get the ith argument from current funcOp through entryBlock
    mlir::Value arg = entryBlock.getArgument(i);


    /// If the source type is not index, cast it to index then.
    if (scop->isParameterSymbol(argSymbol) && arg.getType() != b.getIndexType()) {
      
      mlir::Operation* op = b.create<mlir::arith::IndexCastOp>(sourceFuncOp.getLoc(), b.getIndexType(), arg);
      symbolTable[argSymbol] = op->getResult(0);
    
    } else {
      
      /// Otherwise just add it to the symbolTable
      /// The symbol table that will be built on the fly. And declared inside Importer class
      /// SymbolTable = llvm::StringMap<mlir::Value>;
      /// SymbolTable symbolTable;
      symbolTable[argSymbol] = arg;
    
    }
  
  }



  /// F: Iterate over the symbolTable
  for (const auto& entry : symbolTable) {
    
    
    std::string key = entry.getKey().str();
    mlir::Value value = entry.second;

    std::string valueStr = valueToString(value);

    /// Insert into JSON object, using an array to store multiple values
    trace["processStmt(clast_root *rootStmt)"]["initializeFuncOpInterface()"]["symbolTable"][key].push_back(valueStr);
  }

 

}






/// Translate the root statement as a function. The name of the function is by
/// default "main".
LogicalResult Importer::processStmt(clast_root *rootStmt) {
  
  
  // Create the function.
  initializeFuncOpInterface();
  
  trace["processStmt(clast_root *rootStmt)"]["New funcOp after initializeFuncOpInterface() processes"] = operationToString(func);

  // Initialize several values before start.
  initializeSymbolTable();

  trace["processStmt(clast_root *rootStmt)"]["New funcOp after initializeSymbolTable() processes"] = operationToString(func);

   for (const auto& entry : symbolTable) {
    
    
    std::string key = entry.getKey().str();
    mlir::Value value = entry.second;

    std::string valueStr = valueToString(value);

    /// Insert into JSON object, using an array to store multiple values
    trace["processStmt(clast_root *rootStmt)"]["symbolTable"][key].push_back(valueStr);
  }
  


  return success();

}





LogicalResult Importer::processStmtList(clast_stmt *s) {
  
  /// Declare a file pointer and a buffer for file contents
  FILE *rootStmt_dump;
  char file_contents[1000];
  std::string root_ast;
  std::string counterStr;
  

  /// Helper function to process a statement and update JSON
  auto processAndLogStmt = [&](clast_stmt *stmt, const std::string &stmtType) -> LogicalResult {
  
    rootStmt_dump = fopen("output-files/1.polymer-commit-bda08-forOp/4.rootStmt.txt", "w+");
    
    /// Dump AST to file
    clast_pprint(rootStmt_dump, stmt, 0, options);
    fclose(rootStmt_dump);

    /// Open the file in read mode
    rootStmt_dump = fopen("output-files/1.polymer-commit-bda08-forOp/4.rootStmt.txt", "r");
    
    /// Read the contents of the file into a string
    root_ast.clear(); /// Clear previous contents
    while (fgets(file_contents, sizeof(file_contents), rootStmt_dump)) {
      root_ast += file_contents;
    }
    fclose(rootStmt_dump);

    /// Update JSON trace
    trace["processStmt(" + stmtType + ")"][std::to_string(counter)]["Input"] = root_ast;

    return success();
  };



  /// Loop through each statement in the linked list
  for (; s; s = s->next) {
    
    counter++;
    
    if (CLAST_STMT_IS_A(s, stmt_root)) {
      
      /// Process root statement
      if (failed(processAndLogStmt(s, "clast_root *rootStmt")) ||
    
        failed(processStmt(reinterpret_cast<clast_root *>(s)))) {
        return failure();
      
      }
      
      

    } else if (CLAST_STMT_IS_A(s, stmt_for)) {
      
      /// Process for statement
      if (failed(processAndLogStmt(s, "clast_for *forStmt")) ||
      
        failed(processStmt(reinterpret_cast<clast_for *>(s)))) {
        return failure();
      
      }
    
      

    } else if (CLAST_STMT_IS_A(s, stmt_user)) {
      
      /// Process user statement
      if (failed(processStmt(reinterpret_cast<clast_user_stmt *>(s)))) {
        return failure();
      
      }
    
      
    
    } else {
      
      assert(false && "clast_stmt type not supported");
    
    }
    

  }
  std::printf("INSIDE CLAST_STMT");
  /// Write JSON trace to a file
  std::ofstream o("output-files/1.polymer-commit-bda08-forOp/data_v2.json");
  
  if (!o.is_open()) {
  
    std::cerr << "Failed to open file for writing.\n";
    return failure();
  
  } else {
  
    o << trace.dump(4); /// Pretty-printed JSON with 4 spaces indentation
    o.close();
    std::cout << "Data has been dumped to data_v2.json successfully.\n";
  
  }

  return success();
}









static void unrollJamClastByPlutoProg(clast_stmt *root, 
                                      const PlutoProg *prog,
                                      CloogOptions *cloogOptions,
                                      unsigned ufactor) {
  
  unsigned numPloops;
  
  Ploop **ploops = pluto_get_parallel_loops(prog, &numPloops);

  for (unsigned i = 0; i < numPloops; i++) {
  
    if (!pluto_loop_is_innermost(ploops[i], prog))
  
      continue;

    std::string iter(formatv("t{0}", ploops[i]->depth + 1));

    /// Collect all statements within the current parallel loop.
    SmallVector<int, 8> stmtIds(ploops[i]->nstmts);
  
    for (unsigned j = 0; j < ploops[i]->nstmts; j++)
  
      stmtIds[j] = ploops[i]->stmts[j]->id + 1;

    ClastFilter filter = {/*iter=*/iter.c_str(),
                          /*stmts_filter=*/stmtIds.data(),
                          /*nstmts_filter=*/static_cast<int>(ploops[i]->nstmts),
                          /*filter_type=*/subset};

    clast_for **loops;
    unsigned numLoops, numStmts;
    int *stmts;
  
    clast_filter(root, filter, &loops, (int *)&numLoops, &stmts, (int *)&numStmts);

    // There should be at least one loops.
    if (numLoops == 0) {
    
      free(loops);
      free(stmts);
      continue;
    
    }

    for (unsigned j = 0; j < numLoops; j++)
    
      loops[j]->parallel += CLAST_PARALLEL_VEC;

    free(loops);
    free(stmts);
  
  }

  pluto_loops_free(ploops, numPloops);

  // Call clast transformation.
  clast_unroll_jam(root);

}



static void markParallel(clast_stmt *root, const PlutoProg *prog, CloogOptions *cloogOptions) {
  
  pluto_mark_parallel(root, prog, cloogOptions);

}



static void transformClastByPlutoProg(clast_stmt *root, 
                                      const PlutoProg *prog,
                                      CloogOptions *cloogOptions,
                                      PlutoOptions *plutoOptions) {

  if (plutoOptions->unrolljam)

    unrollJamClastByPlutoProg(root, prog, cloogOptions, plutoOptions->ufactor);
  
  if (plutoOptions->parallel)
  
    markParallel(root, prog, cloogOptions);

}




static void updateCloogOptionsByPlutoProg(CloogOptions *options, const PlutoProg *prog) {

  Stmt **stmts = prog->stmts;
  
  int nstmts = prog->nstmts;

  options->fs = (int *)malloc(nstmts * sizeof(int));
  
  options->ls = (int *)malloc(nstmts * sizeof(int));
  
  options->fs_ls_size = nstmts;

  for (int i = 0; i < nstmts; i++) {
  
    options->fs[i] = -1;
    options->ls[i] = -1;
  
  }

  if (prog->context->options->cloogf >= 1 && prog->context->options->cloogl >= 1) {

    options->f = prog->context->options->cloogf;
    options->l = prog->context->options->cloogl;
  
  } else {
  
    if (prog->context->options->tile) {
  
      for (int i = 0; i < nstmts; i++) {
  
        options->fs[i] = get_first_point_loop(stmts[i], prog) + 1;
        options->ls[i] = prog->num_hyperplanes;
  
      }
  
    } else {
  
      options->f = 1;
      options->l = prog->num_hyperplanes;
  
    }
  
  }

}




/// @brief: Below this point all things belongs to polymer  

mlir::Operation *polymer::createFuncOpFromOpenScop(std::unique_ptr<OslScop> scop, ModuleOp module, OslSymbolTable &symTable,
                                                  MLIRContext *context, PlutoProg *prog, const char *dumpClastAfterPluto) {
  
  
  std::cout << "create Func OP.\n" << std::endl;


  FILE *CloogOut = fopen("output-files/1.polymer-commit-bda08-forOp/1.scop_to_cloog.cloog", "w");
  FILE *ProgramOut = fopen("output-files/1.polymer-commit-bda08-forOp/2.cloog_to_program.txt", "w");
  FILE *ClastOut = fopen("output-files/1.polymer-commit-bda08-forOp/3.program_to_clast.txt", "w");
  // FILE *scop_file = fopen("scop_file.txt", "w");
  // FILE *options1 = fopen("options.txt", "w");


  // TODO: turn these C struct into C++ classes.
  CloogState *state = cloog_state_malloc();
  CloogOptions *options = cloog_options_malloc(state);

  options->openscop = 1; // The input file in the OpenScop format
  options->scop = scop->get(); // Get the raw scop pointer

  // THIS is the culprit
  CloogInput *input = cloog_input_from_osl_scop(options->state, scop->get());

  cloog_options_copy_from_osl_scop(scop->get(), options);
  
  
  
  //+++++++++++++++++++++CLOOG contents printing+++++++++++++++++++
  cloog_input_dump_cloog(CloogOut, input, options);

  
  
  
  if (prog != nullptr)
    
    updateCloogOptionsByPlutoProg(options, prog);

  // Create cloog_program
  CloogProgram *program = cloog_program_alloc(input->context, input->ud, options);
  
  assert(program->loop);

  
  program = cloog_program_generate(program, options);



  // +++++++++++++++++++Print Program+++++++++++++++++++++++++++++++++++
  cloog_program_print(ProgramOut, program);

  
  
  
  if (!program->loop) {
  
    cloog_program_print(stderr, program);
    
    assert(false && "No loop found in the CloogProgram, which may indicate the "
                    "provided OpenScop is malformed.");
  
  }


  // Convert to clast
  clast_stmt *rootStmt = cloog_clast_create(program, options);

  
  assert(rootStmt);
  
  if (prog != nullptr)
    
    transformClastByPlutoProg(rootStmt, prog, options, prog->context->options);

  FILE *clastPrintFile_1 = stderr;
  
  if (dumpClastAfterPluto) {
  
    clastPrintFile_1 = fopen(dumpClastAfterPluto, "w");

    assert(clastPrintFile_1 && "File for clast dump after Pluto cannot be opened.");
  
  }


  if (dumpClastAfterPluto)
    fclose(clastPrintFile_1);



  // +++++++++++++++++Print clast+++++++++++++++++++++++++++
  clast_pprint(ClastOut, rootStmt, 0, options);



  // Process the input.
  Importer deserializer(context, module, &symTable, scop.get(), options);
  
  if (failed(deserializer.processStmtList(rootStmt)))
    return nullptr;

  // Cannot use cloog_input_free, some pointers don't exist.
  free(input);
  cloog_program_free(program);

  options->scop = NULL; // Prevents freeing the scop object.
  cloog_options_free(options);
  cloog_state_free(state);



  fclose(CloogOut);
  fclose(ProgramOut);
  fclose(ClastOut);
  

  return deserializer.getFunc();

}





/**
 * @description Translates an OpenScop description into an MLIR module. This function initializes
 *              an MLIR module and uses the OpenScop description to populate it with function operations
 *              based on the given OpenScop structure.
 *
 * @param scop A unique pointer to an OslScop, which holds the OpenScop description that will be translated.
 * @param context A pointer to an MLIRContext, essential for managing MLIR operations including the handling of dialects and locations.
 *
 * @return Returns an OwningOpRef to a ModuleOp if the translation is successful; otherwise, returns an empty OwningOpRef.
 */
OwningOpRef<ModuleOp> polymer::translateOpenScopToModule(std::unique_ptr<OslScop> scop, MLIRContext *context) {
  

  std::cout << "translate openscope to module of polymer.\n" << std::endl;

  // Load the Affine dialect into the MLIR context. Dialects are collections of operations, types, and attributes
  // necessary for the generation and optimization of MLIR modules
  context->loadDialect<AffineDialect>();

  
  // Create a new MLIR module at a generic file location with default line and column numbers
  // ModuleOp represents a module in MLIR which can contain functions and other modules
  OwningOpRef<ModuleOp> module(ModuleOp::create(FileLineColLoc::get(context, "", /*line=*/0, /*column=*/0)));


  //================beginner version============================
  // Step 1: Obtain a location.
  // The FileLineColLoc function is used to specify where in the source code this module is conceptually located.
  // Since no specific file or line is referenced, we pass empty strings for the file and set line and column to zero.
  // FileLineColLoc location = FileLineColLoc::get(context, "", /*line=*/0, /*column=*/0);

  // Step 2: Create a ModuleOp.
  // ModuleOp::create is called with the location we obtained in the previous step.
  // This operation creates a new module, which is a container that can hold functions, global variables, and other modules.
  // OwningOpRef<ModuleOp> module = ModuleOp::create(location);


  // Initialize a symbol table to manage symbols within the scope of the translation
  OslSymbolTable symTable;


  // Attempt to create an MLIR function operation from the OpenScop description. If the function fails,
  // return an empty OwningOpRef, indicating an error during function creation
  if (!createFuncOpFromOpenScop(std::move(scop), module.get(), symTable, context))

    return {};


  // If the function operation is successfully created and added to the module, return the module
  return module;


}




/**
 * @description Translates OpenScop representation to an MLIR module.
 * This function reads the OpenScop data from the main file managed by sourceMgr,
 * parses it, and then uses another overloaded version of this function
 * to perform the translation into an MLIR module.
 * 
 * @param sourceMgr A reference to an LLVM SourceMgr, which manages source files
 *                  and buffers. It is used to access the input data for OpenScop.
 * @param context A pointer to an MLIRContext, which encapsulates the global state
 *                necessary for MLIR operations, including registered dialects
 *                and types.
 * 
 * @return Returns an OwningOpRef of ModuleOp, which manages the memory of the
 *         created MLIR module, ensuring proper memory management and deletion
 *         when it is no longer needed.
 */
static OwningOpRef<ModuleOp> translateOpenScopToModule(llvm::SourceMgr &sourceMgr, MLIRContext *context) {

  std::cout << "translate openscope to module.\n" << std::endl;


  // Declare an llvm::SMDiagnostic object to hold any diagnostic messages (like errors)
  // that may occur during the operations within this function
  llvm::SMDiagnostic err;


  // Read the OpenScop representation from the main file managed by sourceMgr. 
  // `getMemoryBuffer` retrieves a memory buffer containing the source text, and `getMainFileID`
  // returns the identifier for the main file loaded into sourceMgr
  std::unique_ptr<OslScop> scop = readOpenScop(*sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID()));


  // Call another version of `translateOpenScopToModule` using the OpenScop object `scop` we just read
  // and the MLIR context provided. The `std::move(scop)` is used to transfer ownership of the `scop`
  // object to the called function, preventing the need to copy the object.
  return translateOpenScopToModule(std::move(scop), context);


}




/// @brief polymer namespace
namespace polymer {

  /**
   * @brief: This function is responsible for registering a translation from an OpenScop representation to an 
   * MLIR (Multi-Level Intermediate Representation) module. The translation is registered with a specific name 
   * ("import-scop") and implemented using a lambda function.
   *
   * In other words, when you register a translation, you inform the MLIR framework about a new function or process that 
   * can convert data from one format to another. This function is identified by a specific name (in this case, "import-scop").
   */
  void registerFromOpenScopTranslation() {
    
    // std::cout << "Data has been dumped to data_v_2.json successfully.\n" << std::endl;

    // Create a registration object for the translation
    // "import-scop" is the name used to identify this translation within the MLIR framework
    TranslateToMLIRRegistration fromLLVM(
    
        "import-scop", 

        /// This is a lambda function (an anonymous function defined inline)
        [](llvm::SourceMgr &sourceMgr, MLIRContext *context) {  

          /// This lambda function calls 'translateOpenScopToModule' to perform the actual translation
          /// The '::' scope resolution operator indicates that 'translateOpenScopToModule' is defined in the global namespace
          return ::translateOpenScopToModule(sourceMgr, context); 
    
        }
    
    );

  }

} // namespace polymer


