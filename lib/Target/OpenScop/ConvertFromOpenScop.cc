//===- ConvertFromOpenScop.h ------------------------------------*- C++ -*-===//
//
// This file implements the interfaces for converting OpenScop representation to
// MLIR modules.
//
//===----------------------------------------------------------------------===//

#include <fstream>
#include <iostream>

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




#include "polymer/Support/nlohmann/json.hpp"

using ordered_json = nlohmann::ordered_json;
using json = nlohmann::json;

using namespace polymer;
using namespace mlir;

/// My snippet
// Create a JSON global object
json j_v2;
ordered_json j;
ordered_json symbolTableJson;

typedef llvm::StringMap<mlir::Operation *> StmtOpMap;
typedef llvm::StringMap<mlir::Value> NameValueMap;
typedef llvm::StringMap<std::string> IterScatNameMap;


/// F: My snitch
/// @brief : std::map for dumping the traces
std::map<std::string, std::string> trace;
int counter = 1;

/// @brief Print the trace
/// @param t 
void trace_print(const std::map<std::string, std::string> &t) {

    for (const auto & [key, value] : t)

        std::cout << "[" << key << "] = " << value << ";"; 

}



/// @brief Dump the trace
/// @param t : trace, std::map
/// @param key : key of the map
/// @param value : value of the map
void trace_dump(std::map<std::string, std::string> &t, std::string key, std::string value) {


    t[key] = value;


}

/// @brief these are the helping functions needed for Importer class methods 

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
    if (failed(process(reinterpret_cast<clast_name *>(expr), affExprs)))
      return failure();
    break;
  
  
  case clast_expr_term:
    if (failed(process(reinterpret_cast<clast_term *>(expr), affExprs)))
      return failure();
    break;
  
  
  case clast_expr_bin:
    if (failed(process(reinterpret_cast<clast_binary *>(expr), affExprs)))
      return failure();
    break;
  
  
  case clast_expr_red:
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
  

  /// Check if the Name is a Symbol
  if (scop->isSymbol(expr->name)) {

    std::printf("[process(clast_name)]I AM HIT\n");

    /// Start processing the Symbol
    /// Check if the Symbol is Already in symbolNames map. That means it has been proceesed before.
    ///  This comparison checks if the iterator returned by find is different from end(). 
    /// If it is different, it means the element was found in the map. If it is equal to end(), 
    /// it means the element was not found.
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

  /// F: Iterate over the valueMap
  std::cout << "\n" <<"Printing ValueMap" << "\n";

  for (const auto& entry : valueMap) {
    mlir::Value key = entry.first;
    std::string value = entry.second;

    std::string keyStr = valueToString(key);


    std::cout << "Key: " << keyStr << ", Value: " << value << std::endl;

    /// Insert into JSON object, using an array to store multiple values
    // oslValueJson[keyStr].push_back(value);
  }




  return success();

}

//  void AffineExpr::dump() const {

//   print(llvm::errs());
//   llvm::errs() << "\n";
 
//  }


LogicalResult AffineExprBuilder::process(clast_term *expr, llvm::SmallVectorImpl<AffineExpr> &affExprs) {
  
  // First get the I64 representation of a cloog int.
  int64_t constant;


  /// This extracts the 64-bit integer value from the clast_term's val field using the getI64 function. 
  /// If this extraction fails, the function returns a failure.
  if (failed(getI64(expr->val, &constant)))
    return failure();



  /// Next create a constant AffineExpr.
  /// This creates a constant affine expression using the extracted 64-bit integer value. 
  /// The getAffineConstantExpr method of the OpBuilder (b) is used to create this constant expression.
  AffineExpr affExpr = b.getAffineConstantExpr(constant);


  /// F: Print the initial constant affine expression.
  std::string str;
  llvm::raw_string_ostream rso(str);
  affExpr.print(rso);
  llvm::outs() << "Initial AffineExpr: " << rso.str() << "\n";
  


  /// This checks if the clast_term has a variable part (var). 
  /// If var is not NULL, it means this term is of the form var * val. We should create the
  /// expr that denotes var and multiplies it with the AffineExpr for val.
  if (expr->var) {

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
  
  if (expr->n == 1) {
    if (failed(process(expr->elts[0], affExprs)))
      return failure();
    return success();
  }

  switch (expr->type) {
  case clast_red_sum:
    if (failed(processSumReduction(expr, affExprs)))
      return failure();
    break;
  case clast_red_min:
  case clast_red_max:
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
      void initializeSymbol(mlir::Value val, ordered_json &j);

      LogicalResult processStmt(clast_root *rootStmt);
      LogicalResult processStmt(clast_for *forStmt);

      LogicalResult processStmt(clast_user_stmt *userStmt);
      LogicalResult processStmt(clast_assignment *ass);

      std::string getSourceFuncName(ordered_json &j) const;
      mlir::FuncOp getSourceFuncOp(ordered_json &j);

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

/// We treat the provided the clast_expr as a loop bound. If it is a min/max
/// reduction, we will expand that into multiple expressions.
static LogicalResult processClastLoopBound(clast_expr *expr,
                                           AffineExprBuilder &builder,
                                           SmallVectorImpl<AffineExpr> &exprs,
                                           CloogOptions *options) {


  FILE *expandedExprs_dump = fopen("output-files/processClastLoopBound.txt", "a");
  

  
  SmallVector<clast_expr *, 1> expandedExprs;

  if (expr->type == clast_expr_red) {

    clast_reduction *red = reinterpret_cast<clast_reduction *>(expr);

    if (red->type == clast_red_max || red->type == clast_red_min) {

      for (int i = 0; i < red->n; i++) {

        expandedExprs.push_back(red->elts[i]);

      }
    
    }
  
  }

  
  if (expandedExprs.empty()) // no expansion, just put the original input in.
  
    expandedExprs.push_back(expr);



  
  for (clast_expr *e : expandedExprs) {

    /// F: Snitch to dump expandedExprs vector to file
    fprintf(expandedExprs_dump, "Bounds printing:\n");
    clast_pprint_expr(options, expandedExprs_dump, e);



    if (failed(builder.process(e, exprs)))
  
      return failure();

  }



  fclose(expandedExprs_dump);

  return success();



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




LogicalResult Importer::getAffineLoopBound(clast_expr *expr,
                                           llvm::SmallVectorImpl<mlir::Value> &operands,
                                           AffineMap &affMap, bool isUpper) {




  /// F: My snitch         
  FILE *clast_expr_output = fopen("output-files/6.clast_expr_from_getAffineLoopBound.txt", "a");

  fprintf(clast_expr_output, "\nEntering getAffineLoopBound\n");



  /// An AffineExprBuilder instance is created to help build affine expressions.
  AffineExprBuilder builder(context, symTable, &symbolTable, scop, options);
  /// A vector boundExprs is initialized to store the resulting affine expressions.
  SmallVector<AffineExpr, 4> boundExprs;



  /// F: My snitch
  fprintf(clast_expr_output, "\n");
  
  
  clast_pprint_expr(options, clast_expr_output, expr);
  



  /// The processClastLoopBound function is called to process the clast expression and convert it into one or more affine expressions.
  if (failed(processClastLoopBound(expr, builder, boundExprs, options)))
  
    return failure();



  /// If looking at the upper bound, we should add 1 to all of them.
  if (isUpper)
  
    for (auto &expr : boundExprs)
  
      expr = expr + b.getAffineConstantExpr(1);

  // Insert dim operands.
  unsigned numDims = builder.dimNames.size();
  unsigned numSymbols = builder.symbolNames.size();
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
  affMap = AffineMap::get(numDims, numSymbols, boundExprs, context);

  /// F: My snitch
  // Log exit point
  fprintf(clast_expr_output, "\nExiting getAffineLoopBound\n");
  fclose(clast_expr_output);

  return success();

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





/// Generate the AffineForOp from a clast_for statement. First we create
/// AffineMaps for the lower and upper bounds. Then we decide the step if
/// there is any. And finally, we create the AffineForOp instance and generate
/// its body.
/// @brief : translates CLooG AST (i.e. clast_for) representations of loops into MLIR AffineForOp operations. The function Importer::processStmt is 
/// responsible for processing a CLooG for-loop statement (i.e. clast_for) and converting it into an MLIR affine for-loop.
/// @param forStmt : clast_for *forStmt: A pointer to a CLooG AST for-loop statement
/// @return : 
LogicalResult Importer::processStmt(clast_for *forStmt) {


  FILE *clast_for_output = fopen("output-files/5.clast_for.txt", "a");
  
  
  /// Get loop bounds.
  /// Declares variables of type affine maps for the lower bound (lbMap) and upper bound (ubMap) of the loop.
  AffineMap lbMap, ubMap;


  /// The actual size of each cell in memory depends on the size of the mlir::Value type
  /// It allocates a fixed amount of space for a small number of elements and only falls back to heap allocation if that space is exceeded
  llvm::SmallVector<mlir::Value, 8> lbOperands, ubOperands;
  

  /// Ensures that the loop has both a lower bound (LB) and an upper bound (UB). If either bound is missing, it asserts with an error message
  assert((forStmt->LB && forStmt->UB) && "Unbounded loops are not allowed.");


  // TODO: simplify these sanity checks.
  assert(!(forStmt->LB->type == clast_expr_red &&
           reinterpret_cast<clast_reduction *>(forStmt->LB)->type == clast_red_min) &&
         "If the lower bound is a reduced result, it should not use min for reduction.");

  assert(!(forStmt->UB->type == clast_expr_red &&
           reinterpret_cast<clast_reduction *>(forStmt->UB)->type == clast_red_max) &&
         "If the upper bound is a reduced result, it should not use max for reduction.");
    



  /// F: My snitch
  fprintf(clast_for_output, "\nforStmt->LB\n");

  clast_pprint_expr(options, clast_for_output, forStmt->LB);
  

  if (failed(getAffineLoopBound(forStmt->LB, lbOperands, lbMap))) {
    return failure();
  }

  
  fprintf(clast_for_output, "\nfor->UB\n");
  clast_pprint_expr(options, clast_for_output, forStmt->UB);

  if (failed(getAffineLoopBound(forStmt->UB, ubOperands, ubMap, /*isUpper=*/true))) {
    return failure();
  }



  /// getAffineLoopBound to convert the lower bound (LB) and upper bound (UB) expressions into affine maps and their corresponding operands. 
  /// If either conversion fails, it returns a failure.
  // if (failed(getAffineLoopBound(forStmt->LB, lbOperands, lbMap)) ||
  //     failed(getAffineLoopBound(forStmt->UB, ubOperands, ubMap, /*isUpper=*/true)))

  //   return failure();


  /// Initializes the loop stride to 1. If the stride is greater than 1, 
  /// it converts the stride to an int64_t value. If the conversion fails, it returns a failure
  int64_t stride = 1;

  if (cloog_int_gt_si(forStmt->stride, 1)) {

    if (failed(getI64(forStmt->stride, &stride)))

      return failure();

  }



  /// Create the for operation.
  /// Creates an MLIR affine for-loop operation (AffineForOp) using the lower bound operands and map, upper bound operands and map, and the stride. 
  /// The loop's location is set to an unknown location in the context
  mlir::AffineForOp forOp = b.create<mlir::AffineForOp>(UnknownLoc::get(context), lbOperands, lbMap, ubOperands, ubMap, stride);


  // Update the loop IV mapping.
  auto &entryBlock = *forOp.getLoopBody().getBlocks().begin();
  
  // TODO: confirm is there a case that forOp has multiple operands.
  assert(entryBlock.getNumArguments() == 1 && "affine.for should only have one block argument (iv).");



  symTable->setValue(forStmt->iterator, entryBlock.getArgument(0), OslSymbolTable::LoopIV);

  // Symbol table is mutable.
  // TODO: is there a better way to improve this? Not very safe.
  mlir::Value symValue = symbolTable[forStmt->iterator];

  symbolTable[forStmt->iterator] = entryBlock.getArgument(0);




  // Create the loop body
  b.setInsertionPointToStart(&entryBlock);
  
  entryBlock.walk([&](mlir::AffineYieldOp op) { b.setInsertionPoint(op); });
  
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


  fclose(clast_for_output);



  return success();


}






void Importer::initializeSymbol(mlir::Value val, ordered_json &j) {

  assert(val != nullptr);

  /// Pointer points to valueTable where we have mapping of mlir value to symolic name 
  OslScop::ValueTable *oslValueTable = scop->getValueTable();


  /// func: is an instance of mlir::FuncOp, which represents a function in MLIR.
  /// getBody(): is a method of FuncOp that returns the body of the function. 
  /// The body of a function in MLIR is typically represented as a list of blocks (Block), 
  /// where each block contains a sequence of operations (Operation).
  /// The getBody() method returns a Region. A Region in MLIR is a container for a list of blocks.
  /// begin() is a method of Region that returns an iterator to the first block in the region.
  /// The * operator is used to dereference the iterator, yielding a reference to the first block in the region.
  /// The & symbol indicates that entryBlock is a reference to the block.
  auto &entryBlock = *func.getBody().begin();


  /// Save and restore the insertion point of the OpBuilder
  OpBuilder::InsertionGuard guard(b);

  /// Look up for mlir value in valueTable and collect the corresponding symbolic name of that mlir::Value
  std::string symbol = oslValueTable->lookup(val);



  /// F: push all symbolic name of corresponding mlir::Value
  j["symbol"].push_back(symbol);


  assert(!symbol.empty() && "val to initialize should have a corresponding "
                            "symbol in the original code.");



  /// Symbols that are the block arguments won't be taken care of at this stage.
  /// initializeFuncOpInterface() should already have done that.
  /// it checks if val is something like this `<block argument> of type 'index' at index: 0`
  if (mlir::BlockArgument arg = val.dyn_cast<mlir::BlockArgument>())
    return;


  // This defOp should be cloned to the target function, while its operands
  // may be symbols that are not yet initialized (e.g., IVs in loops not
  // constructed). We should place them into the symbolToDeps map.
  mlir::Operation *defOp = val.getDefiningOp();

  std::string defOpStr = operationToString(defOp);

  j["defOp"].push_back(defOpStr);

  /// if val is something like this: `%2 = memref.alloc() : memref<64x64xf32>`
  /// To verify if defOp is of a specific type, in this case, memref::AllocaOp.
  /// isa<T>(x) is an LLVM utility function used to check if x is an instance of type T or x is of type T
  /// defOp must have zero operands: memory allocation operation does not take any operands
  /// Eg: %0 = memref.alloc() : memref<64x64xf32> This memref.alloc() operation has zero operands because 
  /// it does not require any inputs to allocate memory.
  if (isa<memref::AllocaOp>(defOp) && defOp->getNumOperands() == 0) {

    b.setInsertionPointToStart(&entryBlock);

    /// b.clone(*defOp): Create a copy (clone) of the operation defOp
    /// ->getResult(0):  accesses the first result of the newly cloned operation
    /// while cloning it might get change of register number
    /// Like, %0 = memref.alloc() : memref<64x64xf32> might become %2 = memref.alloc() : memref<64x64xf32>
    symbolTable[symbol] = b.clone(*defOp)->getResult(0);

    return;

  }

  // This indicates whether we have set an insertion point.
  bool hasInsertionPoint = false;



  /// First we examine the AST structure.
  /// parentOp is the parent operation where defOp is present
  /// For example; defOp might be `%0 = memref.alloc() : memref<64x64xf32>` 
  /// Here, this operation is allocated that place or block is parentOp
  mlir::Operation *parentOp = defOp->getParentOp();

  /// F: Convert Value to string
  std::string parentOpStr = operationToString(parentOp);

  /// F: push to json
  j["parentOp"].push_back(parentOpStr);


  /// dyn_cast<mlir::AffineForOp>(parentOp) attempts to cast parentOp to mlir::AffineForOp
  /// if parentOp is not AffineForOp then this part will not work
  if (mlir::AffineForOp forOp = dyn_cast<mlir::AffineForOp>(parentOp)) {


    /// forOp.getInductionVar() gets the induction variable of the affine for-loop
    mlir::Value srcIV = forOp.getInductionVar();


    /// F: Convert Value to string
    std::string srcIVStr = valueToString(srcIV);
    
    /// F: push to json
    j["srcIV"].push_back(srcIVStr);


    /// Retrieves the symbolic name associated with the induction variable
    std::string ivName = oslValueTable->lookup(srcIV);
    

    /// F: push to json
    j["ivName"].push_back(ivName);


    /// checks if the induction variable has been mapped in the symbolTable
    mlir::Value dstIV = symbolTable[ivName];


    /// F: Convert Value to string
    std::string dstIVStr = valueToString(dstIV);
    
    /// F: push to json
    j["dstIVStr"].push_back(dstIVStr);


    if (dstIV == nullptr) {

      symbolToDeps[ivName].insert(val);
      valueToDepSymbols[val].insert(ivName);

    } else {

      // Now the loop IV is there, we just find its owner for loop and clone
      // the op.
      mlir::Block *blockToInsert = dstIV.cast<mlir::BlockArgument>().getOwner();

      hasInsertionPoint = true;

      b.setInsertionPointToStart(blockToInsert);
    }
  
  } 
  
  /// dyn_cast<mlir::FuncOp>(parentOp) attempts to cast parentOp to mlir::FuncOp
  /// if parentOp is FuncOp then this part will work. If the cast is successful, it means parentOp is a function
  else if (mlir::FuncOp funOp = dyn_cast<mlir::FuncOp>(parentOp)) {

    /// Insert at the beginning of this function.
    hasInsertionPoint = true;

    b.setInsertionPointToStart(&entryBlock);


  } 
  
  /// Handle Unsupported Parent Operation
  else {

    assert(false);

  }


  /// Take a small vector for storing new operands
  /// Here, it is used to store up to 8 operands
  SmallVector<mlir::Value, 8> newOperands;

  /// Next, we check whether all operands are in the symbol table
  /// Loop through each operand of the defining operation (defOp)
  for (mlir::Value operand : defOp->getOperands()) {

    /// F: Convert Value to string
    std::string operandStr = valueToString(operand);
    
    /// F: push to json
    j["operandStr"].push_back(operandStr);

    std::string operandSymbol = oslValueTable->lookup(operand);

    /// F: push to json
    j["operandSymbol"].push_back(operandSymbol);


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
  if (newOperands.size() < defOp->getNumOperands())
  
    return;

  // Finally do the initialization.
  if (!hasInsertionPoint)
  
    return;



  BlockAndValueMapping vMap;
  
  /// Create a mapping from the original operands to the new operands
  for (unsigned i = 0; i < newOperands.size(); i++)
  
    vMap.map(defOp->getOperand(i), newOperands[i]);

  

  mlir::Operation *newOp = b.clone(*defOp, vMap);
  
  assert(newOp != nullptr);
  
  assert(newOp->getNumResults() == 1 && "Should only have one result.");

  symbolTable[symbol] = newOp->getResult(0);






  /// F: Open an ofstream to write to the file "data.json"
  std::ofstream o("data.json");
  
  /// F:
  if (!o.is_open()) {
      
      /// F:
      std::cerr << "Failed to open file for writing.\n";
    
  }

  /// F: Write formatted JSON data to the file
  o << j.dump(4); // The argument '4' makes the JSON output pretty-printed with an indentation of 4 spaces

  /// F: Close the file stream
  o.close();



}




void Importer::initializeSymbolTable() {

  OslScop::SymbolTable *oslSymbolTable = scop->getSymbolTable();

  OpBuilder::InsertionGuard guard(b);

  auto &entryBlock = *func.getBody().begin();

  b.setInsertionPointToStart(&entryBlock);

  /// Constants
  symbolTable["zero"] = b.create<mlir::arith::ConstantOp>(b.getUnknownLoc(), b.getIndexType(), b.getIntegerAttr(b.getIndexType(), 0));

  // Check that the symbol was added
  if (symbolTable.find("zero") != symbolTable.end()) {

    std::cout << "Symbol 'zero' was successfully added to the symbolTable." << std::endl;
  
  } else {
    std::cerr << "Failed to add 'zero' to the symbolTable." << std::endl;
  }

  /// F:
  ordered_json& initJson = j["initializeSymbolTable()"];
  ordered_json& oslSymbolJson = initJson["*oslSymbolTable"];
  
  /// F: Iterate over the ValueTable
  // Iterate over the OSL symbol table and populate the JSON object
  for (const auto& entry : *oslSymbolTable) {

    /// F: Insert into JSON object, using an array to store multiple values
    oslSymbolJson[entry.first().str()].push_back(valueToString(entry.second));
  
  }

  ordered_json &symbolTableJson1 = initJson["symbolTableInsideinitializeSymbolTable()"];

  /// F: Print the final contents of the symbolTable
  for (const auto& entry : symbolTable) {

    /// F: Insert into JSON object, using an array to store multiple values
    symbolTableJson1[entry.first().str()].push_back(valueToString(entry.second));

  }

  
  for (const auto &it : *oslSymbolTable)

    initializeSymbol(it.second, initJson["initializeSymbol()"]);


  ordered_json &symbolTableJson2 = initJson["symbolTableAfterinitializeSymbol()call"];

  /// F: Print the final contents of the symbolTable
  for (const auto& entry : symbolTable) {

    /// F: Insert into JSON object, using an array to store multiple values
    symbolTableJson2[entry.first().str()].push_back(valueToString(entry.second));

  }



  // F: Open an ofstream to write to the file "data.json"
  std::ofstream o("data.json");
  
  /// F:
  if (!o.is_open()) {
      
      /// F:
      std::cerr << "Failed to open file for writing.\n";
    
  }

  /// F: Write formatted JSON data to the file
  o << j.dump(4); // The argument '4' makes the JSON output pretty-printed with an indentation of 4 spaces

  /// F: Close the file stream
  o.close();





}



/// If there is anything in the comment, we will use it as a function name.
/// Otherwise, we return an empty string.
std::string Importer::getSourceFuncName(ordered_json &j) const {

  osl_generic_p comment = scop->getExtension("comment");

  /// F:
  FILE *scop_getextension_getSourceFuncName = fopen("scop_getextension_getSourceFuncName.txt", "w");
  osl_generic_idump(scop_getextension_getSourceFuncName, comment, 4);
  /// end of my snippet

  if (comment) {

    char *commentStr = reinterpret_cast<osl_comment_p>(comment->data)->comment;

    /// F:
    j["commentStr"] = commentStr;

    return std::string(commentStr);

  }


  /// F: Open an ofstream to write to the file "data.json"
  std::ofstream o("data.json");
  
  /// F:
  if (!o.is_open()) {
      
      /// F:
      std::cerr << "Failed to open file for writing.\n";
    
  }

  /// F: Write formatted JSON data to the file
  o << j.dump(4); // The argument '4' makes the JSON output pretty-printed with an indentation of 4 spaces

  /// F: Close the file stream
  o.close();

  return std::string("");

}





mlir::FuncOp Importer::getSourceFuncOp(ordered_json &j) {

  std::string sourceFuncName = getSourceFuncName(j["getSourceFuncName()"]);
  
  mlir::Operation *sourceFuncOp = module.lookupSymbol(sourceFuncName);

  
  /// F: Convert operation to string and store in JSON
  j["sourceFuncOp"] = operationToString(sourceFuncOp);

  /// F: Open an ofstream to write to the file "data.json"
  std::ofstream o("data.json");
  
  /// F:
  if (!o.is_open()) {
      
      /// F:
      std::cerr << "Failed to open file for writing.\n";
    
  }

  /// F: Write formatted JSON data to the file
  o << j.dump(4); // The argument '4' makes the JSON output pretty-printed with an indentation of 4 spaces

  /// F: Close the file stream
  o.close();
  

  assert(sourceFuncOp != nullptr && "sourceFuncName cannot be found in the module");
  
  assert(isa<mlir::FuncOp>(sourceFuncOp) && "Found sourceFuncOp should be of type mlir::FuncOp.");

  return cast<mlir::FuncOp>(sourceFuncOp);

}






/// Initialize FuncOpInterface
void Importer::initializeFuncOpInterface() {

  /// Retrieve mapping of mlir value to symbolic names out of valueTable
  /// And this method getValueTable() is written under OslScop class in this format 
  /// `using ValueTable = llvm::DenseMap<mlir::Value, std::string>;` here ValueTable is acting as alias 
  /// for the type `llvm::DenseMap<mlir::Value, std::string>`
  OslScop::ValueTable *oslValueTable = scop->getValueTable();
  

  
  /// F: Populate the oslValueTable section first
  ordered_json& initJson = j["initializeFuncOpInterface()"];
  /// F:
  ordered_json& oslValueJson = initJson["oslValueTable"];

  /// F: Iterate over the ValueTable
  for (const auto& entry : *oslValueTable) {
    mlir::Value key = entry.first;
    std::string value = entry.second;

    std::string keyStr = valueToString(key);

    /// std::cout << "Key: " << keyStr << ", Value: " << value << std::endl;

    /// Insert into JSON object, using an array to store multiple values
    oslValueJson[keyStr].push_back(value);
  }

  /// F: Call getSourceFuncOp() and pass the JSON object by reference
  mlir::FuncOp sourceFuncOp = getSourceFuncOp(initJson["getSourceFuncOp()"]);

  /// OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(module.getBody(), getFuncInsertPt());

  /// The default function name is main.
  std::string funcName("main");

  /// F: If the comment is provided, we will use it as the function name.
  std::string sourceFuncName = getSourceFuncName(initJson["getSourceFuncName()"]);

  if (!sourceFuncName.empty()) {
    funcName = std::string(formatv("{0}_opt", sourceFuncName));
  }

  /// Create the function interface.
  func = b.create<FuncOp>(sourceFuncOp.getLoc(), funcName, sourceFuncOp.getType());

  /// F: Add remaining fields in the desired order
  initJson["funcName"] = funcName;
  initJson["sourceFuncOp location"] = locationToString(sourceFuncOp.getLoc());
  initJson["sourceFuncOp func return type"] = typeToString(sourceFuncOp.getType());

  /// Initialize the symbol table for these entryBlock arguments
  auto& entryBlock = *func.addEntryBlock();

  b.setInsertionPointToStart(&entryBlock);
  b.create<mlir::func::ReturnOp>(UnknownLoc::get(context));
  b.setInsertionPointToStart(&entryBlock);


  /// F:
  ordered_json& argsJson = initJson["arguments"];


  /// This part handles block arguments or function parameters
  for (unsigned i = 0; i < entryBlock.getNumArguments(); i++) {
    
    std::string argSymbol = oslValueTable->lookup(sourceFuncOp.getArgument(i));
    mlir::Value arg = entryBlock.getArgument(i);

    /// F: Use a unique key for each argument
    argsJson[i]["argSymbol"] = argSymbol;
    argsJson[i]["arg"] = valueToString(arg);

    /// If the source type is not index, cast it to index then.
    if (scop->isParameterSymbol(argSymbol) && arg.getType() != b.getIndexType()) {
      
      mlir::Operation* op = b.create<mlir::arith::IndexCastOp>(sourceFuncOp.getLoc(), b.getIndexType(), arg);
      symbolTable[argSymbol] = op->getResult(0);
    
    } else {
    
      symbolTable[argSymbol] = arg;
    
    }
  
  }

  
  /// F: Open an ofstream to write to the file "data.json"
  std::ofstream o("data.json");

  /// F: 
  if (!o.is_open()) {
  
    std::cerr << "Failed to open file for writing.\n";
  
  /// F: 
  } else {
    
    // Write formatted JSON data to the file
    o << j.dump(4); // The argument '4' makes the JSON output pretty-printed with an indentation of 4 spaces
    
    // Close the file stream
    o.close();
    
    // Output success message
    std::cout << "Data has been dumped to data.json successfully.\n";
  
  }

}






/// Translate the root statement as a function. The name of the function is by
/// default "main".
LogicalResult Importer::processStmt(clast_root *rootStmt) {
  
  /// F: My snitch
  printf("[processStmt clast_root *rootstmt] I AM HIT=====================\n");

  trace["processStmt(clast_root *rootStmt)"] = "Start of the rootStmt for creating the funcOp box";

  // Create the function.
  initializeFuncOpInterface();
  
  // Initialize several values before start.
  initializeSymbolTable();

  return success();

}





LogicalResult Importer::processStmtList(clast_stmt *s) {

  /// Declare a file pointer and also some necessary variables
  FILE *rootStmt_dump = fopen("output-files/rootStmt.txt", "w+");
  char file_contents[1000];
  


  /// Check if the file is opened successfully
  if (!rootStmt_dump) {

    std::cerr << "Failed to open file for writing.\n";
    return failure();
  
  }



  /// Loop through each statement in the linked list until the end (NULL)
  for (; s; s = s->next) {
    
    /// Check if the current statement is of type 'stmt_root'
    if (CLAST_STMT_IS_A(s, stmt_root)) {
      
        /// Dump the clast in a file
        clast_pprint(rootStmt_dump, s, 0, options);


        /// Close the file after writing
        fclose(rootStmt_dump);

        /// Reopen the file in read mode
        rootStmt_dump = fopen("output-files/rootStmt.txt", "r");

        /// Read the contents of file and print in console
        printf("Clast is printing from stmt_root\n");
        if (rootStmt_dump != NULL) {

          printf("Counter: %d\n", counter);
          while(fgets(file_contents, 1000, rootStmt_dump)) 

            printf("%s", file_contents);

          /// Clearing the array 
          file_contents[0] = '\0';
          counter++;

        }
        printf("Clast printing from stmt_root done==============\n");
        /// Process the statement and check for failure
        if (failed(processStmt(reinterpret_cast<clast_root *>(s))))

          // Return a failure result if processing failed
          return failure();



    } else if (CLAST_STMT_IS_A(s, stmt_for)) {


        /// Dump the clast in a file
        clast_pprint(rootStmt_dump, s, 0, options);


        /// Close the file after writing
        fclose(rootStmt_dump);

        /// Reopen the file in read mode
        rootStmt_dump = fopen("output-files/rootStmt.txt", "r");

        /// Read the contents of file and print in console
        printf("Clast is printing from stmt_for\n");
        if (rootStmt_dump != NULL) {

          printf("Counter: %d\n", counter);
          while(fgets(file_contents, 1000, rootStmt_dump)) 

            printf("%s", file_contents);

          file_contents[0] = '\0';
          counter++;

        }
        printf("Clast printing from stmt_for done==============\n");

        // Same process for a 'stmt_for' (a for loop statement)
        if (failed(processStmt(reinterpret_cast<clast_for *>(s))))

          return failure();
      
    
    } else if (CLAST_STMT_IS_A(s, stmt_user)) {

        /// Dump the clast in a file
        clast_pprint(rootStmt_dump, s, 0, options);


        /// Close the file after writing
        fclose(rootStmt_dump);

        /// Reopen the file in read mode
        rootStmt_dump = fopen("output-files/rootStmt.txt", "r");

        /// Read the contents of file and print in console
        printf("Clast is printing from stmt_user\n");
        if (rootStmt_dump != NULL) {

          printf("Counter: %d\n", counter);
          while(fgets(file_contents, 1000, rootStmt_dump)) 

            printf("%s", file_contents);

          file_contents[0] = '\0';
          counter++;

        }
        printf("Clast printing from stmt_user done==============\n");

        // Same process for a statement of type 'stmt_user'
        if (failed(processStmt(reinterpret_cast<clast_user_stmt *>(s))))
    
          return failure();

  
    } else {

        // If the statement is not recognized, assert failure (crash)
        assert(false && "clast_stmt type not supported");
    
    }
  
  } // for ends here

  fclose(rootStmt_dump);

  // If all statements were processed successfully, return a success result
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

    // Collect all statements within the current parallel loop.
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


  FILE *CloogOut = fopen("output-files/1. scop_to_cloog.cloog", "w");
  FILE *ProgramOut = fopen("output-files/2. cloog_to_program.txt", "w");
  FILE *ClastOut = fopen("output-files/3. program_to_clast.txt", "w");
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