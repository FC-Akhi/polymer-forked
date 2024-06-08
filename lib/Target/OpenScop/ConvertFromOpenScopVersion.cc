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


#include "polymer/Support/nlohmann/json.hpp"

using ordered_json = nlohmann::ordered_json;

using namespace polymer;
using namespace mlir;

/// My snippet
// Create a JSON global object
ordered_json j;


typedef llvm::StringMap<mlir::Operation *> StmtOpMap;
typedef llvm::StringMap<mlir::Value> NameValueMap;
typedef llvm::StringMap<std::string> IterScatNameMap;

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

  LogicalResult process(clast_name *expr,
                        llvm::SmallVectorImpl<AffineExpr> &affExprs);
  LogicalResult process(clast_term *expr,
                        llvm::SmallVectorImpl<AffineExpr> &affExprs);
  LogicalResult process(clast_binary *expr,
                        llvm::SmallVectorImpl<AffineExpr> &affExprs);
  LogicalResult process(clast_reduction *expr,
                        llvm::SmallVectorImpl<AffineExpr> &affExprs);

  LogicalResult
  processSumReduction(clast_reduction *expr,
                      llvm::SmallVectorImpl<AffineExpr> &affExprs);
  LogicalResult
  processMinOrMaxReduction(clast_reduction *expr,
                           llvm::SmallVectorImpl<AffineExpr> &affExprs);

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

LogicalResult
AffineExprBuilder::process(clast_expr *expr,
                           llvm::SmallVectorImpl<AffineExpr> &affExprs) {

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
LogicalResult
AffineExprBuilder::process(clast_name *expr,
                           llvm::SmallVectorImpl<AffineExpr> &affExprs) {
  if (scop->isSymbol(expr->name)) {
    if (symbolNames.find(expr->name) != symbolNames.end())
      affExprs.push_back(b.getAffineSymbolExpr(symbolNames[expr->name]));
    else {
      affExprs.push_back(b.getAffineSymbolExpr(symbolNames.size()));
      size_t numSymbols = symbolNames.size();
      symbolNames[expr->name] = numSymbols;

      Value v = symbolTable->lookup(expr->name);
      valueMap[v] = expr->name;
    }
  } else if (mlir::Value iv = symbolTable->lookup(expr->name)) {
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

LogicalResult
AffineExprBuilder::process(clast_term *expr,
                           llvm::SmallVectorImpl<AffineExpr> &affExprs) {
  // First get the I64 representation of a cloog int.
  int64_t constant;
  if (failed(getI64(expr->val, &constant)))
    return failure();

  // Next create a constant AffineExpr.
  AffineExpr affExpr = b.getAffineConstantExpr(constant);

  // If var is not NULL, it means this term is var * val. We should create the
  // expr that denotes var and multiplies it with the AffineExpr for val.
  if (expr->var) {
    SmallVector<AffineExpr, 1> varAffExprs;
    if (failed(process(expr->var, varAffExprs)))
      return failure();
    assert(varAffExprs.size() == 1 &&
           "There should be a single expression that stands for the var expr.");

    affExpr = affExpr * varAffExprs[0];
  }

  affExprs.push_back(affExpr);

  return success();
}

LogicalResult
AffineExprBuilder::process(clast_binary *expr,
                           llvm::SmallVectorImpl<AffineExpr> &affExprs) {
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

LogicalResult
AffineExprBuilder::process(clast_reduction *expr,
                           llvm::SmallVectorImpl<AffineExpr> &affExprs) {
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

LogicalResult AffineExprBuilder::processSumReduction(
    clast_reduction *expr, llvm::SmallVectorImpl<AffineExpr> &affExprs) {
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

LogicalResult AffineExprBuilder::processMinOrMaxReduction(
    clast_reduction *expr, llvm::SmallVectorImpl<AffineExpr> &affExprs) {
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
  void visit(clast_guard *guardStmt);
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
    } else if (CLAST_STMT_IS_A(s, stmt_guard)) {
      visit(reinterpret_cast<clast_guard *>(s));
    }
  }
}

void IterScatNameMapper::visit(clast_for *forStmt) {
  visitStmtList(forStmt->body);
}
void IterScatNameMapper::visit(clast_guard *guardStmt) {
  visitStmtList(guardStmt->then);
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






namespace {

  /// Import MLIR code from clast AST
  class Importer {

    public:
            
      /// @brief Importer class constructor
      /// @param context 
      /// @param module 
      /// @param symTable 
      /// @param scop 
      /// @param options 
      Importer(MLIRContext *context, ModuleOp module, OSLSymbolTable *symTable, OslScop *scop, CloogOptions *options);


      /// @brief Main entry module
      /// @param s 
      /// @return 
      LogicalResult processStmtList(clast_stmt *s);


      /// @brief 
      /// @return 
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

      /// Map from an not yet initialized symbol to the Values that depend on it.
      llvm::StringMap<llvm::SetVector<mlir::Value>> symbolToDeps;
            
      /// Map from a value to all the symbols it depends on.
      llvm::DenseMap<mlir::Value, llvm::SetVector<llvm::StringRef>> valueToDepSymbols;

      IterScatNameMap iterScatNameMap;

      llvm::StringMap<clast_stmt *> lhsToAss;

      CloogOptions *options;



      /// They are unknowns to me now
      void initializeSymbolTable();
      void initializeSymbol(mlir::Value val);
      void initializeFuncOpInterface();
            

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

            
      /// @brief process c like syntax each type of statements
      /// @param rootStmt 
      /// @return 
      LogicalResult processStmt(clast_root *rootStmt);
      LogicalResult processStmt(clast_for *forStmt);
      LogicalResult processStmt(clast_guard *guardStmt);
      LogicalResult processStmt(clast_user *userStmt);
      LogicalResult processStmt(clast_assignment *ass);

  };

}


Importer::Importer(MLIRContext *context, ModuleOp module, OslSymbolTable *symTable, OslScop *scop, CloogOptions *options)
            : b(context), context(context), module(module), symTable(symTable), scop(scop), options(options) {

  b.setInsertionPointToStart(module.getBody());

}



/// Initialize the value in the symbol table
void Importer::initializeSymbol(mlir::Value val) {

  assert(val != nullptr);

  OslScop::ValueTable *oslValueTable = scop->getValueTable();

  auto &entryBlock = *func.getBody().begin();

  OpBuilder::InsertionGuard guard(b);

  std::string symbol = oslValueTable->lookup(val);

  assert(!symbol.empty() && "val to initialize should have a corresponding "
                            "symbol in the original code");


  /// Symbols that are the block arguments won't be taken care of at this stage
  /// initializeFuncOpInterface() should already have done that
  if (mlir::BlockArgument arg = val.dyn_cast<mlir::BlockArgument>())
    return;


  /// This defOp should be cloned to the target function, while its operands
  /// may be symbols that are not yet initialized (e.g, IVs in the loops not constructed)
  /// We should place them into the symbolToDeps map
  mlir::Operation *defOp = val.getDefiningOp();

  if (isa<memref::AllocaOp>(defOp) && defOp->getNumOperands() == 0) {

    b.setInsertaionPointToStart(&entryBlock);

    symbolTable[symbol] = b.clone(*defOp)->getResult(0);

    return;

  }

  /// This indicates whether we have set an insertion point
  bool hasInsertionPoint = false;

  /// First we examine the AST structure
  mlir::Operation *parentOp = defOp->getParentOp();


  if (mlir::AffineForOp forOp = dyn_cast<mlir::AffineForOp>(parentOp)) {

    mlir::Value srcIV = forOp.getInductionVar();

    std::string ivName = oslValueTable->loopup(srcIV);

    mlir::Value dstIV = symbolTable[ivName];

    if (dstIV == nullptr) {

      symbolToDeps[ivName].insert(val);
      valueToDepSymbols[val].insert(ivName);

    }

    else {

      // Now the loop IV is there, we just find its owner for loop and clone
      // the op.
      mlir::Block *blockToInsert = dstIV.cast<mlir::BlockArgument>().getOwner();

      hasInsertionPoint = true;

      b.setInsertionPointToStart(blockToInsert);

    }
  
  }

  else if (mlir::FuncOp funcOp = dyn_cast<mlir::FuncOp>(parentOp)) {

    /// Insert at the beginning of this function
    hasInsertionPoint = true;

    b.setInsertionPointToStart(&entryBlock);

  }

  else {

    assert(false);

  }

  SmallVector<mlir::Value, 8> newOperands;

  /// Next, we check whether all operands are in the symbol table
  for (mlir::Value operand : defOp->getOperands()) {

    std::string operandSymbol = oslValueTable->lookup(operand);

    if (operandSymbol.empty()) {

      mlir::Operation *operandDefOp = operand.getDefiningOp();

      if (operandDefOp && isa<mlir::arith::ConstantOp>(operandDefOp)) {

        newOperands.push_back(b.clone(*operandDefOp)->getResult(0));

        continue;

      }

    }

    assert(!operandSymbol.empty() && "operand should be in the original symbol table");

    mlir::Value newOperand = symbolTable[operandSymbol];

    /// If the symbol is not yet initialized, we update the two dependence
    /// tables. Note that here we implicitly assume that the operand symbol
    /// should exist
    assert(newOperand != nullptr);

    newOperands.push_back(newOperand);


  }

  /// The operands are not sufficient, should wait
  if (newOperand.size() < defOp->getNumOperands()) 

    return;

  /// Finally do the initialization
  if (!hasInsertionPoint)

    return;

  BlockAndValueMapping vMap;

  for (unsigned i = 0; i < newOperands.size(); i++)

    vMap.map(defOp->getOperand(i), newOperands[i]);


  mlir::Operation *newOp = b.clone(*defOp, vMap);

  assert(newOp != nullptr);

  assert(newOp->getNumResults() == 1 && "Should only have one result");


  symbolTable[symbol] = newOp->getResult(0);


}






void Importer::initializeSymbolTable() {


  /// TODO need to understand
  OslScop::SymbolTable *oslSymbolTable = scop->getSymbolTable();

  /// My snippet
  /// Iterate over the ValueTable
  for (const auto &entry : *oslSymbolTable) {
      // entry.first gives the key (mlir::Value)
      mlir::Value key = entry.first;

      // entry.second gives the value (std::string)
      std::string value = entry.second;

      // Convert key to string
      std::string keyStr = valueToString(key);

      // Print the key and value
      // std::cout << "Key: " << keyStr << ", Value: " << value << std::endl;

      // Insert into JSON object, using an array to store multiple values
      j[keyStr].push_back(value);

      
  }
  /// end my snippet


  Opbuilder::InsertionGuard guard(b);

  auto &entryBlock = *func.getBody().begin();

  b.setInsertionPointToStart(&entryBlock);

  /// Constants
  symbolTable["zero"] = b.create<mlir::arith::ConstantOp>(b.getUnknownLoc(), b.getIndexType(), b.getIntegerAttr(b.getIndexType(), 0));

  for (const auto &it : *oslSymbolTable)

    initializeSymbol(it.second);

}



/// My snippet
/// Function to print mlir::Value to string
std::string typeToString(mlir::Type type) {
  
  std::string str;
  llvm::raw_string_ostream os(str);
  type.print(os);
  return os.str();

}

/// Function to convert mlir::Location to std::string
std::string locationToString(mlir::Location loc) {
  
  std::string locStr;
  llvm::raw_string_ostream locStream(locStr);
  loc.print(locStream);
  return locStream.str();

}


/// Function to print mlir::Value to string
std::string valueToString(mlir::Value value) {
    
  std::string str;
  llvm::raw_string_ostream os(str);
  value.print(os);
  return os.str();

}

/// Function to print mlir::Operation to string
std::string operationToString(mlir::Operation* op) {
  
  std::string str;
  llvm::raw_string_ostream os(str);
  op->print(os);
  return os.str();

}



/// @brief If there is anything in the comment, we will use it as a function name. Otherwise, we return an empty string
/// @return commentStr or blank string
/// If there is anything in the comment, we will use it as a function name.
/// Otherwise, we return an empty string.
std::string Importer::getSourceFuncName(ordered_json &j) const {

  osl_generic_p comment = scop->getExtension("comment");

  /// My snippet
  FILE *scop_getextension_getSourceFuncName = fopen("scop_getextension_getSourceFuncName.txt", "w");
  osl_generic_idump(scop_getextension_getSourceFuncName, comment, 4);
  /// end of my snippet

  if (comment) {

    char *commentStr = reinterpret_cast<osl_comment_p>(comment->data)->comment;

    /// My snippet
    j["commentStr"] = commentStr;

    return std::string(commentStr);

  }

  /// My snippet
  // Open an ofstream to write to the file "data.json"
  std::ofstream o("data.json");
  
  if (!o.is_open()) {
      
      std::cerr << "Failed to open file for writing.\n";
    
  }

  // Write formatted JSON data to the file
  o << j.dump(4); // The argument '4' makes the JSON output pretty-printed with an indentation of 4 spaces

  // Close the file stream
  o.close();

  return std::string("");

}




mlir::FuncOp Importer::getSourceFuncOp(ordered_json &j) {

  std::string sourceFuncName = getSourceFuncName(j["getSourceFuncName()"]);
  
  mlir::Operation *sourceFuncOp = module.lookupSymbol(sourceFuncName);

  /// My snippet
  /// Convert operation to string and store in JSON
  j["sourceFuncOp"] = operationToString(sourceFuncOp);

  /// Open an ofstream to write to the file "data.json"
  std::ofstream o("data.json");
  
  if (!o.is_open()) {
      
      std::cerr << "Failed to open file for writing.\n";
    
  }

  /// Write formatted JSON data to the file
  o << j.dump(4); // The argument '4' makes the JSON output pretty-printed with an indentation of 4 spaces

  /// Close the file stream
  o.close();
  /// end my snippet

  assert(sourceFuncOp != nullptr && "sourceFuncName cannot be found in the module");
  
  assert(isa<mlir::FuncOp>(sourceFuncOp) && "Found sourceFuncOp should be of type mlir::FuncOp.");

  return cast<mlir::FuncOp>(sourceFuncOp);

}



/// Initialize FuncOpInterface
void Importer::initializeFuncOpInterface() {

  /// Retrieve the value table from the scop (Static Control Part), usually a data structure used in polyhedral models
  OslScop::ValueTable *oslValueTable = scop->getValueTable();
  

  /// My snippet
  /// Populate the oslValueTable section first
  ordered_json& initJson = j["initializeFuncOpInterface()"];
  ordered_json& oslValueJson = initJson["oslValueTable"];

  /// Iterate over the ValueTable
  for (const auto& entry : *oslValueTable) {
    mlir::Value key = entry.first;
    std::string value = entry.second;

    std::string keyStr = valueToString(key);

    std::cout << "Key: " << keyStr << ", Value: " << value << std::endl;

    /// Insert into JSON object, using an array to store multiple values
    oslValueJson[keyStr].push_back(value);
  }

  /// Call getSourceFuncOp() and pass the JSON object by reference
  mlir::FuncOp sourceFuncOp = getSourceFuncOp(initJson["getSourceFuncOp()"]);

  /// OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(module.getBody(), getFuncInsertPt());

  /// The default function name is main.
  std::string funcName("main");

  /// If the comment is provided, we will use it as the function name.
  std::string sourceFuncName = getSourceFuncName(initJson["getSourceFuncName()"]);

  if (!sourceFuncName.empty()) {
    funcName = std::string(formatv("{0}_opt", sourceFuncName));
  }

  /// Create the function interface.
  func = b.create<FuncOp>(sourceFuncOp.getLoc(), funcName, sourceFuncOp.getType());

  /// Add remaining fields in the desired order
  initJson["funcName"] = funcName;
  initJson["sourceFuncOp location"] = locationToString(sourceFuncOp.getLoc());
  initJson["sourceFuncOp func return type"] = typeToString(sourceFuncOp.getType());

  /// Initialize the symbol table for these entryBlock arguments
  auto& entryBlock = *func.addEntryBlock();

  b.setInsertionPointToStart(&entryBlock);
  b.create<mlir::func::ReturnOp>(UnknownLoc::get(context));
  b.setInsertionPointToStart(&entryBlock);

  ordered_json& argsJson = initJson["arguments"];

  for (unsigned i = 0; i < entryBlock.getNumArguments(); i++) {
    std::string argSymbol = oslValueTable->lookup(sourceFuncOp.getArgument(i));
    mlir::Value arg = entryBlock.getArgument(i);

    /// Use a unique key for each argument
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

  /// My snippet
  /// Open an ofstream to write to the file "data.json"
  std::ofstream o("data.json");

  if (!o.is_open()) {
  
    std::cerr << "Failed to open file for writing.\n";
  
  } else {
    
    // Write formatted JSON data to the file
    o << j.dump(4); // The argument '4' makes the JSON output pretty-printed with an indentation of 4 spaces
    
    // Close the file stream
    o.close();
    
    // Output success message
    std::cout << "Data has been dumped to data.json successfully.\n";
  
  }

}




/// @brief Translate the root statement as a function. The name of the function is by default "main"
/// @param rootStmt 
/// @return success
LogicalResult Importer::processStmt(clast_root *rootStmt) {

  /// Create the function
  initializeFuncOpInterface();


  /// Initialize several values before start
  initializeSymbolTable();


  return success();

}







/// @brief Main entry module
/// @param s 
/// @return 
LogicalResult Importer::processStmtList(clast_stmt *s) {


  for (; s; s = s->next) {

    if (CLAST_STMT_IS_A(s, stmt_root)) {

      if (failed(processStmt(reinterpret_cast<clast_root *>(s))))

        return failure();

    }

    else {

      // If the statement is not recognized, assert failure (crash)
      assert(false && "clast_stmt type not supported");
        
    }



  } /// for ends here


  // If all statements were processed successfully, return a success result
  return success();

}




















mlir::Operation *polymer::createFuncOpFromOpenScop(std::unique_ptr<OslScop> scop, ModuleOp module, OslSymbolTable &symTable,
                                                  MLIRContext *context, PlutoProg *prog, const char *dumpClastAfterPluto) {
  
  
  FILE *cloogOutFromScop = fopen("cloog_from_scop.cloog", "w");
  FILE *cloogProgram = fopen("program_from_cloog.txt", "w");
  FILE *clastPrintFile = fopen("clast_from_program.txt", "w");
  // FILE *clastPrintFile2 = fopen("clast2.txt", "w");
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
  cloog_input_dump_cloog(cloogOutFromScop, input, options);

  
  
  if (prog != nullptr)
    
    updateCloogOptionsByPlutoProg(options, prog);

  // Create cloog_program
  CloogProgram *program = cloog_program_alloc(input->context, input->ud, options);
  
  assert(program->loop);

  
  program = cloog_program_generate(program, options);



  // +++++++++++++++++++Print Program+++++++++++++++++++++++++++++++++++
  cloog_program_print(cloogProgram, program);

  
  
  
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
  clast_pprint(clastPrintFile, rootStmt, 0, options);



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

  //==========================================JSON TEST======================================
  // Using initializer lists for JSON
  json ex3 = {
      {"happy", true},
      {"pi", 3.141}
  };

  // Create an ofstream for writing and open "data.json"
  std::ofstream o("data.json");
  if (!o.is_open()) {
  
      std::cerr << "Failed to open file for writing.\n";
        
  }

  // Write formatted JSON data to the file
  o << ex3.dump(4);
  o.close();


  // Output to console that writing was successful
  std::cout << "JSON data has been written to 'data.json'\n";


  // Proceed to reading from the file
  std::ifstream i("data.json");
  if (!i.is_open()) {
      
      std::cerr << "Failed to open file for reading.\n";
      
  }

  // Define a new JSON object to store the read data
  json ex3_read;
  i >> ex3_read;
  i.close();

  // Output the read JSON to the console
  std::cout << "Read JSON data: " << ex3_read.dump(4) << std::endl;
  //==============================================================+++++++++++++++++++++++=============================================



  fclose(cloogOutFromScop);
  fclose(cloogProgram);
  fclose(clastPrintFile);
  

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





namespace polymer {

void registerFromOpenScopTranslation() {

  

  TranslateToMLIRRegistration fromLLVM(
  
      "import-scop", [](llvm::SourceMgr &sourceMgr, MLIRContext *context) {  //lambda function: [], C++ allow for anonymous functions inline

        return ::translateOpenScopToModule(sourceMgr, context); // :: is scope resolution operator indicates that func is defined in global namespace
  
      }
  
  );

  


}

} // namespace polymer
