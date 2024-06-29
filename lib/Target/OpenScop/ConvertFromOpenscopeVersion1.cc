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

ordered_json trace;

typedef llvm::StringMap<mlir::Operation *> StmtOpMap;
typedef llvm::StringMap<mlir::Value> NameValueMap;
typedef llvm::StringMap<std::string> IterScatNameMap;

























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




/// @brief Importer class constructor
/// @param context 
/// @param module 
/// @param symTable 
/// @param scop 
/// @param options 
Importer::Importer(MLIRContext *context, ModuleOp module, OslSymbolTable *symTable, OslScop *scop, CloogOptions *options)
                  : b(context), context(context), module(module), scop(scop), symTable(symTable), options(options) {

  b.setInsertionPointToStart(module.getBody());

}




LogicalResult Importer::getAffineLoopBound(clast_expr *expr,
                                           llvm::SmallVectorImpl<mlir::Value> &operands,
                                           AffineMap &affMap, bool isUpper) {



  /// An AffineExprBuilder instance is created to help build affine expressions.
  AffineExprBuilder builder(context, symTable, &symbolTable, scop, options);

  /// A vector boundExprs is initialized to store the resulting affine expressions.
  SmallVector<AffineExpr, 4> boundExprs;

  

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


  FILE *clast_for_output = fopen("output-files/clast_for.txt", "a");

  fprintf(clast_for_output, "\nclast for the for stmt upper bound is dumping\n");
      
  clast_pprint_expr(options, clast_for_output, forStmt->UB);

  fprintf(clast_for_output, "\nclast for the for stmt lower bound is dumping\n");

  clast_pprint_expr(options, clast_for_output, forStmt->LB);
  
  
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



  /// getAffineLoopBound to convert the lower bound (LB) and upper bound (UB) expressions into affine maps and their corresponding operands. 
  /// If either conversion fails, it returns a failure.
  if (failed(getAffineLoopBound(forStmt->LB, lbOperands, lbMap)) ||
      failed(getAffineLoopBound(forStmt->UB, ubOperands, ubMap, /*isUpper=*/true)))

    return failure();


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







/// @brief unrollJam Clast by Pluto prog. This is one of the clast transform. NEED TO SEE
/// @param root 
/// @param prog 
/// @param cloogOptions 
/// @param ufactor 
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


/// @brief markParallel. Its not any transform but marking the parallel in clast. NEED TO SEE
/// @param root 
/// @param prog 
/// @param cloogOptions 
static void markParallel(clast_stmt *root, const PlutoProg *prog, CloogOptions *cloogOptions) {
  
  pluto_mark_parallel(root, prog, cloogOptions);

}



/// @brief Transformation for clast. NEED TO SEE
/// @param root 
/// @param prog 
/// @param cloogOptions 
/// @param plutoOptions 
static void transformClastByPlutoProg(clast_stmt *root, 
                                      const PlutoProg *prog,
                                      CloogOptions *cloogOptions,
                                      PlutoOptions *plutoOptions) {

  if (plutoOptions->unrolljam)

    unrollJamClastByPlutoProg(root, prog, cloogOptions, plutoOptions->ufactor);
  
  if (plutoOptions->parallel)
  
    markParallel(root, prog, cloogOptions);

}



/// @brief Polymer authors wrapper of cloog options updates
/// @param options 
/// @param prog 
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
