//===- PlutoTransform.cc - Transform MLIR code by PLUTO -------------------===//
//
// This file implements the transformation passes on MLIR using PLUTO.
//
//===----------------------------------------------------------------------===//

#include "polymer/Transforms/PlutoTransform.h"
#include "polymer/Support/OslScop.h"
#include "polymer/Support/OslScopStmtOpSet.h"
#include "polymer/Support/OslSymbolTable.h"
#include "polymer/Support/ScopStmt.h"
#include "polymer/Target/OpenScop.h"

#include "pluto/internal/pluto.h"
#include "pluto/osl_pluto.h"
#include "pluto/pluto.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"


#include "polymer/Support/nlohmann/json.hpp"
#include <fstream>
#include <iostream>

using namespace mlir;
using namespace llvm;
using namespace polymer;

#define DEBUG_TYPE "pluto-opt"


using json = nlohmann::json;



/// @brief : This is the namespace where Options for 
/// PassPipelineRegisteration (PassPipelineRegistration provides a global initializer that registers a Pass pipeline builder routine)
/// are declared. All of the options are PLUTO specific
namespace {

  struct PlutoOptPipelineOptions : public mlir::PassPipelineOptions<PlutoOptPipelineOptions> {
    

    Option<std::string> dumpClastAfterPluto{

      *this, 
      "dump-clast-after-pluto",
      llvm::cl::desc("File name for dumping the CLooG AST (clast) after Pluto optimization.")
      
      /// F:
      // llvm::cl::init(false)
    };



    Option<bool> parallelize{
      
      *this, 
      "parallelize",
      llvm::cl::desc("Enable parallelization from Pluto."),
      llvm::cl::init(false)
      
    };
    
    
    
    Option<bool> debug{
      
      *this, 
      "debug",
      llvm::cl::desc("Enable moredebug in Pluto."),
      llvm::cl::init(true)
      
    };
    
    
    
    Option<bool> generateParallel{
        
      *this, 
      "gen-parallel", 
      llvm::cl::desc("Generate parallel affine loops."),
      llvm::cl::init(false)
        
    };



    Option<int> cloogf{
      
      *this, 
      "cloogf", 
      cl::desc("-cloogf option."),
      cl::init(-1)
    
    };


    Option<int> cloogl{
      
      *this, 
      "cloogl", 
      cl::desc("-cloogl option."),
      cl::init(-1)
      
    };
    

    Option<bool> diamondTiling{
    
      *this, 
      "diamond-tiling",
      cl::desc("Enable diamond tiling"),
      cl::init(false)
    
    };
  
  };

} // namespace






// -------------------------- PlutoParallelizePass ----------------------------

/// Find a single affine.for with scop.parallelizable attr.
static mlir::AffineForOp findParallelizableLoop(mlir::FuncOp f) {
  
  mlir::AffineForOp ret = nullptr;
  
  f.walk([&ret](mlir::AffineForOp forOp) {
  
    if (!ret && forOp->hasAttr("scop.parallelizable"))
  
      ret = forOp;
  
  });

  return ret;

}





/// Turns a single affine.for with scop.parallelizable into affine.parallel. The
/// design of this function is almost the same as affineParallelize. The
/// differences are:
///
/// 1. It is not necessary to check whether the parentOp of a parallelizable
/// affine.for has the AffineScop trait.
static void plutoParallelize(mlir::AffineForOp forOp, OpBuilder b) {
  
  assert(forOp->hasAttr("scop.parallelizable"));

  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointAfter(forOp);

  Location loc = forOp.getLoc();

  // If a loop has a 'max' in the lower bound, emit it outside the parallel loop
  // as it does not have implicit 'max' behavior.
  AffineMap lowerBoundMap = forOp.getLowerBoundMap();
  ValueRange lowerBoundOperands = forOp.getLowerBoundOperands();

  
  AffineMap upperBoundMap = forOp.getUpperBoundMap();
  ValueRange upperBoundOperands = forOp.getUpperBoundOperands();

  // Creating empty 1-D affine.parallel op.
  mlir::AffineParallelOp newPloop = b.create<mlir::AffineParallelOp>(
  
                                                                      loc, 
                                                                      llvm::None, 
                                                                      llvm::None, 
                                                                      lowerBoundMap, 
                                                                      lowerBoundOperands, 
                                                                      upperBoundMap, 
                                                                      upperBoundOperands, 
                                                                      1
  
  );
  
  // Steal the body of the old affine for op and erase it.
  newPloop.region().takeBody(forOp.region());

  for (auto user : forOp->getUsers()) {
  
    user->dump();
  
  }
  
  forOp.erase();

}





/// Need to check whether the bounds of the for loop are using top-level values
/// as operands. If not, then the loop cannot be directly turned into
/// affine.parallel.
static bool isBoundParallelizable(mlir::AffineForOp forOp, bool isUpper) {
  
  llvm::SmallVector<mlir::Value, 4> mapOperands = isUpper ? forOp.getUpperBoundOperands() : forOp.getLowerBoundOperands();

  for (mlir::Value operand : mapOperands)
    
    if (!isTopLevelValue(operand))
    
      return false;
  
  return true;

}





static bool isBoundParallelizable(mlir::AffineForOp forOp) {

  return isBoundParallelizable(forOp, true) && isBoundParallelizable(forOp, false);

}





/// Iteratively replace affine.for with scop.parallelizable with
/// affine.parallel.
static void plutoParallelize(mlir::FuncOp f, OpBuilder b) {
  
  mlir::AffineForOp forOp = nullptr;
  
  while ((forOp = findParallelizableLoop(f)) != nullptr) {
  
    if (!isBoundParallelizable(forOp))
  
      llvm_unreachable("Loops marked as parallelizable should have parallelizable bounds.");

    plutoParallelize(forOp, b);
  
  }

}





namespace {
  
  /// Turn affine.for marked as scop.parallelizable by Pluto into actual
  /// affine.parallel operation.
  struct PlutoParallelizePass : public mlir::PassWrapper<PlutoParallelizePass, OperationPass<mlir::FuncOp>> {

    void runOnOperation() override {
    
      FuncOp f = getOperation();
      OpBuilder b(f.getContext());

      plutoParallelize(f, b);
    
    }
  
  };

} // namespace





// -------------------------- PlutoTransformPass ----------------------------

/// The main function that implements the Pluto based optimization.
/// TODO: transform options?
static mlir::FuncOp plutoTransform(mlir::FuncOp f, 
                                   OpBuilder &rewriter,
                                   std::string dumpClastAfterPluto,
                                   bool parallelize = false, 
                                   bool debug = false,
                                   int cloogf = -1, 
                                   int cloogl = -1,
                                   bool diamondTiling = false) {


                                                                       
  
  LLVM_DEBUG(dbgs() << "Pluto transforming: \n");  
  LLVM_DEBUG(f.dump());
  /// F: add the message
  std::printf("mlir::FuncOp is dumped!!!\n");



  /// Pluto Context Allocation & two OslSymbolTable type variable declaration
  PlutoContext *context = pluto_context_alloc();
  OslSymbolTable srcTable, dstTable;





  /// Create OpenScop representation from mlir::FuncOp f
  std::unique_ptr<OslScop> scop = createOpenScopFromFuncOp(f, srcTable);
  
  if (!scop)
    return nullptr;
  
  if (scop->getNumStatements() == 0)
    return nullptr;







  /// Configure Pluto Context or more specifically options related to pluto
  // Should use isldep, candl cannot work well for this case.
  context->options->silent = !debug;
  context->options->moredebug = debug;
  context->options->debug = debug;
  context->options->isldep = 1;
  context->options->readscop = 1;

  context->options->identity = 0;
  context->options->parallel = parallelize;
  context->options->unrolljam = 0;
  context->options->prevector = 0;
  context->options->diamondtile = diamondTiling;

  if (cloogf != -1)
    context->options->cloogf = cloogf;
  
  if (cloogl != -1)
    context->options->cloogl = cloogl;

  std::printf("================================scop after createOpenScopFromFuncOp===========================\n");
  osl_scop_print(stderr, scop->get());





  /// Pluto Transformation operation on scop
  PlutoProg *prog = osl_scop_to_pluto_prog(scop->get(), context); /// Convert the OpenScop to a Pluto program
  pluto_schedule_prog(prog);                                      /// Run transformation Pluto prog
  pluto_populate_scop(scop->get(), prog, context);                /// Replace previous scop contents(populating transformed scop) 
                                                                  /// with new transformed contents with the help of Pluto prog





  /// Debug Flush
  if (debug) { // Otherwise things dumped afterwards will mess up.
  
    fflush(stderr);
    fflush(stdout);
  
  }




  std::printf("================================scop after pluto_populate_scop method===========================\n");
  osl_scop_print(stderr, scop->get());



  const char *dumpClastAfterPlutoStr = nullptr;
  /// method empty() returns true if the string is empty 
  if (!dumpClastAfterPluto.empty()) {
  
    dumpClastAfterPlutoStr = dumpClastAfterPluto.c_str();


  }



  /// Cast the parent operation of f to mlir::ModuleOp.
  mlir::ModuleOp m = dyn_cast<mlir::ModuleOp>(f->getParentOp());
  /// F: add the message
  std::printf("mlir::ModuleOp m is dumping from inside method....\n");
  LLVM_DEBUG(m.dump());

  




  /// This declares a vector argAttrs that will hold DictionaryAttr objects.
  /// DictionaryAttr is a type in MLIR that represents a dictionary of named attributes. 
  /// It is typically used to store a collection of key-value pairs, where keys are strings and values are attributes.
  SmallVector<DictionaryAttr> argAttrs;
  
  /// Collect all argument attributes of f.
  f.getAllArgAttrs(argAttrs);



   

  /// Create a new function g from the transformed OpenScop using createFuncOpFromOpenScop.
  mlir::FuncOp g = cast<mlir::FuncOp>(createFuncOpFromOpenScop(
                                                              std::move(scop), 
                                                              m, 
                                                              dstTable, 
                                                              rewriter.getContext(), 
                                                              prog,
                                                              dumpClastAfterPlutoStr)
  );
  

  /// F:
  std::printf("=========================dumpClastAfterPluto=====================================\n");
  std::printf("%s\n", dumpClastAfterPluto.c_str());


  /// sets all the collected argument attributes from argAttrs on the new function g.
  g.setAllArgAttrs(argAttrs);




  /// F: add the message
  std::printf("mlir::FuncOp g is dumping....\n");
  LLVM_DEBUG(g.dump());



  pluto_context_free(context);
  
  return g;

}





namespace {

  /// @brief : The class PlutoTransformPass inherits from mlir::PassWrapper, which is a template class in MLIR for defining passes. 
  /// This pass operates on mlir::ModuleOp operations.
  class PlutoTransformPass : public mlir::PassWrapper<PlutoTransformPass, OperationPass<mlir::ModuleOp>> {
    
    /// @brief : These member variables are configuration options for the Pluto optimization. 
    /// They control various aspects of the Pluto transformation.
    std::string dumpClastAfterPluto = "";
    bool parallelize = false;
    bool debug = false;
    int cloogf = -1;
    int cloogl = -1;
    bool diamondTiling = false;



  public:
    

    /// Constructors
    /// Default constructor: Initializes the pass with default values.
    /// Useful for testing or when no specific configuration is needed.
    PlutoTransformPass() = default;

    /// Copy constructor: Creates a new instance by copying an existing one.
    /// Useful when the pass needs to be duplicated.
    PlutoTransformPass(const PlutoTransformPass &pass) {}

    /// Constructor with options: Initializes the pass with specific configuration options.
    /// @param options: An instance of PlutoOptPipelineOptions containing the configuration.
    PlutoTransformPass(const PlutoOptPipelineOptions &options)
        : dumpClastAfterPluto(options.dumpClastAfterPluto),
          parallelize(options.parallelize), 
          debug(options.debug),
          cloogf(options.cloogf), 
          cloogl(options.cloogl),
          diamondTiling(options.diamondTiling) {}

    
    /// Main function that contains the logic for the transformation pass.
    /// Overrides the runOnOperation function from mlir::PassWrapper.
    void runOnOperation() override {
      
      /// F: add the message
      std::printf("HIT FROM INSIDE THE runOnOperation() method from PlutoTransformPass class\n");

      std::printf("dumpClastAfterPluto: %s\n", dumpClastAfterPluto.c_str());
      std::printf("parallelize: %d\n", parallelize);
      std::printf("debug: %d\n", debug);
      std::printf("cloogf: %d\n", cloogf);
      std::printf("cloogl: %d\n", cloogl);
      std::printf("diamondTiling: %d\n", diamondTiling);
      



      /// Retrieves the mlir::ModuleOp that this pass will operates on 
      mlir::ModuleOp m = getOperation();
      
      // F: add the message
      std::printf("mlir::ModuleOp m is dumping inside the pass not the method....\n");
      LLVM_DEBUG(m.dump());


      /// Creates an mlir::OpBuilder for creating new operations
      mlir::OpBuilder b(m.getContext());


      /// Vector to collect functions (mlir::FuncOp) that need to be transformed
      SmallVector<mlir::FuncOp, 8> funcOps;
      

      /// Map to keep track of the original and transformed functions
      llvm::DenseMap<mlir::FuncOp, mlir::FuncOp> funcMap;


      /// Walk through all functions in the module.
      /// If a function does not have the 'scop.stmt' attribute and is not marked as scop.ignored,
      /// add it to the funcOps vector.
      m.walk([&](mlir::FuncOp f) {
    
        if (!f->getAttr("scop.stmt") && !f->hasAttr("scop.ignored")) {
          
          /// F: add the message
          std::printf("HIT FROM INSIDE THE forloop after checking scop.stmt and scop.ignored\n");


          funcOps.push_back(f);
    
        }
    
      });


      /// Iterate over each collected function
      /// Apply the Pluto transformation and store the original and transformed functions in funcMap
      for (mlir::FuncOp f : funcOps) {
        
        if (mlir::FuncOp g = plutoTransform(f, b, dumpClastAfterPluto, parallelize, debug, cloogf, cloogl, diamondTiling)) {
          
          /// F: add the message
          std::printf("HIT FROM INSIDE THE 2nd forloop after checking condition\n");  

          funcMap[f] = g;              /// Store the transformed function g of original function f
          g.setPublic();               /// Make the transformed function public
          g->setAttrs(f->getAttrs());  /// Copy attributes from the original function to the transformed function

        
        }

      }
    


      /// F: print the funcMap
      for (auto &entry : funcMap) {
        mlir::FuncOp keyFunc = entry.first;
        mlir::FuncOp valueFunc = entry.second;

        std::printf("mlir::FuncOp key dumping from fucking map....\n");
        LLVM_DEBUG(keyFunc.dump());

        llvm::outs() << "Function: " << keyFunc.getName() << "\n";
        llvm::outs() << "Attributes:\n";

        for (auto attr : keyFunc->getAttrs()) {
          llvm::outs() << "  " << attr.getName() << " = " << attr.getValue() << "\n";
        }




        std::printf("mlir::FuncOp value dumping from fucking map....\n");
        LLVM_DEBUG(valueFunc.dump());

        llvm::outs() << "Function: " << valueFunc.getName() << "\n";
        llvm::outs() << "Attributes:\n";

        for (auto attr : valueFunc->getAttrs()) {
          llvm::outs() << "  " << attr.getName() << " = " << attr.getValue() << "\n";
        }

      }



      
      // Finally, we delete the definition of the original function, and make the
      // Pluto optimized version have the same name.
      for (const auto &it : funcMap) {
        
        /// F: add the message
        std::printf("HIT FROM INSIDE THE 3rd forloop\n");

        mlir::FuncOp from, to;
        std::tie(from, to) = it;

        to.setName(std::string(from.getName()));
        from.erase();
      
      }

      std::printf("After erasing original!!!");

      /// F: print the funcMap
      for (auto &entry : funcMap) {
        mlir::FuncOp keyFunc = entry.first;
        mlir::FuncOp valueFunc = entry.second;

        std::printf("mlir::FuncOp key dumping from fucking map....\n");
        // LLVM_DEBUG(keyFunc.dump());
        std::printf("mlir::FuncOp value dumping from fucking map....\n");
        LLVM_DEBUG(valueFunc.dump());
      }
    
    }
  
  };

} // namespace











/// @brief Remove duplicate index_cast (NEED TO GO THROUGH LATER)
/// @param f : mlir::FuncOp type
static void dedupIndexCast(FuncOp f) {
  
  std::printf("=======================================DEDUP=====================================\n");
  

  if (f.getBlocks().empty())
    return;

  // Dump the function to JSON
  // funcOpToJson(f, "funcOp.json");


  Block &entry = f.getBlocks().front();

  llvm::MapVector<Value, Value> argToCast;
  
  SmallVector<Operation *> toErase;
  
  for (auto &op : entry) {
  
    if (auto indexCast = dyn_cast<arith::IndexCastOp>(&op)) {
  
      auto arg = indexCast.getOperand().dyn_cast<BlockArgument>();

      
  
      if (argToCast.count(arg)) {
  
        LLVM_DEBUG(dbgs() << "Found duplicated index_cast: " << indexCast << '\n');

        indexCast.replaceAllUsesWith(argToCast.lookup(arg));
        toErase.push_back(indexCast);
      
      } else {
      
        argToCast[arg] = indexCast;
      
      }
    
    }
  
  }

  for (auto op : toErase)
  
    op->erase();

}




/// @brief : Anonymous Namespace: The code is inside an anonymous namespace (namespace { ... }). 
/// This means the contents are only visible within the current file. This is often used to avoid name conflicts.
namespace {


  /// @brief : We are defining a new struct called DedupIndexCastPass.
  /// This struct is a type of PassWrapper or inherits from class PassWrapper.
  /// The PassWrapper utility helps us create a pass easily.
  /// Our pass will specifically operate on functions (mlir::FuncOp).

  /// Think of it like this:

  /// Struct Definition: We're creating a new container (DedupIndexCastPass).
  /// Inheritance: This container will have special abilities provided by PassWrapper.
  /// Pass Type: We're telling PassWrapper two things:
  /// The name of our new container (DedupIndexCastPass).
  /// The type of things our pass will work on (mlir::FuncOp).
  struct DedupIndexCastPass : public mlir::PassWrapper<DedupIndexCastPass, OperationPass<mlir::FuncOp>> {

    /// @brief : runOnOperation() is the method which is inherited from base class PassWrapper. But here, we are overriding thsi method
    /// WHY override??? 
    /// overriding allows you to customize or extend the behavior of inherited methods to suit the specific needs of the derived class. 
    /// By overriding runOnOperation(), the derived struct DedupIndexCastPass can specify exactly what should happen when this pass is run, 
    /// such as performing the dedupIndexCast optimization.
    void runOnOperation() override { 
      
      /// dedupIndexCast function will perform its task on the current operation mlir::FuncOp
      /// getOperation() method is provided by the PassWrapper class in MLIR, and it allows your pass to access the current mlir::FuncOp being processed
      // dedupIndexCast(getOperation());
      dedupIndexCast(getOperation());
    
    }

  };

} // namespace




/// @brief: The purpose of this function is to register a custom optimization pass pipeline for 
/// the MLIR framework. This pipeline is named "pluto-opt" and it includes several optimization passes, 
/// some of which are specific to the ---PLUTO framework---
/// @param: No parameter
/// @return: No value return. Its just include some optimization passes
void polymer::registerPlutoTransformPass() {

  /// F: For debugging- this line is printed and this method is called inside polymer-opt
  std::printf("Inside register pluto transformpass()\n");



  /// @brief : PassPipelineRegistration(templated class) is a utility in MLIR for registering a sequence of passes, 
  /// known as a pass pipeline. And this utility helps to register pass pipeline under a specific name (e.g. pluto-opt). 
  /// Passes in MLIR are transformations or analyses applied to the IR of the code. 
  /// A pass pipeline is a predefined sequence of these transformations or analyses that can be applied together as a single unit.
  /// @ref: (Official doc) https://mlir.llvm.org/doxygen/structmlir_1_1PassPipelineRegistration.html
  PassPipelineRegistration<PlutoOptPipelineOptions>(
      
    /// "pluto-opt" is the name of the pipeline. This name is used to identify and invoke the pipeline within the MLIR framework.
    "pluto-opt", 

    /// "Optimization implemented by PLUTO." is a brief description of the pipeline. This description is useful for documentation and help messages.
    "Optimization implemented by PLUTO.",

    /// @brief : lambda function (anonymous function) used to specify the passes that should be included in the pipeline.
    /// @param pm :  A reference to an OpPassManager object. This object manages the list of passes to be run in the pipeline.
    /// @param pipelineOptions : A constant reference to the options for the pipeline. These options can be used to customize 
    /// the behavior of the passes. This options are defined by user at the top of the script. They are PLUTO specific.
    /// potentially using some options (PlutoOptPipelineOptions).
    [](OpPassManager &pm, const PlutoOptPipelineOptions &pipelineOptions) {
        

      /// Adding passes to the pipeline
      /// This pass helps to remove duplicated index_cast
      pm.addPass(std::make_unique<DedupIndexCastPass>());

      /// The canonicalizer pass simplifies and cleans up the IR by applying pattern-based rewrites. 
      /// This standardizes the IR, making it more efficient and easier to further optimize.
      pm.addPass(createCanonicalizerPass());
      
      /// The PlutoTransformPass performs PLUTO-specific transformations based on the options provided in pipelineOptions.
      pm.addPass(std::make_unique<PlutoTransformPass>(pipelineOptions));
      
      /// Running the canonicalizer again ensures that the IR is simplified and cleaned up after the transformations applied by PlutoTransformPass. 
      /// This helps remove any new redundancies or inefficiencies introduced by the transformation.
      pm.addPass(createCanonicalizerPass());
          

      /// Adding passes with condition check
      /// This line checks if the generateParallel option in pipelineOptions is set to true.
      if (pipelineOptions.generateParallel) {
        
        /// The PlutoParallelizePass introduces parallelization transformations, 
        /// potentially leveraging PLUTO's capabilities to parallelize loops or other constructs in the IR.
        pm.addPass(std::make_unique<PlutoParallelizePass>());

        /// Running the canonicalizer again ensures that any new parallel constructs introduced by 
        /// the PlutoParallelizePass are optimized and the IR remains clean and efficient
        pm.addPass(createCanonicalizerPass());
          
      }
      
    }
  
  );

}
