#include <llvm/ADT/StringMap.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Operation.h> // For example purposes
#include <iostream>



#include "cloog/cloog.h"
#include "cloog/pprint.h"
#include "osl/osl.h"
#include "pluto/internal/pluto.h"
#include "pluto/osl_pluto.h"
#include "pluto/pluto.h"


// Hypothetical function to print mlir::Value
void printMlirValue(mlir::Value value) {
    // This is a placeholder; actual implementation will depend on mlir::Value
    if (value.isa<mlir::BlockArgument>()) {
        std::cout << "BlockArgument";
    } else if (auto *op = value.getDefiningOp()) {
        std::cout << "Operation: " << op->getName().getStringRef().str();
    } else {
        std::cout << "Unknown mlir::Value";
    }
}

int main() {

    /// The OpenScop object pointer.
    OslScop *scop;  
    
    /// The symbol table that will be built on the fly.
    SymbolTable symbolTable;


    OslScop::ValueTable *oslValueTable = scop->getValueTable();

    // Iterate over the SymbolTable
    for (const auto &entry : oslValueTable) {
        // entry.first() gives the key (std::string)
        std::string key = entry.getKey().str();

        // entry.second gives the value (mlir::Value)
        mlir::Value value = entry.getValue();

        // Print the key and value
        std::cout << "Key: " << key << ", Value: ";
        printMlirValue(value);
        std::cout << std::endl;
    }

    return 0;
}