public OpRewritePattern<> is a template class in MLIR used to define custom patterns for rewriting operations. 
It allows you to match specific operations and replace them with optimized or transformed versions during MLIR's optimization passes.


rewriter.create<> is a method in MLIR that creates a new operation during a rewrite or transformation pass. 
It takes the operation's location, types, and operands, and inserts the newly created operation into the IR.

mlir::Value is a handle to an SSA (Static Single Assignment) value in the MLIR (Multi-Level Intermediate Representation) framework. 
It represents the result of an operation or a block argument and is used to connect operations in the IR.

rewriter.replaceOp<> replaces the specified operation with new result(s), typically from a transformed or optimized operation, and updates all uses of the old operation to the new result.
rewriter.eraseOp<> deletes the specified operation from the IR, removing it entirely from the current computation graph.

mlir::arith::ConstantOp is an operation in MLIR that creates a constant value in the arithmetic dialect. It can represent scalar or tensor constants, such as integers or floating-point values, and is often used for optimization and transformation purposes.


mlir::DenseElementsAttr is an attribute in MLIR that represents a dense collection of elements, typically used for tensor data. It stores constant values in a compact form for tensors or vectors and allows efficient access to the underlying elements, whether they are integers, floats, or other types.


llvm::APFloat is a class in the LLVM framework that represents arbitrary precision floating-point numbers.
