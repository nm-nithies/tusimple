prediction  0 = 0.9999999
prediction  1 = 6.470239e-17
prediction  2 = 5.3113327e-09
prediction  3 = 2.3830837e-10
prediction  4 = 1.54674e-15
prediction  5 = 1.6361314e-07
prediction  6 = 2.7768482e-11
prediction  7 = 8.211209e-13
prediction  8 = 2.9605862e-09
prediction  9 = 8.650948e-15



[array([[9.9999988e-01, 6.4702637e-17, 5.3113327e-09, 2.3830793e-10,
        1.5467400e-15, 1.6361314e-07, 2.7768482e-11, 8.2112089e-13,
        2.9605862e-09, 8.6509482e-15]], dtype=float32)]




import numpy as np

# Example model output
output = np.array([[9.9999988e-01, 6.4702637e-17, 5.3113327e-09, 2.3830793e-10,
                    1.5467400e-15, 1.6361314e-07, 2.7768482e-11, 8.2112089e-13,
                    2.9605862e-09, 8.6509482e-15]], dtype=np.float32)

# Loop through each prediction and print with formatting
for i, prediction in enumerate(output[0]):
    print(f"prediction {i} = {prediction:.7g}")





../src/Dialect/ONNX/Transforms/temp.cpp:718:61: error: no match for 'operator+' (operand types are 'mlir::FloatAttr' and 'mlir::FloatAttr')
  718 |     FloatAttr momentumAttr =  rewriter.getF32FloatAttr(0.9) + a;
      |                               ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ^ ~
      |                                                       |       |
      |                                                       |       mlir::FloatAttr
      |                                                       mlir::FloatAttr
../src/Dialect/ONNX/Transforms/temp.cpp:718:61: note: candidate: 'operator+(int, int)' <built-in>
  718 |     FloatAttr momentumAttr =  rewriter.getF32FloatAttr(0.9) + a;
      |                               ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^~~
../src/Dialect/ONNX/Transforms/temp.cpp:718:61: note:   no known conversion for argument 2 from 'mlir::FloatAttr' to 'int'
In file included from /usr/include/c++/9/bits/stl_algobase.h:67,
                 from /usr/include/c++/9/memory:62,
                 from /workspace/ONNX_MLIR/llvm-project/llvm/include/llvm/Support/Casting.h:20,
                 from /workspace/ONNX_MLIR/llvm-project/mlir/include/mlir/Support/LLVM.h:23,
                 from /workspace/ONNX_MLIR/llvm-project/mlir/include/mlir/IR/Visitors.h:16,
                 from /workspace/ONNX_MLIR/llvm-project/mlir/include/mlir/IR/AffineExpr.h:17,
                 from /workspace/ONNX_MLIR/llvm-project/mlir/include/mlir/IR/AffineMap.h:17,
                 from /workspace/ONNX_MLIR/llvm-project/mlir/include/mlir/IR/BuiltinAttributeInterfaces.h:12,
                 from /workspace/ONNX_MLIR/llvm-project/mlir/include/mlir/IR/BuiltinAttributes.h:12,
                 from /workspace/ONNX_MLIR/llvm-project/mlir/include/mlir/IR/Matchers.h:18,
                 from ../src/Dialect/ONNX/Transforms/temp.cpp:23:
/usr/include/c++/9/bits/stl_iterator.h:423:5: note: candidate: 'template<class _Iterator> constexpr std::reverse_iterator<_Iterator> std::operator+(typename std::reverse_iterator<_Iterator>::difference_type, const std::reverse_iterator<_Iterator>&)'
  423 |     operator+(typename reverse_iterator<_Iterator>::difference_type __n,
      |     ^~~~~~~~
/usr/include/c++/9/bits/stl_iterator.h:423:5: note:   template argument deduction/substitution failed:
../src/Dialect/ONNX/Transforms/temp.cpp:718:63: note:   'mlir::FloatAttr' is not derived from 'const std::reverse_iterator<_Iterator>'
  718 |     FloatAttr momentumAttr =  rewriter.getF32FloatAttr(0.9) + a;
      |                                                      


auto a = mlir::cast<FloatAttr>(subConstValue);


/usr/bin/c++ -DONNX_ML=1 -DONNX_MLIR_DECOMP_ONNX_CONVTRANSPOSE -DONNX_MLIR_ENABLE_STABLEHLO -DONNX_NAMESPACE=onnx -D_DEBUG -D_GLIBCXX_ASSERTIONS -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS -I/workspace/ONNX_MLIR/llvm-project/llvm/include -I/workspace/ONNX_MLIR/llvm-project/build/include -I/workspace/ONNX_MLIR/llvm-project/mlir/include -I/workspace/ONNX_MLIR/llvm-project/build/tools/mlir/include -I../ -I. -I../include -I../third_party/onnx -Ithird_party/onnx -I_deps/protobuf-src/src -I_deps/abseil-src -fPIC -fno-semantic-interposition -fvisibility-inlines-hidden -Werror=date-time -fno-lifetime-dse -Wall -Wextra -Wno-unused-parameter -Wwrite-strings -Wcast-qual -Wno-missing-field-initializers -Wimplicit-fallthrough -Wno-nonnull -Wno-class-memaccess -Wno-redundant-move -Wno-pessimizing-move -Wno-noexcept-type -Wdelete-non-virtual-dtor -Wsuggest-override -Wno-comment -Wno-misleading-indentation -fdiagnostics-color -ffunction-sections -fdata-sections -DSUPPRESS_THIRD_PARTY_WARNINGS -O3 -DNDEBUG   -D_DEBUG -D_GLIBCXX_ASSERTIONS -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS -UNDEBUG -pthread -std=gnu++17 -MD -MT src/Dialect/ONNX/Transforms/CMakeFiles/OMONNXRewrite.dir/temp.cpp.o -MF src/Dialect/ONNX/Transforms/CMakeFiles/OMONNXRewrite.dir/temp.cpp.o.d -o src/Dialect/ONNX/Transforms/CMakeFiles/OMONNXRewrite.dir/temp.cpp.o -c ../src/Dialect/ONNX/Transforms/temp.cpp
In file included from /workspace/ONNX_MLIR/llvm-project/mlir/include/mlir/IR/AffineMap.h:18,
                 from /workspace/ONNX_MLIR/llvm-project/mlir/include/mlir/IR/BuiltinAttributeInterfaces.h:12,
                 from /workspace/ONNX_MLIR/llvm-project/mlir/include/mlir/IR/BuiltinAttributes.h:12,
                 from /workspace/ONNX_MLIR/llvm-project/mlir/include/mlir/IR/Matchers.h:18,
                 from ../src/Dialect/ONNX/Transforms/temp.cpp:23:
/workspace/ONNX_MLIR/llvm-project/mlir/include/mlir/IR/Value.h: In instantiation of 'static To llvm::CastInfo<To, From, typename std::enable_if<(is_same_v<mlir::Value, typename std::remove_const<From>::type> || is_base_of_v<mlir::Value, From>), void>::type>::doCast(mlir::Value) [with To = mlir::FloatAttr; From = mlir::Value]':
/workspace/ONNX_MLIR/llvm-project/llvm/include/llvm/Support/Casting.h:573:36:   required from 'decltype(auto) llvm::cast(From&) [with To = mlir::FloatAttr; From = mlir::Value]'
../src/Dialect/ONNX/Transforms/temp.cpp:703:49:   required from here
/workspace/ONNX_MLIR/llvm-project/mlir/include/mlir/IR/Value.h:619:55: error: no matching function for call to 'mlir::FloatAttr::FloatAttr(mlir::detail::ValueImpl*)'
  619 |   static inline To doCast(mlir::Value value) { return To(value.getImpl()); }
      |                                                       ^~~~~~~~~~~~~~~~~~~






#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"

// Function to extract a single float value from a tensor if it's a constant.
float getSingleFloatFromTensor(mlir::Value val) {
    // Check if the value is defined by a constant operation.
    if (auto constOp = val.getDefiningOp<mlir::arith::ConstantOp>()) {
        // Extract the attribute from the constant op.
        if (auto denseAttr = constOp.getValue().dyn_cast<mlir::DenseFPElementsAttr>()) {
            // Make sure the tensor has exactly one element.
            if (denseAttr.getNumElements() == 1) {
                // Return the single float value.
                return (*denseAttr.getValues<float>().begin());
            } else {
                throw std::runtime_error("Tensor has more than one element.");
            }
        }
    }

    // If it's not a constant or not a valid tensor, throw an error.
    throw std::runtime_error("Value is not a constant tensor or doesn't contain a single float.");
}


