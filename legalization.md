# Legalization Customized Passes
.. Revision date: 2024.03  

The legalizer has the following types of customized passes: 
*  Non-linear computation  
    Passes to remove redundant layers, or simply fuse and update layer information, 
    such as removing redundant Transpose, or fusing Pad into Conv layer attributes.
* Complex fusion with internal computation changes  
    Passes to update parameters coefficients, or fusion into a high-level operator, 
    for example, fuse_bn_into_prev_conv and fuse_attention. These passes might affect the quantization results dramatically.
  
Both categories can improve the model compile efficiency on hardware.
  
  
## Non-Linear Computation Passes

* AdjustResizeInput
    * This pass modifies the input format of the Resize node.
    * In higher ONNX opset versions, only one of 'scales' and 'sizes' can be specified in the input field of the Resize node.


* ConvertModelFormatToNCHW/ConvertOperatorFormatToNCHW
    * Prepend and append Transpose Layers to convert the data_format from NHWC (TensorFlow default) to NCHW (ONNX 
      default).


* ConvertSpatialPyramidPool
    * Convert parallel pooling layers into a spatial pyramid form.
```
Convert parallel pooling layers with different kernel sizes...
                         input
                        /  |  \
                       /   |   \
                     5x5  9x9  13x13
                     |     |     |
                   out1   out2   out3
...to a spatial pyramid form:
                         input
                           |
                          5x5
                         /   \
                       5x5    out1
                      /   \
                    5x5   out2
                   /
                 out3
```


* FuseIdenticalSiblingLayers
    * Remove sibling nodes that are identical.


* FuseIntializerWithTranspose
    * Fuse Transpose layer into the initalizer of the DequantizeLinear layer.


* FusePad
    * Fuse Pad layer into the pads attributes of pooling layers.


* FuseTransposeIntoGemm
    * Fuse *"Transpose + Reshape + Gemm"* -> *"Reshape + Gemm (weight transposed)"*.


* MoveActivationUpwardToConv
    * Move activation layers up to Conv layer to allow further enhancement for model compile.


* MoveConstantNodeToInitializer
    * Remove initializers in Reshape, Unsqueeze, Squeeze, and Constant layers, and save them in the Optimizer object for
      easily sharing initializers across layers.


* RemoveBroadcastingInEltwiseOps 
    * Try to expand the constant inputs in Eltwise Ops to avoid broadcasting happen.
    * Default disabled.


* RemoveConsecutiveClips
    * Fuse two consecutive Clips layers into one.


* RemoveConsecutiveReduces
    * Remove consecutive reduce nodes, such as ReduceMax, ReduceMean, ReduceMin, and ReduceSum.


* RemoveConsecutiveReshapes
    * Remove redundant consecutive Reshape layers.


* RemoveConsecutiveSqueezes
    * Fuse consecutive Squeeze layers with a different axis into one.


* RemoveExpandBeforeBroadcastableOp
    * Remove Expand layers before broadcastable ops like Mul, Div, Add, Sub, and Sum.


* RemoveIdentity
    * Remove nop layers.


* RemoveLastNode
    * Remove graph output layer if it is one of the following ``op_type`` types: 
      Transpose, Reshape, Unsqueeze, Cast, ArgMax, and Concat layers.
    * This pass is mainly used for debug and is disabled by default.


* RemoveNopConcat
    * Remove any nop Concat layer that has only one input.


* RemoveNopSlice
    * Remove the nop Slice layer.


* RemovePairedReshapeAroundSoftmax
    * Remove paired Reshape layers before and after the Softmax layer, and modify the axis of Softmax.


* RemoveFlattenReshapeAroundSoftmax
    * Remove the redundant Flatten and Reshape around Softmax, and update the axis of Softmax.


* RemovePairedSqueezeUnsqueeze
    * Remove paired Squeeze and Unsqueeze layers if their axes are matched.


* RemovePairedTransposes
    * Remove paired Transpose layers between LayerNorm, broadcastable ops, Resize, reduce ops, or Softmax.


* RemoveReshapeShapeNotChanged
    * Remove nop Reshape layers.


* RemoveSoftmaxBeforeArgmax
    * Remove the Softmax layer before the ArgMax layer and modify the axis of ArgMax.


* RemoveSqueezeAfterReduce
    * Remove the Squeeze layer after Reduce op and modify the keepdims attribute of the Reduce op.


* ReplaceAvgpool
    * Replace AveragePool by GlobalAveragePool if the ``kernel_shapes`` is the same as the ``input_shape`` in spatial dimensions.


* ReplaceMinmaxByClip
    * Replace the Min and Max layers with one Clip layer.


* ReplaceSum
    * Replace Sum layer with Add layer.


* SwapClipMul
    * Transform ``Clip(, a, b) -> Mul(, c)`` to ``Mul(, c) -> Clip(ac, bc)``.


* RemoveNopWhere
    * Remove Where layer with same condition values that only one branch will be chosen.


* FuseParallelActivationBranches
    * Remove redundant activation branches (Sigmoid/Relu...) that are connected to same successor.
```
Pattern1:
Input subgraph:
                    _____input____
                   /       |      \
                  /        |       \
                 O     sigmoid1  sigmoid2
                 |         |        |
                 |         |        |
                  \____ Concat ____/

Output subgraph:
                    _____input
                   /       |
                  /        |
                 O   merged_sigmoid
                 |         |
                 |         |
                  \_____Concat

Pattern2:
    A         B
    |         |
Sigmoid    Sigmoid
     |      /
      Concat

becomes
    A         B
     |       /
      Concat
        |
      Sigmoid
```


* ReplaceScatterND
    * Remove or replace ScatterND Ops with other Ops. Only the 5 patterns of `BEVFormer`` models are supported.
      * Pattern 1: (6) (Gather + Reshape) + 6 ScatterND -> Unsqueeze + Expand
      * Pattern 2: (Gather + Div + (2) Reshape + 2 ScatterND) * 2 -> DIV
      * Pattern 3: Gather + Reshape + ScatterND -> Identity
      * Pattern 4: ((Add + ScatterND + ScatterND) + Add ) -> ( Add + Add ) + Concat
      * Pattern 5: Several Slice + Add + ScatterND + Sigmoid + Mul/Add subgraph -> Remove ScatterND & add Concat


* FuseReshapeUnsqueezeFlatten
    * Fuse Reshape + Unsqueeze + Flatten into a single Reshape.


* FuseLogReshapeExp
    * Fuse Log + Reshape + Exp into a single Reshape.


* RemoveUnsqueezeGather
    * Remove redundant Unsqueeze + Gather layer that do not affect the output shape.


* ReplaceSqueezeByFlatten
    * Replace Squeeze layer that squeezes the input tensor into 2D tensor by Flatten layer, which is already supported by backend.


* ModifyShareInitializersName
    * We rename the shared tensors for the purpose of supporting different quantization and tiling params in generated compiler product.
    Copy and rename these tensors with different names by adding suffix 0, 1 etc. to make them unique.


* RemoveConsecutiveResizes
    * Remove redundant consecutive Resize layers.


* MoveTransposeToEnd
    * Move intermediate Transpose layer to the end of the graph to ease output layout handling in compiler.
    * Currently support moving Transpose downward in patterns of:
        * Transpose + Reshape
        * Transpose + Split (refer to "yolov4.onnx" and "yolov7_nonms.onnx")
        * Transpose + Transpose
        * Transpose + ReduceOps


* FuseReluClip
    * Remove Relu layer if the following Clip layer includes the range of Relu.


* FuseChannelShuffle
    * Use our defined local function node to represent the general pattern of channel shuffling.
    * Pattern1: Transpose - Reshape - Transpose - Reshape - Transpose
    * Pattern2: Reshape - Transpose - Reshape (without Transpose ops NCHW -> NHWC -> NCHW)


* ReplaceEinsumByMatMul
    * Replace the Einsum by MatMul when the equation of Einsum is a valid MatMul computation, also insert extra Transpose/Reshape layer if needed.
    * Support simple patterns for Einsum like:
        * MatMul(a.transpose, b): b c m, b c n -> b m n
        * MatMul(b.transpose, a): b c m, b c n -> b n m
        * MatMul(a, b.transpose): b c m, b n m -> b c n
        * MatMul(b, a.transpose): b c m, b n m -> b n c
    * Also supports complicated case including Transpose/Reshape layers insertions, like:
        * "bijk,binm->bjknm"
        * "nkctv,kvw->nctw"


* FuseInverseSigmoid
    * Use our defined local function node to represent the general pattern of inverse Sigmoid.
```
formula: x = ln(y / (1 - y))
pattern:

                            ／   Clip(m=epsilon (1e-5))              ＼
            Clip(m=0, M=1)                                              Div - Log
                            ＼   Sub(A=1) - Clip(m=epsilon (1e-5))   ／
```


* RemoveSigmoidInverseSigmoid
    * Remove Sigmoid + InverseSigmoid paired Ops.


* SplitMatMulAdd
    * Split the MatMul layer and Add layer by separating it into multiple MatMul/Add layers instead of using Split layer.


* ConvertDynamicReshapeToStatic
    * Convert all the dynamic Reshape (whose 2nd input is not a Constant, the shape values depend on the model input and might vary in different cases) to static Reshape (shape values are fixed).
    * This pass is disabled by default.


* FuseConsecutiveAddReshape
    * The consecutive Add+Reshape pattern is usually seen in Attention with Mask (Swin_tiny) model, and can be handled directly in FuseAttention pass.
      But if FuseAttention pass is disabled, this extra pass might be needed.


* RemoveConcatSlice
    * Remove Concat + multiple Slice Ops if Slice just revert Concat.
```
Input subgraph:
A --- Concat --- Slice --- A1
B ___|      |___ Slice --- B1

Output subgraph:
A --- A1
B --- B1

```

## Complex Fusion with Internal Computation Changes

* AdjustLargeConvWeight
    * Reduce the weights of Conv layers to improve quantization performance after applying quantization on Relu6 layers.
    * This pass is disabled by default.


* FuseAttention(Customized)
    * Use a customized local function to represent a subgraph of certain Attention pattern. Only some patterns of ``Vit``,
      ``Swin_tiny`` and ``Bert_large`` models are supported.


* FuseBNAsBias
    * Fuse BN layer into successive Conv layer when ``scale``, ``var``, and ``mean`` of BN layer are nop. Also remove nop
      BN layer.


* FuseBNIntoInstancenorm
    * Fuse small ops including ReduceMean, ReduceSumSquare, and BN into an InstanceNorm layer.


* FuseBNIntoPrevConv
    * Fuse BN layer into previous Conv/ConvTranspose layer.
    * Also support pattern of *Convs + Concat + BN* fusion.


* FuseBNIntoPrevGemm
    * Fuse BN layer into previous Gemm layer.


* FuseBNIntoSuccGemm
    * Fuse BN layer into following Gemm layer.


* FuseBNIntoPrevMatmul
    * Fuse BN layer into previous MatMul layer.


* FuseBNIntoSuccConv
    * Fuse BN layer into following Conv layer. 
    * Also supports pattern of *BNs + Concat + Conv* fusion.


* FuseExpandToResize
    * Fuse *Reshape + Expand + Reshape* into a Resize layer.


* FuseGeLU
    * fuse small ops into a Gelu layer.
    * Support two different types of Gelu approximation.


* FuseGemmAdd
    * Fuse Add with constant values into previous Gemm layer.


* FuseIntoHardsigmoid
    * Fuse *Add(3) + Clip + Div(6)* or *Add(3) + Div(6) + Clip* into a HardSigmoid layer.


* FuseIntoHardswish
    * Fuse small ops into a HardSwish layer.
    * Supports three patterns.


* FuseIntoSoftmax
    * Fuse small ops into a Softmax layer.
    * Supports two patterns.


* FuseIntoSpaceToDepth
    * Fuse serial Reshape and Transpose layers into a SpaceToDepth layer.


* FuseLpnormalization
    * Fuse small ops into a LpNormalization layer.
    * Supports three patterns.


* FuseMatMulAddAsGemm
    * Fuse *"MatMul + Add"* into a Gemm layer.


* FuseMish
    * Fuse small ops into a Mish layer.
    * Supports two patterns.


* FuseMulAddIntoNormalization
    * Fuse constant Add/Mul layers into Conv, BN, InstanceNorm.


* FusePrimitives
    * Contains several fusion passes. Only *fuse_mul_max_into_leakyrelu* is enabled.


* FusePrimitivesIntoLayernorm
    * Fuse small ops into a Layernorm layer.


* FuseSplitAdd
    * Fuse *"MatMul + Split + Adds"* to *MatMul + single Add + Split*.


* FuseSplitConv
    * Fuse *Split + Convs + Concat* into a group Conv.


* FuseSwish
    * Fuse small ops into a Swish layer.


* FuseIntoDepthToSpace
    * Fuse Reshape + Transpose + Reshape into a DepthToSpace layer.


* FuseMulIntoMatMulAdd
    * Fuse Mul initializer into previous MatMul and Add (and optionally with one or more intermediate layers from Reshape + Transpose + Reshape) layers.


* FuseMulIntoConv
    * Fuse Mul initializer into previous Conv (and optionally with Relu) layer.


* ConvertArithmeticOps
    * Simplify a series of arithmetic operators.
    * fuse_consecutive_add: Add + Add -> Add
    * fuse_consecutive_mul: Mul + Mul -> Mul
    * fuse_mul_div_to_single_mul: Mul + Div -> Mul
    * pattern1: Sub + Mul + Add -> Mul + Add
    * pattern2: Mul + Pow + Mul -> Pow + Mul
    * pattern3: Neg + Add(A) + Mul(B) + Sub(C) -> Sub(A - B * C, Input) + Mul(B)
    * pattern4: Mul+4x(Gather+Add/Sub)+Concat -> Mul + Split + (Add & Sub) + Concat

* FuseAddIntoSuccConv
    * Fuse Add into following Conv layer.

* FuseSubAndDivIntoBN
    * Fuse Sub and Div into BatchNorm layer.

## Additional Utility
 
* SingleLayerTransforms
    * Util function for graph/nodes handling operations.
