DTPP_decoder:
Created MR. Resolved the MR comments
 
Qwen2_VL model:
Found the reason for the bug mentioned in the jira. 
This reason is same dtpp_decoder that history_len and ids_len input nodes is needed positive values as the NNAC set these input values as zero (as default). Fix it by setting its as 1
 
During legalization, layer validation part is skipped due to larger size
One output node is falied due max_diff value around 1.00
 
This diff is not caused by any passes
Yvonne Chen Whether is this caused by input values or any other reasons?
 
And also faced version issues






 NITHIES:
DTPP Decoder model
Added a new pattern in ReplaceEinsumByMatmul pass
Able to legalize the model
Added unit test case for this pattern and comments to the pass
Cleaned, Optmized and pushed the code




DTPP Decoder model:
Found the ReduceSum Op got updated with the wrong axis values. Updated RemoveConsecutiveReduces pass accordingly
In ReplaceEinsumByMatmul pass, found the matmul gets improper input shapes, Added additional tranpose before matmul to fix this issue. But these changes are messing with the ReplaceEinsumByMatmul_pattern2


DTPP Decoder model :
 
The issue is arraised due to timesteps input node
 
InNNAC , 0 is set to timesteps node by default. I think the input(timesteps) need to be greater than 1 for this case due to the sub node which contains constant value 1 .
the issue mentioned in the jira is ressolved after setting the value to 5 to timesteps node
Later Found some issue with ReplaceEinsumByMul RemoveConsecutiveReduces passes . Currently debugging this
 
After that I had faced another issue when tried to legalize the model (disabling the above passes for now)
There is ConstantOfShape Op with infinity values
I tried with onnx slim to remove this, but it doesn't work
 
Updated the jira and moved the bevformer files to customer folder





Fcn_resnet50:
The resize node does not have the tensor data for given fcn_resnet50-12-qdq model


Tried to run these models mentioned in the jira https://jira.internal.synopsys.com/browse/P10019563-75410 and https://jira.internal.synopsys.com/browse/P10019563-75411
Faced the exact issue and currently debugging it
