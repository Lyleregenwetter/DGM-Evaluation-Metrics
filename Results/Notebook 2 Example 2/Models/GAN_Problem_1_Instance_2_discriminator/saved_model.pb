Ѕя
фЙ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%ЭЬL>"
Ttype0:
2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
С
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ј
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.9.12v2.9.0-18-gd8ce9f9c3018ны

discriminator_11/dense_145/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!discriminator_11/dense_145/bias

3discriminator_11/dense_145/bias/Read/ReadVariableOpReadVariableOpdiscriminator_11/dense_145/bias*
_output_shapes
:*
dtype0

!discriminator_11/dense_145/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*2
shared_name#!discriminator_11/dense_145/kernel

5discriminator_11/dense_145/kernel/Read/ReadVariableOpReadVariableOp!discriminator_11/dense_145/kernel*
_output_shapes

:d*
dtype0

discriminator_11/dense_144/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*0
shared_name!discriminator_11/dense_144/bias

3discriminator_11/dense_144/bias/Read/ReadVariableOpReadVariableOpdiscriminator_11/dense_144/bias*
_output_shapes
:d*
dtype0

!discriminator_11/dense_144/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*2
shared_name#!discriminator_11/dense_144/kernel

5discriminator_11/dense_144/kernel/Read/ReadVariableOpReadVariableOp!discriminator_11/dense_144/kernel*
_output_shapes

:dd*
dtype0

discriminator_11/dense_143/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*0
shared_name!discriminator_11/dense_143/bias

3discriminator_11/dense_143/bias/Read/ReadVariableOpReadVariableOpdiscriminator_11/dense_143/bias*
_output_shapes
:d*
dtype0

!discriminator_11/dense_143/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*2
shared_name#!discriminator_11/dense_143/kernel

5discriminator_11/dense_143/kernel/Read/ReadVariableOpReadVariableOp!discriminator_11/dense_143/kernel*
_output_shapes

:d*
dtype0

NoOpNoOp
З 
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ђ
valueшBх Bо
љ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

Dense1

	LReLU1


Dense2

LReLU2

Dense3

signatures*
.
0
1
2
3
4
5*
.
0
1
2
3
4
5*
* 
А
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
* 
І
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

kernel
bias*

#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses* 
І
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses

kernel
bias*

/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses* 
І
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses

kernel
bias*

;serving_default* 
a[
VARIABLE_VALUE!discriminator_11/dense_143/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEdiscriminator_11/dense_143/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!discriminator_11/dense_144/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEdiscriminator_11/dense_144/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!discriminator_11/dense_145/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEdiscriminator_11/dense_145/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
* 
'
0
	1

2
3
4*
* 
* 
* 
* 
* 
* 
* 

0
1*

0
1*
* 

<non_trainable_variables

=layers
>metrics
?layer_regularization_losses
@layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses*

Atrace_0* 

Btrace_0* 
* 
* 
* 

Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses* 

Htrace_0* 

Itrace_0* 

0
1*

0
1*
* 

Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses*

Otrace_0* 

Ptrace_0* 
* 
* 
* 

Qnon_trainable_variables

Rlayers
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses* 

Vtrace_0* 

Wtrace_0* 

0
1*

0
1*
* 

Xnon_trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\layer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses*

]trace_0* 

^trace_0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
z
serving_default_input_1Placeholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1!discriminator_11/dense_143/kerneldiscriminator_11/dense_143/bias!discriminator_11/dense_144/kerneldiscriminator_11/dense_144/bias!discriminator_11/dense_145/kerneldiscriminator_11/dense_145/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_950855
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ш
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename5discriminator_11/dense_143/kernel/Read/ReadVariableOp3discriminator_11/dense_143/bias/Read/ReadVariableOp5discriminator_11/dense_144/kernel/Read/ReadVariableOp3discriminator_11/dense_144/bias/Read/ReadVariableOp5discriminator_11/dense_145/kernel/Read/ReadVariableOp3discriminator_11/dense_145/bias/Read/ReadVariableOpConst*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__traced_save_951014
ы
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename!discriminator_11/dense_143/kerneldiscriminator_11/dense_143/bias!discriminator_11/dense_144/kerneldiscriminator_11/dense_144/bias!discriminator_11/dense_145/kerneldiscriminator_11/dense_145/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__traced_restore_951042ЂЎ
Ш	
і
E__inference_dense_144_layer_call_and_return_conditional_losses_950711

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџdr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџdw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
Ч

*__inference_dense_144_layer_call_fn_950934

inputs
unknown:dd
	unknown_0:d
identityЂStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_144_layer_call_and_return_conditional_losses_950711o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџd: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
Ш	
і
E__inference_dense_145_layer_call_and_return_conditional_losses_950973

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
Ч

*__inference_dense_145_layer_call_fn_950963

inputs
unknown:d
	unknown_0:
identityЂStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_145_layer_call_and_return_conditional_losses_950734o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџd: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
Ќ
K
/__inference_leaky_re_lu_45_layer_call_fn_950949

inputs
identityИ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_45_layer_call_and_return_conditional_losses_950722`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџd"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџd:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
д
f
J__inference_leaky_re_lu_45_layer_call_and_return_conditional_losses_950954

inputs
identityG
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:џџџџџџџџџd_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџd"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџd:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs

ђ
__inference__traced_save_951014
file_prefix@
<savev2_discriminator_11_dense_143_kernel_read_readvariableop>
:savev2_discriminator_11_dense_143_bias_read_readvariableop@
<savev2_discriminator_11_dense_144_kernel_read_readvariableop>
:savev2_discriminator_11_dense_144_bias_read_readvariableop@
<savev2_discriminator_11_dense_145_kernel_read_readvariableop>
:savev2_discriminator_11_dense_145_bias_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: њ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ѓ
valueBB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH{
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B Є
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0<savev2_discriminator_11_dense_143_kernel_read_readvariableop:savev2_discriminator_11_dense_143_bias_read_readvariableop<savev2_discriminator_11_dense_144_kernel_read_readvariableop:savev2_discriminator_11_dense_144_bias_read_readvariableop<savev2_discriminator_11_dense_145_kernel_read_readvariableop:savev2_discriminator_11_dense_145_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
	2
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*G
_input_shapes6
4: :d:d:dd:d:d:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:d: 

_output_shapes
:d:$ 

_output_shapes

:dd: 

_output_shapes
:d:$ 

_output_shapes

:d: 

_output_shapes
::

_output_shapes
: 
Ш
ў
$__inference_signature_wrapper_950855
input_1
unknown:d
	unknown_0:d
	unknown_1:dd
	unknown_2:d
	unknown_3:d
	unknown_4:
identityЂStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__wrapped_model_950671o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
Ш	
і
E__inference_dense_143_layer_call_and_return_conditional_losses_950688

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:d
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџdr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџdw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ѓ

L__inference_discriminator_11_layer_call_and_return_conditional_losses_950896

inputs:
(dense_143_matmul_readvariableop_resource:d7
)dense_143_biasadd_readvariableop_resource:d:
(dense_144_matmul_readvariableop_resource:dd7
)dense_144_biasadd_readvariableop_resource:d:
(dense_145_matmul_readvariableop_resource:d7
)dense_145_biasadd_readvariableop_resource:
identityЂ dense_143/BiasAdd/ReadVariableOpЂdense_143/MatMul/ReadVariableOpЂ dense_144/BiasAdd/ReadVariableOpЂdense_144/MatMul/ReadVariableOpЂ dense_145/BiasAdd/ReadVariableOpЂdense_145/MatMul/ReadVariableOp
dense_143/MatMul/ReadVariableOpReadVariableOp(dense_143_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0}
dense_143/MatMulMatMulinputs'dense_143/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd
 dense_143/BiasAdd/ReadVariableOpReadVariableOp)dense_143_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
dense_143/BiasAddBiasAdddense_143/MatMul:product:0(dense_143/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџdj
leaky_re_lu_44/LeakyRelu	LeakyReludense_143/BiasAdd:output:0*'
_output_shapes
:џџџџџџџџџd
dense_144/MatMul/ReadVariableOpReadVariableOp(dense_144_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0
dense_144/MatMulMatMul&leaky_re_lu_44/LeakyRelu:activations:0'dense_144/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd
 dense_144/BiasAdd/ReadVariableOpReadVariableOp)dense_144_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
dense_144/BiasAddBiasAdddense_144/MatMul:product:0(dense_144/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџdj
leaky_re_lu_45/LeakyRelu	LeakyReludense_144/BiasAdd:output:0*'
_output_shapes
:џџџџџџџџџd
dense_145/MatMul/ReadVariableOpReadVariableOp(dense_145_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0
dense_145/MatMulMatMul&leaky_re_lu_45/LeakyRelu:activations:0'dense_145/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
 dense_145/BiasAdd/ReadVariableOpReadVariableOp)dense_145_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_145/BiasAddBiasAdddense_145/MatMul:product:0(dense_145/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџi
IdentityIdentitydense_145/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp!^dense_143/BiasAdd/ReadVariableOp ^dense_143/MatMul/ReadVariableOp!^dense_144/BiasAdd/ReadVariableOp ^dense_144/MatMul/ReadVariableOp!^dense_145/BiasAdd/ReadVariableOp ^dense_145/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : : : : : 2D
 dense_143/BiasAdd/ReadVariableOp dense_143/BiasAdd/ReadVariableOp2B
dense_143/MatMul/ReadVariableOpdense_143/MatMul/ReadVariableOp2D
 dense_144/BiasAdd/ReadVariableOp dense_144/BiasAdd/ReadVariableOp2B
dense_144/MatMul/ReadVariableOpdense_144/MatMul/ReadVariableOp2D
 dense_145/BiasAdd/ReadVariableOp dense_145/BiasAdd/ReadVariableOp2B
dense_145/MatMul/ReadVariableOpdense_145/MatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ш	
і
E__inference_dense_144_layer_call_and_return_conditional_losses_950944

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџdr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџdw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
§

1__inference_discriminator_11_layer_call_fn_950872

inputs
unknown:d
	unknown_0:d
	unknown_1:dd
	unknown_2:d
	unknown_3:d
	unknown_4:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_discriminator_11_layer_call_and_return_conditional_losses_950741o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
	

1__inference_discriminator_11_layer_call_fn_950756
input_1
unknown:d
	unknown_0:d
	unknown_1:dd
	unknown_2:d
	unknown_3:d
	unknown_4:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_discriminator_11_layer_call_and_return_conditional_losses_950741o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1

Ѓ
L__inference_discriminator_11_layer_call_and_return_conditional_losses_950741

inputs"
dense_143_950689:d
dense_143_950691:d"
dense_144_950712:dd
dense_144_950714:d"
dense_145_950735:d
dense_145_950737:
identityЂ!dense_143/StatefulPartitionedCallЂ!dense_144/StatefulPartitionedCallЂ!dense_145/StatefulPartitionedCallї
!dense_143/StatefulPartitionedCallStatefulPartitionedCallinputsdense_143_950689dense_143_950691*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_143_layer_call_and_return_conditional_losses_950688ы
leaky_re_lu_44/PartitionedCallPartitionedCall*dense_143/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_44_layer_call_and_return_conditional_losses_950699
!dense_144/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_44/PartitionedCall:output:0dense_144_950712dense_144_950714*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_144_layer_call_and_return_conditional_losses_950711ы
leaky_re_lu_45/PartitionedCallPartitionedCall*dense_144/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_45_layer_call_and_return_conditional_losses_950722
!dense_145/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_45/PartitionedCall:output:0dense_145_950735dense_145_950737*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_145_layer_call_and_return_conditional_losses_950734y
IdentityIdentity*dense_145/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџВ
NoOpNoOp"^dense_143/StatefulPartitionedCall"^dense_144/StatefulPartitionedCall"^dense_145/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : : : : : 2F
!dense_143/StatefulPartitionedCall!dense_143/StatefulPartitionedCall2F
!dense_144/StatefulPartitionedCall!dense_144/StatefulPartitionedCall2F
!dense_145/StatefulPartitionedCall!dense_145/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
д
f
J__inference_leaky_re_lu_45_layer_call_and_return_conditional_losses_950722

inputs
identityG
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:џџџџџџџџџd_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџd"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџd:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
ф
р
"__inference__traced_restore_951042
file_prefixD
2assignvariableop_discriminator_11_dense_143_kernel:d@
2assignvariableop_1_discriminator_11_dense_143_bias:dF
4assignvariableop_2_discriminator_11_dense_144_kernel:dd@
2assignvariableop_3_discriminator_11_dense_144_bias:dF
4assignvariableop_4_discriminator_11_dense_145_kernel:d@
2assignvariableop_5_discriminator_11_dense_145_bias:

identity_7ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_2ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5§
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ѓ
valueBB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH~
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B С
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*0
_output_shapes
:::::::*
dtypes
	2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp2assignvariableop_discriminator_11_dense_143_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Ё
AssignVariableOp_1AssignVariableOp2assignvariableop_1_discriminator_11_dense_143_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Ѓ
AssignVariableOp_2AssignVariableOp4assignvariableop_2_discriminator_11_dense_144_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Ё
AssignVariableOp_3AssignVariableOp2assignvariableop_3_discriminator_11_dense_144_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Ѓ
AssignVariableOp_4AssignVariableOp4assignvariableop_4_discriminator_11_dense_145_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Ё
AssignVariableOp_5AssignVariableOp2assignvariableop_5_discriminator_11_dense_145_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ж

Identity_6Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_7IdentityIdentity_6:output:0^NoOp_1*
T0*
_output_shapes
: Ф
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5*"
_acd_function_control_output(*
_output_shapes
 "!

identity_7Identity_7:output:0*!
_input_shapes
: : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_5:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
О$
Л
!__inference__wrapped_model_950671
input_1K
9discriminator_11_dense_143_matmul_readvariableop_resource:dH
:discriminator_11_dense_143_biasadd_readvariableop_resource:dK
9discriminator_11_dense_144_matmul_readvariableop_resource:ddH
:discriminator_11_dense_144_biasadd_readvariableop_resource:dK
9discriminator_11_dense_145_matmul_readvariableop_resource:dH
:discriminator_11_dense_145_biasadd_readvariableop_resource:
identityЂ1discriminator_11/dense_143/BiasAdd/ReadVariableOpЂ0discriminator_11/dense_143/MatMul/ReadVariableOpЂ1discriminator_11/dense_144/BiasAdd/ReadVariableOpЂ0discriminator_11/dense_144/MatMul/ReadVariableOpЂ1discriminator_11/dense_145/BiasAdd/ReadVariableOpЂ0discriminator_11/dense_145/MatMul/ReadVariableOpЊ
0discriminator_11/dense_143/MatMul/ReadVariableOpReadVariableOp9discriminator_11_dense_143_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0 
!discriminator_11/dense_143/MatMulMatMulinput_18discriminator_11/dense_143/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџdЈ
1discriminator_11/dense_143/BiasAdd/ReadVariableOpReadVariableOp:discriminator_11_dense_143_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0Ч
"discriminator_11/dense_143/BiasAddBiasAdd+discriminator_11/dense_143/MatMul:product:09discriminator_11/dense_143/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd
)discriminator_11/leaky_re_lu_44/LeakyRelu	LeakyRelu+discriminator_11/dense_143/BiasAdd:output:0*'
_output_shapes
:џџџџџџџџџdЊ
0discriminator_11/dense_144/MatMul/ReadVariableOpReadVariableOp9discriminator_11_dense_144_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0а
!discriminator_11/dense_144/MatMulMatMul7discriminator_11/leaky_re_lu_44/LeakyRelu:activations:08discriminator_11/dense_144/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџdЈ
1discriminator_11/dense_144/BiasAdd/ReadVariableOpReadVariableOp:discriminator_11_dense_144_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0Ч
"discriminator_11/dense_144/BiasAddBiasAdd+discriminator_11/dense_144/MatMul:product:09discriminator_11/dense_144/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd
)discriminator_11/leaky_re_lu_45/LeakyRelu	LeakyRelu+discriminator_11/dense_144/BiasAdd:output:0*'
_output_shapes
:џџџџџџџџџdЊ
0discriminator_11/dense_145/MatMul/ReadVariableOpReadVariableOp9discriminator_11_dense_145_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0а
!discriminator_11/dense_145/MatMulMatMul7discriminator_11/leaky_re_lu_45/LeakyRelu:activations:08discriminator_11/dense_145/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЈ
1discriminator_11/dense_145/BiasAdd/ReadVariableOpReadVariableOp:discriminator_11_dense_145_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ч
"discriminator_11/dense_145/BiasAddBiasAdd+discriminator_11/dense_145/MatMul:product:09discriminator_11/dense_145/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџz
IdentityIdentity+discriminator_11/dense_145/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџћ
NoOpNoOp2^discriminator_11/dense_143/BiasAdd/ReadVariableOp1^discriminator_11/dense_143/MatMul/ReadVariableOp2^discriminator_11/dense_144/BiasAdd/ReadVariableOp1^discriminator_11/dense_144/MatMul/ReadVariableOp2^discriminator_11/dense_145/BiasAdd/ReadVariableOp1^discriminator_11/dense_145/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : : : : : 2f
1discriminator_11/dense_143/BiasAdd/ReadVariableOp1discriminator_11/dense_143/BiasAdd/ReadVariableOp2d
0discriminator_11/dense_143/MatMul/ReadVariableOp0discriminator_11/dense_143/MatMul/ReadVariableOp2f
1discriminator_11/dense_144/BiasAdd/ReadVariableOp1discriminator_11/dense_144/BiasAdd/ReadVariableOp2d
0discriminator_11/dense_144/MatMul/ReadVariableOp0discriminator_11/dense_144/MatMul/ReadVariableOp2f
1discriminator_11/dense_145/BiasAdd/ReadVariableOp1discriminator_11/dense_145/BiasAdd/ReadVariableOp2d
0discriminator_11/dense_145/MatMul/ReadVariableOp0discriminator_11/dense_145/MatMul/ReadVariableOp:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
Ч

*__inference_dense_143_layer_call_fn_950905

inputs
unknown:d
	unknown_0:d
identityЂStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_143_layer_call_and_return_conditional_losses_950688o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

Є
L__inference_discriminator_11_layer_call_and_return_conditional_losses_950836
input_1"
dense_143_950818:d
dense_143_950820:d"
dense_144_950824:dd
dense_144_950826:d"
dense_145_950830:d
dense_145_950832:
identityЂ!dense_143/StatefulPartitionedCallЂ!dense_144/StatefulPartitionedCallЂ!dense_145/StatefulPartitionedCallј
!dense_143/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_143_950818dense_143_950820*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_143_layer_call_and_return_conditional_losses_950688ы
leaky_re_lu_44/PartitionedCallPartitionedCall*dense_143/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_44_layer_call_and_return_conditional_losses_950699
!dense_144/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_44/PartitionedCall:output:0dense_144_950824dense_144_950826*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_144_layer_call_and_return_conditional_losses_950711ы
leaky_re_lu_45/PartitionedCallPartitionedCall*dense_144/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_45_layer_call_and_return_conditional_losses_950722
!dense_145/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_45/PartitionedCall:output:0dense_145_950830dense_145_950832*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_145_layer_call_and_return_conditional_losses_950734y
IdentityIdentity*dense_145/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџВ
NoOpNoOp"^dense_143/StatefulPartitionedCall"^dense_144/StatefulPartitionedCall"^dense_145/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : : : : : 2F
!dense_143/StatefulPartitionedCall!dense_143/StatefulPartitionedCall2F
!dense_144/StatefulPartitionedCall!dense_144/StatefulPartitionedCall2F
!dense_145/StatefulPartitionedCall!dense_145/StatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
д
f
J__inference_leaky_re_lu_44_layer_call_and_return_conditional_losses_950699

inputs
identityG
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:џџџџџџџџџd_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџd"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџd:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
Ш	
і
E__inference_dense_145_layer_call_and_return_conditional_losses_950734

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
Ш	
і
E__inference_dense_143_layer_call_and_return_conditional_losses_950915

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:d
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџdr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџdw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
д
f
J__inference_leaky_re_lu_44_layer_call_and_return_conditional_losses_950925

inputs
identityG
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:џџџџџџџџџd_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџd"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџd:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
Ќ
K
/__inference_leaky_re_lu_44_layer_call_fn_950920

inputs
identityИ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_leaky_re_lu_44_layer_call_and_return_conditional_losses_950699`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџd"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџd:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs"ПL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ћ
serving_default
;
input_10
serving_default_input_1:0џџџџџџџџџ<
output_10
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:§r

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

Dense1

	LReLU1


Dense2

LReLU2

Dense3

signatures"
_tf_keras_model
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Т
trace_0
trace_12
1__inference_discriminator_11_layer_call_fn_950756
1__inference_discriminator_11_layer_call_fn_950872Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1
ј
trace_0
trace_12С
L__inference_discriminator_11_layer_call_and_return_conditional_losses_950896
L__inference_discriminator_11_layer_call_and_return_conditional_losses_950836Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1
ЬBЩ
!__inference__wrapped_model_950671input_1"
В
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Л
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
Ѕ
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses"
_tf_keras_layer
Л
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
Ѕ
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses"
_tf_keras_layer
Л
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
,
;serving_default"
signature_map
3:1d2!discriminator_11/dense_143/kernel
-:+d2discriminator_11/dense_143/bias
3:1dd2!discriminator_11/dense_144/kernel
-:+d2discriminator_11/dense_144/bias
3:1d2!discriminator_11/dense_145/kernel
-:+2discriminator_11/dense_145/bias
 "
trackable_list_wrapper
C
0
	1

2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
цBу
1__inference_discriminator_11_layer_call_fn_950756input_1"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
хBт
1__inference_discriminator_11_layer_call_fn_950872inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B§
L__inference_discriminator_11_layer_call_and_return_conditional_losses_950896inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Bў
L__inference_discriminator_11_layer_call_and_return_conditional_losses_950836input_1"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
<non_trainable_variables

=layers
>metrics
?layer_regularization_losses
@layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
ю
Atrace_02б
*__inference_dense_143_layer_call_fn_950905Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zAtrace_0

Btrace_02ь
E__inference_dense_143_layer_call_and_return_conditional_losses_950915Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zBtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
ѓ
Htrace_02ж
/__inference_leaky_re_lu_44_layer_call_fn_950920Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zHtrace_0

Itrace_02ё
J__inference_leaky_re_lu_44_layer_call_and_return_conditional_losses_950925Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zItrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
ю
Otrace_02б
*__inference_dense_144_layer_call_fn_950934Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zOtrace_0

Ptrace_02ь
E__inference_dense_144_layer_call_and_return_conditional_losses_950944Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zPtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Qnon_trainable_variables

Rlayers
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
ѓ
Vtrace_02ж
/__inference_leaky_re_lu_45_layer_call_fn_950949Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zVtrace_0

Wtrace_02ё
J__inference_leaky_re_lu_45_layer_call_and_return_conditional_losses_950954Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zWtrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Xnon_trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\layer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
ю
]trace_02б
*__inference_dense_145_layer_call_fn_950963Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z]trace_0

^trace_02ь
E__inference_dense_145_layer_call_and_return_conditional_losses_950973Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z^trace_0
ЫBШ
$__inference_signature_wrapper_950855input_1"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
оBл
*__inference_dense_143_layer_call_fn_950905inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
љBі
E__inference_dense_143_layer_call_and_return_conditional_losses_950915inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
уBр
/__inference_leaky_re_lu_44_layer_call_fn_950920inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ўBћ
J__inference_leaky_re_lu_44_layer_call_and_return_conditional_losses_950925inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
оBл
*__inference_dense_144_layer_call_fn_950934inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
љBі
E__inference_dense_144_layer_call_and_return_conditional_losses_950944inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
уBр
/__inference_leaky_re_lu_45_layer_call_fn_950949inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ўBћ
J__inference_leaky_re_lu_45_layer_call_and_return_conditional_losses_950954inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
оBл
*__inference_dense_145_layer_call_fn_950963inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
љBі
E__inference_dense_145_layer_call_and_return_conditional_losses_950973inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
!__inference__wrapped_model_950671o0Ђ-
&Ђ#
!
input_1џџџџџџџџџ
Њ "3Њ0
.
output_1"
output_1џџџџџџџџџЅ
E__inference_dense_143_layer_call_and_return_conditional_losses_950915\/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџd
 }
*__inference_dense_143_layer_call_fn_950905O/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџdЅ
E__inference_dense_144_layer_call_and_return_conditional_losses_950944\/Ђ,
%Ђ"
 
inputsџџџџџџџџџd
Њ "%Ђ"

0џџџџџџџџџd
 }
*__inference_dense_144_layer_call_fn_950934O/Ђ,
%Ђ"
 
inputsџџџџџџџџџd
Њ "џџџџџџџџџdЅ
E__inference_dense_145_layer_call_and_return_conditional_losses_950973\/Ђ,
%Ђ"
 
inputsџџџџџџџџџd
Њ "%Ђ"

0џџџџџџџџџ
 }
*__inference_dense_145_layer_call_fn_950963O/Ђ,
%Ђ"
 
inputsџџџџџџџџџd
Њ "џџџџџџџџџБ
L__inference_discriminator_11_layer_call_and_return_conditional_losses_950836a0Ђ-
&Ђ#
!
input_1џџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 А
L__inference_discriminator_11_layer_call_and_return_conditional_losses_950896`/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 
1__inference_discriminator_11_layer_call_fn_950756T0Ђ-
&Ђ#
!
input_1џџџџџџџџџ
Њ "џџџџџџџџџ
1__inference_discriminator_11_layer_call_fn_950872S/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџІ
J__inference_leaky_re_lu_44_layer_call_and_return_conditional_losses_950925X/Ђ,
%Ђ"
 
inputsџџџџџџџџџd
Њ "%Ђ"

0џџџџџџџџџd
 ~
/__inference_leaky_re_lu_44_layer_call_fn_950920K/Ђ,
%Ђ"
 
inputsџџџџџџџџџd
Њ "џџџџџџџџџdІ
J__inference_leaky_re_lu_45_layer_call_and_return_conditional_losses_950954X/Ђ,
%Ђ"
 
inputsџџџџџџџџџd
Њ "%Ђ"

0џџџџџџџџџd
 ~
/__inference_leaky_re_lu_45_layer_call_fn_950949K/Ђ,
%Ђ"
 
inputsџџџџџџџџџd
Њ "џџџџџџџџџdЂ
$__inference_signature_wrapper_950855z;Ђ8
Ђ 
1Њ.
,
input_1!
input_1џџџџџџџџџ"3Њ0
.
output_1"
output_1џџџџџџџџџ