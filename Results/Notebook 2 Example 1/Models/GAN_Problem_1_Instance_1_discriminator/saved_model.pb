яв
С╣
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ѕ
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
alphafloat%═╠L>"
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
delete_old_dirsbool(ѕ
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
dtypetypeѕ
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
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
┴
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
executor_typestring ѕе
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
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.9.12v2.9.0-18-gd8ce9f9c3018└У
њ
discriminator_4/dense_54/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namediscriminator_4/dense_54/bias
І
1discriminator_4/dense_54/bias/Read/ReadVariableOpReadVariableOpdiscriminator_4/dense_54/bias*
_output_shapes
:*
dtype0
џ
discriminator_4/dense_54/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*0
shared_name!discriminator_4/dense_54/kernel
Њ
3discriminator_4/dense_54/kernel/Read/ReadVariableOpReadVariableOpdiscriminator_4/dense_54/kernel*
_output_shapes

:d*
dtype0
њ
discriminator_4/dense_53/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*.
shared_namediscriminator_4/dense_53/bias
І
1discriminator_4/dense_53/bias/Read/ReadVariableOpReadVariableOpdiscriminator_4/dense_53/bias*
_output_shapes
:d*
dtype0
џ
discriminator_4/dense_53/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*0
shared_name!discriminator_4/dense_53/kernel
Њ
3discriminator_4/dense_53/kernel/Read/ReadVariableOpReadVariableOpdiscriminator_4/dense_53/kernel*
_output_shapes

:dd*
dtype0
њ
discriminator_4/dense_52/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*.
shared_namediscriminator_4/dense_52/bias
І
1discriminator_4/dense_52/bias/Read/ReadVariableOpReadVariableOpdiscriminator_4/dense_52/bias*
_output_shapes
:d*
dtype0
џ
discriminator_4/dense_52/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*0
shared_name!discriminator_4/dense_52/kernel
Њ
3discriminator_4/dense_52/kernel/Read/ReadVariableOpReadVariableOpdiscriminator_4/dense_52/kernel*
_output_shapes

:d*
dtype0

NoOpNoOp
Ф 
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Т
value▄B┘ Bм
щ
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
░
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
д
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

kernel
bias*
ј
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses* 
д
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses

kernel
bias*
ј
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses* 
д
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
_Y
VARIABLE_VALUEdiscriminator_4/dense_52/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdiscriminator_4/dense_52/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEdiscriminator_4/dense_53/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdiscriminator_4/dense_53/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEdiscriminator_4/dense_54/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdiscriminator_4/dense_54/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
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
Њ
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
Љ
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
Њ
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
Љ
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
Њ
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
:         *
dtype0*
shape:         
Ђ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1discriminator_4/dense_52/kerneldiscriminator_4/dense_52/biasdiscriminator_4/dense_53/kerneldiscriminator_4/dense_53/biasdiscriminator_4/dense_54/kerneldiscriminator_4/dense_54/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8ѓ *-
f(R&
$__inference_signature_wrapper_199085
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
▄
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename3discriminator_4/dense_52/kernel/Read/ReadVariableOp1discriminator_4/dense_52/bias/Read/ReadVariableOp3discriminator_4/dense_53/kernel/Read/ReadVariableOp1discriminator_4/dense_53/bias/Read/ReadVariableOp3discriminator_4/dense_54/kernel/Read/ReadVariableOp1discriminator_4/dense_54/bias/Read/ReadVariableOpConst*
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
GPU2*0J 8ѓ *(
f#R!
__inference__traced_save_199244
▀
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamediscriminator_4/dense_52/kerneldiscriminator_4/dense_52/biasdiscriminator_4/dense_53/kerneldiscriminator_4/dense_53/biasdiscriminator_4/dense_54/kerneldiscriminator_4/dense_54/bias*
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
GPU2*0J 8ѓ *+
f&R$
"__inference__traced_restore_199272тФ
┬#
Б
!__inference__wrapped_model_198901
input_1I
7discriminator_4_dense_52_matmul_readvariableop_resource:dF
8discriminator_4_dense_52_biasadd_readvariableop_resource:dI
7discriminator_4_dense_53_matmul_readvariableop_resource:ddF
8discriminator_4_dense_53_biasadd_readvariableop_resource:dI
7discriminator_4_dense_54_matmul_readvariableop_resource:dF
8discriminator_4_dense_54_biasadd_readvariableop_resource:
identityѕб/discriminator_4/dense_52/BiasAdd/ReadVariableOpб.discriminator_4/dense_52/MatMul/ReadVariableOpб/discriminator_4/dense_53/BiasAdd/ReadVariableOpб.discriminator_4/dense_53/MatMul/ReadVariableOpб/discriminator_4/dense_54/BiasAdd/ReadVariableOpб.discriminator_4/dense_54/MatMul/ReadVariableOpд
.discriminator_4/dense_52/MatMul/ReadVariableOpReadVariableOp7discriminator_4_dense_52_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0ю
discriminator_4/dense_52/MatMulMatMulinput_16discriminator_4/dense_52/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dц
/discriminator_4/dense_52/BiasAdd/ReadVariableOpReadVariableOp8discriminator_4_dense_52_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0┴
 discriminator_4/dense_52/BiasAddBiasAdd)discriminator_4/dense_52/MatMul:product:07discriminator_4/dense_52/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dЅ
(discriminator_4/leaky_re_lu_16/LeakyRelu	LeakyRelu)discriminator_4/dense_52/BiasAdd:output:0*'
_output_shapes
:         dд
.discriminator_4/dense_53/MatMul/ReadVariableOpReadVariableOp7discriminator_4_dense_53_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0╦
discriminator_4/dense_53/MatMulMatMul6discriminator_4/leaky_re_lu_16/LeakyRelu:activations:06discriminator_4/dense_53/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dц
/discriminator_4/dense_53/BiasAdd/ReadVariableOpReadVariableOp8discriminator_4_dense_53_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0┴
 discriminator_4/dense_53/BiasAddBiasAdd)discriminator_4/dense_53/MatMul:product:07discriminator_4/dense_53/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dЅ
(discriminator_4/leaky_re_lu_17/LeakyRelu	LeakyRelu)discriminator_4/dense_53/BiasAdd:output:0*'
_output_shapes
:         dд
.discriminator_4/dense_54/MatMul/ReadVariableOpReadVariableOp7discriminator_4_dense_54_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0╦
discriminator_4/dense_54/MatMulMatMul6discriminator_4/leaky_re_lu_17/LeakyRelu:activations:06discriminator_4/dense_54/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ц
/discriminator_4/dense_54/BiasAdd/ReadVariableOpReadVariableOp8discriminator_4_dense_54_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0┴
 discriminator_4/dense_54/BiasAddBiasAdd)discriminator_4/dense_54/MatMul:product:07discriminator_4/dense_54/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         x
IdentityIdentity)discriminator_4/dense_54/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         №
NoOpNoOp0^discriminator_4/dense_52/BiasAdd/ReadVariableOp/^discriminator_4/dense_52/MatMul/ReadVariableOp0^discriminator_4/dense_53/BiasAdd/ReadVariableOp/^discriminator_4/dense_53/MatMul/ReadVariableOp0^discriminator_4/dense_54/BiasAdd/ReadVariableOp/^discriminator_4/dense_54/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : : : : : 2b
/discriminator_4/dense_52/BiasAdd/ReadVariableOp/discriminator_4/dense_52/BiasAdd/ReadVariableOp2`
.discriminator_4/dense_52/MatMul/ReadVariableOp.discriminator_4/dense_52/MatMul/ReadVariableOp2b
/discriminator_4/dense_53/BiasAdd/ReadVariableOp/discriminator_4/dense_53/BiasAdd/ReadVariableOp2`
.discriminator_4/dense_53/MatMul/ReadVariableOp.discriminator_4/dense_53/MatMul/ReadVariableOp2b
/discriminator_4/dense_54/BiasAdd/ReadVariableOp/discriminator_4/dense_54/BiasAdd/ReadVariableOp2`
.discriminator_4/dense_54/MatMul/ReadVariableOp.discriminator_4/dense_54/MatMul/ReadVariableOp:P L
'
_output_shapes
:         
!
_user_specified_name	input_1
┼
ќ
)__inference_dense_53_layer_call_fn_199164

inputs
unknown:dd
	unknown_0:d
identityѕбStatefulPartitionedCall▄
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dense_53_layer_call_and_return_conditional_losses_198941o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
Э
Т
__inference__traced_save_199244
file_prefix>
:savev2_discriminator_4_dense_52_kernel_read_readvariableop<
8savev2_discriminator_4_dense_52_bias_read_readvariableop>
:savev2_discriminator_4_dense_53_kernel_read_readvariableop<
8savev2_discriminator_4_dense_53_bias_read_readvariableop>
:savev2_discriminator_4_dense_54_kernel_read_readvariableop<
8savev2_discriminator_4_dense_54_bias_read_readvariableop
savev2_const

identity_1ѕбMergeV2Checkpointsw
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
_temp/partЂ
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
value	B : Њ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Щ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Б
valueЎBќB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH{
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B ў
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0:savev2_discriminator_4_dense_52_kernel_read_readvariableop8savev2_discriminator_4_dense_52_bias_read_readvariableop:savev2_discriminator_4_dense_53_kernel_read_readvariableop8savev2_discriminator_4_dense_53_bias_read_readvariableop:savev2_discriminator_4_dense_54_kernel_read_readvariableop8savev2_discriminator_4_dense_54_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
	2љ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:І
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
┼
ќ
)__inference_dense_54_layer_call_fn_199193

inputs
unknown:d
	unknown_0:
identityѕбStatefulPartitionedCall▄
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dense_54_layer_call_and_return_conditional_losses_198964o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
г
K
/__inference_leaky_re_lu_17_layer_call_fn_199179

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
:         d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_leaky_re_lu_17_layer_call_and_return_conditional_losses_198952`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         d:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
н
f
J__inference_leaky_re_lu_17_layer_call_and_return_conditional_losses_199184

inputs
identityG
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:         d_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:         d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         d:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
╚
■
$__inference_signature_wrapper_199085
input_1
unknown:d
	unknown_0:d
	unknown_1:dd
	unknown_2:d
	unknown_3:d
	unknown_4:
identityѕбStatefulPartitionedCallЬ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8ѓ **
f%R#
!__inference__wrapped_model_198901o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_1
ж
џ
K__inference_discriminator_4_layer_call_and_return_conditional_losses_199066
input_1!
dense_52_199048:d
dense_52_199050:d!
dense_53_199054:dd
dense_53_199056:d!
dense_54_199060:d
dense_54_199062:
identityѕб dense_52/StatefulPartitionedCallб dense_53/StatefulPartitionedCallб dense_54/StatefulPartitionedCallЗ
 dense_52/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_52_199048dense_52_199050*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dense_52_layer_call_and_return_conditional_losses_198918Ж
leaky_re_lu_16/PartitionedCallPartitionedCall)dense_52/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_leaky_re_lu_16_layer_call_and_return_conditional_losses_198929ћ
 dense_53/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_16/PartitionedCall:output:0dense_53_199054dense_53_199056*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dense_53_layer_call_and_return_conditional_losses_198941Ж
leaky_re_lu_17/PartitionedCallPartitionedCall)dense_53/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_leaky_re_lu_17_layer_call_and_return_conditional_losses_198952ћ
 dense_54/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_17/PartitionedCall:output:0dense_54_199060dense_54_199062*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dense_54_layer_call_and_return_conditional_losses_198964x
IdentityIdentity)dense_54/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         »
NoOpNoOp!^dense_52/StatefulPartitionedCall!^dense_53/StatefulPartitionedCall!^dense_54/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : : : : : 2D
 dense_52/StatefulPartitionedCall dense_52/StatefulPartitionedCall2D
 dense_53/StatefulPartitionedCall dense_53/StatefulPartitionedCall2D
 dense_54/StatefulPartitionedCall dense_54/StatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_1
Т
ї
K__inference_discriminator_4_layer_call_and_return_conditional_losses_199126

inputs9
'dense_52_matmul_readvariableop_resource:d6
(dense_52_biasadd_readvariableop_resource:d9
'dense_53_matmul_readvariableop_resource:dd6
(dense_53_biasadd_readvariableop_resource:d9
'dense_54_matmul_readvariableop_resource:d6
(dense_54_biasadd_readvariableop_resource:
identityѕбdense_52/BiasAdd/ReadVariableOpбdense_52/MatMul/ReadVariableOpбdense_53/BiasAdd/ReadVariableOpбdense_53/MatMul/ReadVariableOpбdense_54/BiasAdd/ReadVariableOpбdense_54/MatMul/ReadVariableOpє
dense_52/MatMul/ReadVariableOpReadVariableOp'dense_52_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0{
dense_52/MatMulMatMulinputs&dense_52/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dё
dense_52/BiasAdd/ReadVariableOpReadVariableOp(dense_52_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0Љ
dense_52/BiasAddBiasAdddense_52/MatMul:product:0'dense_52/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         di
leaky_re_lu_16/LeakyRelu	LeakyReludense_52/BiasAdd:output:0*'
_output_shapes
:         dє
dense_53/MatMul/ReadVariableOpReadVariableOp'dense_53_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0Џ
dense_53/MatMulMatMul&leaky_re_lu_16/LeakyRelu:activations:0&dense_53/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dё
dense_53/BiasAdd/ReadVariableOpReadVariableOp(dense_53_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0Љ
dense_53/BiasAddBiasAdddense_53/MatMul:product:0'dense_53/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         di
leaky_re_lu_17/LeakyRelu	LeakyReludense_53/BiasAdd:output:0*'
_output_shapes
:         dє
dense_54/MatMul/ReadVariableOpReadVariableOp'dense_54_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0Џ
dense_54/MatMulMatMul&leaky_re_lu_17/LeakyRelu:activations:0&dense_54/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ё
dense_54/BiasAdd/ReadVariableOpReadVariableOp(dense_54_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Љ
dense_54/BiasAddBiasAdddense_54/MatMul:product:0'dense_54/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         h
IdentityIdentitydense_54/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         Ј
NoOpNoOp ^dense_52/BiasAdd/ReadVariableOp^dense_52/MatMul/ReadVariableOp ^dense_53/BiasAdd/ReadVariableOp^dense_53/MatMul/ReadVariableOp ^dense_54/BiasAdd/ReadVariableOp^dense_54/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : : : : : 2B
dense_52/BiasAdd/ReadVariableOpdense_52/BiasAdd/ReadVariableOp2@
dense_52/MatMul/ReadVariableOpdense_52/MatMul/ReadVariableOp2B
dense_53/BiasAdd/ReadVariableOpdense_53/BiasAdd/ReadVariableOp2@
dense_53/MatMul/ReadVariableOpdense_53/MatMul/ReadVariableOp2B
dense_54/BiasAdd/ReadVariableOpdense_54/BiasAdd/ReadVariableOp2@
dense_54/MatMul/ReadVariableOpdense_54/MatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
г
K
/__inference_leaky_re_lu_16_layer_call_fn_199150

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
:         d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_leaky_re_lu_16_layer_call_and_return_conditional_losses_198929`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         d:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
┼
ќ
)__inference_dense_52_layer_call_fn_199135

inputs
unknown:d
	unknown_0:d
identityѕбStatefulPartitionedCall▄
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dense_52_layer_call_and_return_conditional_losses_198918o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
н
f
J__inference_leaky_re_lu_16_layer_call_and_return_conditional_losses_198929

inputs
identityG
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:         d_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:         d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         d:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
К	
ш
D__inference_dense_53_layer_call_and_return_conditional_losses_198941

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
К	
ш
D__inference_dense_54_layer_call_and_return_conditional_losses_198964

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
н
f
J__inference_leaky_re_lu_17_layer_call_and_return_conditional_losses_198952

inputs
identityG
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:         d_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:         d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         d:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
■
і
0__inference_discriminator_4_layer_call_fn_198986
input_1
unknown:d
	unknown_0:d
	unknown_1:dd
	unknown_2:d
	unknown_3:d
	unknown_4:
identityѕбStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_discriminator_4_layer_call_and_return_conditional_losses_198971o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_1
К	
ш
D__inference_dense_53_layer_call_and_return_conditional_losses_199174

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
К	
ш
D__inference_dense_54_layer_call_and_return_conditional_losses_199203

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
К	
ш
D__inference_dense_52_layer_call_and_return_conditional_losses_199145

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:d
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
К	
ш
D__inference_dense_52_layer_call_and_return_conditional_losses_198918

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:d
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
н
f
J__inference_leaky_re_lu_16_layer_call_and_return_conditional_losses_199155

inputs
identityG
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:         d_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:         d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         d:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
╠
н
"__inference__traced_restore_199272
file_prefixB
0assignvariableop_discriminator_4_dense_52_kernel:d>
0assignvariableop_1_discriminator_4_dense_52_bias:dD
2assignvariableop_2_discriminator_4_dense_53_kernel:dd>
0assignvariableop_3_discriminator_4_dense_53_bias:dD
2assignvariableop_4_discriminator_4_dense_54_kernel:d>
0assignvariableop_5_discriminator_4_dense_54_bias:

identity_7ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_2бAssignVariableOp_3бAssignVariableOp_4бAssignVariableOp_5§
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Б
valueЎBќB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH~
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B ┴
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*0
_output_shapes
:::::::*
dtypes
	2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOpAssignVariableOp0assignvariableop_discriminator_4_dense_52_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_1AssignVariableOp0assignvariableop_1_discriminator_4_dense_52_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:А
AssignVariableOp_2AssignVariableOp2assignvariableop_2_discriminator_4_dense_53_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_3AssignVariableOp0assignvariableop_3_discriminator_4_dense_53_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:А
AssignVariableOp_4AssignVariableOp2assignvariableop_4_discriminator_4_dense_54_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_5AssignVariableOp0assignvariableop_5_discriminator_4_dense_54_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 о

Identity_6Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_7IdentityIdentity_6:output:0^NoOp_1*
T0*
_output_shapes
: ─
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
ч
Ѕ
0__inference_discriminator_4_layer_call_fn_199102

inputs
unknown:d
	unknown_0:d
	unknown_1:dd
	unknown_2:d
	unknown_3:d
	unknown_4:
identityѕбStatefulPartitionedCallЌ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_discriminator_4_layer_call_and_return_conditional_losses_198971o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Т
Ў
K__inference_discriminator_4_layer_call_and_return_conditional_losses_198971

inputs!
dense_52_198919:d
dense_52_198921:d!
dense_53_198942:dd
dense_53_198944:d!
dense_54_198965:d
dense_54_198967:
identityѕб dense_52/StatefulPartitionedCallб dense_53/StatefulPartitionedCallб dense_54/StatefulPartitionedCallз
 dense_52/StatefulPartitionedCallStatefulPartitionedCallinputsdense_52_198919dense_52_198921*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dense_52_layer_call_and_return_conditional_losses_198918Ж
leaky_re_lu_16/PartitionedCallPartitionedCall)dense_52/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_leaky_re_lu_16_layer_call_and_return_conditional_losses_198929ћ
 dense_53/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_16/PartitionedCall:output:0dense_53_198942dense_53_198944*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dense_53_layer_call_and_return_conditional_losses_198941Ж
leaky_re_lu_17/PartitionedCallPartitionedCall)dense_53/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_leaky_re_lu_17_layer_call_and_return_conditional_losses_198952ћ
 dense_54/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_17/PartitionedCall:output:0dense_54_198965dense_54_198967*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dense_54_layer_call_and_return_conditional_losses_198964x
IdentityIdentity)dense_54/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         »
NoOpNoOp!^dense_52/StatefulPartitionedCall!^dense_53/StatefulPartitionedCall!^dense_54/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : : : : : 2D
 dense_52/StatefulPartitionedCall dense_52/StatefulPartitionedCall2D
 dense_53/StatefulPartitionedCall dense_53/StatefulPartitionedCall2D
 dense_54/StatefulPartitionedCall dense_54/StatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs"┐L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ф
serving_defaultЌ
;
input_10
serving_default_input_1:0         <
output_10
StatefulPartitionedCall:0         tensorflow/serving/predict:Мr
ј
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
╩
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
└
trace_0
trace_12Ѕ
0__inference_discriminator_4_layer_call_fn_198986
0__inference_discriminator_4_layer_call_fn_199102б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 ztrace_0ztrace_1
Ш
trace_0
trace_12┐
K__inference_discriminator_4_layer_call_and_return_conditional_losses_199126
K__inference_discriminator_4_layer_call_and_return_conditional_losses_199066б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 ztrace_0ztrace_1
╠B╔
!__inference__wrapped_model_198901input_1"ў
Љ▓Ї
FullArgSpec
argsџ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
╗
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
Ц
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses"
_tf_keras_layer
╗
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
Ц
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses"
_tf_keras_layer
╗
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
1:/d2discriminator_4/dense_52/kernel
+:)d2discriminator_4/dense_52/bias
1:/dd2discriminator_4/dense_53/kernel
+:)d2discriminator_4/dense_53/bias
1:/d2discriminator_4/dense_54/kernel
+:)2discriminator_4/dense_54/bias
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
тBР
0__inference_discriminator_4_layer_call_fn_198986input_1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
СBр
0__inference_discriminator_4_layer_call_fn_199102inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 BЧ
K__inference_discriminator_4_layer_call_and_return_conditional_losses_199126inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ђB§
K__inference_discriminator_4_layer_call_and_return_conditional_losses_199066input_1"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
Г
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
ь
Atrace_02л
)__inference_dense_52_layer_call_fn_199135б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zAtrace_0
ѕ
Btrace_02в
D__inference_dense_52_layer_call_and_return_conditional_losses_199145б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zBtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Г
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
з
Htrace_02о
/__inference_leaky_re_lu_16_layer_call_fn_199150б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zHtrace_0
ј
Itrace_02ы
J__inference_leaky_re_lu_16_layer_call_and_return_conditional_losses_199155б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
Г
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
ь
Otrace_02л
)__inference_dense_53_layer_call_fn_199164б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zOtrace_0
ѕ
Ptrace_02в
D__inference_dense_53_layer_call_and_return_conditional_losses_199174б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zPtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Г
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
з
Vtrace_02о
/__inference_leaky_re_lu_17_layer_call_fn_199179б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zVtrace_0
ј
Wtrace_02ы
J__inference_leaky_re_lu_17_layer_call_and_return_conditional_losses_199184б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
Г
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
ь
]trace_02л
)__inference_dense_54_layer_call_fn_199193б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z]trace_0
ѕ
^trace_02в
D__inference_dense_54_layer_call_and_return_conditional_losses_199203б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z^trace_0
╦B╚
$__inference_signature_wrapper_199085input_1"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
ПB┌
)__inference_dense_52_layer_call_fn_199135inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЭBш
D__inference_dense_52_layer_call_and_return_conditional_losses_199145inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
сBЯ
/__inference_leaky_re_lu_16_layer_call_fn_199150inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
■Bч
J__inference_leaky_re_lu_16_layer_call_and_return_conditional_losses_199155inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
ПB┌
)__inference_dense_53_layer_call_fn_199164inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЭBш
D__inference_dense_53_layer_call_and_return_conditional_losses_199174inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
сBЯ
/__inference_leaky_re_lu_17_layer_call_fn_199179inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
■Bч
J__inference_leaky_re_lu_17_layer_call_and_return_conditional_losses_199184inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
ПB┌
)__inference_dense_54_layer_call_fn_199193inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЭBш
D__inference_dense_54_layer_call_and_return_conditional_losses_199203inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 ћ
!__inference__wrapped_model_198901o0б-
&б#
!і
input_1         
ф "3ф0
.
output_1"і
output_1         ц
D__inference_dense_52_layer_call_and_return_conditional_losses_199145\/б,
%б"
 і
inputs         
ф "%б"
і
0         d
џ |
)__inference_dense_52_layer_call_fn_199135O/б,
%б"
 і
inputs         
ф "і         dц
D__inference_dense_53_layer_call_and_return_conditional_losses_199174\/б,
%б"
 і
inputs         d
ф "%б"
і
0         d
џ |
)__inference_dense_53_layer_call_fn_199164O/б,
%б"
 і
inputs         d
ф "і         dц
D__inference_dense_54_layer_call_and_return_conditional_losses_199203\/б,
%б"
 і
inputs         d
ф "%б"
і
0         
џ |
)__inference_dense_54_layer_call_fn_199193O/б,
%б"
 і
inputs         d
ф "і         ░
K__inference_discriminator_4_layer_call_and_return_conditional_losses_199066a0б-
&б#
!і
input_1         
ф "%б"
і
0         
џ »
K__inference_discriminator_4_layer_call_and_return_conditional_losses_199126`/б,
%б"
 і
inputs         
ф "%б"
і
0         
џ ѕ
0__inference_discriminator_4_layer_call_fn_198986T0б-
&б#
!і
input_1         
ф "і         Є
0__inference_discriminator_4_layer_call_fn_199102S/б,
%б"
 і
inputs         
ф "і         д
J__inference_leaky_re_lu_16_layer_call_and_return_conditional_losses_199155X/б,
%б"
 і
inputs         d
ф "%б"
і
0         d
џ ~
/__inference_leaky_re_lu_16_layer_call_fn_199150K/б,
%б"
 і
inputs         d
ф "і         dд
J__inference_leaky_re_lu_17_layer_call_and_return_conditional_losses_199184X/б,
%б"
 і
inputs         d
ф "%б"
і
0         d
џ ~
/__inference_leaky_re_lu_17_layer_call_fn_199179K/б,
%б"
 і
inputs         d
ф "і         dб
$__inference_signature_wrapper_199085z;б8
б 
1ф.
,
input_1!і
input_1         "3ф0
.
output_1"і
output_1         