До
дє
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
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
alphafloat%ЌћL>"
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
delete_old_dirsbool(И
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
dtypetypeИ
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
Ѕ
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
executor_typestring И®
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
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.9.12v2.9.0-18-gd8ce9f9c3018ќк
Ф
discriminator_8/dense_106/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name discriminator_8/dense_106/bias
Н
2discriminator_8/dense_106/bias/Read/ReadVariableOpReadVariableOpdiscriminator_8/dense_106/bias*
_output_shapes
:*
dtype0
Ь
 discriminator_8/dense_106/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*1
shared_name" discriminator_8/dense_106/kernel
Х
4discriminator_8/dense_106/kernel/Read/ReadVariableOpReadVariableOp discriminator_8/dense_106/kernel*
_output_shapes

:d*
dtype0
Ф
discriminator_8/dense_105/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*/
shared_name discriminator_8/dense_105/bias
Н
2discriminator_8/dense_105/bias/Read/ReadVariableOpReadVariableOpdiscriminator_8/dense_105/bias*
_output_shapes
:d*
dtype0
Ь
 discriminator_8/dense_105/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*1
shared_name" discriminator_8/dense_105/kernel
Х
4discriminator_8/dense_105/kernel/Read/ReadVariableOpReadVariableOp discriminator_8/dense_105/kernel*
_output_shapes

:dd*
dtype0
Ф
discriminator_8/dense_104/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*/
shared_name discriminator_8/dense_104/bias
Н
2discriminator_8/dense_104/bias/Read/ReadVariableOpReadVariableOpdiscriminator_8/dense_104/bias*
_output_shapes
:d*
dtype0
Ь
 discriminator_8/dense_104/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*1
shared_name" discriminator_8/dense_104/kernel
Х
4discriminator_8/dense_104/kernel/Read/ReadVariableOpReadVariableOp discriminator_8/dense_104/kernel*
_output_shapes

:d*
dtype0

NoOpNoOp
± 
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*м
valueвBя BЎ
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
∞
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
¶
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

kernel
bias*
О
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses* 
¶
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses

kernel
bias*
О
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses* 
¶
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
`Z
VARIABLE_VALUE discriminator_8/dense_104/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEdiscriminator_8/dense_104/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUE discriminator_8/dense_105/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEdiscriminator_8/dense_105/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUE discriminator_8/dense_106/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEdiscriminator_8/dense_106/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
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
У
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
С
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
У
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
С
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
У
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
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
З
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1 discriminator_8/dense_104/kerneldiscriminator_8/dense_104/bias discriminator_8/dense_105/kerneldiscriminator_8/dense_105/bias discriminator_8/dense_106/kerneldiscriminator_8/dense_106/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *-
f(R&
$__inference_signature_wrapper_816665
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
в
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename4discriminator_8/dense_104/kernel/Read/ReadVariableOp2discriminator_8/dense_104/bias/Read/ReadVariableOp4discriminator_8/dense_105/kernel/Read/ReadVariableOp2discriminator_8/dense_105/bias/Read/ReadVariableOp4discriminator_8/dense_106/kernel/Read/ReadVariableOp2discriminator_8/dense_106/bias/Read/ReadVariableOpConst*
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
GPU2*0J 8В *(
f#R!
__inference__traced_save_816824
е
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename discriminator_8/dense_104/kerneldiscriminator_8/dense_104/bias discriminator_8/dense_105/kerneldiscriminator_8/dense_105/bias discriminator_8/dense_106/kerneldiscriminator_8/dense_106/bias*
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
GPU2*0J 8В *+
f&R$
"__inference__traced_restore_816852√≠
ђ
K
/__inference_leaky_re_lu_32_layer_call_fn_816730

inputs
identityЄ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_leaky_re_lu_32_layer_call_and_return_conditional_losses_816509`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€d:O K
'
_output_shapes
:€€€€€€€€€d
 
_user_specified_nameinputs
ю#
ѓ
!__inference__wrapped_model_816481
input_1J
8discriminator_8_dense_104_matmul_readvariableop_resource:dG
9discriminator_8_dense_104_biasadd_readvariableop_resource:dJ
8discriminator_8_dense_105_matmul_readvariableop_resource:ddG
9discriminator_8_dense_105_biasadd_readvariableop_resource:dJ
8discriminator_8_dense_106_matmul_readvariableop_resource:dG
9discriminator_8_dense_106_biasadd_readvariableop_resource:
identityИҐ0discriminator_8/dense_104/BiasAdd/ReadVariableOpҐ/discriminator_8/dense_104/MatMul/ReadVariableOpҐ0discriminator_8/dense_105/BiasAdd/ReadVariableOpҐ/discriminator_8/dense_105/MatMul/ReadVariableOpҐ0discriminator_8/dense_106/BiasAdd/ReadVariableOpҐ/discriminator_8/dense_106/MatMul/ReadVariableOp®
/discriminator_8/dense_104/MatMul/ReadVariableOpReadVariableOp8discriminator_8_dense_104_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0Ю
 discriminator_8/dense_104/MatMulMatMulinput_17discriminator_8/dense_104/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d¶
0discriminator_8/dense_104/BiasAdd/ReadVariableOpReadVariableOp9discriminator_8_dense_104_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0ƒ
!discriminator_8/dense_104/BiasAddBiasAdd*discriminator_8/dense_104/MatMul:product:08discriminator_8/dense_104/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dК
(discriminator_8/leaky_re_lu_32/LeakyRelu	LeakyRelu*discriminator_8/dense_104/BiasAdd:output:0*'
_output_shapes
:€€€€€€€€€d®
/discriminator_8/dense_105/MatMul/ReadVariableOpReadVariableOp8discriminator_8_dense_105_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0Ќ
 discriminator_8/dense_105/MatMulMatMul6discriminator_8/leaky_re_lu_32/LeakyRelu:activations:07discriminator_8/dense_105/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d¶
0discriminator_8/dense_105/BiasAdd/ReadVariableOpReadVariableOp9discriminator_8_dense_105_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0ƒ
!discriminator_8/dense_105/BiasAddBiasAdd*discriminator_8/dense_105/MatMul:product:08discriminator_8/dense_105/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dК
(discriminator_8/leaky_re_lu_33/LeakyRelu	LeakyRelu*discriminator_8/dense_105/BiasAdd:output:0*'
_output_shapes
:€€€€€€€€€d®
/discriminator_8/dense_106/MatMul/ReadVariableOpReadVariableOp8discriminator_8_dense_106_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0Ќ
 discriminator_8/dense_106/MatMulMatMul6discriminator_8/leaky_re_lu_33/LeakyRelu:activations:07discriminator_8/dense_106/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€¶
0discriminator_8/dense_106/BiasAdd/ReadVariableOpReadVariableOp9discriminator_8_dense_106_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ƒ
!discriminator_8/dense_106/BiasAddBiasAdd*discriminator_8/dense_106/MatMul:product:08discriminator_8/dense_106/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€y
IdentityIdentity*discriminator_8/dense_106/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€х
NoOpNoOp1^discriminator_8/dense_104/BiasAdd/ReadVariableOp0^discriminator_8/dense_104/MatMul/ReadVariableOp1^discriminator_8/dense_105/BiasAdd/ReadVariableOp0^discriminator_8/dense_105/MatMul/ReadVariableOp1^discriminator_8/dense_106/BiasAdd/ReadVariableOp0^discriminator_8/dense_106/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : : : : : 2d
0discriminator_8/dense_104/BiasAdd/ReadVariableOp0discriminator_8/dense_104/BiasAdd/ReadVariableOp2b
/discriminator_8/dense_104/MatMul/ReadVariableOp/discriminator_8/dense_104/MatMul/ReadVariableOp2d
0discriminator_8/dense_105/BiasAdd/ReadVariableOp0discriminator_8/dense_105/BiasAdd/ReadVariableOp2b
/discriminator_8/dense_105/MatMul/ReadVariableOp/discriminator_8/dense_105/MatMul/ReadVariableOp2d
0discriminator_8/dense_106/BiasAdd/ReadVariableOp0discriminator_8/dense_106/BiasAdd/ReadVariableOp2b
/discriminator_8/dense_106/MatMul/ReadVariableOp/discriminator_8/dense_106/MatMul/ReadVariableOp:P L
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
‘
f
J__inference_leaky_re_lu_33_layer_call_and_return_conditional_losses_816764

inputs
identityG
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:€€€€€€€€€d_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€d:O K
'
_output_shapes
:€€€€€€€€€d
 
_user_specified_nameinputs
»	
ц
E__inference_dense_105_layer_call_and_return_conditional_losses_816754

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€d
 
_user_specified_nameinputs
Ў
Џ
"__inference__traced_restore_816852
file_prefixC
1assignvariableop_discriminator_8_dense_104_kernel:d?
1assignvariableop_1_discriminator_8_dense_104_bias:dE
3assignvariableop_2_discriminator_8_dense_105_kernel:dd?
1assignvariableop_3_discriminator_8_dense_105_bias:dE
3assignvariableop_4_discriminator_8_dense_106_kernel:d?
1assignvariableop_5_discriminator_8_dense_106_bias:

identity_7ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_2ҐAssignVariableOp_3ҐAssignVariableOp_4ҐAssignVariableOp_5э
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*£
valueЩBЦB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH~
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B Ѕ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*0
_output_shapes
:::::::*
dtypes
	2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOpAssignVariableOp1assignvariableop_discriminator_8_dense_104_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:†
AssignVariableOp_1AssignVariableOp1assignvariableop_1_discriminator_8_dense_104_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Ґ
AssignVariableOp_2AssignVariableOp3assignvariableop_2_discriminator_8_dense_105_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:†
AssignVariableOp_3AssignVariableOp1assignvariableop_3_discriminator_8_dense_105_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Ґ
AssignVariableOp_4AssignVariableOp3assignvariableop_4_discriminator_8_dense_106_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:†
AssignVariableOp_5AssignVariableOp1assignvariableop_5_discriminator_8_dense_106_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ÷

Identity_6Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_7IdentityIdentity_6:output:0^NoOp_1*
T0*
_output_shapes
: ƒ
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
‘
f
J__inference_leaky_re_lu_32_layer_call_and_return_conditional_losses_816509

inputs
identityG
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:€€€€€€€€€d_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€d:O K
'
_output_shapes
:€€€€€€€€€d
 
_user_specified_nameinputs
ы
Й
0__inference_discriminator_8_layer_call_fn_816682

inputs
unknown:d
	unknown_0:d
	unknown_1:dd
	unknown_2:d
	unknown_3:d
	unknown_4:
identityИҐStatefulPartitionedCallЧ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_discriminator_8_layer_call_and_return_conditional_losses_816551o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Д
м
__inference__traced_save_816824
file_prefix?
;savev2_discriminator_8_dense_104_kernel_read_readvariableop=
9savev2_discriminator_8_dense_104_bias_read_readvariableop?
;savev2_discriminator_8_dense_105_kernel_read_readvariableop=
9savev2_discriminator_8_dense_105_bias_read_readvariableop?
;savev2_discriminator_8_dense_106_kernel_read_readvariableop=
9savev2_discriminator_8_dense_106_bias_read_readvariableop
savev2_const

identity_1ИҐMergeV2Checkpointsw
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
_temp/partБ
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
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ъ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*£
valueЩBЦB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH{
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B Ю
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0;savev2_discriminator_8_dense_104_kernel_read_readvariableop9savev2_discriminator_8_dense_104_bias_read_readvariableop;savev2_discriminator_8_dense_105_kernel_read_readvariableop9savev2_discriminator_8_dense_105_bias_read_readvariableop;savev2_discriminator_8_dense_106_kernel_read_readvariableop9savev2_discriminator_8_dense_106_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
	2Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Л
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
»	
ц
E__inference_dense_105_layer_call_and_return_conditional_losses_816521

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€d
 
_user_specified_nameinputs
‘
f
J__inference_leaky_re_lu_33_layer_call_and_return_conditional_losses_816532

inputs
identityG
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:€€€€€€€€€d_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€d:O K
'
_output_shapes
:€€€€€€€€€d
 
_user_specified_nameinputs
‘
f
J__inference_leaky_re_lu_32_layer_call_and_return_conditional_losses_816735

inputs
identityG
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:€€€€€€€€€d_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€d:O K
'
_output_shapes
:€€€€€€€€€d
 
_user_specified_nameinputs
»	
ц
E__inference_dense_104_layer_call_and_return_conditional_losses_816498

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:d
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ю
К
0__inference_discriminator_8_layer_call_fn_816566
input_1
unknown:d
	unknown_0:d
	unknown_1:dd
	unknown_2:d
	unknown_3:d
	unknown_4:
identityИҐStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_discriminator_8_layer_call_and_return_conditional_losses_816551o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
З
Ґ
K__inference_discriminator_8_layer_call_and_return_conditional_losses_816551

inputs"
dense_104_816499:d
dense_104_816501:d"
dense_105_816522:dd
dense_105_816524:d"
dense_106_816545:d
dense_106_816547:
identityИҐ!dense_104/StatefulPartitionedCallҐ!dense_105/StatefulPartitionedCallҐ!dense_106/StatefulPartitionedCallч
!dense_104/StatefulPartitionedCallStatefulPartitionedCallinputsdense_104_816499dense_104_816501*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dense_104_layer_call_and_return_conditional_losses_816498л
leaky_re_lu_32/PartitionedCallPartitionedCall*dense_104/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_leaky_re_lu_32_layer_call_and_return_conditional_losses_816509Ш
!dense_105/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_32/PartitionedCall:output:0dense_105_816522dense_105_816524*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dense_105_layer_call_and_return_conditional_losses_816521л
leaky_re_lu_33/PartitionedCallPartitionedCall*dense_105/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_leaky_re_lu_33_layer_call_and_return_conditional_losses_816532Ш
!dense_106/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_33/PartitionedCall:output:0dense_106_816545dense_106_816547*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dense_106_layer_call_and_return_conditional_losses_816544y
IdentityIdentity*dense_106/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€≤
NoOpNoOp"^dense_104/StatefulPartitionedCall"^dense_105/StatefulPartitionedCall"^dense_106/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : : : : : 2F
!dense_104/StatefulPartitionedCall!dense_104/StatefulPartitionedCall2F
!dense_105/StatefulPartitionedCall!dense_105/StatefulPartitionedCall2F
!dense_106/StatefulPartitionedCall!dense_106/StatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ґ
Ш
K__inference_discriminator_8_layer_call_and_return_conditional_losses_816706

inputs:
(dense_104_matmul_readvariableop_resource:d7
)dense_104_biasadd_readvariableop_resource:d:
(dense_105_matmul_readvariableop_resource:dd7
)dense_105_biasadd_readvariableop_resource:d:
(dense_106_matmul_readvariableop_resource:d7
)dense_106_biasadd_readvariableop_resource:
identityИҐ dense_104/BiasAdd/ReadVariableOpҐdense_104/MatMul/ReadVariableOpҐ dense_105/BiasAdd/ReadVariableOpҐdense_105/MatMul/ReadVariableOpҐ dense_106/BiasAdd/ReadVariableOpҐdense_106/MatMul/ReadVariableOpИ
dense_104/MatMul/ReadVariableOpReadVariableOp(dense_104_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0}
dense_104/MatMulMatMulinputs'dense_104/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dЖ
 dense_104/BiasAdd/ReadVariableOpReadVariableOp)dense_104_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0Ф
dense_104/BiasAddBiasAdddense_104/MatMul:product:0(dense_104/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dj
leaky_re_lu_32/LeakyRelu	LeakyReludense_104/BiasAdd:output:0*'
_output_shapes
:€€€€€€€€€dИ
dense_105/MatMul/ReadVariableOpReadVariableOp(dense_105_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0Э
dense_105/MatMulMatMul&leaky_re_lu_32/LeakyRelu:activations:0'dense_105/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dЖ
 dense_105/BiasAdd/ReadVariableOpReadVariableOp)dense_105_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0Ф
dense_105/BiasAddBiasAdddense_105/MatMul:product:0(dense_105/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dj
leaky_re_lu_33/LeakyRelu	LeakyReludense_105/BiasAdd:output:0*'
_output_shapes
:€€€€€€€€€dИ
dense_106/MatMul/ReadVariableOpReadVariableOp(dense_106_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0Э
dense_106/MatMulMatMul&leaky_re_lu_33/LeakyRelu:activations:0'dense_106/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ж
 dense_106/BiasAdd/ReadVariableOpReadVariableOp)dense_106_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ф
dense_106/BiasAddBiasAdddense_106/MatMul:product:0(dense_106/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€i
IdentityIdentitydense_106/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Х
NoOpNoOp!^dense_104/BiasAdd/ReadVariableOp ^dense_104/MatMul/ReadVariableOp!^dense_105/BiasAdd/ReadVariableOp ^dense_105/MatMul/ReadVariableOp!^dense_106/BiasAdd/ReadVariableOp ^dense_106/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : : : : : 2D
 dense_104/BiasAdd/ReadVariableOp dense_104/BiasAdd/ReadVariableOp2B
dense_104/MatMul/ReadVariableOpdense_104/MatMul/ReadVariableOp2D
 dense_105/BiasAdd/ReadVariableOp dense_105/BiasAdd/ReadVariableOp2B
dense_105/MatMul/ReadVariableOpdense_105/MatMul/ReadVariableOp2D
 dense_106/BiasAdd/ReadVariableOp dense_106/BiasAdd/ReadVariableOp2B
dense_106/MatMul/ReadVariableOpdense_106/MatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
»	
ц
E__inference_dense_106_layer_call_and_return_conditional_losses_816544

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€d
 
_user_specified_nameinputs
ђ
K
/__inference_leaky_re_lu_33_layer_call_fn_816759

inputs
identityЄ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_leaky_re_lu_33_layer_call_and_return_conditional_losses_816532`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€d:O K
'
_output_shapes
:€€€€€€€€€d
 
_user_specified_nameinputs
»
ю
$__inference_signature_wrapper_816665
input_1
unknown:d
	unknown_0:d
	unknown_1:dd
	unknown_2:d
	unknown_3:d
	unknown_4:
identityИҐStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В **
f%R#
!__inference__wrapped_model_816481o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
»	
ц
E__inference_dense_104_layer_call_and_return_conditional_losses_816725

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:d
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
«
Ч
*__inference_dense_105_layer_call_fn_816744

inputs
unknown:dd
	unknown_0:d
identityИҐStatefulPartitionedCallЁ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dense_105_layer_call_and_return_conditional_losses_816521o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€d
 
_user_specified_nameinputs
К
£
K__inference_discriminator_8_layer_call_and_return_conditional_losses_816646
input_1"
dense_104_816628:d
dense_104_816630:d"
dense_105_816634:dd
dense_105_816636:d"
dense_106_816640:d
dense_106_816642:
identityИҐ!dense_104/StatefulPartitionedCallҐ!dense_105/StatefulPartitionedCallҐ!dense_106/StatefulPartitionedCallш
!dense_104/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_104_816628dense_104_816630*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dense_104_layer_call_and_return_conditional_losses_816498л
leaky_re_lu_32/PartitionedCallPartitionedCall*dense_104/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_leaky_re_lu_32_layer_call_and_return_conditional_losses_816509Ш
!dense_105/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_32/PartitionedCall:output:0dense_105_816634dense_105_816636*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dense_105_layer_call_and_return_conditional_losses_816521л
leaky_re_lu_33/PartitionedCallPartitionedCall*dense_105/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_leaky_re_lu_33_layer_call_and_return_conditional_losses_816532Ш
!dense_106/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_33/PartitionedCall:output:0dense_106_816640dense_106_816642*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dense_106_layer_call_and_return_conditional_losses_816544y
IdentityIdentity*dense_106/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€≤
NoOpNoOp"^dense_104/StatefulPartitionedCall"^dense_105/StatefulPartitionedCall"^dense_106/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : : : : : 2F
!dense_104/StatefulPartitionedCall!dense_104/StatefulPartitionedCall2F
!dense_105/StatefulPartitionedCall!dense_105/StatefulPartitionedCall2F
!dense_106/StatefulPartitionedCall!dense_106/StatefulPartitionedCall:P L
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
«
Ч
*__inference_dense_106_layer_call_fn_816773

inputs
unknown:d
	unknown_0:
identityИҐStatefulPartitionedCallЁ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dense_106_layer_call_and_return_conditional_losses_816544o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€d
 
_user_specified_nameinputs
»	
ц
E__inference_dense_106_layer_call_and_return_conditional_losses_816783

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€d
 
_user_specified_nameinputs
«
Ч
*__inference_dense_104_layer_call_fn_816715

inputs
unknown:d
	unknown_0:d
identityИҐStatefulPartitionedCallЁ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dense_104_layer_call_and_return_conditional_losses_816498o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs"њL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ђ
serving_defaultЧ
;
input_10
serving_default_input_1:0€€€€€€€€€<
output_10
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:лr
О
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
 
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
ј
trace_0
trace_12Й
0__inference_discriminator_8_layer_call_fn_816566
0__inference_discriminator_8_layer_call_fn_816682Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 ztrace_0ztrace_1
ц
trace_0
trace_12њ
K__inference_discriminator_8_layer_call_and_return_conditional_losses_816706
K__inference_discriminator_8_layer_call_and_return_conditional_losses_816646Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 ztrace_0ztrace_1
ћB…
!__inference__wrapped_model_816481input_1"Ш
С≤Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ї
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
•
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses"
_tf_keras_layer
ї
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
•
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses"
_tf_keras_layer
ї
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
2:0d2 discriminator_8/dense_104/kernel
,:*d2discriminator_8/dense_104/bias
2:0dd2 discriminator_8/dense_105/kernel
,:*d2discriminator_8/dense_105/bias
2:0d2 discriminator_8/dense_106/kernel
,:*2discriminator_8/dense_106/bias
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
еBв
0__inference_discriminator_8_layer_call_fn_816566input_1"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
дBб
0__inference_discriminator_8_layer_call_fn_816682inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
€Bь
K__inference_discriminator_8_layer_call_and_return_conditional_losses_816706inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
АBэ
K__inference_discriminator_8_layer_call_and_return_conditional_losses_816646input_1"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
≠
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
о
Atrace_02—
*__inference_dense_104_layer_call_fn_816715Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zAtrace_0
Й
Btrace_02м
E__inference_dense_104_layer_call_and_return_conditional_losses_816725Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zBtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≠
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
у
Htrace_02÷
/__inference_leaky_re_lu_32_layer_call_fn_816730Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zHtrace_0
О
Itrace_02с
J__inference_leaky_re_lu_32_layer_call_and_return_conditional_losses_816735Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
≠
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
о
Otrace_02—
*__inference_dense_105_layer_call_fn_816744Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zOtrace_0
Й
Ptrace_02м
E__inference_dense_105_layer_call_and_return_conditional_losses_816754Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zPtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≠
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
у
Vtrace_02÷
/__inference_leaky_re_lu_33_layer_call_fn_816759Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zVtrace_0
О
Wtrace_02с
J__inference_leaky_re_lu_33_layer_call_and_return_conditional_losses_816764Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
≠
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
о
]trace_02—
*__inference_dense_106_layer_call_fn_816773Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z]trace_0
Й
^trace_02м
E__inference_dense_106_layer_call_and_return_conditional_losses_816783Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z^trace_0
ЋB»
$__inference_signature_wrapper_816665input_1"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
ёBџ
*__inference_dense_104_layer_call_fn_816715inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
щBц
E__inference_dense_104_layer_call_and_return_conditional_losses_816725inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
гBа
/__inference_leaky_re_lu_32_layer_call_fn_816730inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
юBы
J__inference_leaky_re_lu_32_layer_call_and_return_conditional_losses_816735inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
ёBџ
*__inference_dense_105_layer_call_fn_816744inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
щBц
E__inference_dense_105_layer_call_and_return_conditional_losses_816754inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
гBа
/__inference_leaky_re_lu_33_layer_call_fn_816759inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
юBы
J__inference_leaky_re_lu_33_layer_call_and_return_conditional_losses_816764inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
ёBџ
*__inference_dense_106_layer_call_fn_816773inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
щBц
E__inference_dense_106_layer_call_and_return_conditional_losses_816783inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 Ф
!__inference__wrapped_model_816481o0Ґ-
&Ґ#
!К
input_1€€€€€€€€€
™ "3™0
.
output_1"К
output_1€€€€€€€€€•
E__inference_dense_104_layer_call_and_return_conditional_losses_816725\/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "%Ґ"
К
0€€€€€€€€€d
Ъ }
*__inference_dense_104_layer_call_fn_816715O/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "К€€€€€€€€€d•
E__inference_dense_105_layer_call_and_return_conditional_losses_816754\/Ґ,
%Ґ"
 К
inputs€€€€€€€€€d
™ "%Ґ"
К
0€€€€€€€€€d
Ъ }
*__inference_dense_105_layer_call_fn_816744O/Ґ,
%Ґ"
 К
inputs€€€€€€€€€d
™ "К€€€€€€€€€d•
E__inference_dense_106_layer_call_and_return_conditional_losses_816783\/Ґ,
%Ґ"
 К
inputs€€€€€€€€€d
™ "%Ґ"
К
0€€€€€€€€€
Ъ }
*__inference_dense_106_layer_call_fn_816773O/Ґ,
%Ґ"
 К
inputs€€€€€€€€€d
™ "К€€€€€€€€€∞
K__inference_discriminator_8_layer_call_and_return_conditional_losses_816646a0Ґ-
&Ґ#
!К
input_1€€€€€€€€€
™ "%Ґ"
К
0€€€€€€€€€
Ъ ѓ
K__inference_discriminator_8_layer_call_and_return_conditional_losses_816706`/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "%Ґ"
К
0€€€€€€€€€
Ъ И
0__inference_discriminator_8_layer_call_fn_816566T0Ґ-
&Ґ#
!К
input_1€€€€€€€€€
™ "К€€€€€€€€€З
0__inference_discriminator_8_layer_call_fn_816682S/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "К€€€€€€€€€¶
J__inference_leaky_re_lu_32_layer_call_and_return_conditional_losses_816735X/Ґ,
%Ґ"
 К
inputs€€€€€€€€€d
™ "%Ґ"
К
0€€€€€€€€€d
Ъ ~
/__inference_leaky_re_lu_32_layer_call_fn_816730K/Ґ,
%Ґ"
 К
inputs€€€€€€€€€d
™ "К€€€€€€€€€d¶
J__inference_leaky_re_lu_33_layer_call_and_return_conditional_losses_816764X/Ґ,
%Ґ"
 К
inputs€€€€€€€€€d
™ "%Ґ"
К
0€€€€€€€€€d
Ъ ~
/__inference_leaky_re_lu_33_layer_call_fn_816759K/Ґ,
%Ґ"
 К
inputs€€€€€€€€€d
™ "К€€€€€€€€€dҐ
$__inference_signature_wrapper_816665z;Ґ8
Ґ 
1™.
,
input_1!К
input_1€€€€€€€€€"3™0
.
output_1"К
output_1€€€€€€€€€