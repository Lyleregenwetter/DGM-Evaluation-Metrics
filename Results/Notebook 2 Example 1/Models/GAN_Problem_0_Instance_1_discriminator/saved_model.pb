√к
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
 И"serve*2.9.12v2.9.0-18-gd8ce9f9c3018яз
Т
discriminator_1/dense_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namediscriminator_1/dense_15/bias
Л
1discriminator_1/dense_15/bias/Read/ReadVariableOpReadVariableOpdiscriminator_1/dense_15/bias*
_output_shapes
:*
dtype0
Ъ
discriminator_1/dense_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*0
shared_name!discriminator_1/dense_15/kernel
У
3discriminator_1/dense_15/kernel/Read/ReadVariableOpReadVariableOpdiscriminator_1/dense_15/kernel*
_output_shapes

:d*
dtype0
Т
discriminator_1/dense_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*.
shared_namediscriminator_1/dense_14/bias
Л
1discriminator_1/dense_14/bias/Read/ReadVariableOpReadVariableOpdiscriminator_1/dense_14/bias*
_output_shapes
:d*
dtype0
Ъ
discriminator_1/dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*0
shared_name!discriminator_1/dense_14/kernel
У
3discriminator_1/dense_14/kernel/Read/ReadVariableOpReadVariableOpdiscriminator_1/dense_14/kernel*
_output_shapes

:dd*
dtype0
Т
discriminator_1/dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*.
shared_namediscriminator_1/dense_13/bias
Л
1discriminator_1/dense_13/bias/Read/ReadVariableOpReadVariableOpdiscriminator_1/dense_13/bias*
_output_shapes
:d*
dtype0
Ъ
discriminator_1/dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*0
shared_name!discriminator_1/dense_13/kernel
У
3discriminator_1/dense_13/kernel/Read/ReadVariableOpReadVariableOpdiscriminator_1/dense_13/kernel*
_output_shapes

:d*
dtype0

NoOpNoOp
Ђ 
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ж
value№Bў B“
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
_Y
VARIABLE_VALUEdiscriminator_1/dense_13/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdiscriminator_1/dense_13/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEdiscriminator_1/dense_14/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdiscriminator_1/dense_14/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEdiscriminator_1/dense_15/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdiscriminator_1/dense_15/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
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
А
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1discriminator_1/dense_13/kerneldiscriminator_1/dense_13/biasdiscriminator_1/dense_14/kerneldiscriminator_1/dense_14/biasdiscriminator_1/dense_15/kerneldiscriminator_1/dense_15/bias*
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
GPU2*0J 8В *,
f'R%
#__inference_signature_wrapper_64895
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
џ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename3discriminator_1/dense_13/kernel/Read/ReadVariableOp1discriminator_1/dense_13/bias/Read/ReadVariableOp3discriminator_1/dense_14/kernel/Read/ReadVariableOp1discriminator_1/dense_14/bias/Read/ReadVariableOp3discriminator_1/dense_15/kernel/Read/ReadVariableOp1discriminator_1/dense_15/bias/Read/ReadVariableOpConst*
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
GPU2*0J 8В *'
f"R 
__inference__traced_save_65054
ё
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamediscriminator_1/dense_13/kerneldiscriminator_1/dense_13/biasdiscriminator_1/dense_14/kerneldiscriminator_1/dense_14/biasdiscriminator_1/dense_15/kerneldiscriminator_1/dense_15/bias*
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
GPU2*0J 8В **
f%R#
!__inference__traced_restore_65082ЗЂ
“
d
H__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_64965

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
√
Х
(__inference_dense_15_layer_call_fn_65003

inputs
unknown:d
	unknown_0:
identityИҐStatefulPartitionedCallџ
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
GPU2*0J 8В *L
fGRE
C__inference_dense_15_layer_call_and_return_conditional_losses_64774o
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
√
Х
(__inference_dense_14_layer_call_fn_64974

inputs
unknown:dd
	unknown_0:d
identityИҐStatefulPartitionedCallџ
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
GPU2*0J 8В *L
fGRE
C__inference_dense_14_layer_call_and_return_conditional_losses_64751o
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
®
I
-__inference_leaky_re_lu_5_layer_call_fn_64989

inputs
identityґ
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
GPU2*0J 8В *Q
fLRJ
H__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_64762`
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
∆	
ф
C__inference_dense_15_layer_call_and_return_conditional_losses_65013

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
—
У
J__inference_discriminator_1_layer_call_and_return_conditional_losses_64876
input_1 
dense_13_64858:d
dense_13_64860:d 
dense_14_64864:dd
dense_14_64866:d 
dense_15_64870:d
dense_15_64872:
identityИҐ dense_13/StatefulPartitionedCallҐ dense_14/StatefulPartitionedCallҐ dense_15/StatefulPartitionedCallс
 dense_13/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_13_64858dense_13_64860*
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
GPU2*0J 8В *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_64728з
leaky_re_lu_4/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
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
GPU2*0J 8В *Q
fLRJ
H__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_64739Р
 dense_14/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0dense_14_64864dense_14_64866*
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
GPU2*0J 8В *L
fGRE
C__inference_dense_14_layer_call_and_return_conditional_losses_64751з
leaky_re_lu_5/PartitionedCallPartitionedCall)dense_14/StatefulPartitionedCall:output:0*
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
GPU2*0J 8В *Q
fLRJ
H__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_64762Р
 dense_15/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_5/PartitionedCall:output:0dense_15_64870dense_15_64872*
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
GPU2*0J 8В *L
fGRE
C__inference_dense_15_layer_call_and_return_conditional_losses_64774x
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ѓ
NoOpNoOp!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : : : : : 2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall:P L
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
“
d
H__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_64762

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
“
d
H__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_64739

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
“
d
H__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_64994

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
щ
И
/__inference_discriminator_1_layer_call_fn_64912

inputs
unknown:d
	unknown_0:d
	unknown_1:dd
	unknown_2:d
	unknown_3:d
	unknown_4:
identityИҐStatefulPartitionedCallЦ
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
GPU2*0J 8В *S
fNRL
J__inference_discriminator_1_layer_call_and_return_conditional_losses_64781o
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
Ћ
”
!__inference__traced_restore_65082
file_prefixB
0assignvariableop_discriminator_1_dense_13_kernel:d>
0assignvariableop_1_discriminator_1_dense_13_bias:dD
2assignvariableop_2_discriminator_1_dense_14_kernel:dd>
0assignvariableop_3_discriminator_1_dense_14_bias:dD
2assignvariableop_4_discriminator_1_dense_15_kernel:d>
0assignvariableop_5_discriminator_1_dense_15_bias:

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
:Ы
AssignVariableOpAssignVariableOp0assignvariableop_discriminator_1_dense_13_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_1AssignVariableOp0assignvariableop_1_discriminator_1_dense_13_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_2AssignVariableOp2assignvariableop_2_discriminator_1_dense_14_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_3AssignVariableOp0assignvariableop_3_discriminator_1_dense_14_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_4AssignVariableOp2assignvariableop_4_discriminator_1_dense_15_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_5AssignVariableOp0assignvariableop_5_discriminator_1_dense_15_biasIdentity_5:output:0"/device:CPU:0*
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
ь
Й
/__inference_discriminator_1_layer_call_fn_64796
input_1
unknown:d
	unknown_0:d
	unknown_1:dd
	unknown_2:d
	unknown_3:d
	unknown_4:
identityИҐStatefulPartitionedCallЧ
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
GPU2*0J 8В *S
fNRL
J__inference_discriminator_1_layer_call_and_return_conditional_losses_64781o
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
∆	
ф
C__inference_dense_13_layer_call_and_return_conditional_losses_64955

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
∆	
ф
C__inference_dense_14_layer_call_and_return_conditional_losses_64751

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
б
Л
J__inference_discriminator_1_layer_call_and_return_conditional_losses_64936

inputs9
'dense_13_matmul_readvariableop_resource:d6
(dense_13_biasadd_readvariableop_resource:d9
'dense_14_matmul_readvariableop_resource:dd6
(dense_14_biasadd_readvariableop_resource:d9
'dense_15_matmul_readvariableop_resource:d6
(dense_15_biasadd_readvariableop_resource:
identityИҐdense_13/BiasAdd/ReadVariableOpҐdense_13/MatMul/ReadVariableOpҐdense_14/BiasAdd/ReadVariableOpҐdense_14/MatMul/ReadVariableOpҐdense_15/BiasAdd/ReadVariableOpҐdense_15/MatMul/ReadVariableOpЖ
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0{
dense_13/MatMulMatMulinputs&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dД
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0С
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dh
leaky_re_lu_4/LeakyRelu	LeakyReludense_13/BiasAdd:output:0*'
_output_shapes
:€€€€€€€€€dЖ
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0Ъ
dense_14/MatMulMatMul%leaky_re_lu_4/LeakyRelu:activations:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dД
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0С
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dh
leaky_re_lu_5/LeakyRelu	LeakyReludense_14/BiasAdd:output:0*'
_output_shapes
:€€€€€€€€€dЖ
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0Ъ
dense_15/MatMulMatMul%leaky_re_lu_5/LeakyRelu:activations:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Д
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€h
IdentityIdentitydense_15/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€П
NoOpNoOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : : : : : 2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
®
I
-__inference_leaky_re_lu_4_layer_call_fn_64960

inputs
identityґ
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
GPU2*0J 8В *Q
fLRJ
H__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_64739`
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
∆	
ф
C__inference_dense_15_layer_call_and_return_conditional_losses_64774

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
∆	
ф
C__inference_dense_14_layer_call_and_return_conditional_losses_64984

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
ч
е
__inference__traced_save_65054
file_prefix>
:savev2_discriminator_1_dense_13_kernel_read_readvariableop<
8savev2_discriminator_1_dense_13_bias_read_readvariableop>
:savev2_discriminator_1_dense_14_kernel_read_readvariableop<
8savev2_discriminator_1_dense_14_bias_read_readvariableop>
:savev2_discriminator_1_dense_15_kernel_read_readvariableop<
8savev2_discriminator_1_dense_15_bias_read_readvariableop
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
valueBB B B B B B B Ш
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0:savev2_discriminator_1_dense_13_kernel_read_readvariableop8savev2_discriminator_1_dense_13_bias_read_readvariableop:savev2_discriminator_1_dense_14_kernel_read_readvariableop8savev2_discriminator_1_dense_14_bias_read_readvariableop:savev2_discriminator_1_dense_15_kernel_read_readvariableop8savev2_discriminator_1_dense_15_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
љ#
Ґ
 __inference__wrapped_model_64711
input_1I
7discriminator_1_dense_13_matmul_readvariableop_resource:dF
8discriminator_1_dense_13_biasadd_readvariableop_resource:dI
7discriminator_1_dense_14_matmul_readvariableop_resource:ddF
8discriminator_1_dense_14_biasadd_readvariableop_resource:dI
7discriminator_1_dense_15_matmul_readvariableop_resource:dF
8discriminator_1_dense_15_biasadd_readvariableop_resource:
identityИҐ/discriminator_1/dense_13/BiasAdd/ReadVariableOpҐ.discriminator_1/dense_13/MatMul/ReadVariableOpҐ/discriminator_1/dense_14/BiasAdd/ReadVariableOpҐ.discriminator_1/dense_14/MatMul/ReadVariableOpҐ/discriminator_1/dense_15/BiasAdd/ReadVariableOpҐ.discriminator_1/dense_15/MatMul/ReadVariableOp¶
.discriminator_1/dense_13/MatMul/ReadVariableOpReadVariableOp7discriminator_1_dense_13_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0Ь
discriminator_1/dense_13/MatMulMatMulinput_16discriminator_1/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d§
/discriminator_1/dense_13/BiasAdd/ReadVariableOpReadVariableOp8discriminator_1_dense_13_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0Ѕ
 discriminator_1/dense_13/BiasAddBiasAdd)discriminator_1/dense_13/MatMul:product:07discriminator_1/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dИ
'discriminator_1/leaky_re_lu_4/LeakyRelu	LeakyRelu)discriminator_1/dense_13/BiasAdd:output:0*'
_output_shapes
:€€€€€€€€€d¶
.discriminator_1/dense_14/MatMul/ReadVariableOpReadVariableOp7discriminator_1_dense_14_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0 
discriminator_1/dense_14/MatMulMatMul5discriminator_1/leaky_re_lu_4/LeakyRelu:activations:06discriminator_1/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d§
/discriminator_1/dense_14/BiasAdd/ReadVariableOpReadVariableOp8discriminator_1_dense_14_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0Ѕ
 discriminator_1/dense_14/BiasAddBiasAdd)discriminator_1/dense_14/MatMul:product:07discriminator_1/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€dИ
'discriminator_1/leaky_re_lu_5/LeakyRelu	LeakyRelu)discriminator_1/dense_14/BiasAdd:output:0*'
_output_shapes
:€€€€€€€€€d¶
.discriminator_1/dense_15/MatMul/ReadVariableOpReadVariableOp7discriminator_1_dense_15_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0 
discriminator_1/dense_15/MatMulMatMul5discriminator_1/leaky_re_lu_5/LeakyRelu:activations:06discriminator_1/dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€§
/discriminator_1/dense_15/BiasAdd/ReadVariableOpReadVariableOp8discriminator_1_dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ѕ
 discriminator_1/dense_15/BiasAddBiasAdd)discriminator_1/dense_15/MatMul:product:07discriminator_1/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€x
IdentityIdentity)discriminator_1/dense_15/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€п
NoOpNoOp0^discriminator_1/dense_13/BiasAdd/ReadVariableOp/^discriminator_1/dense_13/MatMul/ReadVariableOp0^discriminator_1/dense_14/BiasAdd/ReadVariableOp/^discriminator_1/dense_14/MatMul/ReadVariableOp0^discriminator_1/dense_15/BiasAdd/ReadVariableOp/^discriminator_1/dense_15/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : : : : : 2b
/discriminator_1/dense_13/BiasAdd/ReadVariableOp/discriminator_1/dense_13/BiasAdd/ReadVariableOp2`
.discriminator_1/dense_13/MatMul/ReadVariableOp.discriminator_1/dense_13/MatMul/ReadVariableOp2b
/discriminator_1/dense_14/BiasAdd/ReadVariableOp/discriminator_1/dense_14/BiasAdd/ReadVariableOp2`
.discriminator_1/dense_14/MatMul/ReadVariableOp.discriminator_1/dense_14/MatMul/ReadVariableOp2b
/discriminator_1/dense_15/BiasAdd/ReadVariableOp/discriminator_1/dense_15/BiasAdd/ReadVariableOp2`
.discriminator_1/dense_15/MatMul/ReadVariableOp.discriminator_1/dense_15/MatMul/ReadVariableOp:P L
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
∆
э
#__inference_signature_wrapper_64895
input_1
unknown:d
	unknown_0:d
	unknown_1:dd
	unknown_2:d
	unknown_3:d
	unknown_4:
identityИҐStatefulPartitionedCallн
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
GPU2*0J 8В *)
f$R"
 __inference__wrapped_model_64711o
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
ќ
Т
J__inference_discriminator_1_layer_call_and_return_conditional_losses_64781

inputs 
dense_13_64729:d
dense_13_64731:d 
dense_14_64752:dd
dense_14_64754:d 
dense_15_64775:d
dense_15_64777:
identityИҐ dense_13/StatefulPartitionedCallҐ dense_14/StatefulPartitionedCallҐ dense_15/StatefulPartitionedCallр
 dense_13/StatefulPartitionedCallStatefulPartitionedCallinputsdense_13_64729dense_13_64731*
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
GPU2*0J 8В *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_64728з
leaky_re_lu_4/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
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
GPU2*0J 8В *Q
fLRJ
H__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_64739Р
 dense_14/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0dense_14_64752dense_14_64754*
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
GPU2*0J 8В *L
fGRE
C__inference_dense_14_layer_call_and_return_conditional_losses_64751з
leaky_re_lu_5/PartitionedCallPartitionedCall)dense_14/StatefulPartitionedCall:output:0*
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
GPU2*0J 8В *Q
fLRJ
H__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_64762Р
 dense_15/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_5/PartitionedCall:output:0dense_15_64775dense_15_64777*
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
GPU2*0J 8В *L
fGRE
C__inference_dense_15_layer_call_and_return_conditional_losses_64774x
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ѓ
NoOpNoOp!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : : : : : 2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
∆	
ф
C__inference_dense_13_layer_call_and_return_conditional_losses_64728

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
√
Х
(__inference_dense_13_layer_call_fn_64945

inputs
unknown:d
	unknown_0:d
identityИҐStatefulPartitionedCallџ
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
GPU2*0J 8В *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_64728o
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
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:Щr
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
Њ
trace_0
trace_12З
/__inference_discriminator_1_layer_call_fn_64796
/__inference_discriminator_1_layer_call_fn_64912Ґ
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
ф
trace_0
trace_12љ
J__inference_discriminator_1_layer_call_and_return_conditional_losses_64936
J__inference_discriminator_1_layer_call_and_return_conditional_losses_64876Ґ
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
ЋB»
 __inference__wrapped_model_64711input_1"Ш
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
1:/d2discriminator_1/dense_13/kernel
+:)d2discriminator_1/dense_13/bias
1:/dd2discriminator_1/dense_14/kernel
+:)d2discriminator_1/dense_14/bias
1:/d2discriminator_1/dense_15/kernel
+:)2discriminator_1/dense_15/bias
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
дBб
/__inference_discriminator_1_layer_call_fn_64796input_1"Ґ
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
гBа
/__inference_discriminator_1_layer_call_fn_64912inputs"Ґ
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
J__inference_discriminator_1_layer_call_and_return_conditional_losses_64936inputs"Ґ
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
J__inference_discriminator_1_layer_call_and_return_conditional_losses_64876input_1"Ґ
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
м
Atrace_02ѕ
(__inference_dense_13_layer_call_fn_64945Ґ
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
З
Btrace_02к
C__inference_dense_13_layer_call_and_return_conditional_losses_64955Ґ
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
с
Htrace_02‘
-__inference_leaky_re_lu_4_layer_call_fn_64960Ґ
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
М
Itrace_02п
H__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_64965Ґ
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
м
Otrace_02ѕ
(__inference_dense_14_layer_call_fn_64974Ґ
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
З
Ptrace_02к
C__inference_dense_14_layer_call_and_return_conditional_losses_64984Ґ
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
с
Vtrace_02‘
-__inference_leaky_re_lu_5_layer_call_fn_64989Ґ
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
М
Wtrace_02п
H__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_64994Ґ
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
м
]trace_02ѕ
(__inference_dense_15_layer_call_fn_65003Ґ
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
З
^trace_02к
C__inference_dense_15_layer_call_and_return_conditional_losses_65013Ґ
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
 B«
#__inference_signature_wrapper_64895input_1"Ф
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
№Bў
(__inference_dense_13_layer_call_fn_64945inputs"Ґ
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
чBф
C__inference_dense_13_layer_call_and_return_conditional_losses_64955inputs"Ґ
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
бBё
-__inference_leaky_re_lu_4_layer_call_fn_64960inputs"Ґ
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
ьBщ
H__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_64965inputs"Ґ
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
№Bў
(__inference_dense_14_layer_call_fn_64974inputs"Ґ
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
чBф
C__inference_dense_14_layer_call_and_return_conditional_losses_64984inputs"Ґ
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
бBё
-__inference_leaky_re_lu_5_layer_call_fn_64989inputs"Ґ
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
ьBщ
H__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_64994inputs"Ґ
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
№Bў
(__inference_dense_15_layer_call_fn_65003inputs"Ґ
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
чBф
C__inference_dense_15_layer_call_and_return_conditional_losses_65013inputs"Ґ
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
 У
 __inference__wrapped_model_64711o0Ґ-
&Ґ#
!К
input_1€€€€€€€€€
™ "3™0
.
output_1"К
output_1€€€€€€€€€£
C__inference_dense_13_layer_call_and_return_conditional_losses_64955\/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "%Ґ"
К
0€€€€€€€€€d
Ъ {
(__inference_dense_13_layer_call_fn_64945O/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "К€€€€€€€€€d£
C__inference_dense_14_layer_call_and_return_conditional_losses_64984\/Ґ,
%Ґ"
 К
inputs€€€€€€€€€d
™ "%Ґ"
К
0€€€€€€€€€d
Ъ {
(__inference_dense_14_layer_call_fn_64974O/Ґ,
%Ґ"
 К
inputs€€€€€€€€€d
™ "К€€€€€€€€€d£
C__inference_dense_15_layer_call_and_return_conditional_losses_65013\/Ґ,
%Ґ"
 К
inputs€€€€€€€€€d
™ "%Ґ"
К
0€€€€€€€€€
Ъ {
(__inference_dense_15_layer_call_fn_65003O/Ґ,
%Ґ"
 К
inputs€€€€€€€€€d
™ "К€€€€€€€€€ѓ
J__inference_discriminator_1_layer_call_and_return_conditional_losses_64876a0Ґ-
&Ґ#
!К
input_1€€€€€€€€€
™ "%Ґ"
К
0€€€€€€€€€
Ъ Ѓ
J__inference_discriminator_1_layer_call_and_return_conditional_losses_64936`/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "%Ґ"
К
0€€€€€€€€€
Ъ З
/__inference_discriminator_1_layer_call_fn_64796T0Ґ-
&Ґ#
!К
input_1€€€€€€€€€
™ "К€€€€€€€€€Ж
/__inference_discriminator_1_layer_call_fn_64912S/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "К€€€€€€€€€§
H__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_64965X/Ґ,
%Ґ"
 К
inputs€€€€€€€€€d
™ "%Ґ"
К
0€€€€€€€€€d
Ъ |
-__inference_leaky_re_lu_4_layer_call_fn_64960K/Ґ,
%Ґ"
 К
inputs€€€€€€€€€d
™ "К€€€€€€€€€d§
H__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_64994X/Ґ,
%Ґ"
 К
inputs€€€€€€€€€d
™ "%Ґ"
К
0€€€€€€€€€d
Ъ |
-__inference_leaky_re_lu_5_layer_call_fn_64989K/Ґ,
%Ґ"
 К
inputs€€€€€€€€€d
™ "К€€€€€€€€€d°
#__inference_signature_wrapper_64895z;Ґ8
Ґ 
1™.
,
input_1!К
input_1€€€€€€€€€"3™0
.
output_1"К
output_1€€€€€€€€€