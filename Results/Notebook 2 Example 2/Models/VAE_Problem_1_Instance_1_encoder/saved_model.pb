он
╓л
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
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
,
Exp
x"T
y"T"
Ttype:

2
.
Identity

input"T
output"T"	
Ttype
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
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р
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
Е
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
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
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
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
executor_typestring Ии
@
StaticRegexFullMatch	
input

output
"
patternstring
ў
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
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
 И"serve*2.9.12v2.9.0-18-gd8ce9f9c3018╙∙
t
z_log_var/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namez_log_var/bias
m
"z_log_var/bias/Read/ReadVariableOpReadVariableOpz_log_var/bias*
_output_shapes
:*
dtype0
|
z_log_var/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*!
shared_namez_log_var/kernel
u
$z_log_var/kernel/Read/ReadVariableOpReadVariableOpz_log_var/kernel*
_output_shapes

:d*
dtype0
n
z_mean/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namez_mean/bias
g
z_mean/bias/Read/ReadVariableOpReadVariableOpz_mean/bias*
_output_shapes
:*
dtype0
v
z_mean/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*
shared_namez_mean/kernel
o
!z_mean/kernel/Read/ReadVariableOpReadVariableOpz_mean/kernel*
_output_shapes

:d*
dtype0
t
dense_138/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_138/bias
m
"dense_138/bias/Read/ReadVariableOpReadVariableOpdense_138/bias*
_output_shapes
:d*
dtype0
|
dense_138/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*!
shared_namedense_138/kernel
u
$dense_138/kernel/Read/ReadVariableOpReadVariableOpdense_138/kernel*
_output_shapes

:dd*
dtype0
t
dense_137/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_137/bias
m
"dense_137/bias/Read/ReadVariableOpReadVariableOpdense_137/bias*
_output_shapes
:d*
dtype0
|
dense_137/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*!
shared_namedense_137/kernel
u
$dense_137/kernel/Read/ReadVariableOpReadVariableOpdense_137/kernel*
_output_shapes

:dd*
dtype0
t
dense_136/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_136/bias
m
"dense_136/bias/Read/ReadVariableOpReadVariableOpdense_136/bias*
_output_shapes
:d*
dtype0
|
dense_136/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*!
shared_namedense_136/kernel
u
$dense_136/kernel/Read/ReadVariableOpReadVariableOpdense_136/kernel*
_output_shapes

:d*
dtype0

NoOpNoOp
╖*
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Є)
valueш)Bх) B▐)
Ъ
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer-6
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
ж
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
ж
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
ж
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

&kernel
'bias*
ж
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses

.kernel
/bias*
ж
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses

6kernel
7bias*
О
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses* 
J
0
1
2
3
&4
'5
.6
/7
68
79*
J
0
1
2
3
&4
'5
.6
/7
68
79*
* 
░
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Ctrace_0
Dtrace_1
Etrace_2
Ftrace_3* 
6
Gtrace_0
Htrace_1
Itrace_2
Jtrace_3* 
* 

Kserving_default* 

0
1*

0
1*
* 
У
Lnon_trainable_variables

Mlayers
Nmetrics
Olayer_regularization_losses
Player_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Qtrace_0* 

Rtrace_0* 
`Z
VARIABLE_VALUEdense_136/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_136/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
У
Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Xtrace_0* 

Ytrace_0* 
`Z
VARIABLE_VALUEdense_137/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_137/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

&0
'1*

&0
'1*
* 
У
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses*

_trace_0* 

`trace_0* 
`Z
VARIABLE_VALUEdense_138/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_138/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

.0
/1*

.0
/1*
* 
У
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses*

ftrace_0* 

gtrace_0* 
]W
VARIABLE_VALUEz_mean/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEz_mean/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

60
71*

60
71*
* 
У
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses*

mtrace_0* 

ntrace_0* 
`Z
VARIABLE_VALUEz_log_var/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEz_log_var/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
С
onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses* 

ttrace_0* 

utrace_0* 
* 
5
0
1
2
3
4
5
6*
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
{
serving_default_input_21Placeholder*'
_output_shapes
:         *
dtype0*
shape:         
Ц
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_21dense_136/kerneldense_136/biasdense_137/kerneldense_137/biasdense_138/kerneldense_138/biasz_mean/kernelz_mean/biasz_log_var/kernelz_log_var/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8В *-
f(R&
$__inference_signature_wrapper_929599
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ф
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense_136/kernel/Read/ReadVariableOp"dense_136/bias/Read/ReadVariableOp$dense_137/kernel/Read/ReadVariableOp"dense_137/bias/Read/ReadVariableOp$dense_138/kernel/Read/ReadVariableOp"dense_138/bias/Read/ReadVariableOp!z_mean/kernel/Read/ReadVariableOpz_mean/bias/Read/ReadVariableOp$z_log_var/kernel/Read/ReadVariableOp"z_log_var/bias/Read/ReadVariableOpConst*
Tin
2*
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
__inference__traced_save_929962
╟
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_136/kerneldense_136/biasdense_137/kerneldense_137/biasdense_138/kerneldense_138/biasz_mean/kernelz_mean/biasz_log_var/kernelz_log_var/bias*
Tin
2*
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
"__inference__traced_restore_930002Оо
┼	
є
B__inference_z_mean_layer_call_and_return_conditional_losses_929248

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
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
Ь

Ў
E__inference_dense_138_layer_call_and_return_conditional_losses_929837

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
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
:         dP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         da
IdentityIdentityRelu:activations:0^NoOp*
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
Р#
з
C__inference_encoder_layer_call_and_return_conditional_losses_929448

inputs"
dense_136_929419:d
dense_136_929421:d"
dense_137_929424:dd
dense_137_929426:d"
dense_138_929429:dd
dense_138_929431:d
z_mean_929434:d
z_mean_929436:"
z_log_var_929439:d
z_log_var_929441:
identity

identity_1

identity_2Ив!dense_136/StatefulPartitionedCallв!dense_137/StatefulPartitionedCallв!dense_138/StatefulPartitionedCallв#sampling_10/StatefulPartitionedCallв!z_log_var/StatefulPartitionedCallвz_mean/StatefulPartitionedCallў
!dense_136/StatefulPartitionedCallStatefulPartitionedCallinputsdense_136_929419dense_136_929421*
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
GPU2*0J 8В *N
fIRG
E__inference_dense_136_layer_call_and_return_conditional_losses_929198Ы
!dense_137/StatefulPartitionedCallStatefulPartitionedCall*dense_136/StatefulPartitionedCall:output:0dense_137_929424dense_137_929426*
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
GPU2*0J 8В *N
fIRG
E__inference_dense_137_layer_call_and_return_conditional_losses_929215Ы
!dense_138/StatefulPartitionedCallStatefulPartitionedCall*dense_137/StatefulPartitionedCall:output:0dense_138_929429dense_138_929431*
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
GPU2*0J 8В *N
fIRG
E__inference_dense_138_layer_call_and_return_conditional_losses_929232П
z_mean/StatefulPartitionedCallStatefulPartitionedCall*dense_138/StatefulPartitionedCall:output:0z_mean_929434z_mean_929436*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_z_mean_layer_call_and_return_conditional_losses_929248Ы
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall*dense_138/StatefulPartitionedCall:output:0z_log_var_929439z_log_var_929441*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_z_log_var_layer_call_and_return_conditional_losses_929264Я
#sampling_10/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_sampling_10_layer_call_and_return_conditional_losses_929296v
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         {

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         }

Identity_2Identity,sampling_10/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Э
NoOpNoOp"^dense_136/StatefulPartitionedCall"^dense_137/StatefulPartitionedCall"^dense_138/StatefulPartitionedCall$^sampling_10/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : : : 2F
!dense_136/StatefulPartitionedCall!dense_136/StatefulPartitionedCall2F
!dense_137/StatefulPartitionedCall!dense_137/StatefulPartitionedCall2F
!dense_138/StatefulPartitionedCall!dense_138/StatefulPartitionedCall2J
#sampling_10/StatefulPartitionedCall#sampling_10/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
¤
u
,__inference_sampling_10_layer_call_fn_929881
inputs_0
inputs_1
identityИвStatefulPartitionedCall╥
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_sampling_10_layer_call_and_return_conditional_losses_929296o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         :         22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:         
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1
н+
Щ
"__inference__traced_restore_930002
file_prefix3
!assignvariableop_dense_136_kernel:d/
!assignvariableop_1_dense_136_bias:d5
#assignvariableop_2_dense_137_kernel:dd/
!assignvariableop_3_dense_137_bias:d5
#assignvariableop_4_dense_138_kernel:dd/
!assignvariableop_5_dense_138_bias:d2
 assignvariableop_6_z_mean_kernel:d,
assignvariableop_7_z_mean_bias:5
#assignvariableop_8_z_log_var_kernel:d/
!assignvariableop_9_z_log_var_bias:
identity_11ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_2вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9│
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*┘
value╧B╠B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЖ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B ╒
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*@
_output_shapes.
,:::::::::::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOpAssignVariableOp!assignvariableop_dense_136_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_136_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_2AssignVariableOp#assignvariableop_2_dense_137_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_137_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_4AssignVariableOp#assignvariableop_4_dense_138_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_138_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_6AssignVariableOp assignvariableop_6_z_mean_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_7AssignVariableOpassignvariableop_7_z_mean_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_8AssignVariableOp#assignvariableop_8_z_log_var_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_9AssignVariableOp!assignvariableop_9_z_log_var_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 л
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_11IdentityIdentity_10:output:0^NoOp_1*
T0*
_output_shapes
: Ш
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_11Identity_11:output:0*)
_input_shapes
: : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ь

Ў
E__inference_dense_136_layer_call_and_return_conditional_losses_929198

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:d
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
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
:         dP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         da
IdentityIdentityRelu:activations:0^NoOp*
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
Ц#
й
C__inference_encoder_layer_call_and_return_conditional_losses_929536
input_21"
dense_136_929507:d
dense_136_929509:d"
dense_137_929512:dd
dense_137_929514:d"
dense_138_929517:dd
dense_138_929519:d
z_mean_929522:d
z_mean_929524:"
z_log_var_929527:d
z_log_var_929529:
identity

identity_1

identity_2Ив!dense_136/StatefulPartitionedCallв!dense_137/StatefulPartitionedCallв!dense_138/StatefulPartitionedCallв#sampling_10/StatefulPartitionedCallв!z_log_var/StatefulPartitionedCallвz_mean/StatefulPartitionedCall∙
!dense_136/StatefulPartitionedCallStatefulPartitionedCallinput_21dense_136_929507dense_136_929509*
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
GPU2*0J 8В *N
fIRG
E__inference_dense_136_layer_call_and_return_conditional_losses_929198Ы
!dense_137/StatefulPartitionedCallStatefulPartitionedCall*dense_136/StatefulPartitionedCall:output:0dense_137_929512dense_137_929514*
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
GPU2*0J 8В *N
fIRG
E__inference_dense_137_layer_call_and_return_conditional_losses_929215Ы
!dense_138/StatefulPartitionedCallStatefulPartitionedCall*dense_137/StatefulPartitionedCall:output:0dense_138_929517dense_138_929519*
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
GPU2*0J 8В *N
fIRG
E__inference_dense_138_layer_call_and_return_conditional_losses_929232П
z_mean/StatefulPartitionedCallStatefulPartitionedCall*dense_138/StatefulPartitionedCall:output:0z_mean_929522z_mean_929524*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_z_mean_layer_call_and_return_conditional_losses_929248Ы
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall*dense_138/StatefulPartitionedCall:output:0z_log_var_929527z_log_var_929529*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_z_log_var_layer_call_and_return_conditional_losses_929264Я
#sampling_10/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_sampling_10_layer_call_and_return_conditional_losses_929296v
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         {

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         }

Identity_2Identity,sampling_10/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Э
NoOpNoOp"^dense_136/StatefulPartitionedCall"^dense_137/StatefulPartitionedCall"^dense_138/StatefulPartitionedCall$^sampling_10/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : : : 2F
!dense_136/StatefulPartitionedCall!dense_136/StatefulPartitionedCall2F
!dense_137/StatefulPartitionedCall!dense_137/StatefulPartitionedCall2F
!dense_138/StatefulPartitionedCall!dense_138/StatefulPartitionedCall2J
#sampling_10/StatefulPartitionedCall#sampling_10/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:Q M
'
_output_shapes
:         
"
_user_specified_name
input_21
З
Н
(__inference_encoder_layer_call_fn_929657

inputs
unknown:d
	unknown_0:d
	unknown_1:dd
	unknown_2:d
	unknown_3:dd
	unknown_4:d
	unknown_5:d
	unknown_6:
	unknown_7:d
	unknown_8:
identity

identity_1

identity_2ИвStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_929448o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╟
Ч
*__inference_z_log_var_layer_call_fn_929865

inputs
unknown:d
	unknown_0:
identityИвStatefulPartitionedCall▌
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_z_log_var_layer_call_and_return_conditional_losses_929264o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
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
Р#
з
C__inference_encoder_layer_call_and_return_conditional_losses_929301

inputs"
dense_136_929199:d
dense_136_929201:d"
dense_137_929216:dd
dense_137_929218:d"
dense_138_929233:dd
dense_138_929235:d
z_mean_929249:d
z_mean_929251:"
z_log_var_929265:d
z_log_var_929267:
identity

identity_1

identity_2Ив!dense_136/StatefulPartitionedCallв!dense_137/StatefulPartitionedCallв!dense_138/StatefulPartitionedCallв#sampling_10/StatefulPartitionedCallв!z_log_var/StatefulPartitionedCallвz_mean/StatefulPartitionedCallў
!dense_136/StatefulPartitionedCallStatefulPartitionedCallinputsdense_136_929199dense_136_929201*
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
GPU2*0J 8В *N
fIRG
E__inference_dense_136_layer_call_and_return_conditional_losses_929198Ы
!dense_137/StatefulPartitionedCallStatefulPartitionedCall*dense_136/StatefulPartitionedCall:output:0dense_137_929216dense_137_929218*
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
GPU2*0J 8В *N
fIRG
E__inference_dense_137_layer_call_and_return_conditional_losses_929215Ы
!dense_138/StatefulPartitionedCallStatefulPartitionedCall*dense_137/StatefulPartitionedCall:output:0dense_138_929233dense_138_929235*
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
GPU2*0J 8В *N
fIRG
E__inference_dense_138_layer_call_and_return_conditional_losses_929232П
z_mean/StatefulPartitionedCallStatefulPartitionedCall*dense_138/StatefulPartitionedCall:output:0z_mean_929249z_mean_929251*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_z_mean_layer_call_and_return_conditional_losses_929248Ы
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall*dense_138/StatefulPartitionedCall:output:0z_log_var_929265z_log_var_929267*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_z_log_var_layer_call_and_return_conditional_losses_929264Я
#sampling_10/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_sampling_10_layer_call_and_return_conditional_losses_929296v
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         {

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         }

Identity_2Identity,sampling_10/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Э
NoOpNoOp"^dense_136/StatefulPartitionedCall"^dense_137/StatefulPartitionedCall"^dense_138/StatefulPartitionedCall$^sampling_10/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : : : 2F
!dense_136/StatefulPartitionedCall!dense_136/StatefulPartitionedCall2F
!dense_137/StatefulPartitionedCall!dense_137/StatefulPartitionedCall2F
!dense_138/StatefulPartitionedCall!dense_138/StatefulPartitionedCall2J
#sampling_10/StatefulPartitionedCall#sampling_10/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
З
Н
(__inference_encoder_layer_call_fn_929628

inputs
unknown:d
	unknown_0:d
	unknown_1:dd
	unknown_2:d
	unknown_3:dd
	unknown_4:d
	unknown_5:d
	unknown_6:
	unknown_7:d
	unknown_8:
identity

identity_1

identity_2ИвStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_929301o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ч
Л
$__inference_signature_wrapper_929599
input_21
unknown:d
	unknown_0:d
	unknown_1:dd
	unknown_2:d
	unknown_3:dd
	unknown_4:d
	unknown_5:d
	unknown_6:
	unknown_7:d
	unknown_8:
identity

identity_1

identity_2ИвStatefulPartitionedCall╦
StatefulPartitionedCallStatefulPartitionedCallinput_21unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8В **
f%R#
!__inference__wrapped_model_929180o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:         
"
_user_specified_name
input_21
Ь

Ў
E__inference_dense_136_layer_call_and_return_conditional_losses_929797

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:d
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
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
:         dP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         da
IdentityIdentityRelu:activations:0^NoOp*
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
Н
П
(__inference_encoder_layer_call_fn_929504
input_21
unknown:d
	unknown_0:d
	unknown_1:dd
	unknown_2:d
	unknown_3:dd
	unknown_4:d
	unknown_5:d
	unknown_6:
	unknown_7:d
	unknown_8:
identity

identity_1

identity_2ИвStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinput_21unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_929448o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:         
"
_user_specified_name
input_21
╚	
Ў
E__inference_z_log_var_layer_call_and_return_conditional_losses_929875

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
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
╟
Ч
*__inference_dense_136_layer_call_fn_929786

inputs
unknown:d
	unknown_0:d
identityИвStatefulPartitionedCall▌
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
GPU2*0J 8В *N
fIRG
E__inference_dense_136_layer_call_and_return_conditional_losses_929198o
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
╚	
Ў
E__inference_z_log_var_layer_call_and_return_conditional_losses_929264

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
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
Ь

Ў
E__inference_dense_137_layer_call_and_return_conditional_losses_929817

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
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
:         dP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         da
IdentityIdentityRelu:activations:0^NoOp*
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
Ц#
й
C__inference_encoder_layer_call_and_return_conditional_losses_929568
input_21"
dense_136_929539:d
dense_136_929541:d"
dense_137_929544:dd
dense_137_929546:d"
dense_138_929549:dd
dense_138_929551:d
z_mean_929554:d
z_mean_929556:"
z_log_var_929559:d
z_log_var_929561:
identity

identity_1

identity_2Ив!dense_136/StatefulPartitionedCallв!dense_137/StatefulPartitionedCallв!dense_138/StatefulPartitionedCallв#sampling_10/StatefulPartitionedCallв!z_log_var/StatefulPartitionedCallвz_mean/StatefulPartitionedCall∙
!dense_136/StatefulPartitionedCallStatefulPartitionedCallinput_21dense_136_929539dense_136_929541*
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
GPU2*0J 8В *N
fIRG
E__inference_dense_136_layer_call_and_return_conditional_losses_929198Ы
!dense_137/StatefulPartitionedCallStatefulPartitionedCall*dense_136/StatefulPartitionedCall:output:0dense_137_929544dense_137_929546*
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
GPU2*0J 8В *N
fIRG
E__inference_dense_137_layer_call_and_return_conditional_losses_929215Ы
!dense_138/StatefulPartitionedCallStatefulPartitionedCall*dense_137/StatefulPartitionedCall:output:0dense_138_929549dense_138_929551*
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
GPU2*0J 8В *N
fIRG
E__inference_dense_138_layer_call_and_return_conditional_losses_929232П
z_mean/StatefulPartitionedCallStatefulPartitionedCall*dense_138/StatefulPartitionedCall:output:0z_mean_929554z_mean_929556*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_z_mean_layer_call_and_return_conditional_losses_929248Ы
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall*dense_138/StatefulPartitionedCall:output:0z_log_var_929559z_log_var_929561*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_z_log_var_layer_call_and_return_conditional_losses_929264Я
#sampling_10/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_sampling_10_layer_call_and_return_conditional_losses_929296v
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         {

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         }

Identity_2Identity,sampling_10/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Э
NoOpNoOp"^dense_136/StatefulPartitionedCall"^dense_137/StatefulPartitionedCall"^dense_138/StatefulPartitionedCall$^sampling_10/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : : : 2F
!dense_136/StatefulPartitionedCall!dense_136/StatefulPartitionedCall2F
!dense_137/StatefulPartitionedCall!dense_137/StatefulPartitionedCall2F
!dense_138/StatefulPartitionedCall!dense_138/StatefulPartitionedCall2J
#sampling_10/StatefulPartitionedCall#sampling_10/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:Q M
'
_output_shapes
:         
"
_user_specified_name
input_21
╛D
Ш
C__inference_encoder_layer_call_and_return_conditional_losses_929717

inputs:
(dense_136_matmul_readvariableop_resource:d7
)dense_136_biasadd_readvariableop_resource:d:
(dense_137_matmul_readvariableop_resource:dd7
)dense_137_biasadd_readvariableop_resource:d:
(dense_138_matmul_readvariableop_resource:dd7
)dense_138_biasadd_readvariableop_resource:d7
%z_mean_matmul_readvariableop_resource:d4
&z_mean_biasadd_readvariableop_resource::
(z_log_var_matmul_readvariableop_resource:d7
)z_log_var_biasadd_readvariableop_resource:
identity

identity_1

identity_2Ив dense_136/BiasAdd/ReadVariableOpвdense_136/MatMul/ReadVariableOpв dense_137/BiasAdd/ReadVariableOpвdense_137/MatMul/ReadVariableOpв dense_138/BiasAdd/ReadVariableOpвdense_138/MatMul/ReadVariableOpв z_log_var/BiasAdd/ReadVariableOpвz_log_var/MatMul/ReadVariableOpвz_mean/BiasAdd/ReadVariableOpвz_mean/MatMul/ReadVariableOpИ
dense_136/MatMul/ReadVariableOpReadVariableOp(dense_136_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0}
dense_136/MatMulMatMulinputs'dense_136/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dЖ
 dense_136/BiasAdd/ReadVariableOpReadVariableOp)dense_136_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0Ф
dense_136/BiasAddBiasAdddense_136/MatMul:product:0(dense_136/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dd
dense_136/ReluReludense_136/BiasAdd:output:0*
T0*'
_output_shapes
:         dИ
dense_137/MatMul/ReadVariableOpReadVariableOp(dense_137_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0У
dense_137/MatMulMatMuldense_136/Relu:activations:0'dense_137/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dЖ
 dense_137/BiasAdd/ReadVariableOpReadVariableOp)dense_137_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0Ф
dense_137/BiasAddBiasAdddense_137/MatMul:product:0(dense_137/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dd
dense_137/ReluReludense_137/BiasAdd:output:0*
T0*'
_output_shapes
:         dИ
dense_138/MatMul/ReadVariableOpReadVariableOp(dense_138_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0У
dense_138/MatMulMatMuldense_137/Relu:activations:0'dense_138/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dЖ
 dense_138/BiasAdd/ReadVariableOpReadVariableOp)dense_138_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0Ф
dense_138/BiasAddBiasAdddense_138/MatMul:product:0(dense_138/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dd
dense_138/ReluReludense_138/BiasAdd:output:0*
T0*'
_output_shapes
:         dВ
z_mean/MatMul/ReadVariableOpReadVariableOp%z_mean_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0Н
z_mean/MatMulMatMuldense_138/Relu:activations:0$z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         А
z_mean/BiasAdd/ReadVariableOpReadVariableOp&z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Л
z_mean/BiasAddBiasAddz_mean/MatMul:product:0%z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         И
z_log_var/MatMul/ReadVariableOpReadVariableOp(z_log_var_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0У
z_log_var/MatMulMatMuldense_138/Relu:activations:0'z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ж
 z_log_var/BiasAdd/ReadVariableOpReadVariableOp)z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ф
z_log_var/BiasAddBiasAddz_log_var/MatMul:product:0(z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         X
sampling_10/ShapeShapez_mean/BiasAdd:output:0*
T0*
_output_shapes
:i
sampling_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!sampling_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!sampling_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
sampling_10/strided_sliceStridedSlicesampling_10/Shape:output:0(sampling_10/strided_slice/stack:output:0*sampling_10/strided_slice/stack_1:output:0*sampling_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
sampling_10/Shape_1Shapez_mean/BiasAdd:output:0*
T0*
_output_shapes
:k
!sampling_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:m
#sampling_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#sampling_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ч
sampling_10/strided_slice_1StridedSlicesampling_10/Shape_1:output:0*sampling_10/strided_slice_1/stack:output:0,sampling_10/strided_slice_1/stack_1:output:0,sampling_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЯ
sampling_10/random_normal/shapePack"sampling_10/strided_slice:output:0$sampling_10/strided_slice_1:output:0*
N*
T0*
_output_shapes
:c
sampling_10/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    e
 sampling_10/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╬
.sampling_10/random_normal/RandomStandardNormalRandomStandardNormal(sampling_10/random_normal/shape:output:0*
T0*'
_output_shapes
:         *
dtype0*
seed▒ х)*
seed2їо╘║
sampling_10/random_normal/mulMul7sampling_10/random_normal/RandomStandardNormal:output:0)sampling_10/random_normal/stddev:output:0*
T0*'
_output_shapes
:         а
sampling_10/random_normalAddV2!sampling_10/random_normal/mul:z:0'sampling_10/random_normal/mean:output:0*
T0*'
_output_shapes
:         V
sampling_10/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?А
sampling_10/mulMulsampling_10/mul/x:output:0z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:         ]
sampling_10/ExpExpsampling_10/mul:z:0*
T0*'
_output_shapes
:         ~
sampling_10/mul_1Mulsampling_10/Exp:y:0sampling_10/random_normal:z:0*
T0*'
_output_shapes
:         z
sampling_10/addAddV2z_mean/BiasAdd:output:0sampling_10/mul_1:z:0*
T0*'
_output_shapes
:         f
IdentityIdentityz_mean/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         k

Identity_1Identityz_log_var/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         d

Identity_2Identitysampling_10/add:z:0^NoOp*
T0*'
_output_shapes
:         Щ
NoOpNoOp!^dense_136/BiasAdd/ReadVariableOp ^dense_136/MatMul/ReadVariableOp!^dense_137/BiasAdd/ReadVariableOp ^dense_137/MatMul/ReadVariableOp!^dense_138/BiasAdd/ReadVariableOp ^dense_138/MatMul/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : : : 2D
 dense_136/BiasAdd/ReadVariableOp dense_136/BiasAdd/ReadVariableOp2B
dense_136/MatMul/ReadVariableOpdense_136/MatMul/ReadVariableOp2D
 dense_137/BiasAdd/ReadVariableOp dense_137/BiasAdd/ReadVariableOp2B
dense_137/MatMul/ReadVariableOpdense_137/MatMul/ReadVariableOp2D
 dense_138/BiasAdd/ReadVariableOp dense_138/BiasAdd/ReadVariableOp2B
dense_138/MatMul/ReadVariableOpdense_138/MatMul/ReadVariableOp2D
 z_log_var/BiasAdd/ReadVariableOp z_log_var/BiasAdd/ReadVariableOp2B
z_log_var/MatMul/ReadVariableOpz_log_var/MatMul/ReadVariableOp2>
z_mean/BiasAdd/ReadVariableOpz_mean/BiasAdd/ReadVariableOp2<
z_mean/MatMul/ReadVariableOpz_mean/MatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
¤
t
G__inference_sampling_10_layer_call_and_return_conditional_losses_929296

inputs
inputs_1
identityИ;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask=
Shape_1Shapeinputs*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
random_normal/shapePackstrided_slice:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:W
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╡
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*'
_output_shapes
:         *
dtype0*
seed▒ х)*
seed2Ът;Ц
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:         |
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:         J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?V
mulMulmul/x:output:0inputs_1*
T0*'
_output_shapes
:         E
ExpExpmul:z:0*
T0*'
_output_shapes
:         Z
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:         Q
addAddV2inputs	mul_1:z:0*
T0*'
_output_shapes
:         O
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         :         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs
Ь

Ў
E__inference_dense_138_layer_call_and_return_conditional_losses_929232

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
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
:         dP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         da
IdentityIdentityRelu:activations:0^NoOp*
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
Ы 
╞
__inference__traced_save_929962
file_prefix/
+savev2_dense_136_kernel_read_readvariableop-
)savev2_dense_136_bias_read_readvariableop/
+savev2_dense_137_kernel_read_readvariableop-
)savev2_dense_137_bias_read_readvariableop/
+savev2_dense_138_kernel_read_readvariableop-
)savev2_dense_138_bias_read_readvariableop,
(savev2_z_mean_kernel_read_readvariableop*
&savev2_z_mean_bias_read_readvariableop/
+savev2_z_log_var_kernel_read_readvariableop-
)savev2_z_log_var_bias_read_readvariableop
savev2_const

identity_1ИвMergeV2Checkpointsw
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
: ░
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*┘
value╧B╠B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHГ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B ь
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense_136_kernel_read_readvariableop)savev2_dense_136_bias_read_readvariableop+savev2_dense_137_kernel_read_readvariableop)savev2_dense_137_bias_read_readvariableop+savev2_dense_138_kernel_read_readvariableop)savev2_dense_138_bias_read_readvariableop(savev2_z_mean_kernel_read_readvariableop&savev2_z_mean_bias_read_readvariableop+savev2_z_log_var_kernel_read_readvariableop)savev2_z_log_var_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2Р
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

identity_1Identity_1:output:0*g
_input_shapesV
T: :d:d:dd:d:dd:d:d::d:: 2(
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

:dd: 

_output_shapes
:d:$ 

_output_shapes

:d: 

_output_shapes
::$	 

_output_shapes

:d: 


_output_shapes
::

_output_shapes
: 
╟
Ч
*__inference_dense_137_layer_call_fn_929806

inputs
unknown:dd
	unknown_0:d
identityИвStatefulPartitionedCall▌
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
GPU2*0J 8В *N
fIRG
E__inference_dense_137_layer_call_and_return_conditional_losses_929215o
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
┼	
є
B__inference_z_mean_layer_call_and_return_conditional_losses_929856

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
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
Н
П
(__inference_encoder_layer_call_fn_929328
input_21
unknown:d
	unknown_0:d
	unknown_1:dd
	unknown_2:d
	unknown_3:dd
	unknown_4:d
	unknown_5:d
	unknown_6:
	unknown_7:d
	unknown_8:
identity

identity_1

identity_2ИвStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinput_21unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_929301o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:         
"
_user_specified_name
input_21
Ь

Ў
E__inference_dense_137_layer_call_and_return_conditional_losses_929215

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
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
:         dP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         da
IdentityIdentityRelu:activations:0^NoOp*
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
╜D
Ш
C__inference_encoder_layer_call_and_return_conditional_losses_929777

inputs:
(dense_136_matmul_readvariableop_resource:d7
)dense_136_biasadd_readvariableop_resource:d:
(dense_137_matmul_readvariableop_resource:dd7
)dense_137_biasadd_readvariableop_resource:d:
(dense_138_matmul_readvariableop_resource:dd7
)dense_138_biasadd_readvariableop_resource:d7
%z_mean_matmul_readvariableop_resource:d4
&z_mean_biasadd_readvariableop_resource::
(z_log_var_matmul_readvariableop_resource:d7
)z_log_var_biasadd_readvariableop_resource:
identity

identity_1

identity_2Ив dense_136/BiasAdd/ReadVariableOpвdense_136/MatMul/ReadVariableOpв dense_137/BiasAdd/ReadVariableOpвdense_137/MatMul/ReadVariableOpв dense_138/BiasAdd/ReadVariableOpвdense_138/MatMul/ReadVariableOpв z_log_var/BiasAdd/ReadVariableOpвz_log_var/MatMul/ReadVariableOpвz_mean/BiasAdd/ReadVariableOpвz_mean/MatMul/ReadVariableOpИ
dense_136/MatMul/ReadVariableOpReadVariableOp(dense_136_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0}
dense_136/MatMulMatMulinputs'dense_136/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dЖ
 dense_136/BiasAdd/ReadVariableOpReadVariableOp)dense_136_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0Ф
dense_136/BiasAddBiasAdddense_136/MatMul:product:0(dense_136/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dd
dense_136/ReluReludense_136/BiasAdd:output:0*
T0*'
_output_shapes
:         dИ
dense_137/MatMul/ReadVariableOpReadVariableOp(dense_137_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0У
dense_137/MatMulMatMuldense_136/Relu:activations:0'dense_137/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dЖ
 dense_137/BiasAdd/ReadVariableOpReadVariableOp)dense_137_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0Ф
dense_137/BiasAddBiasAdddense_137/MatMul:product:0(dense_137/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dd
dense_137/ReluReludense_137/BiasAdd:output:0*
T0*'
_output_shapes
:         dИ
dense_138/MatMul/ReadVariableOpReadVariableOp(dense_138_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0У
dense_138/MatMulMatMuldense_137/Relu:activations:0'dense_138/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dЖ
 dense_138/BiasAdd/ReadVariableOpReadVariableOp)dense_138_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0Ф
dense_138/BiasAddBiasAdddense_138/MatMul:product:0(dense_138/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dd
dense_138/ReluReludense_138/BiasAdd:output:0*
T0*'
_output_shapes
:         dВ
z_mean/MatMul/ReadVariableOpReadVariableOp%z_mean_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0Н
z_mean/MatMulMatMuldense_138/Relu:activations:0$z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         А
z_mean/BiasAdd/ReadVariableOpReadVariableOp&z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Л
z_mean/BiasAddBiasAddz_mean/MatMul:product:0%z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         И
z_log_var/MatMul/ReadVariableOpReadVariableOp(z_log_var_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0У
z_log_var/MatMulMatMuldense_138/Relu:activations:0'z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ж
 z_log_var/BiasAdd/ReadVariableOpReadVariableOp)z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ф
z_log_var/BiasAddBiasAddz_log_var/MatMul:product:0(z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         X
sampling_10/ShapeShapez_mean/BiasAdd:output:0*
T0*
_output_shapes
:i
sampling_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!sampling_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!sampling_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
sampling_10/strided_sliceStridedSlicesampling_10/Shape:output:0(sampling_10/strided_slice/stack:output:0*sampling_10/strided_slice/stack_1:output:0*sampling_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
sampling_10/Shape_1Shapez_mean/BiasAdd:output:0*
T0*
_output_shapes
:k
!sampling_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:m
#sampling_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#sampling_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ч
sampling_10/strided_slice_1StridedSlicesampling_10/Shape_1:output:0*sampling_10/strided_slice_1/stack:output:0,sampling_10/strided_slice_1/stack_1:output:0,sampling_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЯ
sampling_10/random_normal/shapePack"sampling_10/strided_slice:output:0$sampling_10/strided_slice_1:output:0*
N*
T0*
_output_shapes
:c
sampling_10/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    e
 sampling_10/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?═
.sampling_10/random_normal/RandomStandardNormalRandomStandardNormal(sampling_10/random_normal/shape:output:0*
T0*'
_output_shapes
:         *
dtype0*
seed▒ х)*
seed2└М║
sampling_10/random_normal/mulMul7sampling_10/random_normal/RandomStandardNormal:output:0)sampling_10/random_normal/stddev:output:0*
T0*'
_output_shapes
:         а
sampling_10/random_normalAddV2!sampling_10/random_normal/mul:z:0'sampling_10/random_normal/mean:output:0*
T0*'
_output_shapes
:         V
sampling_10/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?А
sampling_10/mulMulsampling_10/mul/x:output:0z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:         ]
sampling_10/ExpExpsampling_10/mul:z:0*
T0*'
_output_shapes
:         ~
sampling_10/mul_1Mulsampling_10/Exp:y:0sampling_10/random_normal:z:0*
T0*'
_output_shapes
:         z
sampling_10/addAddV2z_mean/BiasAdd:output:0sampling_10/mul_1:z:0*
T0*'
_output_shapes
:         f
IdentityIdentityz_mean/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         k

Identity_1Identityz_log_var/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         d

Identity_2Identitysampling_10/add:z:0^NoOp*
T0*'
_output_shapes
:         Щ
NoOpNoOp!^dense_136/BiasAdd/ReadVariableOp ^dense_136/MatMul/ReadVariableOp!^dense_137/BiasAdd/ReadVariableOp ^dense_137/MatMul/ReadVariableOp!^dense_138/BiasAdd/ReadVariableOp ^dense_138/MatMul/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : : : 2D
 dense_136/BiasAdd/ReadVariableOp dense_136/BiasAdd/ReadVariableOp2B
dense_136/MatMul/ReadVariableOpdense_136/MatMul/ReadVariableOp2D
 dense_137/BiasAdd/ReadVariableOp dense_137/BiasAdd/ReadVariableOp2B
dense_137/MatMul/ReadVariableOpdense_137/MatMul/ReadVariableOp2D
 dense_138/BiasAdd/ReadVariableOp dense_138/BiasAdd/ReadVariableOp2B
dense_138/MatMul/ReadVariableOpdense_138/MatMul/ReadVariableOp2D
 z_log_var/BiasAdd/ReadVariableOp z_log_var/BiasAdd/ReadVariableOp2B
z_log_var/MatMul/ReadVariableOpz_log_var/MatMul/ReadVariableOp2>
z_mean/BiasAdd/ReadVariableOpz_mean/BiasAdd/ReadVariableOp2<
z_mean/MatMul/ReadVariableOpz_mean/MatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
эM
Ш	
!__inference__wrapped_model_929180
input_21B
0encoder_dense_136_matmul_readvariableop_resource:d?
1encoder_dense_136_biasadd_readvariableop_resource:dB
0encoder_dense_137_matmul_readvariableop_resource:dd?
1encoder_dense_137_biasadd_readvariableop_resource:dB
0encoder_dense_138_matmul_readvariableop_resource:dd?
1encoder_dense_138_biasadd_readvariableop_resource:d?
-encoder_z_mean_matmul_readvariableop_resource:d<
.encoder_z_mean_biasadd_readvariableop_resource:B
0encoder_z_log_var_matmul_readvariableop_resource:d?
1encoder_z_log_var_biasadd_readvariableop_resource:
identity

identity_1

identity_2Ив(encoder/dense_136/BiasAdd/ReadVariableOpв'encoder/dense_136/MatMul/ReadVariableOpв(encoder/dense_137/BiasAdd/ReadVariableOpв'encoder/dense_137/MatMul/ReadVariableOpв(encoder/dense_138/BiasAdd/ReadVariableOpв'encoder/dense_138/MatMul/ReadVariableOpв(encoder/z_log_var/BiasAdd/ReadVariableOpв'encoder/z_log_var/MatMul/ReadVariableOpв%encoder/z_mean/BiasAdd/ReadVariableOpв$encoder/z_mean/MatMul/ReadVariableOpШ
'encoder/dense_136/MatMul/ReadVariableOpReadVariableOp0encoder_dense_136_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0П
encoder/dense_136/MatMulMatMulinput_21/encoder/dense_136/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dЦ
(encoder/dense_136/BiasAdd/ReadVariableOpReadVariableOp1encoder_dense_136_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0м
encoder/dense_136/BiasAddBiasAdd"encoder/dense_136/MatMul:product:00encoder/dense_136/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dt
encoder/dense_136/ReluRelu"encoder/dense_136/BiasAdd:output:0*
T0*'
_output_shapes
:         dШ
'encoder/dense_137/MatMul/ReadVariableOpReadVariableOp0encoder_dense_137_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0л
encoder/dense_137/MatMulMatMul$encoder/dense_136/Relu:activations:0/encoder/dense_137/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dЦ
(encoder/dense_137/BiasAdd/ReadVariableOpReadVariableOp1encoder_dense_137_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0м
encoder/dense_137/BiasAddBiasAdd"encoder/dense_137/MatMul:product:00encoder/dense_137/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dt
encoder/dense_137/ReluRelu"encoder/dense_137/BiasAdd:output:0*
T0*'
_output_shapes
:         dШ
'encoder/dense_138/MatMul/ReadVariableOpReadVariableOp0encoder_dense_138_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0л
encoder/dense_138/MatMulMatMul$encoder/dense_137/Relu:activations:0/encoder/dense_138/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dЦ
(encoder/dense_138/BiasAdd/ReadVariableOpReadVariableOp1encoder_dense_138_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0м
encoder/dense_138/BiasAddBiasAdd"encoder/dense_138/MatMul:product:00encoder/dense_138/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dt
encoder/dense_138/ReluRelu"encoder/dense_138/BiasAdd:output:0*
T0*'
_output_shapes
:         dТ
$encoder/z_mean/MatMul/ReadVariableOpReadVariableOp-encoder_z_mean_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0е
encoder/z_mean/MatMulMatMul$encoder/dense_138/Relu:activations:0,encoder/z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Р
%encoder/z_mean/BiasAdd/ReadVariableOpReadVariableOp.encoder_z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0г
encoder/z_mean/BiasAddBiasAddencoder/z_mean/MatMul:product:0-encoder/z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ш
'encoder/z_log_var/MatMul/ReadVariableOpReadVariableOp0encoder_z_log_var_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0л
encoder/z_log_var/MatMulMatMul$encoder/dense_138/Relu:activations:0/encoder/z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ц
(encoder/z_log_var/BiasAdd/ReadVariableOpReadVariableOp1encoder_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0м
encoder/z_log_var/BiasAddBiasAdd"encoder/z_log_var/MatMul:product:00encoder/z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         h
encoder/sampling_10/ShapeShapeencoder/z_mean/BiasAdd:output:0*
T0*
_output_shapes
:q
'encoder/sampling_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)encoder/sampling_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)encoder/sampling_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╡
!encoder/sampling_10/strided_sliceStridedSlice"encoder/sampling_10/Shape:output:00encoder/sampling_10/strided_slice/stack:output:02encoder/sampling_10/strided_slice/stack_1:output:02encoder/sampling_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
encoder/sampling_10/Shape_1Shapeencoder/z_mean/BiasAdd:output:0*
T0*
_output_shapes
:s
)encoder/sampling_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+encoder/sampling_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+encoder/sampling_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┐
#encoder/sampling_10/strided_slice_1StridedSlice$encoder/sampling_10/Shape_1:output:02encoder/sampling_10/strided_slice_1/stack:output:04encoder/sampling_10/strided_slice_1/stack_1:output:04encoder/sampling_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask╖
'encoder/sampling_10/random_normal/shapePack*encoder/sampling_10/strided_slice:output:0,encoder/sampling_10/strided_slice_1:output:0*
N*
T0*
_output_shapes
:k
&encoder/sampling_10/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    m
(encoder/sampling_10/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?▐
6encoder/sampling_10/random_normal/RandomStandardNormalRandomStandardNormal0encoder/sampling_10/random_normal/shape:output:0*
T0*'
_output_shapes
:         *
dtype0*
seed▒ х)*
seed2┌╔с╥
%encoder/sampling_10/random_normal/mulMul?encoder/sampling_10/random_normal/RandomStandardNormal:output:01encoder/sampling_10/random_normal/stddev:output:0*
T0*'
_output_shapes
:         ╕
!encoder/sampling_10/random_normalAddV2)encoder/sampling_10/random_normal/mul:z:0/encoder/sampling_10/random_normal/mean:output:0*
T0*'
_output_shapes
:         ^
encoder/sampling_10/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ш
encoder/sampling_10/mulMul"encoder/sampling_10/mul/x:output:0"encoder/z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:         m
encoder/sampling_10/ExpExpencoder/sampling_10/mul:z:0*
T0*'
_output_shapes
:         Ц
encoder/sampling_10/mul_1Mulencoder/sampling_10/Exp:y:0%encoder/sampling_10/random_normal:z:0*
T0*'
_output_shapes
:         Т
encoder/sampling_10/addAddV2encoder/z_mean/BiasAdd:output:0encoder/sampling_10/mul_1:z:0*
T0*'
_output_shapes
:         j
IdentityIdentityencoder/sampling_10/add:z:0^NoOp*
T0*'
_output_shapes
:         s

Identity_1Identity"encoder/z_log_var/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         p

Identity_2Identityencoder/z_mean/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         щ
NoOpNoOp)^encoder/dense_136/BiasAdd/ReadVariableOp(^encoder/dense_136/MatMul/ReadVariableOp)^encoder/dense_137/BiasAdd/ReadVariableOp(^encoder/dense_137/MatMul/ReadVariableOp)^encoder/dense_138/BiasAdd/ReadVariableOp(^encoder/dense_138/MatMul/ReadVariableOp)^encoder/z_log_var/BiasAdd/ReadVariableOp(^encoder/z_log_var/MatMul/ReadVariableOp&^encoder/z_mean/BiasAdd/ReadVariableOp%^encoder/z_mean/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : : : 2T
(encoder/dense_136/BiasAdd/ReadVariableOp(encoder/dense_136/BiasAdd/ReadVariableOp2R
'encoder/dense_136/MatMul/ReadVariableOp'encoder/dense_136/MatMul/ReadVariableOp2T
(encoder/dense_137/BiasAdd/ReadVariableOp(encoder/dense_137/BiasAdd/ReadVariableOp2R
'encoder/dense_137/MatMul/ReadVariableOp'encoder/dense_137/MatMul/ReadVariableOp2T
(encoder/dense_138/BiasAdd/ReadVariableOp(encoder/dense_138/BiasAdd/ReadVariableOp2R
'encoder/dense_138/MatMul/ReadVariableOp'encoder/dense_138/MatMul/ReadVariableOp2T
(encoder/z_log_var/BiasAdd/ReadVariableOp(encoder/z_log_var/BiasAdd/ReadVariableOp2R
'encoder/z_log_var/MatMul/ReadVariableOp'encoder/z_log_var/MatMul/ReadVariableOp2N
%encoder/z_mean/BiasAdd/ReadVariableOp%encoder/z_mean/BiasAdd/ReadVariableOp2L
$encoder/z_mean/MatMul/ReadVariableOp$encoder/z_mean/MatMul/ReadVariableOp:Q M
'
_output_shapes
:         
"
_user_specified_name
input_21
К
v
G__inference_sampling_10_layer_call_and_return_conditional_losses_929907
inputs_0
inputs_1
identityИ=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Shape_1Shapeinputs_0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
random_normal/shapePackstrided_slice:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:W
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╢
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*'
_output_shapes
:         *
dtype0*
seed▒ х)*
seed2ў▄юЦ
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:         |
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:         J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?V
mulMulmul/x:output:0inputs_1*
T0*'
_output_shapes
:         E
ExpExpmul:z:0*
T0*'
_output_shapes
:         Z
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:         S
addAddV2inputs_0	mul_1:z:0*
T0*'
_output_shapes
:         O
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         :         :Q M
'
_output_shapes
:         
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1
┴
Ф
'__inference_z_mean_layer_call_fn_929846

inputs
unknown:d
	unknown_0:
identityИвStatefulPartitionedCall┌
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_z_mean_layer_call_and_return_conditional_losses_929248o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
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
╟
Ч
*__inference_dense_138_layer_call_fn_929826

inputs
unknown:dd
	unknown_0:d
identityИвStatefulPartitionedCall▌
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
GPU2*0J 8В *N
fIRG
E__inference_dense_138_layer_call_and_return_conditional_losses_929232o
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
 
_user_specified_nameinputs"┐L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*л
serving_defaultЧ
=
input_211
serving_default_input_21:0         ?
sampling_100
StatefulPartitionedCall:0         =
	z_log_var0
StatefulPartitionedCall:1         :
z_mean0
StatefulPartitionedCall:2         tensorflow/serving/predict:ЭЭ
▒
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer-6
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
╗
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
╗
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
╗
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

&kernel
'bias"
_tf_keras_layer
╗
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses

.kernel
/bias"
_tf_keras_layer
╗
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses

6kernel
7bias"
_tf_keras_layer
е
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses"
_tf_keras_layer
f
0
1
2
3
&4
'5
.6
/7
68
79"
trackable_list_wrapper
f
0
1
2
3
&4
'5
.6
/7
68
79"
trackable_list_wrapper
 "
trackable_list_wrapper
╩
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
╓
Ctrace_0
Dtrace_1
Etrace_2
Ftrace_32ы
(__inference_encoder_layer_call_fn_929328
(__inference_encoder_layer_call_fn_929628
(__inference_encoder_layer_call_fn_929657
(__inference_encoder_layer_call_fn_929504└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 zCtrace_0zDtrace_1zEtrace_2zFtrace_3
┬
Gtrace_0
Htrace_1
Itrace_2
Jtrace_32╫
C__inference_encoder_layer_call_and_return_conditional_losses_929717
C__inference_encoder_layer_call_and_return_conditional_losses_929777
C__inference_encoder_layer_call_and_return_conditional_losses_929536
C__inference_encoder_layer_call_and_return_conditional_losses_929568└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 zGtrace_0zHtrace_1zItrace_2zJtrace_3
═B╩
!__inference__wrapped_model_929180input_21"Ш
С▓Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
,
Kserving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
н
Lnon_trainable_variables

Mlayers
Nmetrics
Olayer_regularization_losses
Player_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ю
Qtrace_02╤
*__inference_dense_136_layer_call_fn_929786в
Щ▓Х
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
annotationsк *
 zQtrace_0
Й
Rtrace_02ь
E__inference_dense_136_layer_call_and_return_conditional_losses_929797в
Щ▓Х
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
annotationsк *
 zRtrace_0
": d2dense_136/kernel
:d2dense_136/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
н
Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ю
Xtrace_02╤
*__inference_dense_137_layer_call_fn_929806в
Щ▓Х
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
annotationsк *
 zXtrace_0
Й
Ytrace_02ь
E__inference_dense_137_layer_call_and_return_conditional_losses_929817в
Щ▓Х
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
annotationsк *
 zYtrace_0
": dd2dense_137/kernel
:d2dense_137/bias
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
н
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
ю
_trace_02╤
*__inference_dense_138_layer_call_fn_929826в
Щ▓Х
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
annotationsк *
 z_trace_0
Й
`trace_02ь
E__inference_dense_138_layer_call_and_return_conditional_losses_929837в
Щ▓Х
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
annotationsк *
 z`trace_0
": dd2dense_138/kernel
:d2dense_138/bias
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
н
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
ы
ftrace_02╬
'__inference_z_mean_layer_call_fn_929846в
Щ▓Х
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
annotationsк *
 zftrace_0
Ж
gtrace_02щ
B__inference_z_mean_layer_call_and_return_conditional_losses_929856в
Щ▓Х
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
annotationsк *
 zgtrace_0
:d2z_mean/kernel
:2z_mean/bias
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
н
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
ю
mtrace_02╤
*__inference_z_log_var_layer_call_fn_929865в
Щ▓Х
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
annotationsк *
 zmtrace_0
Й
ntrace_02ь
E__inference_z_log_var_layer_call_and_return_conditional_losses_929875в
Щ▓Х
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
annotationsк *
 zntrace_0
": d2z_log_var/kernel
:2z_log_var/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
Ё
ttrace_02╙
,__inference_sampling_10_layer_call_fn_929881в
Щ▓Х
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
annotationsк *
 zttrace_0
Л
utrace_02ю
G__inference_sampling_10_layer_call_and_return_conditional_losses_929907в
Щ▓Х
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
annotationsк *
 zutrace_0
 "
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
№B∙
(__inference_encoder_layer_call_fn_929328input_21"└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
·Bў
(__inference_encoder_layer_call_fn_929628inputs"└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
·Bў
(__inference_encoder_layer_call_fn_929657inputs"└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
№B∙
(__inference_encoder_layer_call_fn_929504input_21"└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ХBТ
C__inference_encoder_layer_call_and_return_conditional_losses_929717inputs"└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ХBТ
C__inference_encoder_layer_call_and_return_conditional_losses_929777inputs"└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ЧBФ
C__inference_encoder_layer_call_and_return_conditional_losses_929536input_21"└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ЧBФ
C__inference_encoder_layer_call_and_return_conditional_losses_929568input_21"└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╠B╔
$__inference_signature_wrapper_929599input_21"Ф
Н▓Й
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
annotationsк *
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
▐B█
*__inference_dense_136_layer_call_fn_929786inputs"в
Щ▓Х
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
annotationsк *
 
∙BЎ
E__inference_dense_136_layer_call_and_return_conditional_losses_929797inputs"в
Щ▓Х
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
annotationsк *
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
▐B█
*__inference_dense_137_layer_call_fn_929806inputs"в
Щ▓Х
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
annotationsк *
 
∙BЎ
E__inference_dense_137_layer_call_and_return_conditional_losses_929817inputs"в
Щ▓Х
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
annotationsк *
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
▐B█
*__inference_dense_138_layer_call_fn_929826inputs"в
Щ▓Х
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
annotationsк *
 
∙BЎ
E__inference_dense_138_layer_call_and_return_conditional_losses_929837inputs"в
Щ▓Х
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
annotationsк *
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
█B╪
'__inference_z_mean_layer_call_fn_929846inputs"в
Щ▓Х
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
annotationsк *
 
ЎBє
B__inference_z_mean_layer_call_and_return_conditional_losses_929856inputs"в
Щ▓Х
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
annotationsк *
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
▐B█
*__inference_z_log_var_layer_call_fn_929865inputs"в
Щ▓Х
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
annotationsк *
 
∙BЎ
E__inference_z_log_var_layer_call_and_return_conditional_losses_929875inputs"в
Щ▓Х
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
annotationsк *
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
ьBщ
,__inference_sampling_10_layer_call_fn_929881inputs/0inputs/1"в
Щ▓Х
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
annotationsк *
 
ЗBД
G__inference_sampling_10_layer_call_and_return_conditional_losses_929907inputs/0inputs/1"в
Щ▓Х
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
annotationsк *
 А
!__inference__wrapped_model_929180┌
&'./671в.
'в$
"К
input_21         
к "ШкФ
4
sampling_10%К"
sampling_10         
0
	z_log_var#К 
	z_log_var         
*
z_mean К
z_mean         е
E__inference_dense_136_layer_call_and_return_conditional_losses_929797\/в,
%в"
 К
inputs         
к "%в"
К
0         d
Ъ }
*__inference_dense_136_layer_call_fn_929786O/в,
%в"
 К
inputs         
к "К         dе
E__inference_dense_137_layer_call_and_return_conditional_losses_929817\/в,
%в"
 К
inputs         d
к "%в"
К
0         d
Ъ }
*__inference_dense_137_layer_call_fn_929806O/в,
%в"
 К
inputs         d
к "К         dе
E__inference_dense_138_layer_call_and_return_conditional_losses_929837\&'/в,
%в"
 К
inputs         d
к "%в"
К
0         d
Ъ }
*__inference_dense_138_layer_call_fn_929826O&'/в,
%в"
 К
inputs         d
к "К         d√
C__inference_encoder_layer_call_and_return_conditional_losses_929536│
&'./679в6
/в,
"К
input_21         
p 

 
к "jвg
`Ъ]
К
0/0         
К
0/1         
К
0/2         
Ъ √
C__inference_encoder_layer_call_and_return_conditional_losses_929568│
&'./679в6
/в,
"К
input_21         
p

 
к "jвg
`Ъ]
К
0/0         
К
0/1         
К
0/2         
Ъ ∙
C__inference_encoder_layer_call_and_return_conditional_losses_929717▒
&'./677в4
-в*
 К
inputs         
p 

 
к "jвg
`Ъ]
К
0/0         
К
0/1         
К
0/2         
Ъ ∙
C__inference_encoder_layer_call_and_return_conditional_losses_929777▒
&'./677в4
-в*
 К
inputs         
p

 
к "jвg
`Ъ]
К
0/0         
К
0/1         
К
0/2         
Ъ ╨
(__inference_encoder_layer_call_fn_929328г
&'./679в6
/в,
"К
input_21         
p 

 
к "ZЪW
К
0         
К
1         
К
2         ╨
(__inference_encoder_layer_call_fn_929504г
&'./679в6
/в,
"К
input_21         
p

 
к "ZЪW
К
0         
К
1         
К
2         ╬
(__inference_encoder_layer_call_fn_929628б
&'./677в4
-в*
 К
inputs         
p 

 
к "ZЪW
К
0         
К
1         
К
2         ╬
(__inference_encoder_layer_call_fn_929657б
&'./677в4
-в*
 К
inputs         
p

 
к "ZЪW
К
0         
К
1         
К
2         ╧
G__inference_sampling_10_layer_call_and_return_conditional_losses_929907ГZвW
PвM
KЪH
"К
inputs/0         
"К
inputs/1         
к "%в"
К
0         
Ъ ж
,__inference_sampling_10_layer_call_fn_929881vZвW
PвM
KЪH
"К
inputs/0         
"К
inputs/1         
к "К         П
$__inference_signature_wrapper_929599ц
&'./67=в:
в 
3к0
.
input_21"К
input_21         "ШкФ
4
sampling_10%К"
sampling_10         
0
	z_log_var#К 
	z_log_var         
*
z_mean К
z_mean         е
E__inference_z_log_var_layer_call_and_return_conditional_losses_929875\67/в,
%в"
 К
inputs         d
к "%в"
К
0         
Ъ }
*__inference_z_log_var_layer_call_fn_929865O67/в,
%в"
 К
inputs         d
к "К         в
B__inference_z_mean_layer_call_and_return_conditional_losses_929856\.//в,
%в"
 К
inputs         d
к "%в"
К
0         
Ъ z
'__inference_z_mean_layer_call_fn_929846O.//в,
%в"
 К
inputs         d
к "К         