ШЈ
жЋ
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
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
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	
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

RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
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
ї
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.9.12v2.9.0-18-gd8ce9f9c3018ѕ
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
r
dense_86/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_86/bias
k
!dense_86/bias/Read/ReadVariableOpReadVariableOpdense_86/bias*
_output_shapes
:d*
dtype0
z
dense_86/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd* 
shared_namedense_86/kernel
s
#dense_86/kernel/Read/ReadVariableOpReadVariableOpdense_86/kernel*
_output_shapes

:dd*
dtype0
r
dense_85/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_85/bias
k
!dense_85/bias/Read/ReadVariableOpReadVariableOpdense_85/bias*
_output_shapes
:d*
dtype0
z
dense_85/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd* 
shared_namedense_85/kernel
s
#dense_85/kernel/Read/ReadVariableOpReadVariableOpdense_85/kernel*
_output_shapes

:dd*
dtype0
r
dense_84/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_84/bias
k
!dense_84/bias/Read/ReadVariableOpReadVariableOpdense_84/bias*
_output_shapes
:d*
dtype0
z
dense_84/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d* 
shared_namedense_84/kernel
s
#dense_84/kernel/Read/ReadVariableOpReadVariableOpdense_84/kernel*
_output_shapes

:d*
dtype0

NoOpNoOp
Б*
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ь)
valueт)Bп) Bи)

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
І
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
І
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
І
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

&kernel
'bias*
І
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses

.kernel
/bias*
І
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses

6kernel
7bias*

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
А
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

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
_Y
VARIABLE_VALUEdense_84/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_84/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

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
_Y
VARIABLE_VALUEdense_85/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_85/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

&0
'1*

&0
'1*
* 

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
_Y
VARIABLE_VALUEdense_86/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_86/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

.0
/1*

.0
/1*
* 

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

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

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
serving_default_input_13Placeholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_13dense_84/kerneldense_84/biasdense_85/kerneldense_85/biasdense_86/kerneldense_86/biasz_mean/kernelz_mean/biasz_log_var/kernelz_log_var/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_750679
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_84/kernel/Read/ReadVariableOp!dense_84/bias/Read/ReadVariableOp#dense_85/kernel/Read/ReadVariableOp!dense_85/bias/Read/ReadVariableOp#dense_86/kernel/Read/ReadVariableOp!dense_86/bias/Read/ReadVariableOp!z_mean/kernel/Read/ReadVariableOpz_mean/bias/Read/ReadVariableOp$z_log_var/kernel/Read/ReadVariableOp"z_log_var/bias/Read/ReadVariableOpConst*
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
GPU2*0J 8 *(
f#R!
__inference__traced_save_751042
С
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_84/kerneldense_84/biasdense_85/kerneldense_85/biasdense_86/kerneldense_86/biasz_mean/kernelz_mean/biasz_log_var/kernelz_log_var/bias*
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
GPU2*0J 8 *+
f&R$
"__inference__traced_restore_751082ћЉ
аC

C__inference_encoder_layer_call_and_return_conditional_losses_750857

inputs9
'dense_84_matmul_readvariableop_resource:d6
(dense_84_biasadd_readvariableop_resource:d9
'dense_85_matmul_readvariableop_resource:dd6
(dense_85_biasadd_readvariableop_resource:d9
'dense_86_matmul_readvariableop_resource:dd6
(dense_86_biasadd_readvariableop_resource:d7
%z_mean_matmul_readvariableop_resource:d4
&z_mean_biasadd_readvariableop_resource::
(z_log_var_matmul_readvariableop_resource:d7
)z_log_var_biasadd_readvariableop_resource:
identity

identity_1

identity_2Ђdense_84/BiasAdd/ReadVariableOpЂdense_84/MatMul/ReadVariableOpЂdense_85/BiasAdd/ReadVariableOpЂdense_85/MatMul/ReadVariableOpЂdense_86/BiasAdd/ReadVariableOpЂdense_86/MatMul/ReadVariableOpЂ z_log_var/BiasAdd/ReadVariableOpЂz_log_var/MatMul/ReadVariableOpЂz_mean/BiasAdd/ReadVariableOpЂz_mean/MatMul/ReadVariableOp
dense_84/MatMul/ReadVariableOpReadVariableOp'dense_84_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0{
dense_84/MatMulMatMulinputs&dense_84/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd
dense_84/BiasAdd/ReadVariableOpReadVariableOp(dense_84_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
dense_84/BiasAddBiasAdddense_84/MatMul:product:0'dense_84/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџdb
dense_84/ReluReludense_84/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџd
dense_85/MatMul/ReadVariableOpReadVariableOp'dense_85_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0
dense_85/MatMulMatMuldense_84/Relu:activations:0&dense_85/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd
dense_85/BiasAdd/ReadVariableOpReadVariableOp(dense_85_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
dense_85/BiasAddBiasAdddense_85/MatMul:product:0'dense_85/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџdb
dense_85/ReluReludense_85/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџd
dense_86/MatMul/ReadVariableOpReadVariableOp'dense_86_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0
dense_86/MatMulMatMuldense_85/Relu:activations:0&dense_86/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd
dense_86/BiasAdd/ReadVariableOpReadVariableOp(dense_86_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
dense_86/BiasAddBiasAdddense_86/MatMul:product:0'dense_86/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџdb
dense_86/ReluReludense_86/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџd
z_mean/MatMul/ReadVariableOpReadVariableOp%z_mean_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0
z_mean/MatMulMatMuldense_86/Relu:activations:0$z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
z_mean/BiasAdd/ReadVariableOpReadVariableOp&z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
z_mean/BiasAddBiasAddz_mean/MatMul:product:0%z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
z_log_var/MatMul/ReadVariableOpReadVariableOp(z_log_var_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0
z_log_var/MatMulMatMuldense_86/Relu:activations:0'z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
 z_log_var/BiasAdd/ReadVariableOpReadVariableOp)z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
z_log_var/BiasAddBiasAddz_log_var/MatMul:product:0(z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџW
sampling_6/ShapeShapez_mean/BiasAdd:output:0*
T0*
_output_shapes
:h
sampling_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 sampling_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 sampling_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
sampling_6/strided_sliceStridedSlicesampling_6/Shape:output:0'sampling_6/strided_slice/stack:output:0)sampling_6/strided_slice/stack_1:output:0)sampling_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
sampling_6/Shape_1Shapez_mean/BiasAdd:output:0*
T0*
_output_shapes
:j
 sampling_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:l
"sampling_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"sampling_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
sampling_6/strided_slice_1StridedSlicesampling_6/Shape_1:output:0)sampling_6/strided_slice_1/stack:output:0+sampling_6/strided_slice_1/stack_1:output:0+sampling_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
sampling_6/random_normal/shapePack!sampling_6/strided_slice:output:0#sampling_6/strided_slice_1:output:0*
N*
T0*
_output_shapes
:b
sampling_6/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    d
sampling_6/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ь
-sampling_6/random_normal/RandomStandardNormalRandomStandardNormal'sampling_6/random_normal/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2ЊлЫЗ
sampling_6/random_normal/mulMul6sampling_6/random_normal/RandomStandardNormal:output:0(sampling_6/random_normal/stddev:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
sampling_6/random_normalAddV2 sampling_6/random_normal/mul:z:0&sampling_6/random_normal/mean:output:0*
T0*'
_output_shapes
:џџџџџџџџџU
sampling_6/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?~
sampling_6/mulMulsampling_6/mul/x:output:0z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ[
sampling_6/ExpExpsampling_6/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ{
sampling_6/mul_1Mulsampling_6/Exp:y:0sampling_6/random_normal:z:0*
T0*'
_output_shapes
:џџџџџџџџџx
sampling_6/addAddV2z_mean/BiasAdd:output:0sampling_6/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџf
IdentityIdentityz_mean/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџk

Identity_1Identityz_log_var/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџc

Identity_2Identitysampling_6/add:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp ^dense_84/BiasAdd/ReadVariableOp^dense_84/MatMul/ReadVariableOp ^dense_85/BiasAdd/ReadVariableOp^dense_85/MatMul/ReadVariableOp ^dense_86/BiasAdd/ReadVariableOp^dense_86/MatMul/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ: : : : : : : : : : 2B
dense_84/BiasAdd/ReadVariableOpdense_84/BiasAdd/ReadVariableOp2@
dense_84/MatMul/ReadVariableOpdense_84/MatMul/ReadVariableOp2B
dense_85/BiasAdd/ReadVariableOpdense_85/BiasAdd/ReadVariableOp2@
dense_85/MatMul/ReadVariableOpdense_85/MatMul/ReadVariableOp2B
dense_86/BiasAdd/ReadVariableOpdense_86/BiasAdd/ReadVariableOp2@
dense_86/MatMul/ReadVariableOpdense_86/MatMul/ReadVariableOp2D
 z_log_var/BiasAdd/ReadVariableOp z_log_var/BiasAdd/ReadVariableOp2B
z_log_var/MatMul/ReadVariableOpz_log_var/MatMul/ReadVariableOp2>
z_mean/BiasAdd/ReadVariableOpz_mean/BiasAdd/ReadVariableOp2<
z_mean/MatMul/ReadVariableOpz_mean/MatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
С

'__inference_z_mean_layer_call_fn_750926

inputs
unknown:d
	unknown_0:
identityЂStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_z_mean_layer_call_and_return_conditional_losses_750328o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
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
Х	
ѓ
B__inference_z_mean_layer_call_and_return_conditional_losses_750936

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
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


(__inference_encoder_layer_call_fn_750708

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

identity_2ЂStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_750381o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


(__inference_encoder_layer_call_fn_750737

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

identity_2ЂStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_750528o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ч

*__inference_z_log_var_layer_call_fn_750945

inputs
unknown:d
	unknown_0:
identityЂStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_z_log_var_layer_call_and_return_conditional_losses_750344o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
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
э"

C__inference_encoder_layer_call_and_return_conditional_losses_750648
input_13!
dense_84_750619:d
dense_84_750621:d!
dense_85_750624:dd
dense_85_750626:d!
dense_86_750629:dd
dense_86_750631:d
z_mean_750634:d
z_mean_750636:"
z_log_var_750639:d
z_log_var_750641:
identity

identity_1

identity_2Ђ dense_84/StatefulPartitionedCallЂ dense_85/StatefulPartitionedCallЂ dense_86/StatefulPartitionedCallЂ"sampling_6/StatefulPartitionedCallЂ!z_log_var/StatefulPartitionedCallЂz_mean/StatefulPartitionedCallѕ
 dense_84/StatefulPartitionedCallStatefulPartitionedCallinput_13dense_84_750619dense_84_750621*
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
GPU2*0J 8 *M
fHRF
D__inference_dense_84_layer_call_and_return_conditional_losses_750278
 dense_85/StatefulPartitionedCallStatefulPartitionedCall)dense_84/StatefulPartitionedCall:output:0dense_85_750624dense_85_750626*
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
GPU2*0J 8 *M
fHRF
D__inference_dense_85_layer_call_and_return_conditional_losses_750295
 dense_86/StatefulPartitionedCallStatefulPartitionedCall)dense_85/StatefulPartitionedCall:output:0dense_86_750629dense_86_750631*
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
GPU2*0J 8 *M
fHRF
D__inference_dense_86_layer_call_and_return_conditional_losses_750312
z_mean/StatefulPartitionedCallStatefulPartitionedCall)dense_86/StatefulPartitionedCall:output:0z_mean_750634z_mean_750636*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_z_mean_layer_call_and_return_conditional_losses_750328
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall)dense_86/StatefulPartitionedCall:output:0z_log_var_750639z_log_var_750641*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_z_log_var_layer_call_and_return_conditional_losses_750344
"sampling_6/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sampling_6_layer_call_and_return_conditional_losses_750376v
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ{

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ|

Identity_2Identity+sampling_6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp!^dense_84/StatefulPartitionedCall!^dense_85/StatefulPartitionedCall!^dense_86/StatefulPartitionedCall#^sampling_6/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ: : : : : : : : : : 2D
 dense_84/StatefulPartitionedCall dense_84/StatefulPartitionedCall2D
 dense_85/StatefulPartitionedCall dense_85/StatefulPartitionedCall2D
 dense_86/StatefulPartitionedCall dense_86/StatefulPartitionedCall2H
"sampling_6/StatefulPartitionedCall"sampling_6/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
input_13
§
s
F__inference_sampling_6_layer_call_and_return_conditional_losses_750376

inputs
inputs_1
identity;
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
valueB:б
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
valueB:л
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
 *  ?Ж
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2ПЙ
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:џџџџџџџџџ|
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:џџџџџџџџџJ
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?V
mulMulmul/x:output:0inputs_1*
T0*'
_output_shapes
:џџџџџџџџџE
ExpExpmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџZ
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:џџџџџџџџџQ
addAddV2inputs	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџO
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџ:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Х	
ѓ
B__inference_z_mean_layer_call_and_return_conditional_losses_750328

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
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


ѕ
D__inference_dense_86_layer_call_and_return_conditional_losses_750312

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
:џџџџџџџџџdP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџda
IdentityIdentityRelu:activations:0^NoOp*
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


ѕ
D__inference_dense_85_layer_call_and_return_conditional_losses_750295

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
:џџџџџџџџџdP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџda
IdentityIdentityRelu:activations:0^NoOp*
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


ѕ
D__inference_dense_85_layer_call_and_return_conditional_losses_750897

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
:џџџџџџџџџdP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџda
IdentityIdentityRelu:activations:0^NoOp*
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
ч"

C__inference_encoder_layer_call_and_return_conditional_losses_750528

inputs!
dense_84_750499:d
dense_84_750501:d!
dense_85_750504:dd
dense_85_750506:d!
dense_86_750509:dd
dense_86_750511:d
z_mean_750514:d
z_mean_750516:"
z_log_var_750519:d
z_log_var_750521:
identity

identity_1

identity_2Ђ dense_84/StatefulPartitionedCallЂ dense_85/StatefulPartitionedCallЂ dense_86/StatefulPartitionedCallЂ"sampling_6/StatefulPartitionedCallЂ!z_log_var/StatefulPartitionedCallЂz_mean/StatefulPartitionedCallѓ
 dense_84/StatefulPartitionedCallStatefulPartitionedCallinputsdense_84_750499dense_84_750501*
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
GPU2*0J 8 *M
fHRF
D__inference_dense_84_layer_call_and_return_conditional_losses_750278
 dense_85/StatefulPartitionedCallStatefulPartitionedCall)dense_84/StatefulPartitionedCall:output:0dense_85_750504dense_85_750506*
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
GPU2*0J 8 *M
fHRF
D__inference_dense_85_layer_call_and_return_conditional_losses_750295
 dense_86/StatefulPartitionedCallStatefulPartitionedCall)dense_85/StatefulPartitionedCall:output:0dense_86_750509dense_86_750511*
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
GPU2*0J 8 *M
fHRF
D__inference_dense_86_layer_call_and_return_conditional_losses_750312
z_mean/StatefulPartitionedCallStatefulPartitionedCall)dense_86/StatefulPartitionedCall:output:0z_mean_750514z_mean_750516*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_z_mean_layer_call_and_return_conditional_losses_750328
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall)dense_86/StatefulPartitionedCall:output:0z_log_var_750519z_log_var_750521*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_z_log_var_layer_call_and_return_conditional_losses_750344
"sampling_6/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sampling_6_layer_call_and_return_conditional_losses_750376v
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ{

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ|

Identity_2Identity+sampling_6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp!^dense_84/StatefulPartitionedCall!^dense_85/StatefulPartitionedCall!^dense_86/StatefulPartitionedCall#^sampling_6/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ: : : : : : : : : : 2D
 dense_84/StatefulPartitionedCall dense_84/StatefulPartitionedCall2D
 dense_85/StatefulPartitionedCall dense_85/StatefulPartitionedCall2D
 dense_86/StatefulPartitionedCall dense_86/StatefulPartitionedCall2H
"sampling_6/StatefulPartitionedCall"sampling_6/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
 
Р
__inference__traced_save_751042
file_prefix.
*savev2_dense_84_kernel_read_readvariableop,
(savev2_dense_84_bias_read_readvariableop.
*savev2_dense_85_kernel_read_readvariableop,
(savev2_dense_85_bias_read_readvariableop.
*savev2_dense_86_kernel_read_readvariableop,
(savev2_dense_86_bias_read_readvariableop,
(savev2_z_mean_kernel_read_readvariableop*
&savev2_z_mean_bias_read_readvariableop/
+savev2_z_log_var_kernel_read_readvariableop-
)savev2_z_log_var_bias_read_readvariableop
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
: А
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*й
valueЯBЬB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B ц
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_84_kernel_read_readvariableop(savev2_dense_84_bias_read_readvariableop*savev2_dense_85_kernel_read_readvariableop(savev2_dense_85_bias_read_readvariableop*savev2_dense_86_kernel_read_readvariableop(savev2_dense_86_bias_read_readvariableop(savev2_z_mean_kernel_read_readvariableop&savev2_z_mean_bias_read_readvariableop+savev2_z_log_var_kernel_read_readvariableop)savev2_z_log_var_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2
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
M
	
!__inference__wrapped_model_750260
input_13A
/encoder_dense_84_matmul_readvariableop_resource:d>
0encoder_dense_84_biasadd_readvariableop_resource:dA
/encoder_dense_85_matmul_readvariableop_resource:dd>
0encoder_dense_85_biasadd_readvariableop_resource:dA
/encoder_dense_86_matmul_readvariableop_resource:dd>
0encoder_dense_86_biasadd_readvariableop_resource:d?
-encoder_z_mean_matmul_readvariableop_resource:d<
.encoder_z_mean_biasadd_readvariableop_resource:B
0encoder_z_log_var_matmul_readvariableop_resource:d?
1encoder_z_log_var_biasadd_readvariableop_resource:
identity

identity_1

identity_2Ђ'encoder/dense_84/BiasAdd/ReadVariableOpЂ&encoder/dense_84/MatMul/ReadVariableOpЂ'encoder/dense_85/BiasAdd/ReadVariableOpЂ&encoder/dense_85/MatMul/ReadVariableOpЂ'encoder/dense_86/BiasAdd/ReadVariableOpЂ&encoder/dense_86/MatMul/ReadVariableOpЂ(encoder/z_log_var/BiasAdd/ReadVariableOpЂ'encoder/z_log_var/MatMul/ReadVariableOpЂ%encoder/z_mean/BiasAdd/ReadVariableOpЂ$encoder/z_mean/MatMul/ReadVariableOp
&encoder/dense_84/MatMul/ReadVariableOpReadVariableOp/encoder_dense_84_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0
encoder/dense_84/MatMulMatMulinput_13.encoder/dense_84/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd
'encoder/dense_84/BiasAdd/ReadVariableOpReadVariableOp0encoder_dense_84_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0Љ
encoder/dense_84/BiasAddBiasAdd!encoder/dense_84/MatMul:product:0/encoder/dense_84/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџdr
encoder/dense_84/ReluRelu!encoder/dense_84/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџd
&encoder/dense_85/MatMul/ReadVariableOpReadVariableOp/encoder_dense_85_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0Ј
encoder/dense_85/MatMulMatMul#encoder/dense_84/Relu:activations:0.encoder/dense_85/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd
'encoder/dense_85/BiasAdd/ReadVariableOpReadVariableOp0encoder_dense_85_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0Љ
encoder/dense_85/BiasAddBiasAdd!encoder/dense_85/MatMul:product:0/encoder/dense_85/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџdr
encoder/dense_85/ReluRelu!encoder/dense_85/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџd
&encoder/dense_86/MatMul/ReadVariableOpReadVariableOp/encoder_dense_86_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0Ј
encoder/dense_86/MatMulMatMul#encoder/dense_85/Relu:activations:0.encoder/dense_86/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd
'encoder/dense_86/BiasAdd/ReadVariableOpReadVariableOp0encoder_dense_86_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0Љ
encoder/dense_86/BiasAddBiasAdd!encoder/dense_86/MatMul:product:0/encoder/dense_86/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџdr
encoder/dense_86/ReluRelu!encoder/dense_86/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџd
$encoder/z_mean/MatMul/ReadVariableOpReadVariableOp-encoder_z_mean_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0Є
encoder/z_mean/MatMulMatMul#encoder/dense_86/Relu:activations:0,encoder/z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
%encoder/z_mean/BiasAdd/ReadVariableOpReadVariableOp.encoder_z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ѓ
encoder/z_mean/BiasAddBiasAddencoder/z_mean/MatMul:product:0-encoder/z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
'encoder/z_log_var/MatMul/ReadVariableOpReadVariableOp0encoder_z_log_var_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0Њ
encoder/z_log_var/MatMulMatMul#encoder/dense_86/Relu:activations:0/encoder/z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
(encoder/z_log_var/BiasAdd/ReadVariableOpReadVariableOp1encoder_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ќ
encoder/z_log_var/BiasAddBiasAdd"encoder/z_log_var/MatMul:product:00encoder/z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџg
encoder/sampling_6/ShapeShapeencoder/z_mean/BiasAdd:output:0*
T0*
_output_shapes
:p
&encoder/sampling_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(encoder/sampling_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(encoder/sampling_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:А
 encoder/sampling_6/strided_sliceStridedSlice!encoder/sampling_6/Shape:output:0/encoder/sampling_6/strided_slice/stack:output:01encoder/sampling_6/strided_slice/stack_1:output:01encoder/sampling_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maski
encoder/sampling_6/Shape_1Shapeencoder/z_mean/BiasAdd:output:0*
T0*
_output_shapes
:r
(encoder/sampling_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:t
*encoder/sampling_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*encoder/sampling_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:К
"encoder/sampling_6/strided_slice_1StridedSlice#encoder/sampling_6/Shape_1:output:01encoder/sampling_6/strided_slice_1/stack:output:03encoder/sampling_6/strided_slice_1/stack_1:output:03encoder/sampling_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskД
&encoder/sampling_6/random_normal/shapePack)encoder/sampling_6/strided_slice:output:0+encoder/sampling_6/strided_slice_1:output:0*
N*
T0*
_output_shapes
:j
%encoder/sampling_6/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    l
'encoder/sampling_6/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?м
5encoder/sampling_6/random_normal/RandomStandardNormalRandomStandardNormal/encoder/sampling_6/random_normal/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2смњЯ
$encoder/sampling_6/random_normal/mulMul>encoder/sampling_6/random_normal/RandomStandardNormal:output:00encoder/sampling_6/random_normal/stddev:output:0*
T0*'
_output_shapes
:џџџџџџџџџЕ
 encoder/sampling_6/random_normalAddV2(encoder/sampling_6/random_normal/mul:z:0.encoder/sampling_6/random_normal/mean:output:0*
T0*'
_output_shapes
:џџџџџџџџџ]
encoder/sampling_6/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
encoder/sampling_6/mulMul!encoder/sampling_6/mul/x:output:0"encoder/z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџk
encoder/sampling_6/ExpExpencoder/sampling_6/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
encoder/sampling_6/mul_1Mulencoder/sampling_6/Exp:y:0$encoder/sampling_6/random_normal:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
encoder/sampling_6/addAddV2encoder/z_mean/BiasAdd:output:0encoder/sampling_6/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџi
IdentityIdentityencoder/sampling_6/add:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџs

Identity_1Identity"encoder/z_log_var/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџp

Identity_2Identityencoder/z_mean/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџу
NoOpNoOp(^encoder/dense_84/BiasAdd/ReadVariableOp'^encoder/dense_84/MatMul/ReadVariableOp(^encoder/dense_85/BiasAdd/ReadVariableOp'^encoder/dense_85/MatMul/ReadVariableOp(^encoder/dense_86/BiasAdd/ReadVariableOp'^encoder/dense_86/MatMul/ReadVariableOp)^encoder/z_log_var/BiasAdd/ReadVariableOp(^encoder/z_log_var/MatMul/ReadVariableOp&^encoder/z_mean/BiasAdd/ReadVariableOp%^encoder/z_mean/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ: : : : : : : : : : 2R
'encoder/dense_84/BiasAdd/ReadVariableOp'encoder/dense_84/BiasAdd/ReadVariableOp2P
&encoder/dense_84/MatMul/ReadVariableOp&encoder/dense_84/MatMul/ReadVariableOp2R
'encoder/dense_85/BiasAdd/ReadVariableOp'encoder/dense_85/BiasAdd/ReadVariableOp2P
&encoder/dense_85/MatMul/ReadVariableOp&encoder/dense_85/MatMul/ReadVariableOp2R
'encoder/dense_86/BiasAdd/ReadVariableOp'encoder/dense_86/BiasAdd/ReadVariableOp2P
&encoder/dense_86/MatMul/ReadVariableOp&encoder/dense_86/MatMul/ReadVariableOp2T
(encoder/z_log_var/BiasAdd/ReadVariableOp(encoder/z_log_var/BiasAdd/ReadVariableOp2R
'encoder/z_log_var/MatMul/ReadVariableOp'encoder/z_log_var/MatMul/ReadVariableOp2N
%encoder/z_mean/BiasAdd/ReadVariableOp%encoder/z_mean/BiasAdd/ReadVariableOp2L
$encoder/z_mean/MatMul/ReadVariableOp$encoder/z_mean/MatMul/ReadVariableOp:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
input_13
ч

$__inference_signature_wrapper_750679
input_13
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

identity_2ЂStatefulPartitionedCallЫ
StatefulPartitionedCallStatefulPartitionedCallinput_13unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__wrapped_model_750260o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
input_13
ћ
t
+__inference_sampling_6_layer_call_fn_750961
inputs_0
inputs_1
identityЂStatefulPartitionedCallб
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sampling_6_layer_call_and_return_conditional_losses_750376o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџ:џџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1
Ш	
і
E__inference_z_log_var_layer_call_and_return_conditional_losses_750344

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
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


(__inference_encoder_layer_call_fn_750408
input_13
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

identity_2ЂStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinput_13unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_750381o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
input_13
Х

)__inference_dense_84_layer_call_fn_750866

inputs
unknown:d
	unknown_0:d
identityЂStatefulPartitionedCallм
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
GPU2*0J 8 *M
fHRF
D__inference_dense_84_layer_call_and_return_conditional_losses_750278o
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


ѕ
D__inference_dense_84_layer_call_and_return_conditional_losses_750877

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
:џџџџџџџџџdP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџda
IdentityIdentityRelu:activations:0^NoOp*
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
Ё+

"__inference__traced_restore_751082
file_prefix2
 assignvariableop_dense_84_kernel:d.
 assignvariableop_1_dense_84_bias:d4
"assignvariableop_2_dense_85_kernel:dd.
 assignvariableop_3_dense_85_bias:d4
"assignvariableop_4_dense_86_kernel:dd.
 assignvariableop_5_dense_86_bias:d2
 assignvariableop_6_z_mean_kernel:d,
assignvariableop_7_z_mean_bias:5
#assignvariableop_8_z_log_var_kernel:d/
!assignvariableop_9_z_log_var_bias:
identity_11ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_2ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9Г
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*й
valueЯBЬB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B е
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*@
_output_shapes.
,:::::::::::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp assignvariableop_dense_84_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_84_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_85_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_85_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_86_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_86_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp assignvariableop_6_z_mean_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_z_mean_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp#assignvariableop_8_z_log_var_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp!assignvariableop_9_z_log_var_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Ћ
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_11IdentityIdentity_10:output:0^NoOp_1*
T0*
_output_shapes
: 
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
э"

C__inference_encoder_layer_call_and_return_conditional_losses_750616
input_13!
dense_84_750587:d
dense_84_750589:d!
dense_85_750592:dd
dense_85_750594:d!
dense_86_750597:dd
dense_86_750599:d
z_mean_750602:d
z_mean_750604:"
z_log_var_750607:d
z_log_var_750609:
identity

identity_1

identity_2Ђ dense_84/StatefulPartitionedCallЂ dense_85/StatefulPartitionedCallЂ dense_86/StatefulPartitionedCallЂ"sampling_6/StatefulPartitionedCallЂ!z_log_var/StatefulPartitionedCallЂz_mean/StatefulPartitionedCallѕ
 dense_84/StatefulPartitionedCallStatefulPartitionedCallinput_13dense_84_750587dense_84_750589*
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
GPU2*0J 8 *M
fHRF
D__inference_dense_84_layer_call_and_return_conditional_losses_750278
 dense_85/StatefulPartitionedCallStatefulPartitionedCall)dense_84/StatefulPartitionedCall:output:0dense_85_750592dense_85_750594*
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
GPU2*0J 8 *M
fHRF
D__inference_dense_85_layer_call_and_return_conditional_losses_750295
 dense_86/StatefulPartitionedCallStatefulPartitionedCall)dense_85/StatefulPartitionedCall:output:0dense_86_750597dense_86_750599*
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
GPU2*0J 8 *M
fHRF
D__inference_dense_86_layer_call_and_return_conditional_losses_750312
z_mean/StatefulPartitionedCallStatefulPartitionedCall)dense_86/StatefulPartitionedCall:output:0z_mean_750602z_mean_750604*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_z_mean_layer_call_and_return_conditional_losses_750328
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall)dense_86/StatefulPartitionedCall:output:0z_log_var_750607z_log_var_750609*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_z_log_var_layer_call_and_return_conditional_losses_750344
"sampling_6/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sampling_6_layer_call_and_return_conditional_losses_750376v
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ{

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ|

Identity_2Identity+sampling_6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp!^dense_84/StatefulPartitionedCall!^dense_85/StatefulPartitionedCall!^dense_86/StatefulPartitionedCall#^sampling_6/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ: : : : : : : : : : 2D
 dense_84/StatefulPartitionedCall dense_84/StatefulPartitionedCall2D
 dense_85/StatefulPartitionedCall dense_85/StatefulPartitionedCall2D
 dense_86/StatefulPartitionedCall dense_86/StatefulPartitionedCall2H
"sampling_6/StatefulPartitionedCall"sampling_6/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
input_13
Х

)__inference_dense_86_layer_call_fn_750906

inputs
unknown:dd
	unknown_0:d
identityЂStatefulPartitionedCallм
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
GPU2*0J 8 *M
fHRF
D__inference_dense_86_layer_call_and_return_conditional_losses_750312o
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
ч"

C__inference_encoder_layer_call_and_return_conditional_losses_750381

inputs!
dense_84_750279:d
dense_84_750281:d!
dense_85_750296:dd
dense_85_750298:d!
dense_86_750313:dd
dense_86_750315:d
z_mean_750329:d
z_mean_750331:"
z_log_var_750345:d
z_log_var_750347:
identity

identity_1

identity_2Ђ dense_84/StatefulPartitionedCallЂ dense_85/StatefulPartitionedCallЂ dense_86/StatefulPartitionedCallЂ"sampling_6/StatefulPartitionedCallЂ!z_log_var/StatefulPartitionedCallЂz_mean/StatefulPartitionedCallѓ
 dense_84/StatefulPartitionedCallStatefulPartitionedCallinputsdense_84_750279dense_84_750281*
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
GPU2*0J 8 *M
fHRF
D__inference_dense_84_layer_call_and_return_conditional_losses_750278
 dense_85/StatefulPartitionedCallStatefulPartitionedCall)dense_84/StatefulPartitionedCall:output:0dense_85_750296dense_85_750298*
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
GPU2*0J 8 *M
fHRF
D__inference_dense_85_layer_call_and_return_conditional_losses_750295
 dense_86/StatefulPartitionedCallStatefulPartitionedCall)dense_85/StatefulPartitionedCall:output:0dense_86_750313dense_86_750315*
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
GPU2*0J 8 *M
fHRF
D__inference_dense_86_layer_call_and_return_conditional_losses_750312
z_mean/StatefulPartitionedCallStatefulPartitionedCall)dense_86/StatefulPartitionedCall:output:0z_mean_750329z_mean_750331*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_z_mean_layer_call_and_return_conditional_losses_750328
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall)dense_86/StatefulPartitionedCall:output:0z_log_var_750345z_log_var_750347*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_z_log_var_layer_call_and_return_conditional_losses_750344
"sampling_6/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sampling_6_layer_call_and_return_conditional_losses_750376v
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ{

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ|

Identity_2Identity+sampling_6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp!^dense_84/StatefulPartitionedCall!^dense_85/StatefulPartitionedCall!^dense_86/StatefulPartitionedCall#^sampling_6/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ: : : : : : : : : : 2D
 dense_84/StatefulPartitionedCall dense_84/StatefulPartitionedCall2D
 dense_85/StatefulPartitionedCall dense_85/StatefulPartitionedCall2D
 dense_86/StatefulPartitionedCall dense_86/StatefulPartitionedCall2H
"sampling_6/StatefulPartitionedCall"sampling_6/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

u
F__inference_sampling_6_layer_call_and_return_conditional_losses_750987
inputs_0
inputs_1
identity=
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
valueB:б
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
valueB:л
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
 *  ?Ж
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2зЖв
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:џџџџџџџџџ|
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:џџџџџџџџџJ
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?V
mulMulmul/x:output:0inputs_1*
T0*'
_output_shapes
:џџџџџџџџџE
ExpExpmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџZ
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:џџџџџџџџџS
addAddV2inputs_0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџO
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџ:џџџџџџџџџ:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1


ѕ
D__inference_dense_86_layer_call_and_return_conditional_losses_750917

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
:џџџџџџџџџdP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџda
IdentityIdentityRelu:activations:0^NoOp*
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
Х

)__inference_dense_85_layer_call_fn_750886

inputs
unknown:dd
	unknown_0:d
identityЂStatefulPartitionedCallм
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
GPU2*0J 8 *M
fHRF
D__inference_dense_85_layer_call_and_return_conditional_losses_750295o
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
E__inference_z_log_var_layer_call_and_return_conditional_losses_750955

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
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


ѕ
D__inference_dense_84_layer_call_and_return_conditional_losses_750278

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
:џџџџџџџџџdP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџda
IdentityIdentityRelu:activations:0^NoOp*
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


(__inference_encoder_layer_call_fn_750584
input_13
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

identity_2ЂStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinput_13unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_750528o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
input_13
аC

C__inference_encoder_layer_call_and_return_conditional_losses_750797

inputs9
'dense_84_matmul_readvariableop_resource:d6
(dense_84_biasadd_readvariableop_resource:d9
'dense_85_matmul_readvariableop_resource:dd6
(dense_85_biasadd_readvariableop_resource:d9
'dense_86_matmul_readvariableop_resource:dd6
(dense_86_biasadd_readvariableop_resource:d7
%z_mean_matmul_readvariableop_resource:d4
&z_mean_biasadd_readvariableop_resource::
(z_log_var_matmul_readvariableop_resource:d7
)z_log_var_biasadd_readvariableop_resource:
identity

identity_1

identity_2Ђdense_84/BiasAdd/ReadVariableOpЂdense_84/MatMul/ReadVariableOpЂdense_85/BiasAdd/ReadVariableOpЂdense_85/MatMul/ReadVariableOpЂdense_86/BiasAdd/ReadVariableOpЂdense_86/MatMul/ReadVariableOpЂ z_log_var/BiasAdd/ReadVariableOpЂz_log_var/MatMul/ReadVariableOpЂz_mean/BiasAdd/ReadVariableOpЂz_mean/MatMul/ReadVariableOp
dense_84/MatMul/ReadVariableOpReadVariableOp'dense_84_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0{
dense_84/MatMulMatMulinputs&dense_84/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd
dense_84/BiasAdd/ReadVariableOpReadVariableOp(dense_84_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
dense_84/BiasAddBiasAdddense_84/MatMul:product:0'dense_84/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџdb
dense_84/ReluReludense_84/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџd
dense_85/MatMul/ReadVariableOpReadVariableOp'dense_85_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0
dense_85/MatMulMatMuldense_84/Relu:activations:0&dense_85/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd
dense_85/BiasAdd/ReadVariableOpReadVariableOp(dense_85_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
dense_85/BiasAddBiasAdddense_85/MatMul:product:0'dense_85/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџdb
dense_85/ReluReludense_85/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџd
dense_86/MatMul/ReadVariableOpReadVariableOp'dense_86_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype0
dense_86/MatMulMatMuldense_85/Relu:activations:0&dense_86/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd
dense_86/BiasAdd/ReadVariableOpReadVariableOp(dense_86_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
dense_86/BiasAddBiasAdddense_86/MatMul:product:0'dense_86/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџdb
dense_86/ReluReludense_86/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџd
z_mean/MatMul/ReadVariableOpReadVariableOp%z_mean_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0
z_mean/MatMulMatMuldense_86/Relu:activations:0$z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
z_mean/BiasAdd/ReadVariableOpReadVariableOp&z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
z_mean/BiasAddBiasAddz_mean/MatMul:product:0%z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
z_log_var/MatMul/ReadVariableOpReadVariableOp(z_log_var_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0
z_log_var/MatMulMatMuldense_86/Relu:activations:0'z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
 z_log_var/BiasAdd/ReadVariableOpReadVariableOp)z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
z_log_var/BiasAddBiasAddz_log_var/MatMul:product:0(z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџW
sampling_6/ShapeShapez_mean/BiasAdd:output:0*
T0*
_output_shapes
:h
sampling_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 sampling_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 sampling_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
sampling_6/strided_sliceStridedSlicesampling_6/Shape:output:0'sampling_6/strided_slice/stack:output:0)sampling_6/strided_slice/stack_1:output:0)sampling_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
sampling_6/Shape_1Shapez_mean/BiasAdd:output:0*
T0*
_output_shapes
:j
 sampling_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:l
"sampling_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"sampling_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
sampling_6/strided_slice_1StridedSlicesampling_6/Shape_1:output:0)sampling_6/strided_slice_1/stack:output:0+sampling_6/strided_slice_1/stack_1:output:0+sampling_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
sampling_6/random_normal/shapePack!sampling_6/strided_slice:output:0#sampling_6/strided_slice_1:output:0*
N*
T0*
_output_shapes
:b
sampling_6/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    d
sampling_6/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ь
-sampling_6/random_normal/RandomStandardNormalRandomStandardNormal'sampling_6/random_normal/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*
seedБџх)*
seed2рЉтЗ
sampling_6/random_normal/mulMul6sampling_6/random_normal/RandomStandardNormal:output:0(sampling_6/random_normal/stddev:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
sampling_6/random_normalAddV2 sampling_6/random_normal/mul:z:0&sampling_6/random_normal/mean:output:0*
T0*'
_output_shapes
:џџџџџџџџџU
sampling_6/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?~
sampling_6/mulMulsampling_6/mul/x:output:0z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ[
sampling_6/ExpExpsampling_6/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ{
sampling_6/mul_1Mulsampling_6/Exp:y:0sampling_6/random_normal:z:0*
T0*'
_output_shapes
:џџџџџџџџџx
sampling_6/addAddV2z_mean/BiasAdd:output:0sampling_6/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџf
IdentityIdentityz_mean/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџk

Identity_1Identityz_log_var/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџc

Identity_2Identitysampling_6/add:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp ^dense_84/BiasAdd/ReadVariableOp^dense_84/MatMul/ReadVariableOp ^dense_85/BiasAdd/ReadVariableOp^dense_85/MatMul/ReadVariableOp ^dense_86/BiasAdd/ReadVariableOp^dense_86/MatMul/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ: : : : : : : : : : 2B
dense_84/BiasAdd/ReadVariableOpdense_84/BiasAdd/ReadVariableOp2@
dense_84/MatMul/ReadVariableOpdense_84/MatMul/ReadVariableOp2B
dense_85/BiasAdd/ReadVariableOpdense_85/BiasAdd/ReadVariableOp2@
dense_85/MatMul/ReadVariableOpdense_85/MatMul/ReadVariableOp2B
dense_86/BiasAdd/ReadVariableOpdense_86/BiasAdd/ReadVariableOp2@
dense_86/MatMul/ReadVariableOpdense_86/MatMul/ReadVariableOp2D
 z_log_var/BiasAdd/ReadVariableOp z_log_var/BiasAdd/ReadVariableOp2B
z_log_var/MatMul/ReadVariableOpz_log_var/MatMul/ReadVariableOp2>
z_mean/BiasAdd/ReadVariableOpz_mean/BiasAdd/ReadVariableOp2<
z_mean/MatMul/ReadVariableOpz_mean/MatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs"ПL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Њ
serving_default
=
input_131
serving_default_input_13:0џџџџџџџџџ>

sampling_60
StatefulPartitionedCall:0џџџџџџџџџ=
	z_log_var0
StatefulPartitionedCall:1џџџџџџџџџ:
z_mean0
StatefulPartitionedCall:2џџџџџџџџџtensorflow/serving/predict:ћ
Б
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
Л
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
Л
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
Л
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

&kernel
'bias"
_tf_keras_layer
Л
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses

.kernel
/bias"
_tf_keras_layer
Л
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses

6kernel
7bias"
_tf_keras_layer
Ѕ
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
Ъ
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
ж
Ctrace_0
Dtrace_1
Etrace_2
Ftrace_32ы
(__inference_encoder_layer_call_fn_750408
(__inference_encoder_layer_call_fn_750708
(__inference_encoder_layer_call_fn_750737
(__inference_encoder_layer_call_fn_750584Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 zCtrace_0zDtrace_1zEtrace_2zFtrace_3
Т
Gtrace_0
Htrace_1
Itrace_2
Jtrace_32з
C__inference_encoder_layer_call_and_return_conditional_losses_750797
C__inference_encoder_layer_call_and_return_conditional_losses_750857
C__inference_encoder_layer_call_and_return_conditional_losses_750616
C__inference_encoder_layer_call_and_return_conditional_losses_750648Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 zGtrace_0zHtrace_1zItrace_2zJtrace_3
ЭBЪ
!__inference__wrapped_model_750260input_13"
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
­
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
э
Qtrace_02а
)__inference_dense_84_layer_call_fn_750866Ђ
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
 zQtrace_0

Rtrace_02ы
D__inference_dense_84_layer_call_and_return_conditional_losses_750877Ђ
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
 zRtrace_0
!:d2dense_84/kernel
:d2dense_84/bias
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
­
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
э
Xtrace_02а
)__inference_dense_85_layer_call_fn_750886Ђ
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
 zXtrace_0

Ytrace_02ы
D__inference_dense_85_layer_call_and_return_conditional_losses_750897Ђ
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
 zYtrace_0
!:dd2dense_85/kernel
:d2dense_85/bias
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
­
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
э
_trace_02а
)__inference_dense_86_layer_call_fn_750906Ђ
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
 z_trace_0

`trace_02ы
D__inference_dense_86_layer_call_and_return_conditional_losses_750917Ђ
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
 z`trace_0
!:dd2dense_86/kernel
:d2dense_86/bias
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
­
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
ftrace_02Ю
'__inference_z_mean_layer_call_fn_750926Ђ
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
 zftrace_0

gtrace_02щ
B__inference_z_mean_layer_call_and_return_conditional_losses_750936Ђ
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
­
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
mtrace_02б
*__inference_z_log_var_layer_call_fn_750945Ђ
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
 zmtrace_0

ntrace_02ь
E__inference_z_log_var_layer_call_and_return_conditional_losses_750955Ђ
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
 zntrace_0
": d2z_log_var/kernel
:2z_log_var/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
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
я
ttrace_02в
+__inference_sampling_6_layer_call_fn_750961Ђ
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
 zttrace_0

utrace_02э
F__inference_sampling_6_layer_call_and_return_conditional_losses_750987Ђ
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
ќBљ
(__inference_encoder_layer_call_fn_750408input_13"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
њBї
(__inference_encoder_layer_call_fn_750708inputs"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
њBї
(__inference_encoder_layer_call_fn_750737inputs"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ќBљ
(__inference_encoder_layer_call_fn_750584input_13"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B
C__inference_encoder_layer_call_and_return_conditional_losses_750797inputs"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B
C__inference_encoder_layer_call_and_return_conditional_losses_750857inputs"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B
C__inference_encoder_layer_call_and_return_conditional_losses_750616input_13"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
B
C__inference_encoder_layer_call_and_return_conditional_losses_750648input_13"Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ЬBЩ
$__inference_signature_wrapper_750679input_13"
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
нBк
)__inference_dense_84_layer_call_fn_750866inputs"Ђ
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
јBѕ
D__inference_dense_84_layer_call_and_return_conditional_losses_750877inputs"Ђ
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
нBк
)__inference_dense_85_layer_call_fn_750886inputs"Ђ
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
јBѕ
D__inference_dense_85_layer_call_and_return_conditional_losses_750897inputs"Ђ
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
нBк
)__inference_dense_86_layer_call_fn_750906inputs"Ђ
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
јBѕ
D__inference_dense_86_layer_call_and_return_conditional_losses_750917inputs"Ђ
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
лBи
'__inference_z_mean_layer_call_fn_750926inputs"Ђ
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
іBѓ
B__inference_z_mean_layer_call_and_return_conditional_losses_750936inputs"Ђ
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
*__inference_z_log_var_layer_call_fn_750945inputs"Ђ
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
E__inference_z_log_var_layer_call_and_return_conditional_losses_750955inputs"Ђ
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
ыBш
+__inference_sampling_6_layer_call_fn_750961inputs/0inputs/1"Ђ
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
B
F__inference_sampling_6_layer_call_and_return_conditional_losses_750987inputs/0inputs/1"Ђ
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
 ў
!__inference__wrapped_model_750260и
&'./671Ђ.
'Ђ$
"
input_13џџџџџџџџџ
Њ "Њ
2

sampling_6$!

sampling_6џџџџџџџџџ
0
	z_log_var# 
	z_log_varџџџџџџџџџ
*
z_mean 
z_meanџџџџџџџџџЄ
D__inference_dense_84_layer_call_and_return_conditional_losses_750877\/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџd
 |
)__inference_dense_84_layer_call_fn_750866O/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџdЄ
D__inference_dense_85_layer_call_and_return_conditional_losses_750897\/Ђ,
%Ђ"
 
inputsџџџџџџџџџd
Њ "%Ђ"

0џџџџџџџџџd
 |
)__inference_dense_85_layer_call_fn_750886O/Ђ,
%Ђ"
 
inputsџџџџџџџџџd
Њ "џџџџџџџџџdЄ
D__inference_dense_86_layer_call_and_return_conditional_losses_750917\&'/Ђ,
%Ђ"
 
inputsџџџџџџџџџd
Њ "%Ђ"

0џџџџџџџџџd
 |
)__inference_dense_86_layer_call_fn_750906O&'/Ђ,
%Ђ"
 
inputsџџџџџџџџџd
Њ "џџџџџџџџџdћ
C__inference_encoder_layer_call_and_return_conditional_losses_750616Г
&'./679Ђ6
/Ђ,
"
input_13џџџџџџџџџ
p 

 
Њ "jЂg
`]

0/0џџџџџџџџџ

0/1џџџџџџџџџ

0/2џџџџџџџџџ
 ћ
C__inference_encoder_layer_call_and_return_conditional_losses_750648Г
&'./679Ђ6
/Ђ,
"
input_13џџџџџџџџџ
p

 
Њ "jЂg
`]

0/0џџџџџџџџџ

0/1џџџџџџџџџ

0/2џџџџџџџџџ
 љ
C__inference_encoder_layer_call_and_return_conditional_losses_750797Б
&'./677Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p 

 
Њ "jЂg
`]

0/0џџџџџџџџџ

0/1џџџџџџџџџ

0/2џџџџџџџџџ
 љ
C__inference_encoder_layer_call_and_return_conditional_losses_750857Б
&'./677Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p

 
Њ "jЂg
`]

0/0џџџџџџџџџ

0/1џџџџџџџџџ

0/2џџџџџџџџџ
 а
(__inference_encoder_layer_call_fn_750408Ѓ
&'./679Ђ6
/Ђ,
"
input_13џџџџџџџџџ
p 

 
Њ "ZW

0џџџџџџџџџ

1џџџџџџџџџ

2џџџџџџџџџа
(__inference_encoder_layer_call_fn_750584Ѓ
&'./679Ђ6
/Ђ,
"
input_13џџџџџџџџџ
p

 
Њ "ZW

0џџџџџџџџџ

1џџџџџџџџџ

2џџџџџџџџџЮ
(__inference_encoder_layer_call_fn_750708Ё
&'./677Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p 

 
Њ "ZW

0џџџџџџџџџ

1џџџџџџџџџ

2џџџџџџџџџЮ
(__inference_encoder_layer_call_fn_750737Ё
&'./677Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p

 
Њ "ZW

0џџџџџџџџџ

1џџџџџџџџџ

2џџџџџџџџџЮ
F__inference_sampling_6_layer_call_and_return_conditional_losses_750987ZЂW
PЂM
KH
"
inputs/0џџџџџџџџџ
"
inputs/1џџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 Ѕ
+__inference_sampling_6_layer_call_fn_750961vZЂW
PЂM
KH
"
inputs/0џџџџџџџџџ
"
inputs/1џџџџџџџџџ
Њ "џџџџџџџџџ
$__inference_signature_wrapper_750679ф
&'./67=Ђ:
Ђ 
3Њ0
.
input_13"
input_13џџџџџџџџџ"Њ
2

sampling_6$!

sampling_6џџџџџџџџџ
0
	z_log_var# 
	z_log_varџџџџџџџџџ
*
z_mean 
z_meanџџџџџџџџџЅ
E__inference_z_log_var_layer_call_and_return_conditional_losses_750955\67/Ђ,
%Ђ"
 
inputsџџџџџџџџџd
Њ "%Ђ"

0џџџџџџџџџ
 }
*__inference_z_log_var_layer_call_fn_750945O67/Ђ,
%Ђ"
 
inputsџџџџџџџџџd
Њ "џџџџџџџџџЂ
B__inference_z_mean_layer_call_and_return_conditional_losses_750936\.//Ђ,
%Ђ"
 
inputsџџџџџџџџџd
Њ "%Ђ"

0џџџџџџџџџ
 z
'__inference_z_mean_layer_call_fn_750926O.//Ђ,
%Ђ"
 
inputsџџџџџџџџџd
Њ "џџџџџџџџџ