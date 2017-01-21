'From Cuis 5.0 of 7 November 2016 [latest update: #3040] on 21 January 2017 at 5:25:13 am'!
'Description Please enter a description for this package'!
!provides: 'TensorFlow' 1 222!
!requires: 'FFI' 1 18 nil!
!classDefinition: #ExternalWordArrayTest category: #'TensorFlow-Kernel'!
TestCase subclass: #ExternalWordArrayTest
	instanceVariableNames: 'library'
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Kernel'!
!classDefinition: 'ExternalWordArrayTest class' category: #'TensorFlow-Kernel'!
ExternalWordArrayTest class
	instanceVariableNames: ''!

!classDefinition: #TensorFlowCAPITest category: #'TensorFlow-Kernel'!
TestCase subclass: #TensorFlowCAPITest
	instanceVariableNames: 'library'
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Kernel'!
!classDefinition: 'TensorFlowCAPITest class' category: #'TensorFlow-Kernel'!
TensorFlowCAPITest class
	instanceVariableNames: ''!

!classDefinition: #TensorFlowOperationsTest category: #'TensorFlow-Kernel'!
TestCase subclass: #TensorFlowOperationsTest
	instanceVariableNames: 'library'
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Kernel'!
!classDefinition: 'TensorFlowOperationsTest class' category: #'TensorFlow-Kernel'!
TensorFlowOperationsTest class
	instanceVariableNames: ''!

!classDefinition: #TensorFlowExamplesTest category: #'TensorFlow-Examples'!
TestCase subclass: #TensorFlowExamplesTest
	instanceVariableNames: 'nn'
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Examples'!
!classDefinition: 'TensorFlowExamplesTest class' category: #'TensorFlow-Examples'!
TensorFlowExamplesTest class
	instanceVariableNames: ''!

!classDefinition: #TensorFlowExamplesWithOperationsTest category: #'TensorFlow-Examples'!
TensorFlowExamplesTest subclass: #TensorFlowExamplesWithOperationsTest
	instanceVariableNames: ''
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Examples'!
!classDefinition: 'TensorFlowExamplesWithOperationsTest class' category: #'TensorFlow-Examples'!
TensorFlowExamplesWithOperationsTest class
	instanceVariableNames: ''!

!classDefinition: #TensorFlowCAPI category: #'TensorFlow-Kernel'!
ExternalLibrary subclass: #TensorFlowCAPI
	instanceVariableNames: ''
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Kernel'!
!classDefinition: 'TensorFlowCAPI class' category: #'TensorFlow-Kernel'!
TensorFlowCAPI class
	instanceVariableNames: 'current'!

!classDefinition: #TF_Structure category: #'TensorFlow-Kernel'!
ExternalStructure subclass: #TF_Structure
	instanceVariableNames: ''
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Kernel'!
!classDefinition: 'TF_Structure class' category: #'TensorFlow-Kernel'!
TF_Structure class
	instanceVariableNames: ''!

!classDefinition: #ExternalWordArray category: #'TensorFlow-Kernel'!
TF_Structure subclass: #ExternalWordArray
	instanceVariableNames: ''
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Kernel'!
!classDefinition: 'ExternalWordArray class' category: #'TensorFlow-Kernel'!
ExternalWordArray class
	instanceVariableNames: ''!

!classDefinition: #Int64Array category: #'TensorFlow-Kernel'!
ExternalWordArray subclass: #Int64Array
	instanceVariableNames: ''
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Kernel'!
!classDefinition: 'Int64Array class' category: #'TensorFlow-Kernel'!
Int64Array class
	instanceVariableNames: ''!

!classDefinition: #TF_OperationPtrArray category: #'TensorFlow-Kernel'!
ExternalWordArray subclass: #TF_OperationPtrArray
	instanceVariableNames: ''
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Kernel'!
!classDefinition: 'TF_OperationPtrArray class' category: #'TensorFlow-Kernel'!
TF_OperationPtrArray class
	instanceVariableNames: ''!

!classDefinition: #TF_OutputArray category: #'TensorFlow-Kernel'!
ExternalWordArray subclass: #TF_OutputArray
	instanceVariableNames: ''
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Kernel'!
!classDefinition: 'TF_OutputArray class' category: #'TensorFlow-Kernel'!
TF_OutputArray class
	instanceVariableNames: ''!

!classDefinition: #TF_TensorPtrArray category: #'TensorFlow-Kernel'!
ExternalWordArray subclass: #TF_TensorPtrArray
	instanceVariableNames: ''
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Kernel'!
!classDefinition: 'TF_TensorPtrArray class' category: #'TensorFlow-Kernel'!
TF_TensorPtrArray class
	instanceVariableNames: ''!

!classDefinition: #TF_AttrMetadata category: #'TensorFlow-Kernel'!
TF_Structure subclass: #TF_AttrMetadata
	instanceVariableNames: ''
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Kernel'!
!classDefinition: 'TF_AttrMetadata class' category: #'TensorFlow-Kernel'!
TF_AttrMetadata class
	instanceVariableNames: ''!

!classDefinition: #TF_Buffer category: #'TensorFlow-Kernel'!
TF_Structure subclass: #TF_Buffer
	instanceVariableNames: ''
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Kernel'!
!classDefinition: 'TF_Buffer class' category: #'TensorFlow-Kernel'!
TF_Buffer class
	instanceVariableNames: ''!

!classDefinition: #TF_Graph category: #'TensorFlow-Kernel'!
TF_Structure subclass: #TF_Graph
	instanceVariableNames: ''
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Kernel'!
!classDefinition: 'TF_Graph class' category: #'TensorFlow-Kernel'!
TF_Graph class
	instanceVariableNames: ''!

!classDefinition: #TF_ImportGraphDefOptions category: #'TensorFlow-Kernel'!
TF_Structure subclass: #TF_ImportGraphDefOptions
	instanceVariableNames: ''
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Kernel'!
!classDefinition: 'TF_ImportGraphDefOptions class' category: #'TensorFlow-Kernel'!
TF_ImportGraphDefOptions class
	instanceVariableNames: ''!

!classDefinition: #TF_Operation category: #'TensorFlow-Kernel'!
TF_Structure subclass: #TF_Operation
	instanceVariableNames: 'graph'
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Kernel'!
!classDefinition: 'TF_Operation class' category: #'TensorFlow-Kernel'!
TF_Operation class
	instanceVariableNames: ''!

!classDefinition: #TF_OperationDescription category: #'TensorFlow-Kernel'!
TF_Structure subclass: #TF_OperationDescription
	instanceVariableNames: ''
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Kernel'!
!classDefinition: 'TF_OperationDescription class' category: #'TensorFlow-Kernel'!
TF_OperationDescription class
	instanceVariableNames: ''!

!classDefinition: #TF_Output category: #'TensorFlow-Kernel'!
TF_Structure subclass: #TF_Output
	instanceVariableNames: ''
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Kernel'!
!classDefinition: 'TF_Output class' category: #'TensorFlow-Kernel'!
TF_Output class
	instanceVariableNames: ''!

!classDefinition: #TF_Input category: #'TensorFlow-Kernel'!
TF_Output subclass: #TF_Input
	instanceVariableNames: ''
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Kernel'!
!classDefinition: 'TF_Input class' category: #'TensorFlow-Kernel'!
TF_Input class
	instanceVariableNames: ''!

!classDefinition: #TF_Session category: #'TensorFlow-Kernel'!
TF_Structure subclass: #TF_Session
	instanceVariableNames: ''
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Kernel'!
!classDefinition: 'TF_Session class' category: #'TensorFlow-Kernel'!
TF_Session class
	instanceVariableNames: ''!

!classDefinition: #TF_SessionOptions category: #'TensorFlow-Kernel'!
TF_Structure subclass: #TF_SessionOptions
	instanceVariableNames: ''
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Kernel'!
!classDefinition: 'TF_SessionOptions class' category: #'TensorFlow-Kernel'!
TF_SessionOptions class
	instanceVariableNames: ''!

!classDefinition: #TF_Status category: #'TensorFlow-Kernel'!
TF_Structure subclass: #TF_Status
	instanceVariableNames: ''
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Kernel'!
!classDefinition: 'TF_Status class' category: #'TensorFlow-Kernel'!
TF_Status class
	instanceVariableNames: ''!

!classDefinition: #TF_Tensor category: #'TensorFlow-Kernel'!
TF_Structure subclass: #TF_Tensor
	instanceVariableNames: ''
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Kernel'!
!classDefinition: 'TF_Tensor class' category: #'TensorFlow-Kernel'!
TF_Tensor class
	instanceVariableNames: ''!

!classDefinition: #ExamplePlan category: #'TensorFlow-Examples'!
Object subclass: #ExamplePlan
	instanceVariableNames: 'graph inputs outputs operation'
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Examples'!
!classDefinition: 'ExamplePlan class' category: #'TensorFlow-Examples'!
ExamplePlan class
	instanceVariableNames: ''!

!classDefinition: #ExampleNearestNeighborPlan category: #'TensorFlow-Examples'!
ExamplePlan subclass: #ExampleNearestNeighborPlan
	instanceVariableNames: ''
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Examples'!
!classDefinition: 'ExampleNearestNeighborPlan class' category: #'TensorFlow-Examples'!
ExampleNearestNeighborPlan class
	instanceVariableNames: ''!

!classDefinition: #ExampleOLSPlan category: #'TensorFlow-Examples'!
ExamplePlan subclass: #ExampleOLSPlan
	instanceVariableNames: ''
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Examples'!
!classDefinition: 'ExampleOLSPlan class' category: #'TensorFlow-Examples'!
ExampleOLSPlan class
	instanceVariableNames: ''!

!classDefinition: #NearestNeighbor category: #'TensorFlow-Examples'!
Object subclass: #NearestNeighbor
	instanceVariableNames: 'plan'
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Examples'!
!classDefinition: 'NearestNeighbor class' category: #'TensorFlow-Examples'!
NearestNeighbor class
	instanceVariableNames: ''!

!classDefinition: #NearestNeighborWithOperations category: #'TensorFlow-Examples'!
NearestNeighbor subclass: #NearestNeighborWithOperations
	instanceVariableNames: ''
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Examples'!
!classDefinition: 'NearestNeighborWithOperations class' category: #'TensorFlow-Examples'!
NearestNeighborWithOperations class
	instanceVariableNames: ''!

!classDefinition: #OLSExample category: #'TensorFlow-Examples'!
Object subclass: #OLSExample
	instanceVariableNames: 'plan'
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Examples'!
!classDefinition: 'OLSExample class' category: #'TensorFlow-Examples'!
OLSExample class
	instanceVariableNames: ''!


!Object methodsFor: '*TensorFlow' stamp: 'gera 1/7/2017 20:18:45'!
ignoreFinalization
	self finalizationRegistry remove: self ifAbsent: [].
! !

!Object methodsFor: '*TensorFlow' stamp: 'gera 1/7/2017 20:18:16'!
useFinalization
	self finalizationRegistry add: self! !

!String methodsFor: '*TensorFlow' stamp: 'gera 1/7/2017 04:33:10'!
asAsciiZ
	^self, Character null asString! !

!ExternalWordArrayTest methodsFor: 'testing' stamp: 'gera 12/29/2016 14:51:01'!
testArray: anExternalWordArraySubclass sampleElement: anObject
	| array size  |
	size _ 4.
	array _ anExternalWordArraySubclass new: 4.
	self assert: size * Smalltalk wordSize = array getHandle size.
	self assert: (array first isKindOf: anObject class).
	1 to: size do: [:index |
		self assert: ((array at: index) isKindOf: anObject class)].
! !

!ExternalWordArrayTest methodsFor: 'testing' stamp: 'gera 1/13/2017 12:17:48'!
testArrayAtPutFor: anExternalWordArraySubclass elementSample: anObject
	| array size one  |
	size _ 4.
	array _ anExternalWordArraySubclass new: 4.
	one _ array first.
	
	array first: anObject.
	self assert: array first equals: anObject.
	self assert: (array first isKindOf: anObject class).
	array first: one.
	self assert: array first equals: one.
		
	1 to: size do: [:index |
		array at: index put: anObject.
		self assert: (array at: index) equals: anObject.
		self assert: ((array at: index) isKindOf: anObject class).
		array at: index put: one.
		self assert: (array at: index) equals: one.
	]
	! !

!ExternalWordArrayTest methodsFor: 'testing' stamp: 'gera 1/13/2017 13:08:20'!
testExternalFromArrayOperation
	| op1 op2 array |
	op1 _ TF_Operation fromHandle: (ExternalAddress new + 11111).
	op2 _ TF_Operation fromHandle: (ExternalAddress new + 22222).
	array _ TF_OperationPtrArray externalFromArray: (Array with: op1 with: op2).

	self assert: array first getHandle = op1 getHandle.
	self assert: (array at: 2) getHandle = op2 getHandle.
! !

!ExternalWordArrayTest methodsFor: 'testing' stamp: 'gera 1/13/2017 13:08:25'!
testExternalFromArrayOutput
	| o1 o2 array |
	o1 _ TF_Output new index: 16r8070605040302010.
	o2 _ TF_Output new index: 16rFEDCBA9876543210.
	array _ TF_OutputArray externalFromArray: (Array with: o1 with: o2).

	self assert: array first index = o1 index.
	self assert: (array at: 2) index = o2 index.
! !

!ExternalWordArrayTest methodsFor: 'testing' stamp: 'gera 1/13/2017 13:08:14'!
testExternalFromArrayTensor
	| t1 t2 array |
	t1 _ TF_Tensor fromInt64: 16r8070605040302010.
	t2 _ TF_Tensor fromInt64: 16rFEDCBA9876543210.
	array _ TF_TensorPtrArray externalFromArray: (Array with: t1 with: t2).
	
	self assert: array first getHandle = t1 getHandle.
	self assert: (array at: 2) getHandle = t2 getHandle.
	! !

!ExternalWordArrayTest methodsFor: 'testing' stamp: 'gera 1/20/2017 18:08:26'!
testOperationArrayAtPut
	| graph sample |
	graph _ TF_Graph create.
	sample _ graph placeholder: 'var' type: TF_Tensor typeInt64.
	
	self testArrayAtPutFor: TF_OperationPtrArray elementSample: sample.
! !

!ExternalWordArrayTest methodsFor: 'testing' stamp: 'gera 12/27/2016 19:15:43'!
testOperationArrayElementsOfRightClass
	^ self testArray: TF_OperationPtrArray sampleElement: TF_Operation new.
! !

!ExternalWordArrayTest methodsFor: 'testing' stamp: 'gera 12/29/2016 15:37:49'!
testOutputArrayAtPut
	| array size one other operation |
	size _ 4.
	array _ TF_OutputArray externalNew: size.
	operation _ TF_Operation fromHandle: ExternalAddress new + 16rF1F2F3F4.
	
	one _ TF_Output new.
	one index: 16rF1F2F3F4F5F6F7F8.
	one operation: operation.

	other _ array first.
	array first: one.

	self deny: one getHandle = other getHandle.	
	
	self assert: one index = other index.
	self assert: one operation = other operation.
	
	operation _ TF_Operation fromHandle: ExternalAddress new + 16r12345678.
	one index: 16rF0F1F2F3F4F5F6F7.
	one operation: operation.

	1 to: size do: [:index |
		other _ array at: index.
		array at: index put: one.

		self deny: one getHandle = other getHandle.	
		self assert: one index = other index.
		self assert: one operation = other operation.
	]! !

!ExternalWordArrayTest methodsFor: 'testing' stamp: 'gera 12/29/2016 15:04:16'!
testOutputArrayElementsAreAliases
	| array size one other |
	size _ 4.
	array _ TF_OutputArray externalNew: size.

	one _ array first.
	other _ array at: 1.
	
	one index: 12345678.
	self assert: one index = 12345678.
	self assert: other index = 12345678.

	other index: 77665544.
	self assert: one index = 77665544.
	self assert: other index = 77665544.

	1 to: size do: [:index |
		one _ array at: index.
		other _ array at: index.

		one index: 12348765.
		self assert: one index = 12348765.
		self assert: other index = 12348765.

		other index: 77222244.
		self assert: one index = 77222244.
		self assert: other index = 77222244].
! !

!ExternalWordArrayTest methodsFor: 'testing' stamp: 'gera 12/29/2016 14:53:08'!
testOutputArrayElementsOfRightClass
	| array size  |
	size _ 4.
	array _ TF_OutputArray externalNew: size.
	self assert: (array first isKindOf: TF_Output).
	1 to: size do: [:index |
		self assert: ((array at: index) isKindOf: TF_Output)].
! !

!ExternalWordArrayTest methodsFor: 'testing' stamp: 'gera 1/13/2017 12:50:45'!
testTensorArrayAtPut
	| sample |
	sample _ TF_Tensor fromInt64: 16r12341234.
	
	self testArrayAtPutFor: TF_TensorPtrArray elementSample: sample.! !

!ExternalWordArrayTest methodsFor: 'testing' stamp: 'gera 12/27/2016 19:16:40'!
testTensorArrayElementsOfRightClass
	^ self testArray: TF_TensorPtrArray sampleElement: TF_Tensor new.! !

!ExternalWordArrayTest methodsFor: 'testing' stamp: 'gera 12/25/2016 20:54:13'!
testUIn64ArrayfromArray
	self testUIn64ArrayfromArray: #(1).
	self testUIn64ArrayfromArray: #(16r12345678 2 16r76543210 4 16rDEADBEAF 6 16rFFFFFFFF).
! !

!ExternalWordArrayTest methodsFor: 'testing' stamp: 'gera 1/13/2017 13:08:30'!
testUIn64ArrayfromArray: template
	| array |
	array _ Int64Array externalFromArray: template.
	
	template withIndexDo: [:each :index |
		self assert: ((template at: index) = (array at: index))
	].
! !

!ExternalWordArrayTest methodsFor: 'testing' stamp: 'gera 12/27/2016 18:13:43'!
testUInt64ArrayAtPut
	self testArrayAtPutFor: Int64Array elementSample: 12345.
	self testArrayAtPutFor: Int64Array elementSample: 16rFE00FE00.
	self testArrayAtPutFor: Int64Array elementSample: 16r7E00FE00FE00FE00.
	self testArrayAtPutFor: Int64Array elementSample: 16r-7E00FE00FE00FE00.
! !

!ExternalWordArrayTest methodsFor: 'testing' stamp: 'gera 12/27/2016 18:11:59'!
testUInt64ArrayElementsOfRightClass
	^ self testArray: Int64Array sampleElement: 1234.! !

!TensorFlowCAPITest methodsFor: 'graphs' stamp: 'gera 1/5/2017 23:56:12'!
addGraphTwoInputsInt64
	| graph in1 in2  |
	graph _ TF_Graph create.
	
	in1 _ graph placeholder: 'in1' type: TF_Tensor typeInt64.
	in2 _ graph placeholder: 'in2' type: TF_Tensor typeInt64.
	graph add: 'add' described: [:description |
		description addInput: (in1 output: 0).
		description addInput: (in2 output: 0)].
	^ graph! !

!TensorFlowCAPITest methodsFor: 'graphs' stamp: 'gera 1/6/2017 17:31:10'!
asStringGraphType: type
	| graph in |
	graph _ TF_Graph create.
	in _ graph placeholder: 'in' type: type.
	graph asString: 'out' described: [:description |
		description addInput: (in output: 0)].
	^ graph! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 1/13/2017 19:53:12'!
assertDimensionsOf: aMultidimensionalTensor is: anInteger
	| dimensions |
	dimensions _ TF_Tensor dimensionsOf: aMultidimensionalTensor.
	self
		assert: dimensions = anInteger
		description: 'The dimension is ', dimensions printString, ' and should have been ', anInteger printString! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 1/13/2017 19:48:59'!
assertRankOf: aMultidimensionalTensor is: anInteger
	| rank |
	rank _ TF_Tensor rankOf: aMultidimensionalTensor.
	self
		assert: rank = anInteger
		description: 'The rank is ', rank printString, ' and should have been ', anInteger printString! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 1/18/2017 00:03:38'!
assertSizeOf: aMultidimensionalTensor is: anInteger
	| size |
	size _ TF_Tensor sizeOf: aMultidimensionalTensor.
	self
		assert: size = anInteger
		description: 'The size is ', size printString, ' and should have been ', anInteger printString! !

!TensorFlowCAPITest methodsFor: 'graphs' stamp: 'gera 1/2/2017 15:40:57'!
concatGraphInputList
	| graph in1 in2 concat dimension dimensionValue inputs |
	graph _ TF_Graph create.
	dimensionValue _ TF_Tensor fromInt32: 0.
	dimension _ graph const: 'const' value: dimensionValue.
	
	in1 _ graph placeholder: 'in1' type: TF_Tensor typeInt64.
	in2 _ graph placeholder: 'in2' type: TF_Tensor typeInt64.
	inputs _ Array with: (in1 output: 0) with: (in2 output: 0).
	concat _ graph concat: 'concat' described: [:description |
		description addInput: (dimension output: 0).
		description addInputs: inputs.
		description at: 'N' putInt: 2.
		description at: 'T' putType: TF_Tensor typeInt64.
	].
	concat.
	^ graph! !

!TensorFlowCAPITest methodsFor: 'graphs' stamp: 'gera 1/2/2017 15:40:41'!
concatGraphInputListNoSizeNoType
	| graph in1 in2 concat dimension dimensionValue inputs |
	graph _ TF_Graph create.
	dimensionValue _ TF_Tensor fromInt32: 0.
	dimension _ graph const: 'const' value: dimensionValue.
	
	in1 _ graph placeholder: 'in1' type: TF_Tensor typeInt64.
	in2 _ graph placeholder: 'in2' type: TF_Tensor typeInt64.
	inputs _ Array with: (in1 output: 0) with: (in2 output: 0).
	concat _ graph concat: 'concat' described: [:description |
		description addInput: (dimension output: 0).
		description addInputs: inputs.
	].
	concat.
	^ graph! !

!TensorFlowCAPITest methodsFor: 'graphs' stamp: 'gera 1/2/2017 15:38:00'!
concatGraphInputListWrongSize
	| graph in1 in2 concat dimension dimensionValue inputs |
	graph _ TF_Graph create.
	dimensionValue _ TF_Tensor fromInt32: 0.
	dimension _ graph const: 'const' value: dimensionValue.
	
	in1 _ graph placeholder: 'in1' type: TF_Tensor typeInt64.
	in2 _ graph placeholder: 'in2' type: TF_Tensor typeInt64.
	inputs _ Array with: (in1 output: 0) with: (in2 output: 0).
	concat _ graph concat: 'concat' described: [:description |
		description addInput: (dimension output: 0).
		description addInputs: inputs.
		description at: 'N' putInt: 0.
	].
	concat.
	^ graph! !

!TensorFlowCAPITest methodsFor: 'graphs' stamp: 'gera 1/2/2017 15:39:49'!
concatGraphInputListWrongType
	| graph in1 in2 concat dimension dimensionValue inputs |
	graph _ TF_Graph create.
	dimensionValue _ TF_Tensor fromInt32: 0.
	dimension _ graph const: 'const' value: dimensionValue.
	
	in1 _ graph placeholder: 'in1' type: TF_Tensor typeInt64.
	in2 _ graph placeholder: 'in2' type: TF_Tensor typeInt64.
	inputs _ Array with: (in1 output: 0) with: (in2 output: 0).
	concat _ graph concat: 'concat' described: [:description |
		description addInput: (dimension output: 0).
		description addInputs: inputs.
		description at: 'T' putType: TF_Tensor typeInt32.
	].
	concat.
	^ graph! !

!TensorFlowCAPITest methodsFor: 'graphs' stamp: 'gera 1/18/2017 00:50:25'!
constant2x2FloatGraphDef
	" This GraphDef corresponds to simple Graph, defined as
	
		a = tf.constant([[-1.1, -2.1],[-1.2,-2.2]], name='a')
		
	 saved as ProtoBuf "
	^ #[
		 16r0A 16r42 16r0A 16r01 16r61 16r12 16r05 16r43 16r6F 16r6E 16r73 16r74 16r2A 16r29 16r0A 16r05
		 16r76 16r61 16r6C 16r75 16r65 16r12 16r20 16r42 16r1E 16r08 16r01 16r12 16r08 16r12 16r02 16r08
		 16r02 16r12 16r02 16r08 16r02 16r22 16r10 16rCD 16rCC 16r8C 16rBF 16r66 16r66 16r06 16rC0 16r9A
		 16r99 16r99 16rBF 16rCD 16rCC 16r0C 16rC0 16r2A 16r0B 16r0A 16r05 16r64 16r74 16r79 16r70 16r65
		 16r12 16r02 16r30 16r01 16r0A 16r0C 16r0A 16r04 16r69 16r6E 16r69 16r74 16r12 16r04 16r4E 16r6F
		 16r4F 16r70 16r22 16r02 16r08 16r11
	] asString! !

!TensorFlowCAPITest methodsFor: 'graphs' stamp: 'gera 1/18/2017 00:51:13'!
constant2x2FloatGraphFromDef
	^ TF_Graph fromString: self constant2x2FloatGraphDef! !

!TensorFlowCAPITest methodsFor: 'graphs' stamp: 'gera 12/26/2016 10:58:56'!
constantFloatGraphDef
	" This GraphDef corresponds to simple Graph, defined as
	
		tf.constant(0.42, name='a')
		
	 saved as ProtoBuf "
	^ #[
 16r0A 16r2E 16r0A 16r01 16r61 16r12 16r05 16r43 16r6F 16r6E 16r73 16r74 16r2A 16r15 16r0A 16r05
 16r76 16r61 16r6C 16r75 16r65 16r12 16r0C 16r42 16r0A 16r08 16r01 16r12 16r00 16r2A 16r04 16r3D
 16r0A 16rD7 16r3E 16r2A 16r0B 16r0A 16r05 16r64 16r74 16r79 16r70 16r65 16r12 16r02 16r30 16r01
 16r0A 16r0C 16r0A 16r04 16r69 16r6E 16r69 16r74 16r12 16r04 16r4E 16r6F 16r4F 16r70 16r22 16r02
 16r08 16r0F] asString! !

!TensorFlowCAPITest methodsFor: 'graphs' stamp: 'gera 1/6/2017 00:04:03'!
constantFloatGraphFromDef
	^ TF_Graph fromString: self constantFloatGraphDef! !

!TensorFlowCAPITest methodsFor: 'graphs' stamp: 'gera 12/26/2016 11:13:29'!
constantInt32GraphDef
	" This GraphDef corresponds to simple Graph, defined as
	
		tf.constant(0.42, name='a')
		
	 saved as ProtoBuf "
	^ #[
 16r0A 16r2B 16r0A 16r01 16r61 16r12 16r05 16r43 16r6F 16r6E 16r73 16r74 16r2A 16r12 16r0A 16r05
 16r76 16r61 16r6C 16r75 16r65 16r12 16r09 16r42 16r07 16r08 16r03 16r12 16r00 16r3A 16r01 16r2A
 16r2A 16r0B 16r0A 16r05 16r64 16r74 16r79 16r70 16r65 16r12 16r02 16r30 16r03 16r0A 16r0C 16r0A
 16r04 16r69 16r6E 16r69 16r74 16r12 16r04 16r4E 16r6F 16r4F 16r70 16r22 16r02 16r08 16r0F] asString! !

!TensorFlowCAPITest methodsFor: 'graphs' stamp: 'gera 1/6/2017 00:03:26'!
constantInt32GraphFromDef
	^ TF_Graph fromString: self constantInt32GraphDef! !

!TensorFlowCAPITest methodsFor: 'graphs' stamp: 'gera 1/6/2017 00:00:14'!
constantInt64Graph
	| graph operation constant |
	graph _ TF_Graph create.
	constant _ TF_Tensor fromInt64: 16r4242424242424242.
	
	self shouldnt: [
		operation _ graph const: 'a' value: constant
	] raise: Error.
	self deny: operation isNull.
	^ graph! !

!TensorFlowCAPITest methodsFor: 'graphs' stamp: 'gera 12/26/2016 12:23:03'!
constantInt64GraphDef
	" This GraphDef corresponds to simple Graph, defined as
	
		tf.constant(0.42, name='a')
		
	 saved as ProtoBuf "
	^ #[
 16r0A 16r33 16r0A 16r01 16r61 16r12 16r05 16r43 16r6F 16r6E 16r73 16r74 16r2A 16r1A 16r0A 16r05
 16r76 16r61 16r6C 16r75 16r65 16r12 16r11 16r42 16r0F 16r08 16r09 16r12 16r00 16r52 16r09 16rC2
 16r84 16r89 16r92 16rA4 16rC8 16r90 16rA1 16r42 16r2A 16r0B 16r0A 16r05 16r64 16r74 16r79 16r70
 16r65 16r12 16r02 16r30 16r09 16r0A 16r0C 16r0A 16r04 16r69 16r6E 16r69 16r74 16r12 16r04 16r4E
 16r6F 16r4F 16r70 16r22 16r02 16r08 16r0F] asString! !

!TensorFlowCAPITest methodsFor: 'graphs' stamp: 'gera 1/6/2017 00:02:53'!
constantInt64GraphFromDef
	^ TF_Graph fromString: self constantInt64GraphDef! !

!TensorFlowCAPITest methodsFor: 'graphs' stamp: 'gera 1/13/2017 01:28:16'!
decodeCSVGraphDefaults: anArrayOfTF_Tensors
	| graph records defaults |
	
	graph _ TF_Graph create.
	records _ (graph placeholder: 'records' type: TF_Tensor typeString) output: 0.
	defaults _ Array new: anArrayOfTF_Tensors size.
	
	anArrayOfTF_Tensors withIndexDo: [:each :index |
		| one |
		one _ (graph const: 'default',index printString value: each) output: 0.
		defaults at: index put: one].
	graph newOperation: 'DecodeCSV' named: 'output' described: [:description |
		description addInput: records.
		description addInputs: defaults].
	
	^ graph! !

!TensorFlowCAPITest methodsFor: 'graphs' stamp: 'gera 12/22/2016 02:20:25'!
emptyGraph
	^ TF_Graph fromString: self emptyGraphDef.
	! !

!TensorFlowCAPITest methodsFor: 'graphs' stamp: 'gera 12/22/2016 01:51:30'!
emptyGraphDef
	" This GraphDef corresponds to an Empty Graph (no operations), saved as ProtoBuf "
	^ #[16r0A 16r0C 16r0A 16r04 16r69 16r6E 16r69 16r74 16r12 16r04 16r4E 16r6F 16r4F 16r70 16r22 16r02 16r08 16r0F] asString! !

!TensorFlowCAPITest methodsFor: 'graphs' stamp: 'gera 1/6/2017 10:48:25'!
floatAsStringGraph
	| graph const |
	graph _ self constantFloatGraphFromDef.
	const _ graph operationNamed: 'a'.
	graph asString: 'output' described: [:description |
		description addInput: (const output: 0)].
	^ graph! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 1/18/2017 15:07:29'!
get2x2FloatFromGraphDef
	| graph session const result |
	graph _ self constant2x2FloatGraphFromDef.
	graph useFinalization.
	
	const _ graph operationNamed: 'a'.
	session _ TF_Session onGraph: graph.
	result _ session
		runOperation: const
		output: (const output: 0).
	
	^ result! !

!TensorFlowCAPITest methodsFor: 'graphs' stamp: 'gera 1/5/2017 23:57:31'!
mulGraphOneInputInt64
	| graph constant const in |
	graph _ TF_Graph create.
	constant _ TF_Tensor fromInt64: 16r0606060606060606.
	
	in _ graph placeholder: 'in' type: constant type.
	const _ graph const: 'const' value: constant.
	graph mul: 'mul' described: [:description |
		description addInput: (in output: 0).
		description addInput: (const output: 0)].
	
	^ graph! !

!TensorFlowCAPITest methodsFor: 'graphs' stamp: 'gera 1/5/2017 23:57:52'!
mulGraphTwoInputsInt64
	| graph constant const in1 in2 mul1 |
	graph _ TF_Graph create.
	constant _ TF_Tensor fromInt64: 16r0101010101010101.
	
	in1 _ graph placeholder: 'in1' type: constant type.
	in2 _ graph placeholder: 'in2' type: constant type.
	const _ graph const: 'const' value: constant.
	mul1 _ graph mul: 'mul1' described: [:description |
		description addInput: (const output: 0).
		description addInput: (in1 output: 0)].
	graph mul: 'mul2' described: [:description |
		description addInput: (mul1 output: 0).
		description addInput: (in2 output: 0)].
	
	^ graph! !

!TensorFlowCAPITest methodsFor: 'graphs' stamp: 'gera 1/13/2017 11:12:16'!
mulGraphTwoInputsInt64ConstTensorDeleted
	| graph constant const in1 in2 mul1 |
	graph _ TF_Graph create.
	constant _ TF_Tensor fromInt64: 16r2121212121212121.
	
	in1 _ graph placeholder: 'in1' type: constant type.
	in2 _ graph placeholder: 'in2' type: constant type.
	const _ graph const: 'const' value: constant.

	constant delete.	
	constant _ TF_Tensor fromInt64: 16r2222222222222222.
	constant delete.	

	mul1 _ graph mul: 'mul1' described: [:description |
		description addInput: (const output: 0).
		description addInput: (in1 output: 0)].

	graph mul: 'mul2' described: [:description |
		description addInput: (mul1 output: 0).
		description addInput: (in2 output: 0)].
	
	^ graph! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/7/2017 20:28:39'!
runFloatAsStringGraph
	| session graph output result |
	
	graph _ self floatAsStringGraph.
	session _ TF_Session onGraph: graph.
	output _ graph operationNamed: 'output'.
	result _ session
		runOperation: output
		output: (output output: 0).
	
	graph delete.
	^ result
! !

!TensorFlowCAPITest methodsFor: 'initialization' stamp: 'gera 12/21/2016 02:11:17'!
setUp
	library _ TensorFlowCAPI current! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/7/2017 20:28:44'!
testAsStringGraphRunOn: tensor
	| graph session in out result |
	graph _ self asStringGraphType: tensor type.
	session _ TF_Session onGraph: graph.
	
	in _ graph operationNamed: 'in'.
	out _ graph operationNamed: 'out'.
	
	result _ session
		runOperation: out
		input: (in input: 0)
		value: tensor
		output: (out output: 0).

	graph delete.

	^ result! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/20/2017 18:09:16'!
testAttrGetBoolFalse
	| graph in op |
	graph _ TF_Graph create.
	in _ graph placeholder: 'in' type: TF_Tensor typeDouble.
	op _ graph asString: 'out' described: [:description |
		description addInput: (in output: 0)].
	
	self assert: (op boolAt: 'scientific') equals: false.
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/20/2017 18:09:23'!
testAttrGetBoolTrue
	| graph in op |
	graph _ TF_Graph create.
	in _ graph placeholder: 'in' type: TF_Tensor typeDouble.
	op _ graph newOperation: 'QuantizeAndDequantize' named: 'out' described: [:description |
		description addInput: (in output: 0)].
	
	self assert: (op boolAt: 'signed_input') equals: true.
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/20/2017 18:09:27'!
testAttrGetFloat
	| graph in op |
	graph _ TF_Graph create.
	in _ graph placeholder: 'in' type: TF_Tensor typeFloat.
	op _ graph
		newOperation: 'FakeQuantWithMinMaxArgs'
		named: 'out'
		described: [:description |
			description addInput: (in output: 0)].
	
	self assert: (op floatAt: 'min') equals: -6.0.
	self assert: (op floatAt: 'max') equals: 6.0.
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/13/2017 10:08:33'!
testAttrGetInt
	| op graph |
	graph _ self concatGraphInputList.
	op _ graph operationNamed: 'concat'.
	
	self assert: (op intAt: 'N') equals: 2.
	
	graph delete! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/20/2017 18:09:31'!
testAttrGetShape
	| graph op |
	graph _ TF_Graph create.
	op _ graph placeholder: 'in' type: TF_Tensor typeDouble.
	
	self assert: (op shapeAt: 'shape') equals: #().
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/20/2017 18:09:36'!
testAttrGetString
	| graph in op |
	graph _ TF_Graph create.
	in _ graph placeholder: 'in' type: TF_Tensor typeDouble.
	op _ graph newOperation: 'AsString' named: 'out' described: [:description |
		description addInput: (in output: 0)].
	
	self assert: (op stringAt: 'fill') equals: ''.
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/13/2017 10:08:53'!
testAttrGetTensor
	| op graph tensor |
	graph _ self constantInt64Graph.
	op _ graph operationNamed: 'a'.

	tensor _ op tensorAt: 'value'.
	
	self assert: tensor type equals: TF_Tensor typeInt64.
	self assert: tensor dimensions equals: #().
	self assert: tensor allInt64s equals: #(16r4242424242424242).
	
	graph delete! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/13/2017 10:09:03'!
testAttrGetType
	| op graph |
	graph _ self concatGraphInputList.
	op _ graph operationNamed: 'concat'.
	
	self assert: (op typeAt: 'T') equals: TF_Tensor typeInt64.
	
	graph delete! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/20/2017 18:09:39'!
testAttrSetBoolFalse
	| graph in op |
	graph _ TF_Graph create.
	in _ graph placeholder: 'in' type: TF_Tensor typeDouble.
	op _ graph
		newOperation: 'QuantizeAndDequantize'
		named: 'out'
		described: [:description |
			description at: 'signed_input' putBoolean: false.
			description addInput: (in output: 0)].
	
	self assert: (op boolAt: 'signed_input') equals: false.
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/20/2017 18:09:43'!
testAttrSetBoolTrue
	| graph in op |
	graph _ TF_Graph create.
	in _ graph placeholder: 'in' type: TF_Tensor typeDouble.
	op _ graph asString: 'out' described: [:description |
		description at: 'scientific' putBoolean: true.
		description addInput: (in output: 0)].
	
	self assert: (op boolAt: 'scientific') equals: true.
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/20/2017 18:09:48'!
testAttrSetFloat
	| graph in op min max |
	min _ -1234.5678e10.
	max _ 12345678e-10 asFraction.
	graph _ TF_Graph create.
	in _ graph placeholder: 'in' type: TF_Tensor typeFloat.
	op _ graph
		newOperation: 'FakeQuantWithMinMaxArgs'
		named: 'out'
		described: [:description |
			description at: 'min' putFloat: min.
			description at: 'max' putFloat: max.
			description addInput: (in output: 0)].
	
	self assert: ((op floatAt: 'min') closeTo: min).
	self assert: ((op floatAt: 'max') closeTo: max).
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/11/2017 23:38:13'!
testAttrSetShape
	self testAttrSetShape: #().
	self testAttrSetShape: #(16r1234567890ABCDEF).
	self testAttrSetShape: #(1 2 3 4).
	self testAttrSetShape: (1 to: 200) asArray.! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/20/2017 18:09:53'!
testAttrSetShape: anIntegerArray
	| graph op |
	graph _ TF_Graph create.
	op _ graph
		newOperation: 'Placeholder'
		named: 'const'
		described: [:description |
			description at: 'shape' putShape: anIntegerArray.
			description at: 'dtype' putType: TF_Tensor typeInt64].
		
	self assert: (op shapeAt: 'shape') equals: anIntegerArray.
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/10/2017 20:59:12'!
testAttrSetString
	self testAttrSetString: '1'.
	self testAttrSetString: '12'.
	self testAttrSetString: '1234'.
	self testAttrSetString: '1234567'.
	self testAttrSetString: '12345678'.
	self testAttrSetString: '123456789'.
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/20/2017 18:09:57'!
testAttrSetString: aString
	| graph in op |
	graph _ TF_Graph create.
	in _ graph placeholder: 'in' type: TF_Tensor typeDouble.
	op _ graph newOperation: 'AsString' named: 'out' described: [:description |
		description at: 'fill' putString: aString.
		description addInput: (in output: 0)].
	
	Smalltalk garbageCollect.	" Force releasing the string, to see if the API makes a copy or not"
	
	self assert: (op stringAt: 'fill') equals: aString.
! !

!TensorFlowCAPITest methodsFor: 'testing buffer' stamp: 'gera 1/13/2017 10:10:20'!
testBufferDataBytes
	| buffer string data |
	string _ ' hola manola'.
	buffer _ TF_Buffer fromString: string.

	data _ buffer dataBytes.
	self assert: string equals: data asString.
	
	buffer delete.
! !

!TensorFlowCAPITest methodsFor: 'testing buffer' stamp: 'gera 1/13/2017 10:10:24'!
testBufferNoNeedExternalize
	| buffer string data |
	string _ ' hola manola'.
	buffer _ TF_Buffer fromString: string.
	
	string _ string copy.
	Smalltalk garbageCollect.
	
	data _ buffer dataBytes.
	self assert: string equals: data asString.
	
	buffer delete.
! !

!TensorFlowCAPITest methodsFor: 'testing session' stamp: 'gera 1/7/2017 20:29:13'!
testCloseSession
	| graph session |
	graph _ TF_Graph create.
	session _ TF_Session onGraph: graph.
	self deny: session isNull.
	session ignoreFinalization.
	self shouldnt: [session close] raise: Error.
	self shouldnt: [session delete] raise: Error.
	self assert: session isNull.! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/2/2017 15:40:33'!
testConcatGraphInputList
	self concatGraphInputListNoSizeNoType.
	self concatGraphInputList.
	self should: [self concatGraphInputListWrongSize] raise: Error.
	self should: [self concatGraphInputListWrongType] raise: Error.
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/5/2017 23:56:22'!
testCreateGraphAddTwoInputs
	| graph |
	graph _ self addGraphTwoInputsInt64.
	graph delete.! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/13/2017 10:10:32'!
testCreateGraphMulOneInput
	| graph input mul |
	graph _ self mulGraphOneInputInt64.
	
	self shouldnt: [input _ graph operationNamed: 'in'] raise: Error.
	self shouldnt: [mul _ graph operationNamed: 'mul'] raise: Error.
	
	self assert: input name equals: 'in'.
	self assert: mul name equals: 'mul'.

	graph delete.
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/13/2017 01:29:13'!
testDecodeCSVGraphCreate
	| defaults |
	defaults _ {
		TF_Tensor fromInt64Array: #(-1).
		TF_Tensor fromInt64Array: #(-1).
		TF_Tensor fromInt64Array: #(-1).
		TF_Tensor fromInt64Array: #(-1)}.
		
	self decodeCSVGraphDefaults: defaults.! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/13/2017 04:12:40'!
testDecodeCSVGraphRunCSV: csvLines
	| defaults graph output records session results values |
	defaults _ {
		TF_Tensor fromInt64Array: #(-1).
		TF_Tensor fromInt64Array: #(-1).
		TF_Tensor fromInt64Array: #(-1).
		TF_Tensor fromInt64Array: #(-1)}.
		
	graph _ self decodeCSVGraphDefaults: defaults.
	records _ (graph operationNamed: 'records') input: 0.
	output _ graph operationNamed: 'output'.
	values _ TF_Tensor fromStringArray: csvLines.
	
	session _ TF_Session onGraph: graph.
	results _ session
		runOperations: {output}
		inputs: {records}
		values: {values}
		outputs: {
			(output output: 0).
			(output output: 1).
			(output output: 2).
			(output output: 3)}.
		
	graph delete.
	
	^ (1 to: 4) collect: [:i |
		(results at: i) allInt64s].
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/13/2017 10:06:38'!
testDecodeCSVGraphRunManyLines
	| cols |
	cols _ self testDecodeCSVGraphRunCSV:
	'1,2,3,4
	 11,22,33,44
	 111,222,333,444
	 1111,2222,3333,4444' lines.
	
	
	self assert: cols first equals: #(1 11 111 1111).
	self assert: cols second equals: #(2 22 222 2222).
	self assert: cols third equals: #(3 33 333 3333).
	self assert: cols fourth equals: #(4 44 444 4444).
	! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/13/2017 10:10:47'!
testDecodeCSVGraphRunOneLine
	| cols |
	cols _ self testDecodeCSVGraphRunCSV: '11111111111,22222222,33333333,44444444' lines.
	
	self assert: cols first equals: #(11111111111).
	self assert: cols second equals: #(22222222).
	self assert: cols third equals: #(33333333).
	self assert: cols fourth equals: #(44444444).
	! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 1/13/2017 19:56:50'!
testDimensions
	self assertDimensionsOf: -13123213 is: #().
	self assertDimensionsOf: #(123 123 123 123) is: #(4).
	self assertDimensionsOf: #((1 2 3) (4 5 6) (7 8 9) (10 11 12)) is: #(4 3).
	self assertDimensionsOf: #(((1) (2) (3)) ((4) (5) (6)) ((7) (8) (9))) is: #(3 3 1).! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 1/17/2017 18:58:04'!
testElementsOf: aTensor sum: aNumber
	| sum |
	sum _ 0.
	TF_Tensor elementsOf: aTensor do: [:each |
		sum _ sum + each].
	
	self assert: sum equals: aNumber! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 1/17/2017 18:58:20'!
testElementsOfTensorDoIteratesAll
	self testElementsOf: -13123213 sum: -13123213.
	self testElementsOf: #(123 123 123 123) sum: 123*4.
	self testElementsOf: #((1 2 3) (4 5 6) (7 8 9) (10 11 12)) sum: 12*13/2.
	self testElementsOf: #(((1) (2) (3)) ((4) (5) (6)) ((7) (8) (9))) sum: 9*10/2.
	self
		testElementsOf: #(((1 100) (2 100) (3 100)) ((4 100) (5 100) (6 100)) ((7 100) (8 100) (9 100)))
		sum: 9*10/2+(100*9).! !

!TensorFlowCAPITest methodsFor: 'testing options' stamp: 'gera 1/13/2017 10:10:56'!
testExternalizeString
	| original copy |
	original _ 'hola manola'.
	copy _ library externalizeString: original.
	original withIndexDo: [:each :index |
		self assert: each asciiValue equals: (copy byteAt: index)].
	self assert: (copy byteAt: original size + 1) equals: 0.
	copy free.! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/6/2017 10:43:49'!
testFloatAsStringGraphCreate
	self floatAsStringGraph
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/13/2017 10:11:07'!
testFloatAsStringGraphRun
	| result str expected |
	
	expected _ '0.420000'.
	result _ self runFloatAsStringGraph.
	
	self deny: result isNull.
	self deny: result data isNull.
	str _ result dataBytes.
	
	self assert: 8+1+ expected size equals: str size.
	self assert: (str unsignedLongLongAt: 1) equals: 0.
	self assert: (str at: 9) equals: expected size.
	self assert: (str copyFrom: 10 count: expected size) asString equals: expected.
	
	result delete.
! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 1/18/2017 15:19:54'!
testGet2x2FloatFromGraphDef
	| templates consts |
	templates _ #(-1.1 -2.1 -1.2 -2.2).
	consts _ self get2x2FloatFromGraphDef allFloats.
	
	templates with: consts do: [:temp :const |
		self assert: (temp closeTo: const)]! !

!TensorFlowCAPITest methodsFor: 'testing library' stamp: 'gera 1/3/2017 09:37:31'!
testGetAllOps
	| ops |
	ops _ library getAllOps.
	self assert: (ops data fromCString includesSubString: 'tensor').
	ops delete! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/20/2017 18:10:06'!
testGetAttrMetadataBoolean
	| graph in op template metadata |
	template _ '1234567890abc'.
	graph _ TF_Graph create.
	in _ graph placeholder: 'in' type: TF_Tensor typeDouble.
	op _ graph newOperation: 'AsString' named: 'out' described: [:description |
		description at: 'fill' putString: template.
		description addInput: (in output: 0)].
		
	metadata _ op attrMetadata: 'scientific'.
	self assert: metadata isBoolean.
	self assert: metadata isList equals: false.
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/20/2017 18:10:12'!
testGetAttrMetadataFloat
	| graph in op metadata |
	graph _ TF_Graph create.
	in _ graph placeholder: 'in' type: TF_Tensor typeFloat.
	op _ graph
		newOperation: 'FakeQuantWithMinMaxArgs'
		named: 'out'
		described: [:description |
			description addInput: (in output: 0)].
	
	metadata _ op attrMetadata: 'min'.
	self assert: metadata isFloat.
	self assert: metadata isList equals: false.! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/20/2017 18:10:17'!
testGetAttrMetadataInt
	| graph in op template metadata |
	template _ '1234567890abc'.
	graph _ TF_Graph create.
	in _ graph placeholder: 'in' type: TF_Tensor typeDouble.
	op _ graph newOperation: 'AsString' named: 'out' described: [:description |
		description at: 'fill' putString: template.
		description addInput: (in output: 0)].
		
	metadata _ op attrMetadata: 'precision'.
	self assert: metadata isInt.
	self assert: metadata isList equals: false.
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/20/2017 18:10:22'!
testGetAttrMetadataShape
	| graph op template metadata |
	template _ #(1 2 3 4 5).
	graph _ TF_Graph create.
	op _ graph 
		newOperation: 'Placeholder'
		named: 'const'
		described: [:description |
			description at: 'shape' putShape: template.
			description at: 'dtype' putType: TF_Tensor typeInt64].
		
	metadata _ op attrMetadata: 'shape'.
	self assert: metadata isShape.
	self assert: metadata isList equals: false.
	self assert: metadata totalSize equals: template size.
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/20/2017 18:10:27'!
testGetAttrMetadataString
	| graph in op template metadata |
	template _ '1234567890abc'.
	graph _ TF_Graph create.
	in _ graph placeholder: 'in' type: TF_Tensor typeDouble.
	op _ graph newOperation: 'AsString' named: 'out' described: [:description |
		description at: 'fill' putString: template.
		description addInput: (in output: 0)].
		
	self assert: (op stringAt: 'fill') equals: template.
	
	metadata _ op attrMetadata: 'fill'.
	self assert: metadata isString.
	self assert: metadata isList equals: false.
	self assert: metadata totalSize equals: template size.
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/20/2017 18:10:33'!
testGetAttrMetadataTensor
	| graph op template metadata |
	template _ #(1 2 3 4 5).
	graph _ TF_Graph create.
	op _ graph const: 'const' value: (TF_Tensor fromInt64Array: template).
		
	metadata _ op attrMetadata: 'value'.
	self assert: metadata isTensor.
	self assert: metadata isList equals: false.
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/20/2017 18:10:37'!
testGetAttrMetadataType
	| graph op template metadata |
	template _ #(1 2 3 4 5).
	graph _ TF_Graph create.
	op _ graph 
		newOperation: 'Placeholder'
		named: 'const'
		described: [:description |
			description at: 'shape' putShape: template.
			description at: 'dtype' putType: TF_Tensor typeInt64].
		
	metadata _ op attrMetadata: 'dtype'.
	self assert: metadata isType.
	self assert: metadata isList equals: false.
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/13/2017 10:12:06'!
testGetOperationOnConstantGraph
	| graph op |
	graph _ self constantFloatGraphFromDef.
	self shouldnt: [op _ graph operationNamed: 'a'] raise: Error.
	self assert: op name equals: 'a'.
	self assert: op type equals: 'Const'.
	self assert: op InputsCount equals: 0.
	self assert: op outputsCount equals: 1.! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 12/22/2016 03:44:19'!
testGetOperationOnEmptyGraph
	| graph |
	graph _ self emptyGraph.
	self should: [graph operationNamed: 'something'] raise: Error! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/13/2017 10:12:12'!
testGraph: aTF_Graph outputType: anInteger
	| operation output |
	operation _ aTF_Graph operationNamed: 'a'.
	output _ operation output: 0.
	self assert: output type equals: anInteger.
	output free.! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/13/2017 10:12:23'!
testGraphCreationConst
	| graph operation |
	graph _ self constantInt64Graph.
	
	operation _ graph operationNamed: 'a'.
	self assert: operation type equals: 'Const'.
	self assert: operation name equals: 'a'.
	self assert: operation InputsCount equals: 0.
	self assert: operation outputsCount equals: 1.
! !

!TensorFlowCAPITest methodsFor: 'testing graph' stamp: 'gera 1/13/2017 10:12:31'!
testGraphDefinition
	| definition operations |
	definition _ self mulGraphTwoInputsInt64 definition.
	operations _ (TF_Graph fromString: definition) allVariables.
		
	self assert: operations size equals: 2.
	self assert: operations first equals: 'in1'.
	self assert: operations second equals: 'in2'.
! !

!TensorFlowCAPITest methodsFor: 'testing graph' stamp: 'gera 1/13/2017 10:12:48'!
testGraphDeletionDoesntBreakOperations
	| graph in1 in2 add |
	graph _ self addGraphTwoInputsInt64.
	graph ignoreFinalization.
	
	in1 _ graph operationNamed: 'in1'.
	in2 _ graph operationNamed: 'in2'.
	add _ graph operationNamed: 'add'.

	self assert: in1 name equals: 'in1'.
	self assert: in2 name equals: 'in2'.
	self assert: add name equals: 'add'.
	
	graph delete.
	
	self assert: false description: 'This test actually fails, and by failing it corrupts external memory and leads to a crash. This means when a TF_Graph is deleted, all the TF_Operations composing it are also deleted, hence pointers held to them (from Smalltalk or otherwise) become invalid'.
	
	self assert: in1 name equals: 'in1'.
	self assert: in2 name equals: 'in2'.
	self assert: add name equals: 'add'.
	
	graph _ self mulGraphTwoInputsInt64.
	graph ignoreFinalization.
	graph delete.

	self assert: in1 name equals: 'in1'.
	self assert: in2 name equals: 'in2'.
	self assert: add name equals: 'add'.! !

!TensorFlowCAPITest methodsFor: 'testing graph' stamp: 'gera 1/9/2017 19:10:33'!
testGraphDeletionDoesntBreakSessions
	| graph inputs inputValues add output session results |
	graph _ self addGraphTwoInputsInt64.
	graph ignoreFinalization.
	
	inputs _ Array
		with: ((graph operationNamed: 'in1') input: 0)
		with: ((graph operationNamed: 'in2') input: 0).
	inputValues _ Array
		with: (TF_Tensor fromInt64: 16r2021222021222021)
		with: (TF_Tensor fromInt64: 16r2221202221202221).
	add _ graph operationNamed: 'add'.
	output _ add output: 0.
	session _ TF_Session onGraph: graph.
	graph delete.

	graph _ self addGraphTwoInputsInt64.
	graph ignoreFinalization.
	graph delete.
		
	results _ session
		runOperations: (Array with: add)
		inputs: inputs
		values: inputValues
		outputs: (Array with: output).
	
	self deny: results isNull.
	self deny: results first isNull.
	self deny: results first data isNull.
	self assert: (results first data getHandle signedLongLongAt: 1) = 16r4242424242424242.

	results first delete.
! !

!TensorFlowCAPITest methodsFor: 'testing graph' stamp: 'gera 1/7/2017 21:00:14'!
testGraphFinalizationReleasesExternalMemory
	" WeakArray restartFinalizationProcess "
	| handles tries |
	tries _ 10.
	handles _ (1 to: tries) collect: [:i |
		Smalltalk garbageCollect.
		TF_Graph create useFinalization getHandle].

	self assert: (handles asSet size) < tries.
! !

!TensorFlowCAPITest methodsFor: 'testing graph' stamp: 'gera 1/20/2017 18:04:24'!
testGraphNotFinalizedWhenHeldByOperations
	| graph in1 in2 add |
	graph _ self addGraphTwoInputsInt64.
	graph useFinalization.
	
	in1 _ graph operationNamed: 'in1'.
	in2 _ graph operationNamed: 'in2'.
	add _ graph operationNamed: 'add'.

	self assert: in1 name equals: 'in1'.
	self assert: in2 name equals: 'in2'.
	self assert: add name equals: 'add'.
		
	graph _ nil.
	Smalltalk garbageCollect.
	
	self assert: in1 name equals: 'in1'.
	self assert: in2 name equals: 'in2'.
	self assert: add name equals: 'add'.
	
	graph _ self mulGraphTwoInputsInt64.
	graph delete.

	self assert: in1 name equals: 'in1'.
	self assert: in2 name equals: 'in2'.
	self assert: add name equals: 'add'.! !

!TensorFlowCAPITest methodsFor: 'testing graph' stamp: 'gera 1/13/2017 10:13:07'!
testGraphOperationAt
	| graph operation context |
	graph _ self mulGraphTwoInputsInt64.
	context _ graph newOperationIteratorContext.
	
	operation _ graph operationAt: context.
	self assert: operation name equals: 'in1'.
	self assert: operation type equals: 'Placeholder'.

	operation _ graph operationAt: context.
	self assert: operation name equals: 'in2'.
	self assert: operation type equals: 'Placeholder'.

	operation _ graph operationAt: context.
	self assert: operation name equals: 'const'.
	self assert: operation type equals: 'Const'.

	operation _ graph operationAt: context.
	self assert: operation name equals: 'mul1'.
	self assert: operation type equals: 'Mul'.

	operation _ graph operationAt: context.
	self assert: operation name equals: 'mul2'.
	self assert: operation type equals: 'Mul'.

	operation _ graph operationAt: context.
	self assert: operation isNull.
! !

!TensorFlowCAPITest methodsFor: 'testing graph' stamp: 'gera 1/19/2017 11:11:05'!
testGraphOperationsCount
	| graph |
	graph _ self mulGraphTwoInputsInt64.
	
	self assert: graph operationsCount equals: 5.! !

!TensorFlowCAPITest methodsFor: 'testing graph' stamp: 'gera 1/13/2017 10:13:21'!
testGraphOperationsDo
	| graph operations |
	graph _ self mulGraphTwoInputsInt64.
	operations _ OrderedCollection new.
	
	graph operationsDo: [:op |
		operations add: op name].
	
	self assert: operations size equals: 5.
	self assert: operations first equals: 'in1'.
	self assert: operations second equals: 'in2'.
	self assert: operations third equals: 'const'.
	self assert: operations fourth equals: 'mul1'.
	self assert: operations fifth equals: 'mul2'.
! !

!TensorFlowCAPITest methodsFor: 'testing graph' stamp: 'gera 1/13/2017 10:13:32'!
testGraphOperationsSelect
	| operations |
	operations _ self mulGraphTwoInputsInt64 allVariables.
		
	self assert: operations size equals: 2.
	self assert: operations first equals: 'in1'.
	self assert: operations second equals: 'in2'.
! !

!TensorFlowCAPITest methodsFor: 'testing graph' stamp: 'gera 1/13/2017 10:13:43'!
testGraphOperationsSelectEmpty
	| graph operations |
	graph _ self mulGraphTwoInputsInt64.
	
	operations _ graph operationsSelect: [:op | false].
	
	self assert: operations size equals: 0.
! !

!TensorFlowCAPITest methodsFor: 'testing graph' stamp: 'gera 1/20/2017 18:16:44'!
testImportBad
	| graph buffer |
	graph _ TF_Graph create.
	buffer _ TF_Buffer fromString: 'ouch'.
	
	self should: [graph import: buffer] raise: Error.
	buffer delete.! !

!TensorFlowCAPITest methodsFor: 'testing graph' stamp: 'gera 1/20/2017 18:06:41'!
testImportConstantGraph
	self shouldnt: [
		self constantFloatGraphFromDef.
		self constantInt32GraphFromDef.
		self constantInt64GraphFromDef.
	] raise: Error.
	! !

!TensorFlowCAPITest methodsFor: 'testing graph' stamp: 'gera 1/20/2017 18:06:23'!
testImportEmpty
	self shouldnt: [self emptyGraph] raise: Error.
	! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/13/2017 12:23:08'!
testInt64AsStringGraph
	| result tensor |
	tensor _ TF_Tensor fromInt64: 101010101.
	result _ self testAsStringGraphRunOn: tensor.
	
	self assert: result allStrings first equals: '101010101'.

	result delete.! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/13/2017 12:23:33'!
testInt64rrayAsStringGraph
	| result tensor strings template |
	
	template _ #(101010101 -123321 1 2 3 4).
	tensor _ TF_Tensor fromInt64Array: template.
	result _ self testAsStringGraphRunOn: tensor.
	strings _ result allStrings.
	
	strings withIndexDo: [:value :index |
		self assert: value equals: (template at: index) asString].
	
	result delete.! !

!TensorFlowCAPITest methodsFor: 'testing buffer' stamp: 'gera 1/13/2017 10:14:02'!
testNewBufferFromFileNamed
	| buffer string data temporaryFile |
	string _ ' hola manola'.
	temporaryFile _ 'temporaryFile.txt'.
	temporaryFile asFileEntry writeStream nextPutAll: string; close.

	buffer _ TF_Buffer fromFileNamed: temporaryFile.
	temporaryFile asFileEntry delete.
	
	self deny: buffer isNull.
	self assert: buffer length equals: string size.
	
	data _ buffer data fromCString first: string size.
	self assert: string equals: data.
	
	buffer delete.
	self assert: buffer isNull.
! !

!TensorFlowCAPITest methodsFor: 'testing buffer' stamp: 'gera 1/13/2017 10:14:07'!
testNewBufferFromString
	| buffer string data |
	string _ ' hola manola'.
	buffer _ TF_Buffer fromString: string.
	self deny: buffer isNull.
	self assert: buffer length equals: string size.
	
	data _ buffer data fromCString first: string size.
	self assert: string equals: data.
	
	buffer delete.
	self assert: buffer isNull.! !

!TensorFlowCAPITest methodsFor: 'testing graph' stamp: 'gera 12/28/2016 00:09:54'!
testNewGraph
	| graph |
	graph _ TF_Graph create.
	self deny: graph isNull.
	graph delete.
	self assert: graph isNull.! !

!TensorFlowCAPITest methodsFor: 'testing options' stamp: 'gera 12/28/2016 00:11:14'!
testNewImportGraphDefOptions
	| options |
	options _ TF_ImportGraphDefOptions create.
	self deny: options isNull.
	options delete.
	self assert: options isNull.! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 12/28/2016 00:24:17'!
testNewOperationDescription
	| graph description |
	graph _ TF_Graph create.
	description _ graph newOperation: 'Const' named: 'first_operation'.
	self deny: description isNull.
	self should: [description finish] raise: Error description: 'This should have complained of missing attributes'.! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/13/2017 10:14:47'!
testNewOperationMul
	| graph operation a b |
	graph _ TF_Graph create.
	a _ graph placeholder: 'a' type: TF_Tensor typeInt64.
	b _ graph placeholder: 'b' type: TF_Tensor typeInt64.
	
	operation _ graph
		mul: 'aMultiplication'
		described: [:description |
			description addInputFromOutput: 0 of: a.
			description addInputFromOutput: 0 of: b].

	self assert: operation type equals: 'Mul'.
	self assert: operation name equals: 'aMultiplication'.
	self assert: operation InputsCount equals: 2.
	self assert: operation outputsCount equals: 1.
	
	operation _ graph operationNamed: 'aMultiplication'.
	self assert: operation type equals: 'Mul'.
	self assert: operation name equals: 'aMultiplication'.
	self assert: operation InputsCount equals: 2.
	self assert: operation outputsCount equals: 1.
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/13/2017 10:14:59'!
testNewOperationPlaceholder
	| graph operation |
	graph _ TF_Graph create.
	self shouldnt: [
		operation _ graph placeholder: 'aPlaceholder' type: TF_Tensor typeInt64.
	] raise: Error.
	
	self assert: operation type equals: 'Placeholder'.
	self assert: operation name equals: 'aPlaceholder'.
	self assert: operation InputsCount equals: 0.
	self assert: operation outputsCount equals: 1.
	
	operation _ graph operationNamed: 'aPlaceholder'.
	self assert: operation type equals: 'Placeholder'.
	self assert: operation name equals: 'aPlaceholder'.
	self assert: operation InputsCount equals: 0.
	self assert: operation outputsCount equals: 1.
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/5/2017 05:11:32'!
testNewOperationPlaceholderNoType
	| graph |
	graph _ TF_Graph create.
	self
		should: [(graph newOperation: 'Placeholder' named: 'placeholder') finish]
		raise: Error
	! !

!TensorFlowCAPITest methodsFor: 'testing session' stamp: 'gera 1/7/2017 20:29:25'!
testNewSession
	| graph session |
	graph _ TF_Graph create.
	session _ TF_Session onGraph: graph.
	session ignoreFinalization.
	self deny: session isNull.
	session delete.
	self assert: session isNull.! !

!TensorFlowCAPITest methodsFor: 'testing options' stamp: 'gera 12/28/2016 00:12:36'!
testNewSessionOptions
	| options |
	options _ TF_SessionOptions create.
	self deny: options isNull.
	options delete.
	self assert: options isNull.! !

!TensorFlowCAPITest methodsFor: 'testing status' stamp: 'gera 1/13/2017 12:29:09'!
testNewStatus
	| status |
	status _ TF_Status create.
	status ignoreFinalization.
	
	self deny: status isNull.
	status delete.
	self assert: status isNull.! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/6/2017 00:04:22'!
testOperationOutputTypeFloat
	^self testGraph: self constantFloatGraphFromDef outputType: TF_Tensor typeFloat! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/6/2017 00:03:41'!
testOperationOutputTypeInt32
	^self testGraph: self constantInt32GraphFromDef outputType: TF_Tensor typeInt32! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/6/2017 00:05:25'!
testOperationOutputTypeInt64
	^self testGraph: self constantInt64GraphFromDef outputType: TF_Tensor typeInt64! !

!TensorFlowCAPITest methodsFor: 'testing graph' stamp: 'gera 1/20/2017 18:07:50'!
testOutputDims
	| graph operation output |
	graph _ self constantInt64GraphFromDef.
	operation _ graph operationNamed: 'a'.
	output _ operation output: 0.
	self assert: (graph outputDimenssionsCount: output) equals: 0.
	output free.! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/20/2017 18:12:39'!
testPlaceholderType: type
	| graph var session result tensor abs |
	graph _ TF_Graph create.
	tensor _ TF_Tensor type: type dimensions: #().
	var _ graph placeholder: 'var' type: type.
	abs _ graph newOperation: 'Abs' named: 'abs' described: [:description |
		description addInput: (var output: 0)].
	session _ TF_Session onGraph: graph.
	
	result _ session
		runOperation: abs
		input: (var input: 0)
		value: tensor
		output: (abs output: 0).

	result delete.
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/6/2017 17:21:34'!
testPlaceholderTypes
	self testPlaceholderType: TF_Tensor typeInt64.
	self testPlaceholderType: TF_Tensor typeInt32.
	self testPlaceholderType: TF_Tensor typeFloat.
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/13/2017 10:15:18'!
testPrintOn
	| graph printString |
	graph _ self addGraphTwoInputsInt64.

	printString _ (graph operationNamed: 'in1') printString substrings.
	self assert: printString second equals: 'TF_Operation'.
	self assert: printString third equals: '@'.
	self assert: (printString fourth beginsWith: '16r').
	self assert: printString fifth equals: '''Placeholder'''.
	self assert: printString last equals: '''in1'''.

	printString _ (graph operationNamed: 'add') printString substrings.
	self assert: printString second equals: 'TF_Operation'.
	self assert: printString third equals: '@'.
	self assert: (printString fourth beginsWith: '16r').
	self assert: printString fifth equals: '''Add'''.
	self assert: printString last equals: '''add'''.
! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 1/13/2017 19:50:59'!
testRanks
	self assertRankOf: -13123213 is: 0.
	self assertRankOf: #(123 123 123 123) is: 1.
	self assertRankOf: #((1 2 3) (4 5 6) (7 8 9)) is: 2.
	self assertRankOf: #(((1) (2) (3)) ((4) (5) (6)) ((7) (8) (9))) is: 3.! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/13/2017 10:15:32'!
testRunGraphAddTwoInputs
	| graph inputs inputValues add output session results |
	graph _ self addGraphTwoInputsInt64.
	
	inputs _ Array
		with: ((graph operationNamed: 'in1') input: 0)
		with: ((graph operationNamed: 'in2') input: 0).
	inputValues _ Array
		with: (TF_Tensor fromInt64: 16r2021222021222021)
		with: (TF_Tensor fromInt64: 16r2221202221202221).
	add _ graph operationNamed: 'add'.
	output _ add output: 0.
	session _ TF_Session onGraph: graph.
	results _ session
		runOperations: (Array with: add)
		inputs: inputs
		values: inputValues
		outputs: (Array with: output).
	
	self deny: results isNull.
	self deny: results first isNull.
	self deny: results first data isNull.
	self assert: (results first data getHandle signedLongLongAt: 1) equals: 16r4242424242424242.

	graph delete.
	results first delete.
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/6/2017 00:06:22'!
testRunGraphConst
	| graph |
	graph _ self constantInt64GraphFromDef.

	self testSessionRunTargetOutputOnGraph: graph! !

!TensorFlowCAPITest methodsFor: 'testing session' stamp: 'gera 1/13/2017 12:24:10'!
testRunGraphMulOneInput
	| graph input inputValue result mul output session |
	graph _ self mulGraphOneInputInt64.
	
	input _ (graph operationNamed: 'in') input: 0.
	inputValue _ TF_Tensor fromInt64: 11.
	mul _ graph operationNamed: 'mul'.
	output _ mul output: 0.
	session _ TF_Session onGraph: graph.
	result _ session runOperation: mul input: input value: inputValue output: output.
	
	self deny: result isNull.
	self deny: result data isNull.
	self assert: (result data getHandle signedLongLongAt: 1) equals: 16r4242424242424242.

	result delete.
	input free.
	output free.
	graph delete.! !

!TensorFlowCAPITest methodsFor: 'testing session' stamp: 'gera 1/13/2017 10:15:44'!
testRunGraphMulTwoInputs
	| graph inputs inputValues mul output session results |
	graph _ self mulGraphTwoInputsInt64.
	
	inputs _ Array
		with: ((graph operationNamed: 'in1') input: 0)
		with: ((graph operationNamed: 'in2') input: 0).
	inputValues _ Array
		with: (TF_Tensor fromInt64: 6)
		with: (TF_Tensor fromInt64: 11).
	mul _ graph operationNamed: 'mul2'.
	output _ mul output: 0.
	session _ TF_Session onGraph: graph.
	results _ session
		runOperations: (Array with: mul)
		inputs: inputs
		values: inputValues
		outputs: (Array with: output).
	
	self deny: results isNull.
	self deny: results first isNull.
	self deny: results first data isNull.
	self assert: (results first data getHandle signedLongLongAt: 1) equals: 16r4242424242424242.

	graph delete.
	results first delete.
! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 1/13/2017 11:18:12'!
testRunGraphMulTwoInputsConstTensorDeleted
	| graph inputs inputValues mul output session results |
	graph _ self mulGraphTwoInputsInt64ConstTensorDeleted.
	
	inputs _ Array
		with: ((graph operationNamed: 'in1') input: 0)
		with: ((graph operationNamed: 'in2') input: 0).
		
	inputValues _ Array
		with: (TF_Tensor fromInt64: 16r23) useFinalization
		with: (TF_Tensor fromInt64: 16r24) useFinalization.

	Smalltalk garbageCollect.
	(TF_Tensor fromInt64: 16r1234123412341234) useFinalization.
	Smalltalk garbageCollect.
	
	mul _ graph operationNamed: 'mul2'.
	output _ mul output: 0.
	session _ TF_Session onGraph: graph.
	results _ session
		runOperations: (Array with: mul)
		inputs: inputs
		values: inputValues
		outputs: (Array with: output).
	
	self deny: results isNull.
	self deny: results first isNull.
	self deny: results first data isNull.
	self assert: (results first data getHandle signedLongLongAt: 1) equals: 16rF0F0F0F0F0F0E6C.

	graph delete.
	results first delete.
! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 1/13/2017 11:18:24'!
testRunGraphMulTwoInputsConstTensorDeletedManyTimes
	20 timesRepeat: [self testRunGraphMulTwoInputsConstTensorDeleted]! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/7/2017 20:30:34'!
testRunOperationArray
	| graph operation session |
	graph _ self constantFloatGraphFromDef.
	session _ TF_Session onGraph: graph.
	operation _ graph operationNamed: 'a'.
	
	self shouldnt: [session runOperations: (Array with: operation)] raise: Error.

	graph delete.! !

!TensorFlowCAPITest methodsFor: 'testing graph' stamp: 'gera 1/20/2017 18:12:50'!
testSessionDeletionDoesntDeleteGraphs
	| session graph allocatedObjects |
	graph _ TF_Graph create.
	session _ TF_Session onGraph: graph.
	session ignoreFinalization.
	session close.
	session delete.

	" Allocate some external objects using the library, if the graph was released, we expect its space to be reused "
	allocatedObjects _ OrderedCollection new: 10.
	10 timesRepeat: [allocatedObjects add: (TF_Status create)].
	
	self
		shouldnt: [graph placeholder: 'a' type: TF_Tensor typeInt64]
		raise: Error
		description: 'The FFI call would crash if the graph was released by deleting the session'.
! !

!TensorFlowCAPITest methodsFor: 'testing session' stamp: 'gera 1/20/2017 18:12:58'!
testSessionFinalizationReleasesExternalMemory 
	" WeakArray restartFinalizationProcess "
	| tries total handles graph |
	total _ 0.
	tries _ 20.
	graph _ TF_Graph create.
	handles _ Set new.
	
	20 timesRepeat: [
		total _ total + tries.
		handles addAll: ((1 to: tries) collect: [:i |
			Smalltalk garbageCollect.
			(TF_Session onGraph: graph) getHandle]).
		handles size < total ifTrue: [
			^ self]].
	
	self assert: (handles size) < total.
! !

!TensorFlowCAPITest methodsFor: 'testing session' stamp: 'gera 1/7/2017 20:30:46'!
testSessionOnEmptyGraph
	| session |
	session _ TF_Session onGraph: self emptyGraph.
	self should: [session run] raise: Error.
! !

!TensorFlowCAPITest methodsFor: 'testing options' stamp: 'gera 12/28/2016 00:12:49'!
testSessionOptionsSetTarget
	| options |
	options _ library newSessionOptions.
	self deny: options isNull.
	options target: 'local'.
	options delete.
	self assert: options isNull.! !

!TensorFlowCAPITest methodsFor: 'testing session' stamp: 'gera 1/7/2017 20:30:57'!
testSessionRunTarget
	| graph operation session |
	graph _ self constantFloatGraphFromDef.
	session _ TF_Session onGraph: graph.
	operation _ graph operationNamed: 'a'.
	
	self shouldnt: [session runOperation: operation] raise: Error.

	graph delete.! !

!TensorFlowCAPITest methodsFor: 'testing session' stamp: 'gera 1/6/2017 00:06:16'!
testSessionRunTargetOutput
	^ self testSessionRunTargetOutputOnGraph: self constantInt64GraphFromDef.
! !

!TensorFlowCAPITest methodsFor: 'testing session' stamp: 'gera 1/13/2017 12:29:42'!
testSessionRunTargetOutputOnGraph: graph
	| operation session output tensor |
	session _ TF_Session onGraph: graph.
	operation _ graph operationNamed: 'a'.
	output _ operation output: 0.
	
	tensor _ session runOperation: operation output: output.
	
	self deny: tensor isNull.
	self deny: tensor data isNull.
	self assert: (tensor data getHandle signedLongLongAt: 1) equals: 16r4242424242424242.

	output free.
	graph delete.! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 1/18/2017 00:05:31'!
testSizes
	self assertSizeOf: -13123213 is: 1.
	self assertSizeOf: #(123 123 123 123) is: 4.
	self assertSizeOf: #((1 2 3) (4 5 6) (7 8 9) (10 11 12)) is: 4*3.
	self assertSizeOf: #(((1) (2) (3)) ((4) (5) (6)) ((7) (8) (9))) is: 3*3*1.! !

!TensorFlowCAPITest methodsFor: 'testing status' stamp: 'gera 1/13/2017 10:16:01'!
testStatusCodes
	| status  |
	status _ TF_Status create.
	
	self assert: status isOk.
	self assert: status codeText equals: 'OK'.
	self shouldnt: [status check] raise: Error.
	
	status code: 1 message: 'You cancelled it!!'.
	self assert: status codeText equals: 'CANCELLED'.
	self should: [status check] raise: Error.
! !

!TensorFlowCAPITest methodsFor: 'testing status' stamp: 'gera 1/7/2017 20:21:25'!
testStatusFinalizationReleasesExternalMemory
	" WeakArray restartFinalizationProcess "
	| handles |
	handles _ (1 to: 11) collect: [:i |
		Smalltalk garbageCollect.
		TF_Status create getHandle].

	self assert: (handles asSet size) < 11.! !

!TensorFlowCAPITest methodsFor: 'testing status' stamp: 'gera 1/13/2017 10:16:12'!
testStatusGetMessage
	| status message |
	status _ TF_Status create.
	
	status code: 1 message: 'All is one'.
	message _ status message.
	self assert: message equals: 'All is one'.

	status code: 7 message: 'Something is very seven'.
	message _ status message.
	self assert: message equals: 'Something is very seven'.
! !

!TensorFlowCAPITest methodsFor: 'testing status' stamp: 'gera 1/13/2017 10:16:21'!
testStatusSetGetCode
	| status code |
	status _ TF_Status create.
	status code: 1 message: ''.
	code _ status code.
	self assert: code equals: 1.

	status code: 2 message: ''.
	code _ status code.
	self assert: code equals: 2.
! !

!TensorFlowCAPITest methodsFor: 'testing strings' stamp: 'gera 1/13/2017 10:16:37'!
testStringDecode
	| size string encoded decoded status |
	" This test assumes the internal representation of a TensorFlow string. May fail if they change it "
	
	string _ 'a ver como queda este string encodeado?'.
	
	status _ TF_Status create.
	size _ library stringEncodedSize: string.
	encoded _ ExternalAddress gcallocate: size + 20.
	encoded byteAt: size + 1 put: $@ asciiValue.
	library stringEncode: string len: string size destination: encoded len: size + 20 status: status.
	status check.

	decoded _ library stringDecode: encoded.

	self deny: decoded = (string, '@').
	self assert: decoded equals: string.
! !

!TensorFlowCAPITest methodsFor: 'testing strings' stamp: 'gera 1/13/2017 10:16:53'!
testStringEncode
	| size string encoded encodedSize |
	" This test assumes the internal representation of a TensorFlow string. May fail if they change it "
	
	string _ 'a ver como queda este string encodeado?'.
	size _ library stringEncodedSize: string.
	encoded _ ByteArray new: size+20.
	encodedSize _ library stringEncode: string to: encoded.
	
	self assert: encodedSize equals: size.
	self assert: (encoded byteAt: 1) equals: string size.
	self assert: (encoded structAt: 2 length: string size) asString equals: string.
! !

!TensorFlowCAPITest methodsFor: 'testing strings' stamp: 'gera 1/13/2017 10:17:11'!
testStringEncodeExternal
	| size string encoded encodedSize status |
	" This test assumes the internal representation of a TensorFlow string. May fail if they change it "
	
	status _ TF_Status create.
	string _ 'a ver como queda este string encodeado?'.
	
	size _ library stringEncodedSize: string.
	encoded _ ExternalAddress gcallocate: size + 20.
	encodedSize _ library stringEncode: string len: string size destination: encoded len: size + 20 status: status.

	status check.
	self assert: encodedSize equals: size.
	self assert: (encoded unsignedByteAt: 1) equals: string size.
	self assert: (encoded structAt: 2 length: string size) asString equals: string.
! !

!TensorFlowCAPITest methodsFor: 'testing strings' stamp: 'gera 1/13/2017 10:17:31'!
testStringSize
	self assert: (library primStringEncodedSize: 0) equals: (0+1).
	self assert: (library primStringEncodedSize: 127) equals: (127+1).
	self assert: (library primStringEncodedSize: 128) equals: (128+2).
	self assert: (library primStringEncodedSize: 127*127) equals: (127*127+2).
	self assert: (library primStringEncodedSize: 127*127+123) equals: (127*127+123+2).
	self assert: (library primStringEncodedSize: 127*127*127) equals: (127*127*127+3).

	self assert: (library stringEncodedSize: '') equals: (0+1).
	self assert: (library stringEncodedSize: (String new: 127)) equals: (127+1).
	self assert: (library stringEncodedSize: (String new: 128)) equals: (128+2).
! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 1/13/2017 10:17:38'!
testTensorAllStrings
	| result strings expected |
	
	expected _ #('0.420000').
	result _ self runFloatAsStringGraph.
	strings _ result allStrings.
	
	self assert: strings equals: expected.
	
	result delete.
! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 1/13/2017 10:17:42'!
testTensorByteSize
	| tensor |
	tensor _ TF_Tensor
		type: TF_Tensor typeInt64
		dimensions: #(2 3).

	self assert: tensor byteSize equals: (8*2*3)! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 1/13/2017 12:24:45'!
testTensorData
	| tensor |
	tensor _ TF_Tensor
		type: TF_Tensor typeInt64
		dimensions: #(2 3).
	self assert: tensor numberOfDimensions equals: 2.
	self deny: tensor data getHandle asInteger = 0.
	self deny: tensor data getHandle isNil.
! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 12/26/2016 14:44:54'!
testTensorDimensions0D
	^ self testTensorDimensions: #()! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 12/26/2016 14:45:28'!
testTensorDimensions10D
	^ self testTensorDimensions: #(1 2 3 4 5 6 7 8 9 10)! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 12/26/2016 14:44:50'!
testTensorDimensions1D
	^ self testTensorDimensions: #(7)! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 12/26/2016 14:44:40'!
testTensorDimensions2D
	^ self testTensorDimensions: #(1 4)! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 1/13/2017 12:24:56'!
testTensorDimensions: anArray
	| tensor dimensions size |
	tensor _ TF_Tensor
		type: TF_Tensor typeInt64
		dimensions: anArray.
	self assert: tensor numberOfDimensions equals: anArray size.
	anArray withIndexDo: [:each :index |
		self assert: (tensor sizeOn: index - 1) equals: (anArray at: index)].
	dimensions _ tensor dimensions.
	size _ anArray isEmpty
		ifTrue: [1]
		ifFalse: [anArray product].
	self assert: dimensions equals: anArray.
	self assert: tensor size equals: size.
! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 1/20/2017 17:49:12'!
testTensorFinalizationReleasesExternalMemory
	" WeakArray restartFinalizationProcess "
	| handles template tries total |
	total _ 0.
	tries _ 20.
	handles _ Set new.

	template _ {String new: 10. String new: 100. String new: 1000}.
	
	20 timesRepeat: [
		total _ total + tries.
		handles addAll: ((1 to: tries) collect: [:i |
			Smalltalk garbageCollect.
			(TF_Tensor fromStringArray: template) getHandle]).
		handles size < total ifTrue: [
			^ self]
	].

	self assert: (handles size) < total.
! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 1/18/2017 00:36:16'!
testTensorFromFloats
	| tensor template values |
	template _ #(1.23456 0.0 -1.234567).
	tensor _ TF_Tensor fromFloats: template.
	values _ tensor allFloats.
		
	self assert: tensor dimensions equals: #(3).
	self assert: tensor size equals: 3.
	self assert: tensor byteSize equals: (3*4).
	template @ values do: [:point |
		self assert: (point x closeTo: point y)]
! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 1/19/2017 10:13:47'!
testTensorFromFloats2x2
	| tensor template values |
	template _ #(
		(-1.1 -2.1)
		(-1.2 -2.2)).
		
	tensor _ TF_Tensor fromFloats: template.
	values _ tensor allFloats.
		
	self assert: tensor dimensions equals: #(2 2).
	self assert: tensor size equals: 4.
	self assert: tensor byteSize equals: (4*4).
		
	#(-1.1 -2.1 -1.2 -2.2) with: values do: [:reference :value |
		self assert: (reference closeTo: value)]
! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 1/19/2017 10:14:00'!
testTensorFromFloats2x2SameElementsOrder
	| tensor template values constTensor consts |
	template _ #(
		(-1.1 -2.1)
		(-1.2 -2.2)).
		
	tensor _ TF_Tensor fromFloats: template.
	values _ tensor allFloats.
	
	constTensor _ self get2x2FloatFromGraphDef.
	consts _ constTensor allFloats.
	
	consts with: values do: [:const :value |
		self assert: (const closeTo: value)].! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 1/19/2017 11:32:12'!
testTensorFromFloatsScalar
	| tensor template values |
	template _ 3.141516.
	tensor _ TF_Tensor fromFloats: template.
	values _ tensor allFloats.
		
	self assert: tensor dimensions equals: #().
	self assert: tensor size equals: 1.
	self assert: tensor byteSize equals: (1*4).
	
	self assert: (template closeTo: values first).! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 1/13/2017 12:25:10'!
testTensorFromInt64Array
	| tensor template values |
	template _ #(16r1234567812345678 0 -12345678910111213).
	tensor _ TF_Tensor fromInt64Array: template.
	values _ tensor allInt64s.
	
	self assert: tensor dimensions equals: #(3).
	self assert: tensor size equals: 3.
	self assert: tensor byteSize equals: (3*8).
	self assert: template equals: values.
! !

!TensorFlowCAPITest methodsFor: 'testing strings' stamp: 'gera 1/13/2017 10:18:17'!
testTensorFromString
	| tensor template |
	template _ 'hola manola'.
	
	tensor _ TF_Tensor fromString: template.
	
	self assert: tensor dataBytes first equals: template size.
	self assert: tensor dataBytes allButFirst asString equals: template.! !

!TensorFlowCAPITest methodsFor: 'testing strings' stamp: 'gera 1/13/2017 10:18:23'!
testTensorFromStringArray
	| tensor template |
	template _ #('hola manola' 'te traje una lola' 'pamela' 'que pandulce!!').
	
	tensor _ TF_Tensor fromStringArray: template.
	
	self assert: tensor allStrings equals: template.! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 1/13/2017 12:25:32'!
testTensorNewScalar
	| tensor |
	tensor _ TF_Tensor type: TF_Tensor typeInt64 dimensions: #().
	tensor ignoreFinalization.
	
	self deny: tensor isNull.
	tensor delete.
	self assert: tensor isNull.! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 1/13/2017 12:25:40'!
testTensorType
	| tensor |
	tensor _ TF_Tensor type: TF_Tensor typeInt64 dimensions: #().
	self assert: tensor type equals: tensor class typeInt64.

	tensor _ TF_Tensor type: TF_Tensor typeFloat dimensions: #().
	self assert: tensor type equals: tensor class typeFloat.
! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 1/13/2017 10:18:51'!
testTensorTypes
	| types |
	types _ #(
		Float 1
		Double 2
		Int32 3
		UInt8 4
		Int16 5
		Int8 6
		String 7
		Complex64 8
		Int64 9
		Bool 10
		QInt8 11
		QUInt8 12
		QInt32 13
		BFloat16 14
		QInt16 15
		QUInt16 16
		UInt16 17
		Complex128 18
		Half 19
		Resource 20).
	types pairsDo: [:name :value |
		self assert: (TF_Tensor perform: (#type, name) asSymbol) equals: value]! !

!TensorFlowCAPITest methodsFor: 'testing library' stamp: 'gera 1/11/2017 14:12:47'!
testVersion
	| version |
	version _ library version.
	self assert: (#('0.11.head' '0.12.0' '0.12.head' '1.0.0-alpha') includes: version)! !

!TensorFlowOperationsTest methodsFor: 'initialization' stamp: 'gera 1/20/2017 18:23:31'!
setUp
	library _ TensorFlowCAPI current! !

!TensorFlowOperationsTest methodsFor: 'unary operations' stamp: 'gera 1/21/2017 05:21:32'!
testOperationAbs
	| graph template session const result output |
	
	template _ TF_Tensor fromFloats: #((-1.1 1.2) (-2.1 -2.2)).
	graph _ TF_Graph create.
	graph useFinalization.
	
	const _ graph const: 'const' value: template.
	output _ const abs.
	
	session _ TF_Session onGraph: graph.
	result _ session
		runOperation: output
		output: (output output: 0).
	
	self assert: result dimensions equals: template dimensions.
	self assert: result allFloats equals: template allFloats abs.! !

!TensorFlowOperationsTest methodsFor: 'unary operations' stamp: 'gera 1/20/2017 18:24:47'!
testOperationAcos
	| graph template session const result output intput |
	
	template _ #(1.1 0.2 2.34 0.717273).
	intput _ TF_Tensor fromFloats: template cos.
	graph _ TF_Graph create.
	
	const _ graph const: 'const' value: intput.
	output _ const arcCos.
	
	session _ TF_Session onGraph: graph.
	result _ session
		runOperation: output
		output: (output output: 0).
	
	self assert: result dimensions equals: intput dimensions.
	result allFloats with: template do: [:res :temp |
		self assert: (res closeTo: temp)].! !

!TensorFlowOperationsTest methodsFor: 'binary operations' stamp: 'gera 1/21/2017 04:04:39'!
testOperationArgMin
	| graph a session const result output0 output1 axis0 axis1 |
	
	a _ TF_Tensor fromFloats: #(
		(1.1 1.2)
		(2.1 2.2)
		(3.1 0.2)
	).
	
	graph _ TF_Graph create.
	
	const _ graph const: 'a' value: a.
	axis0 _ graph const: 'axis0' value: (TF_Tensor fromInt32: 0).
	axis1 _ graph const: 'axis1' value: (TF_Tensor fromInt32: 1).
	
	output0 _ const findMinOn: axis0.
	output1 _ const findMinOn: axis1.
	
	session _ TF_Session onGraph: graph.
	result _ session
		runOperation: output0
		output: (output0 output: 0).
	
	self assert: result dimensions equals: #(2).
	result allInt64s with: #(0 2) do: [:r :t |
		self assert: r equals: t].
	
	result _ session
		runOperation: output1
		output: (output1 output: 0).
	
	self assert: result dimensions equals: #(3).
	result allInt64s with: #(0 0 1) do: [:r :t |
		self assert: r equals: t].! !

!TensorFlowOperationsTest methodsFor: 'unary operations' stamp: 'gera 1/20/2017 18:24:52'!
testOperationAsin
	| graph template session const result output intput |
	
	template _ #(1.1 0.2 -1.34 0.717273).
	intput _ TF_Tensor fromFloats: template sin.
	graph _ TF_Graph create.
	
	const _ graph const: 'const' value: intput.
	output _ const arcSin.
	
	session _ TF_Session onGraph: graph.
	result _ session
		runOperation: output
		output: (output output: 0).
	
	self assert: result dimensions equals: intput dimensions.
	result allFloats with: template do: [:res :temp |
		self assert: (res closeTo: temp)].! !

!TensorFlowOperationsTest methodsFor: 'unary operations' stamp: 'gera 1/20/2017 18:25:00'!
testOperationAtan
	| graph template session const result output intput |
	
	template _ #(1.1 0.2 -1.34 0.717273).
	intput _ TF_Tensor fromFloats: template tan.
	graph _ TF_Graph create.
	
	const _ graph const: 'const' value: intput.
	output _ const arcTan.
	
	session _ TF_Session onGraph: graph.
	result _ session
		runOperation: output
		output: (output output: 0).
	
	self assert: result dimensions equals: intput dimensions.
	result allFloats with: template do: [:res :temp |
		self assert: (res closeTo: temp)].! !

!TensorFlowOperationsTest methodsFor: 'unary operations' stamp: 'gera 1/20/2017 18:25:08'!
testOperationCos
	| graph template session const result output intput |

	template _ #(0.1 0.0 -0.94 0.717273).
	intput _ TF_Tensor fromFloats: template arcCos.
	graph _ TF_Graph create.
	
	const _ graph const: 'const' value: intput.
	output _ const cos.
	
	session _ TF_Session onGraph: graph.
	result _ session
		runOperation: output
		output: (output output: 0).
	
	self assert: result dimensions equals: intput dimensions.
	result allFloats with: template do: [:res :temp |
		self assert: (res closeTo: temp)].! !

!TensorFlowOperationsTest methodsFor: 'binary operations' stamp: 'gera 1/21/2017 01:38:01'!
testOperationMatMul
	| graph a b session constA constB result output template |
	
	a _ TF_Tensor fromFloats: #((1.1 1.2) (2.1 2.2)).
	b _ TF_Tensor fromFloats: #((2.0 4.0) (8.0 16.0)).
	
	graph _ TF_Graph create.
	
	constA _ graph const: 'a' value: a.
	constB _ graph const: 'b' value: b.
	
	output _ constA * constB.
	template _ {
		1.1*2.0+(1.2*8.0).		1.1*4.0+(1.2*16.0).
		2.1*2.0+(2.2*8.0).		2.1*4.0+(2.2*16.0).
	}.

	session _ TF_Session onGraph: graph.
	result _ session
		runOperation: output
		output: (output output: 0).
	
	self assert: result dimensions equals: a dimensions.
	result allFloats with: template do: [:r :t |
		self assert: (r closeTo: t)].! !

!TensorFlowOperationsTest methodsFor: 'binary operations' stamp: 'gera 1/21/2017 02:28:13'!
testOperationMatMulATransposed
	| graph a b session constA constB result output template |
	
	a _ TF_Tensor fromFloats: #((1.1 1.2) (2.1 2.2)).
	b _ TF_Tensor fromFloats: #((2.0 4.0) (8.0 16.0)).
	
	graph _ TF_Graph create.
	
	constA _ graph const: 'a' value: a.
	constB _ graph const: 'b' value: b.
	
	output _ constA \* constB.
	template _ {
		1.1*2.0+(2.1*8.0).		1.1*4.0+(2.1*16.0).
		1.2*2.0+(2.2*8.0).		1.2*4.0+(2.2*16.0).
	}.

	session _ TF_Session onGraph: graph.
	result _ session
		runOperation: output
		output: (output output: 0).
	
	self assert: result dimensions equals: a dimensions.
	result allFloats with: template do: [:r :t |
		self assert: (r closeTo: t)].! !

!TensorFlowOperationsTest methodsFor: 'binary operations' stamp: 'gera 1/21/2017 02:27:34'!
testOperationMatMulBTransposed
	| graph a b session constA constB result output template |
	
	a _ TF_Tensor fromFloats: #((1.1 1.2) (2.1 2.2)).
	b _ TF_Tensor fromFloats: #((2.0 4.0) (8.0 16.0)).
	
	graph _ TF_Graph create.
	
	constA _ graph const: 'a' value: a.
	constB _ graph const: 'b' value: b.
	
	output _ constA *\ constB.
	template _ {
		1.1*2.0+(1.2*4.0).		1.1*8.0+(1.2*16.0).
		2.1*2.0+(2.2*4.0).		2.1*8.0+(2.2*16.0).
	}.

	session _ TF_Session onGraph: graph.
	result _ session
		runOperation: output
		output: (output output: 0).
	
	self assert: result dimensions equals: a dimensions.
	result allFloats with: template do: [:r :t |
		self assert: (r closeTo: t)].! !

!TensorFlowOperationsTest methodsFor: 'binary operations' stamp: 'gera 1/20/2017 18:25:38'!
testOperationMod
	| graph a b session constA constB result output template |
	
	a _ TF_Tensor fromFloats: #((10.0 11.0) (12.0 13.0)).
	b _ TF_Tensor fromFloats: #((3.0 2.0) (7.0 5.0)).
	
	graph _ TF_Graph create.
	
	constA _ graph const: 'a' value: a.
	constB _ graph const: 'b' value: b.
	
	output _ constA \\ constB.
	template _ a allFloats \\ b allFloats.
	
	session _ TF_Session onGraph: graph.
	result _ session
		runOperation: output
		output: (output output: 0).
	
	self assert: result dimensions equals: a dimensions.
	result allFloats with: template do: [:r :t |
		self assert: (r closeTo: t)].! !

!TensorFlowOperationsTest methodsFor: 'binary operations' stamp: 'gera 1/20/2017 18:25:43'!
testOperationMul
	| graph a b session constA constB result output template |
	
	a _ TF_Tensor fromFloats: #((1.1 1.2) (2.1 2.2)).
	b _ TF_Tensor fromFloats: #((2.0 4.0) (8.0 16.0)).
	
	graph _ TF_Graph create.
	
	constA _ graph const: 'a' value: a.
	constB _ graph const: 'b' value: b.
	
	output _ constA @* constB.
	template _ a allFloats * b allFloats.
	
	session _ TF_Session onGraph: graph.
	result _ session
		runOperation: output
		output: (output output: 0).
	
	self assert: result dimensions equals: a dimensions.
	result allFloats with: template do: [:r :t |
		self assert: (r closeTo: t)].! !

!TensorFlowOperationsTest methodsFor: 'binary operations' stamp: 'gera 1/20/2017 18:25:49'!
testOperationMulScalar
	| graph a b session constA constB result sum template |
	
	a _ TF_Tensor fromFloats: #((1.1 1.2) (2.1 2.2)).
	b _ TF_Tensor fromFloats: 2.0.
	
	graph _ TF_Graph create.
	
	constA _ graph const: 'a' value: a.
	constB _ graph const: 'b' value: b.
	
	sum _ constA @* constB.
	template _ a allFloats * 2.0.
	
	session _ TF_Session onGraph: graph.
	result _ session
		runOperation: sum
		output: (sum output: 0).
	
	self assert: result dimensions equals: a dimensions.
	result allFloats with: template do: [:r :t |
		self assert: (r closeTo: t)].! !

!TensorFlowOperationsTest methodsFor: 'unary operations' stamp: 'gera 1/20/2017 18:25:56'!
testOperationNegated
	| graph template session const result negated |
	
	template _ TF_Tensor fromFloats: #((1.1 1.2) (2.1 2.2)).
	graph _ TF_Graph create.
	
	const _ graph const: 'const' value: template.
	negated _ const negated.
	
	session _ TF_Session onGraph: graph.
	result _ session
		runOperation: negated
		output: (negated output: 0).
	
	self assert: result dimensions equals: template dimensions.
	self assert: result allFloats equals: template allFloats negated.! !

!TensorFlowOperationsTest methodsFor: 'binary operations' stamp: 'gera 1/20/2017 18:26:01'!
testOperationPlus
	| graph a b session constA constB result sum template |
	
	a _ TF_Tensor fromFloats: #((1.1 1.2) (2.1 2.2)).
	b _ TF_Tensor fromFloats: #((3.14 3.14) (3.14 3.14)).
	
	graph _ TF_Graph create.
	
	constA _ graph const: 'a' value: a.
	constB _ graph const: 'b' value: b.
	
	sum _ constA + constB.
	template _ a allFloats + b allFloats.
	
	session _ TF_Session onGraph: graph.
	result _ session
		runOperation: sum
		output: (sum output: 0).
	
	self assert: result dimensions equals: a dimensions.
	result allFloats with: template do: [:r :t |
		self assert: (r closeTo: t)].! !

!TensorFlowOperationsTest methodsFor: 'binary operations' stamp: 'gera 1/20/2017 18:26:09'!
testOperationPlusNegated
	| graph a b session constA constB result sum template |
	
	a _ TF_Tensor fromFloats: #((1.1 1.2) (2.1 2.2)).
	b _ TF_Tensor fromFloats: #((3.14 3.14) (3.14 3.14)).
	
	graph _ TF_Graph create.
	
	constA _ graph const: 'a' value: a.
	constB _ graph const: 'b' value: b.
	
	sum _ constA + constB negated.
	
	session _ TF_Session onGraph: graph.
	result _ session
		runOperation: sum
		output: (sum output: 0).
	
	template _ a allFloats - b allFloats.

	self assert: result dimensions equals: a dimensions.
	result allFloats with: template do: [:r :t |
		self assert: (r closeTo: t)].! !

!TensorFlowOperationsTest methodsFor: 'binary operations' stamp: 'gera 1/20/2017 18:26:13'!
testOperationPlusNegatedNegated
	| graph a b session constA constB result sum template |
	
	a _ TF_Tensor fromFloats: #((1.1 1.2) (2.1 2.2)).
	b _ TF_Tensor fromFloats: #((3.14 3.14) (3.14 3.14)).
	
	graph _ TF_Graph create.
	
	constA _ graph const: 'a' value: a.
	constB _ graph const: 'b' value: b.
	
	sum _ (constA + constB negated) negated.
	template _ (a allFloats + b allFloats negated) negated.
	
	session _ TF_Session onGraph: graph.
	result _ session
		runOperation: sum
		output: (sum output: 0).
	
	self assert: result dimensions equals: a dimensions.
	result allFloats with: template do: [:r :t |
		self assert: (r closeTo: t)].! !

!TensorFlowOperationsTest methodsFor: 'binary operations' stamp: 'gera 1/20/2017 18:26:17'!
testOperationPlusNegatedNegatedPlus
	| graph a b session constA constB result sum template |
	
	a _ TF_Tensor fromFloats: #((1.1 1.2) (2.1 2.2)).
	b _ TF_Tensor fromFloats: #((3.14 3.14) (3.14 3.14)).
	
	graph _ TF_Graph create.
	
	constA _ graph const: 'a' value: a.
	constB _ graph const: 'b' value: b.
	
	sum _ (constA + constB negated) negated + constB negated.
	template _ (a allFloats + b allFloats negated) negated + b allFloats negated.
	
	session _ TF_Session onGraph: graph.
	result _ session
		runOperation: sum
		output: (sum output: 0).
	
	self assert: result dimensions equals: a dimensions.
	
	result allFloats with: a allFloats negated do: [:r :t |
		self assert: (r closeTo: t)].
	
	result allFloats with: template do: [:r :t |
		self assert: (r closeTo: t)].! !

!TensorFlowOperationsTest methodsFor: 'unary operations' stamp: 'gera 1/20/2017 18:26:22'!
testOperationSin
	| graph template session const result output intput |

	template _ #(0.1 0.0 -0.94 0.717273).
	intput _ TF_Tensor fromFloats: template arcSin.
	graph _ TF_Graph create.
	
	const _ graph const: 'const' value: intput.
	output _ const sin.
	
	session _ TF_Session onGraph: graph.
	result _ session
		runOperation: output
		output: (output output: 0).
	
	self assert: result dimensions equals: intput dimensions.
	result allFloats with: template do: [:res :temp |
		self assert: (res closeTo: temp)].! !

!TensorFlowOperationsTest methodsFor: 'binary operations' stamp: 'gera 1/21/2017 04:08:09'!
testOperationSum
	| graph a session const result output0 output1 axis0 axis1 template sumOn1 sumOn0 |
	
	template _ #(
		(1.1 1.2)
		(2.1 2.2)
		(3.1 0.2)
	).
	
	sumOn0 _ template sum.
	sumOn1 _ template collect: [:line | line sum].
	
	a _ TF_Tensor fromFloats: template.
	
	graph _ TF_Graph create.
	
	const _ graph const: 'a' value: a.
	axis0 _ graph const: 'axis0' value: (TF_Tensor fromInt32: 0).
	axis1 _ graph const: 'axis1' value: (TF_Tensor fromInt32: 1).
	
	output0 _ const sumOn: axis0.
	output1 _ const sumOn: axis1.
	
	session _ TF_Session onGraph: graph.
	result _ session
		runOperation: output0
		output: (output0 output: 0).
	
	self assert: result dimensions equals: #(2).
	result allFloats with: sumOn0 do: [:r :t |
		self assert: (r closeTo: t)].
	
	result _ session
		runOperation: output1
		output: (output1 output: 0).
	
	self assert: result dimensions equals: #(3).
	result allFloats with: sumOn1 do: [:r :t |
		self assert: (r closeTo: t)].! !

!TensorFlowOperationsTest methodsFor: 'unary operations' stamp: 'gera 1/20/2017 18:26:27'!
testOperationTan
	| graph template session const result output intput |

	template _ #(0.1 0.0 -0.94 0.717273).
	intput _ TF_Tensor fromFloats: template arcTan.
	graph _ TF_Graph create.
	
	const _ graph const: 'const' value: intput.
	output _ const tan.
	
	session _ TF_Session onGraph: graph.
	result _ session
		runOperation: output
		output: (output output: 0).
	
	self assert: result dimensions equals: intput dimensions.
	result allFloats with: template do: [:res :temp |
		self assert: (res closeTo: temp)].! !

!TensorFlowExamplesTest methodsFor: 'initialization' stamp: 'jb 1/16/2017 22:05:43'!
setUp
	nn _ NearestNeighbor new.! !

!TensorFlowExamplesTest methodsFor: 'testing-NearesNeighbor' stamp: 'JB 1/18/2017 23:03:42'!
tensorFrom: points
	| dimensions transposed |
	dimensions _ points first size.
	transposed _ OrderedCollection new.
	(1 to: dimensions) do: [ :i |
		transposed add: OrderedCollection new ].
	points do: [ :point |
		point withIndexDo: [ :value :coordinate |
			(transposed at: coordinate) add: value ]].
	^ TF_Tensor fromFloats: transposed.! !

!TensorFlowExamplesTest methodsFor: 'testing-NearesNeighbor' stamp: 'gera 1/21/2017 04:32:58'!
testNearesNeighbor
	| first second third closest references tensorReference tensors plan predictor |
	plan _ ExampleNearestNeighborPlan  new.
	first _ #(0 0 0 0 0).
	second _ #(1 1 3 4 2).
	third _ #(8 1 3 4 2).
	references _{
		first.
		second.
		third}.
	tensorReference _ self tensorFrom: references.
	tensors _ references collect: [ :point |
		self tensorFrom: {point} ].
	
	predictor _ [:unknown | | result | 
		result _ plan predict: unknown from: tensorReference.
		result first].
	
	closest _ predictor value: tensors first.
	self assert: closest equals: 1.
	
	closest _ predictor value:  tensors second.
	self assert: closest equals: 2.
	
	closest _ predictor value:  tensors third.
	self assert: closest equals: 3.! !

!TensorFlowExamplesTest methodsFor: 'testing-ols' stamp: 'gera 1/21/2017 04:28:10'!
testOLS
	|ols x y betas |
	ols _ OLSExample new.
	x _ TF_Tensor fromFloats: #(
		(1  2   3)
		(6  7   8)
		(4  5   6)
		(9  2   3)
		(1 10  2)
	).

	y _ TF_Tensor fromFloats: #(
		(14)
		(44)
		(32)
		(22)
		(27)
	).
	
	betas _ ols regress: x on: y.
	self assert: (betas first closeTo: 1);
		assert: (betas second closeTo: 2);
		assert: (betas third closeTo: 3)
	
		
! !

!TensorFlowCAPI methodsFor: 'tensor' stamp: 'gera 12/27/2016 11:56:22'!
allocateTensorType: anInteger dimensions: aLonnLongArray dimensionsCount: dimCount length: len
	<cdecl: TF_Tensor* 'TF_AllocateTensor' (ulong UInt64Array* ulonglong ulonglong)>
	^ self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'session' stamp: 'gera 12/21/2016 23:26:55'!
closeSession: aTF_Session status: aTF_Status
	<cdecl: void 'TF_CloseSession' (TF_Session* TF_Status*)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'buffer' stamp: 'gera 12/22/2016 00:38:51'!
deleteBuffer: aTF_Buffer
	<cdecl: void 'TF_DeleteBuffer' (TF_Buffer*)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'graph' stamp: 'gera 12/21/2016 22:32:51'!
deleteGraph: aTF_Graph
	<cdecl: void 'TF_DeleteGraph' (TF_Graph*)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'options' stamp: 'gera 12/22/2016 01:16:37'!
deleteImportGraphDefOptions: aTF_ImportGraphDefOptions
	<cdecl: void 'TF_DeleteImportGraphDefOptions' (TF_ImportGraphDefOptions*)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'session' stamp: 'gera 12/21/2016 23:23:33'!
deleteSession: aTF_Session status: aTF_Status
	<cdecl: void 'TF_DeleteSession' (TF_Session* TF_Status*)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'options' stamp: 'GerardoRicharte 12/18/2016 21:41'!
deleteSessionOptions: aTF_SessionOptions
	<cdecl: void 'TF_DeleteSessionOptions' (TF_SessionOptions*)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'status' stamp: 'gera 12/21/2016 02:57:15'!
deleteStatus: aTF_Status
	<cdecl: void 'TF_DeleteStatus' (TF_Status*)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'tensor' stamp: 'gera 12/23/2016 20:54:17'!
deleteTensor: aTF_Tensor
	<cdecl: void 'TF_DeleteTensor' (TF_Tensor*)>
	^ self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'operation description' stamp: 'gera 12/28/2016 01:59:08'!
description: aTF_OperationDescription addInput: aTF_Output
	<cdecl: void 'TF_AddInput' (TF_OperationDescription* TF_Output)>
	^ self externalCallFailed.! !

!TensorFlowCAPI methodsFor: 'operation description' stamp: 'gera 1/2/2017 12:36:00'!
description: aTF_OperationDescription addInputs: aTF_OutputArray size: anInteger
	<cdecl: void 'TF_AddInputList' (TF_OperationDescription* TF_OutputArray* longlong)>
	^ self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'operation description' stamp: 'gera 1/9/2017 19:53:21'!
description: aTF_OperationDescription set: aString toBool: anInt
	<cdecl: void 'TF_SetAttrBool' (TF_OperationDescription* char* byte)>
	^ self externalCallFailed.! !

!TensorFlowCAPI methodsFor: 'operation description' stamp: 'gera 1/10/2017 10:32:25'!
description: aTF_OperationDescription set: aString toFloat: valueFloat
	<cdecl: void 'TF_SetAttrFloat' (TF_OperationDescription* char* float)>
	^ self externalCallFailed.! !

!TensorFlowCAPI methodsFor: 'operation description' stamp: 'gera 12/30/2016 12:52:47'!
description: aTF_OperationDescription set: aString toInt64: anInt
	<cdecl: void 'TF_SetAttrInt' (TF_OperationDescription* char* ulonglong)>
	^ self externalCallFailed.! !

!TensorFlowCAPI methodsFor: 'operation description' stamp: 'gera 1/10/2017 20:51:14'!
description: aTF_OperationDescription set: aString toShape: anInt64Array size: size
	<cdecl: void 'TF_SetAttrShape' (TF_OperationDescription* char* Int64Array* ulonglong)>
	^ self externalCallFailed.! !

!TensorFlowCAPI methodsFor: 'operation description' stamp: 'gera 1/10/2017 08:45:47'!
description: aTF_OperationDescription set: aString toString: anInt size: size
	<cdecl: void 'TF_SetAttrString' (TF_OperationDescription* char* char* ulonglong)>
	^ self externalCallFailed.! !

!TensorFlowCAPI methodsFor: 'operation description' stamp: 'gera 12/28/2016 01:02:26'!
description: aTF_OperationDescription set: aString toTensor: aTF_Tensor status: aTF_Status
	<cdecl: void 'TF_SetAttrTensor' (TF_OperationDescription* char* TF_Tensor* TF_Status*)>
	^ self externalCallFailed.! !

!TensorFlowCAPI methodsFor: 'operation description' stamp: 'gera 12/28/2016 00:44:16'!
description: aTF_OperationDescription set: aString toType: anInt
	<cdecl: void 'TF_SetAttrType' (TF_OperationDescription* char* ulonglong)>
	^ self externalCallFailed.! !

!TensorFlowCAPI methodsFor: 'utils' stamp: 'gera 12/21/2016 09:56:40'!
externalizeString: aString
	| answer |
	answer := ExternalAddress allocate: aString size + 1.
	answer byteAt: aString size + 1 put: 0.
	aString withIndexDo: [:char :index |
		answer byteAt:  index put: char asciiValue].
	^ answer! !

!TensorFlowCAPI methodsFor: 'operation description' stamp: 'gera 1/19/2017 23:32:25'!
finishOperation: aTF_OperationDescription status: aTF_Status
	| answer |
	answer _ self finishOperationAsVoid: aTF_OperationDescription status: aTF_Status.
	^ TF_Operation fromHandle: answer getHandle.! !

!TensorFlowCAPI methodsFor: 'operation description' stamp: 'gera 1/19/2017 10:33:08'!
finishOperationAsVoid: aTF_OperationDescription status: aTF_Status
	<cdecl: void* 'TF_FinishOperation' (TF_OperationDescription* TF_Status*)>
	^ self externalCallFailed.! !

!TensorFlowCAPI methodsFor: 'operation' stamp: 'gera 12/26/2016 13:13:55'!
forGraph: aTF_Graph outputDims: aTF_Output status: aTF_Status
	<cdecl: ulonglong 'TF_GraphGetTensorNumDims' (TF_Graph* TF_Output TF_Status*)>
	^ self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'utils' stamp: 'gera 12/29/2016 23:03:34'!
getAllOps
	<cdecl: TF_Buffer* 'TF_GetAllOpList' (void)>
	^ self externalCallFailed.! !

!TensorFlowCAPI methodsFor: 'status' stamp: 'gera 12/21/2016 09:59:30'!
getCode: aTF_Status
	<cdecl: ulong 'TF_GetCode' (TF_Status*)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'graph' stamp: 'gera 1/19/2017 23:27:15'!
graph: aTF_Graph getOperationNamed: aString
	| answer |
	answer _ self graph: aTF_Graph getOperationNamedAsVoid: aString.
	answer _ TF_Operation fromHandle: answer getHandle.
	answer graph: aTF_Graph.
	^ answer! !

!TensorFlowCAPI methodsFor: 'graph' stamp: 'gera 1/19/2017 23:12:26'!
graph: aTF_Graph getOperationNamedAsVoid: aString
	<cdecl: void* 'TF_GraphOperationByName' (TF_Graph* char*)>
	^ self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'operation' stamp: 'gera 1/19/2017 23:29:16'!
graph: aTF_Graph operationAt: contextULongLongPtr
	| answer |
	answer _ self graph: aTF_Graph operationAtAsVoid: contextULongLongPtr.
	answer _ TF_Operation fromHandle: answer getHandle.
	answer graph: aTF_Graph.
	^ answer! !

!TensorFlowCAPI methodsFor: 'operation' stamp: 'gera 1/19/2017 23:28:23'!
graph: aTF_Graph operationAtAsVoid: contextULongLongPtr
	<cdecl: void* 'TF_GraphNextOperation' (TF_Graph* ulonglong*)>
	^ self externalCallFailed
! !

!TensorFlowCAPI methodsFor: 'graph' stamp: 'gera 1/5/2017 20:47:43'!
graph: aTF_Graph toGraphDef: aTF_Buffer status: aTF_Status
	<cdecl: void 'TF_GraphToGraphDef' (TF_Graph* TF_Buffer* TF_Status*)>
	^ self externalCallFailed
! !

!TensorFlowCAPI methodsFor: 'graph' stamp: 'gera 12/22/2016 01:22:22'!
importGraphDefInto: aTF_Graph from: aTF_Buffer options: aTF_ImportGraphDefOptions status: aTF_Status
	<cdecl: void 'TF_GraphImportGraphDef' (TF_Graph* TF_Buffer* TF_ImportGraphDefOptions* TF_Status*)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'status' stamp: 'gera 12/21/2016 10:14:23'!
message: aTF_Status
	<cdecl: char* 'TF_Message' (TF_Status*)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'buffer' stamp: 'gera 12/22/2016 00:38:18'!
newBufferFromString: aString size: anInteger
	<cdecl: TF_Buffer* 'TF_NewBufferFromString' (char* ulong)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'graph' stamp: 'gera 12/21/2016 22:36:38'!
newGraph
	<cdecl: TF_Graph* 'TF_NewGraph' (void)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'options' stamp: 'gera 12/22/2016 01:17:45'!
newImportGraphDefOptions
	<cdecl: TF_ImportGraphDefOptions* 'TF_NewImportGraphDefOptions' (void)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'operation' stamp: 'gera 12/27/2016 23:15:02'!
newOperationOn: graph type: type named: name
	<cdecl: TF_OperationDescription* 'TF_NewOperation' (TF_Graph* char* char*)>
	^ self externalCallFailed
	! !

!TensorFlowCAPI methodsFor: 'session' stamp: 'gera 12/21/2016 23:21:09'!
newSessionGraph: aTF_Graph options:  aTF_SessionOptions status: aTF_Status
	<cdecl: TF_Session* 'TF_NewSession' (TF_Graph* TF_SessionOptions* TF_Status*)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'options' stamp: 'GerardoRicharte 12/18/2016 21:35'!
newSessionOptions
	<cdecl: TF_SessionOptions* 'TF_NewSessionOptions' (void)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'status' stamp: 'gera 12/21/2016 02:56:25'!
newStatus
	<cdecl: TF_Status* 'TF_NewStatus' (void)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'operation' stamp: 'gera 1/9/2017 19:45:05'!
operation: aTF_Operation attr: nameZString getBool: valueBoolPtr status: status
	<cdecl: void 'TF_OperationGetAttrBool' (TF_Operation* char* ulonglong* TF_Status*)>
	^ self externalCallFailed

! !

!TensorFlowCAPI methodsFor: 'operation' stamp: 'gera 1/10/2017 10:21:45'!
operation: aTF_Operation attr: nameZString getFloat: valueFloatPtr status: status
	<cdecl: void 'TF_OperationGetAttrFloat' (TF_Operation* char* float* TF_Status*)>
	^ self externalCallFailed

! !

!TensorFlowCAPI methodsFor: 'operation' stamp: 'gera 1/7/2017 04:24:06'!
operation: aTF_Operation attr: nameZString getInt64: valueLongPtr status: status
	<cdecl: void 'TF_OperationGetAttrInt' (TF_Operation* char* ulonglong* TF_Status*)>
	^ self externalCallFailed

! !

!TensorFlowCAPI methodsFor: 'operation' stamp: 'gera 1/10/2017 16:39:17'!
operation: aTF_Operation attr: nameZString getShape: int64array size: maxSize status: status
	<cdecl: void 'TF_OperationGetAttrShape' (TF_Operation* char* Int64Array* ulonglong TF_Status*)>
	^ self externalCallFailed

! !

!TensorFlowCAPI methodsFor: 'operation' stamp: 'gera 1/10/2017 09:00:54'!
operation: aTF_Operation attr: nameZString getString: valueString size: maxSize status: status
	<cdecl: void 'TF_OperationGetAttrString' (TF_Operation* char* void* ulonglong TF_Status*)>
	^ self externalCallFailed

! !

!TensorFlowCAPI methodsFor: 'operation' stamp: 'gera 1/7/2017 04:59:08'!
operation: aTF_Operation attr: nameZString getTensor: valueLongPtr status: status
	<cdecl: void 'TF_OperationGetAttrTensor' (TF_Operation* char* void* TF_Status*)>
	^ self externalCallFailed

! !

!TensorFlowCAPI methodsFor: 'operation' stamp: 'gera 1/7/2017 04:54:15'!
operation: aTF_Operation attr: nameZString getType: valueLongPtr status: status
	<cdecl: void 'TF_OperationGetAttrType' (TF_Operation* char* ulonglong* TF_Status*)>
	^ self externalCallFailed

! !

!TensorFlowCAPI methodsFor: 'operation' stamp: 'gera 1/11/2017 22:34:01'!
operation: aTF_Operation getMetadataFor: nameZString status: status
	<cdecl: TF_AttrMetadata 'TF_OperationGetAttrMetadata' (TF_Operation* char* TF_Status*)>
	^ self externalCallFailed

! !

!TensorFlowCAPI methodsFor: 'operation' stamp: 'gera 12/22/2016 03:54:06'!
operationDevice: aTF_Operation
	<cdecl: char* 'TF_OperationDevice' (TF_Operation*)>
	^ self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'operation' stamp: 'gera 12/22/2016 03:53:36'!
operationName: aTF_Operation
	<cdecl: char* 'TF_OperationName' (TF_Operation*)>
	^ self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'operation' stamp: 'gera 12/22/2016 03:55:39'!
operationNumInputs: aTF_Operation
	<cdecl: long 'TF_OperationNumInputs' (TF_Operation*)>
	^ self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'operation' stamp: 'gera 12/22/2016 03:54:41'!
operationNumOutputs: aTF_Operation
	<cdecl: long 'TF_OperationNumOutputs' (TF_Operation*)>
	^ self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'operation' stamp: 'gera 12/22/2016 03:53:56'!
operationOpType: aTF_Operation
	<cdecl: char* 'TF_OperationOpType' (TF_Operation*)>
	^ self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'operation' stamp: 'gera 12/23/2016 10:45:09'!
operationOutputType: aTF_Output
	<cdecl: ulonglong 'TF_OperationOutputType' (TF_Output)>
	^ self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'strings' stamp: 'gera 1/3/2017 09:43:38'!
primStringEncodedSize: anInteger
	<cdecl: ulonglong 'TF_StringEncodedSize' (ulonglong)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'session' stamp: 'gera 12/29/2016 18:19:01'!
runSession: aTF_Session
	options: opsTF_Buffer
	inputs: inTF_OutputArray values: inTF_TensorArray count: inCount
	outputs: outTF_OutputArray values: outTF_TensorArrayPtr count: outCount
	targets: aTF_OperationArray count: targetCount
	metadata: metaTF_Buffer
	status: aTF_Status
	
	<cdecl: void 'TF_SessionRun' (
		TF_Session*
		TF_Buffer*				"RunOptions, may be nil"
		TF_OutputArray*			"inputs"
		TF_TensorPtrArray*		"inputs' values"
		long						"inputs count"
		TF_OutputArray*			"outputs"
		TF_TensorPtrArray*		"outputs values (results)"
		long						"outputs count"
		TF_OperationPtrArray*	"target operations array, may be nil"
		long						"taret operations count"
		TF_Buffer*				"RunMetadata"
		TF_Status*)>
	^ self externalCallFailed
! !

!TensorFlowCAPI methodsFor: 'options' stamp: 'gera 12/21/2016 23:47:01'!
sessionOptions: aTF_SessionOptions setConfig: aString configSize: anInteger status: aTF_Status
	" not sure how to use this. Best information found in http://devdocs.io/tensorflow~python/contrib.learn#RunConfig"
	<cdecl: void 'TF_SetConfig' (TF_SessionOptions* void* ulong TF_Status*)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'options' stamp: 'GerardoRicharte 12/19/2016 10:58'!
sessionOptions: aTF_SessionOptions setTarget: aString
	<cdecl: void 'TF_SetTarget' (TF_SessionOptions* char*)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'status' stamp: 'gera 12/21/2016 09:54:03'!
setStatus: aTF_Status code: anInteger message: anExternalString
	<cdecl: void 'TF_SetStatus' (TF_Status* ulong char*)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'strings' stamp: 'gera 1/5/2017 03:14:02'!
stringDecode: src
	| destination status answer dstSize |
	destination _ ByteArray new: 8.
	dstSize _ ByteArray new: 8.
	status _ TF_Status create.
	answer _ self stringDecode: src len: src size destination: destination len: dstSize status: status.
	status check.
	dstSize _ dstSize unsignedLongLongAt: 1.
	destination _ destination pointerAt: 1.
	answer _ (destination structAt: 1 length: dstSize) asString.
	^ answer! !

!TensorFlowCAPI methodsFor: 'strings' stamp: 'gera 1/5/2017 03:11:10'!
stringDecode: srcString len: srcLen destination: dstPointer len: dstLenPointer status: status
	<cdecl: ulonglong 'TF_StringDecode' (char* ulonglong char* ulonglong* TF_Status)>
	^ self externalCallFailed
! !

!TensorFlowCAPI methodsFor: 'strings' stamp: 'gera 1/4/2017 19:02:15'!
stringEncode: srcString len: srcLen destination: dstString len: dstLen status: status
	<cdecl: ulonglong 'TF_StringEncode' (char* ulonglong char* ulonglong TF_Status)>
	^ self externalCallFailed
! !

!TensorFlowCAPI methodsFor: 'strings' stamp: 'gera 1/13/2017 01:50:35'!
stringEncode: src to: dst
	^ self stringEncode: src to: dst size: dst size

! !

!TensorFlowCAPI methodsFor: 'strings' stamp: 'gera 1/13/2017 01:50:12'!
stringEncode: src to: dst size: dstSize
	| status answer |
	status _ TF_Status create.
	answer _ self stringEncode: src len: src size destination: dst len: dstSize status: status.
	status check.
	^ answer
! !

!TensorFlowCAPI methodsFor: 'strings' stamp: 'gera 1/3/2017 09:40:22'!
stringEncodedSize: aString
	^self primStringEncodedSize: aString size! !

!TensorFlowCAPI methodsFor: 'tensor' stamp: 'jb 12/26/2016 12:15:05'!
tensor: aTF_Tensor sizeOn: dimension
	<cdecl: longlong 'TF_Dim' (TF_Tensor* ulonglong)>
	^ self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'tensor' stamp: 'gera 1/5/2017 23:36:32'!
tensorByteSize: aTF_Tensor
	<cdecl: ulonglong 'TF_TensorByteSize' (TF_Tensor*)>
	^ self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'tensor' stamp: 'gera 12/27/2016 09:17:49'!
tensorData: aTF_Tensor
	<cdecl: void* 'TF_TensorData' (TF_Tensor*)>
	^ self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'tensor' stamp: 'jb 12/26/2016 12:04:40'!
tensorNumDimensions: aTF_Tensor
	<cdecl: longlong 'TF_NumDims' (TF_Tensor*)>
	^ self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'tensor' stamp: 'gera 12/23/2016 20:54:40'!
tensorType: aTF_Tensor
	<cdecl: ulonglong 'TF_TensorType' (TF_Tensor*)>
	^ self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'utils' stamp: 'jb 12/26/2016 11:48:09'!
version
	<cdecl: char* 'TF_Version' (void)>
	^ self externalCallFailed.! !

!TensorFlowCAPI class methodsFor: 'as yet unclassified' stamp: 'jb 12/26/2016 10:19:17'!
current
	^ current ifNil: [current _ self new forceLoading].
! !

!TensorFlowCAPI class methodsFor: 'as yet unclassified' stamp: 'GerardoRicharte 12/18/2016 12:18'!
moduleName
	^ 'libtensorflow'! !

!TF_Structure methodsFor: 'testing' stamp: 'gera 12/25/2016 19:14:16'!
= aTF_Structure
	^ handle = self getHandle! !

!TF_Structure methodsFor: 'private' stamp: 'gera 12/21/2016 02:52:42'!
library
	^ TensorFlowCAPI current! !

!TF_Structure methodsFor: 'printing' stamp: 'gera 1/3/2017 10:26:05'!
printOn: strm
	super printOn: strm.
	strm space; print: handle.! !

!TF_Structure class methodsFor: 'field definition' stamp: 'gera 12/22/2016 00:24:05'!
fields
	"self initialize"
	^#()! !

!TF_Structure class methodsFor: 'class initialization' stamp: 'gera 1/19/2017 19:06:52'!
initialize
	super initialize.
	self == TF_Structure ifTrue: [
		self allSubclassesDo: [:cls | cls initialize].
		self allSubclassesDo: [:cls | cls initialize].
	]! !

!TF_Structure class methodsFor: 'field definition' stamp: 'gera 12/22/2016 00:33:15'!
pointerSize
	" Fixed for 64 bit platforms. TensorFlow is only available in 64 bit platforms "
	self assert: 8 == Smalltalk wordSize.
	^ Smalltalk wordSize! !

!ExternalWordArray methodsFor: 'accessing' stamp: 'gera 12/25/2016 13:45:27'!
at: index
	"This method was automatically generated. See PointerArray class>>fields."
	^ self elementAtOffset: index  - 1 * self class byteSize + 1
	! !

!ExternalWordArray methodsFor: 'accessing' stamp: 'gera 12/25/2016 14:10:50'!
at: index put: anExternalStructure
	"This method was automatically generated. See PointerArray class>>fields."
	^ self elementAtOffset: index  - 1 * self class byteSize + 1 put: anExternalStructure
	! !

!ExternalWordArray methodsFor: 'private' stamp: 'gera 12/25/2016 13:44:02'!
elementAtOffset: byteOffset
	^self elementClass fromHandle: (handle longPointerAt: byteOffset)
	! !

!ExternalWordArray methodsFor: 'private' stamp: 'gera 12/25/2016 14:14:46'!
elementAtOffset: byteOffset put: anExternalStructure
	handle longPointerAt: byteOffset put: anExternalStructure getHandle.! !

!ExternalWordArray methodsFor: 'accessing' stamp: 'gera 12/25/2016 13:09:17'!
elementClass
	^self subclassResponsibility! !

!ExternalWordArray methodsFor: 'accessing' stamp: 'gera 12/23/2016 11:10:52'!
first
	^ self at: 1! !

!ExternalWordArray methodsFor: 'accessing' stamp: 'gera 12/23/2016 11:11:29'!
first: anExternalStructure
	self at: 1 put: anExternalStructure
! !

!ExternalWordArray methodsFor: 'accessing' stamp: 'gera 12/23/2016 11:10:15'!
template
	"This method was automatically generated. See PointerArray class>>fields."
	^ExternalData fromHandle: (handle longPointerAt: 1) type: ExternalType void asPointerType! !

!ExternalWordArray methodsFor: 'accessing' stamp: 'gera 12/23/2016 11:10:15'!
template: anObject
	"This method was automatically generated. See PointerArray class>>fields."
	handle longPointerAt: 1 put: anObject getHandle.! !

!ExternalWordArray class methodsFor: 'instance creation' stamp: 'gera 12/29/2016 18:35:06'!
externalFromArray: anArray
	| answer |
	answer _ self externalNew: anArray size.
	anArray withIndexDo: [:each :index |
		answer at: index put: each].
	^ answer! !

!ExternalWordArray class methodsFor: 'instance creation' stamp: 'gera 12/23/2016 11:15:00'!
externalNew
	^ self shouldNotImplement! !

!ExternalWordArray class methodsFor: 'instance creation' stamp: 'gera 1/13/2017 13:11:46'!
externalNew: count
	| address answer |
	address _ ExternalAddress gcallocate: self byteSize * count.
	answer _ self fromHandle: address.
	^ answer! !

!ExternalWordArray class methodsFor: 'field definition' stamp: 'gera 12/23/2016 11:10:13'!
fields
	"self initialize."
	^#((template 'void*'))! !

!ExternalWordArray class methodsFor: 'instance creation' stamp: 'gera 12/23/2016 11:14:45'!
new
	^ self shouldNotImplement! !

!ExternalWordArray class methodsFor: 'instance creation' stamp: 'gera 12/22/2016 20:34:52'!
new: count
	^self fromHandle: (ByteArray new: self byteSize * count)! !

!Int64Array methodsFor: 'accessing' stamp: 'gera 12/27/2016 18:11:32'!
elementAtOffset: offset
	^handle signedLongLongAt: offset
	! !

!Int64Array methodsFor: 'accessing' stamp: 'gera 12/27/2016 18:11:36'!
elementAtOffset: offset put: anInteger
	handle signedLongLongAt: offset put: anInteger! !

!Int64Array class methodsFor: 'instance creation' stamp: 'jb 12/26/2016 09:43:53'!
byteSize
	^ 8! !

!TF_OperationPtrArray methodsFor: 'as yet unclassified' stamp: 'gera 12/25/2016 03:50:55'!
elementClass
	^TF_Operation! !

!TF_OutputArray methodsFor: 'accessing' stamp: 'gera 12/29/2016 09:44:00'!
elementAtOffset: byteOffset
	^self elementClass fromHandle: (handle + (byteOffset - 1))
	
! !

!TF_OutputArray methodsFor: 'accessing' stamp: 'gera 12/29/2016 09:51:07'!
elementAtOffset: byteOffset put: anExternalStructure
	handle structAt: byteOffset put: anExternalStructure getHandle length: 16.! !

!TF_OutputArray methodsFor: 'accessing' stamp: 'gera 12/25/2016 03:50:48'!
elementClass
	^TF_Output! !

!TF_OutputArray methodsFor: 'accessing' stamp: 'gera 12/29/2016 14:09:24'!
template
	"This method was automatically generated. See TF_OutputArray class>>fields."
	^TF_Output fromHandle: (handle structAt: 1 length: 16)! !

!TF_OutputArray methodsFor: 'accessing' stamp: 'gera 12/29/2016 14:09:24'!
template: anObject
	"This method was automatically generated. See TF_OutputArray class>>fields."
	handle structAt: 1 put: anObject getHandle length: 16.! !

!TF_OutputArray class methodsFor: 'as yet unclassified' stamp: 'gera 12/29/2016 09:39:03'!
fields
	"self initialize"
	^#((template 'TF_Output'))! !

!TF_OutputArray class methodsFor: 'as yet unclassified' stamp: 'gera 12/29/2016 09:39:40'!
new: unused
	^ self shouldNotImplement! !

!TF_TensorPtrArray methodsFor: 'as yet unclassified' stamp: 'gera 12/27/2016 09:09:07'!
elementClass
	^TF_Tensor! !

!TF_AttrMetadata methodsFor: 'types' stamp: 'gera 1/11/2017 23:12:07'!
isBoolean
	^ self type = 3! !

!TF_AttrMetadata methodsFor: 'types' stamp: 'gera 1/11/2017 23:11:55'!
isFloat
	^ self type = 2! !

!TF_AttrMetadata methodsFor: 'types' stamp: 'gera 1/11/2017 23:15:35'!
isFunc
	^ self type = 8! !

!TF_AttrMetadata methodsFor: 'types' stamp: 'gera 1/11/2017 23:11:31'!
isInt
	^ self type = 1! !

!TF_AttrMetadata methodsFor: 'accessing' stamp: 'gera 1/11/2017 22:41:49'!
isList
	"This method was automatically generated. See TF_AttrMetadata class>>fields."
	^self _isList \\ 256 ~= 0.! !

!TF_AttrMetadata methodsFor: 'types' stamp: 'gera 1/11/2017 23:12:54'!
isPlaceholder
	^ self type = 7! !

!TF_AttrMetadata methodsFor: 'types' stamp: 'gera 1/11/2017 23:12:27'!
isShape
	^ self type = 5! !

!TF_AttrMetadata methodsFor: 'types' stamp: 'gera 1/11/2017 23:11:09'!
isString
	^ self type = 0
! !

!TF_AttrMetadata methodsFor: 'types' stamp: 'gera 1/11/2017 23:12:41'!
isTensor
	^ self type = 6! !

!TF_AttrMetadata methodsFor: 'types' stamp: 'gera 1/11/2017 23:12:17'!
isType
	^ self type = 4! !

!TF_AttrMetadata methodsFor: 'accessing' stamp: 'gera 1/11/2017 22:43:36'!
listSize
	"This method was automatically generated. See TF_AttrMetadata class>>fields."
	^handle signedLongLongAt: 9! !

!TF_AttrMetadata methodsFor: 'accessing' stamp: 'gera 1/11/2017 22:43:40'!
totalSize
	"This method was automatically generated. See TF_AttrMetadata class>>fields."
	^handle signedLongLongAt: 25! !

!TF_AttrMetadata methodsFor: 'accessing' stamp: 'gera 1/11/2017 22:42:04'!
type
	"This method was automatically generated. See TF_AttrMetadata class>>fields."
	^self _type \\ 256! !

!TF_AttrMetadata methodsFor: 'accessing' stamp: 'gera 1/11/2017 22:40:36'!
_isList
	"This method was automatically generated. See TF_AttrMetadata class>>fields."
	^handle unsignedLongLongAt: 1! !

!TF_AttrMetadata methodsFor: 'accessing' stamp: 'gera 1/11/2017 22:40:36'!
_type
	"This method was automatically generated. See TF_AttrMetadata class>>fields."
	^handle unsignedLongLongAt: 17! !

!TF_AttrMetadata class methodsFor: 'field definition' stamp: 'gera 1/11/2017 22:43:56'!
fields
	"self initialize"
	^#(
		(_isList 'ulonglong')
		(listSize 'longlong')
		(_type 'ulonglong')
		(totalSize 'longlong'))
	! !

!TF_Buffer methodsFor: 'accessing' stamp: 'JB 1/18/2017 23:01:10'!
data
	"This method was automatically generated. See TF_Buffer class>>fields."
	^ExternalData fromHandle: (handle longPointerAt: 1) type: ExternalType void asPointerType! !

!TF_Buffer methodsFor: 'accessing' stamp: 'gera 12/22/2016 00:33:21'!
data: anObject
	"This method was automatically generated. See TF_Buffer class>>fields."
	handle longPointerAt: 1 put: anObject getHandle.! !

!TF_Buffer methodsFor: 'accessing' stamp: 'gera 1/5/2017 20:43:10'!
dataBytes
	^ self data getHandle structAt: 1 length: self length! !

!TF_Buffer methodsFor: 'accessing' stamp: 'gera 12/22/2016 00:34:10'!
deallocator
	"This method was automatically generated. See TF_Buffer class>>fields."
	^ExternalData fromHandle: (handle longPointerAt: 17) type: ExternalType void asPointerType! !

!TF_Buffer methodsFor: 'accessing' stamp: 'gera 12/22/2016 00:34:10'!
deallocator: anObject
	"This method was automatically generated. See TF_Buffer class>>fields."
	handle longPointerAt: 17 put: anObject getHandle.! !

!TF_Buffer methodsFor: 'release' stamp: 'gera 1/19/2017 09:46:20'!
delete
	self isNull ifFalse: [
		self library deleteBuffer: self].
	handle _ nil! !

!TF_Buffer methodsFor: 'accessing' stamp: 'gera 12/22/2016 00:34:10'!
length
	"This method was automatically generated. See TF_Buffer class>>fields."
	^handle unsignedLongLongAt: 9! !

!TF_Buffer methodsFor: 'accessing' stamp: 'gera 12/22/2016 00:34:10'!
length: anObject
	"This method was automatically generated. See TF_Buffer class>>fields."
	handle unsignedLongLongAt: 9 put: anObject! !

!TF_Buffer class methodsFor: 'field definition' stamp: 'gera 1/12/2017 18:45:49'!
fields
	"self initialize.
	self defineFields"
	^#(
		(data 'void*')
		(length 'ulonglong')
		(deallocator 'void*'))
	! !

!TF_Buffer class methodsFor: 'instance creation' stamp: 'gera 12/22/2016 01:05:44'!
fromFileNamed: aString
	| fileContents |
	fileContents _ aString asFileEntry readStream contents.
	^ self fromString: fileContents! !

!TF_Buffer class methodsFor: 'instance creation' stamp: 'gera 1/10/2017 21:00:42'!
fromString: aString
	| answer |
	answer _ TensorFlowCAPI current newBufferFromString: aString size: aString size.
	^ answer! !

!TF_Graph methodsFor: 'operations' stamp: 'gera 1/5/2017 05:21:30'!
add: nameString described: aBlock
	^self newOperation: 'Add' named: nameString described: aBlock! !

!TF_Graph methodsFor: 'accessing' stamp: 'gera 1/5/2017 20:35:36'!
allVariables
	^ self operationsSelect: [:op | op type = 'Placeholder']! !

!TF_Graph methodsFor: 'operations' stamp: 'gera 1/5/2017 23:55:21'!
asString: nameString described: aBlock
	^self newOperation: 'AsString' named: nameString described: aBlock! !

!TF_Graph methodsFor: 'operations' stamp: 'gera 12/30/2016 12:16:54'!
concat: nameString described: aBlock
	^self newOperation: 'Concat' named: nameString described: aBlock! !

!TF_Graph methodsFor: 'operations' stamp: 'gera 12/28/2016 02:07:25'!
const: nameString value: aTF_Tensor
	^ self
		newOperation: 'Const'
		named: nameString
		described: [:description |
			description at: 'dtype' putType: aTF_Tensor type.
			description at: 'value' putTensor: aTF_Tensor]! !

!TF_Graph methodsFor: 'accessing' stamp: 'gera 1/5/2017 20:50:50'!
definition
	| buffer status |
	status _ TF_Status create.
	buffer _ TF_Buffer new.
	self library graph: self toGraphDef: buffer status: status.
	status check.
	^ buffer dataBytes asString! !

!TF_Graph methodsFor: 'initialize-release' stamp: 'gera 1/19/2017 09:45:55'!
delete
	self ignoreFinalization.
	self isNull ifFalse: [
		self library deleteGraph: self].
	handle _ nil! !

!TF_Graph methodsFor: 'initialize-release' stamp: 'gera 1/13/2017 10:32:27'!
finalize
	self delete! !

!TF_Graph methodsFor: 'initialize-release' stamp: 'gera 1/3/2017 17:20:14'!
import: aTF_Buffer
	| options status |
	
	status _ TF_Status create.
	options _ TF_ImportGraphDefOptions create.
	
	self library importGraphDefInto: self from: aTF_Buffer options: options status: status.
	
	options delete.
	status check.! !

!TF_Graph methodsFor: 'initialize-release' stamp: 'gera 12/22/2016 01:27:24'!
importFileNamed: aString
	| buffer |
	buffer _ TF_Buffer fromFileNamed: aString.
	[self import: buffer] ensure: [buffer delete].! !

!TF_Graph methodsFor: 'initialize-release' stamp: 'gera 12/22/2016 01:26:59'!
importString: aString
	| buffer |
	buffer _ TF_Buffer fromString: aString.
	[self import: buffer] ensure: [buffer delete].! !

!TF_Graph methodsFor: 'operations' stamp: 'gera 12/28/2016 02:11:44'!
mul: nameString described: aBlock
	^self newOperation: 'Mul' named: nameString described: aBlock! !

!TF_Graph methodsFor: 'operations' stamp: 'gera 12/27/2016 23:16:19'!
newOperation: typeString named: aString
	^ self library newOperationOn: self type: typeString named: aString
! !

!TF_Graph methodsFor: 'operations' stamp: 'gera 1/19/2017 10:34:58'!
newOperation: typeString named: aString described: aBlock
	| description answer |
	description _ self newOperation: typeString named: aString.
	aBlock value: description.
	answer _ description finish.
	answer graph: self.
	^ answer! !

!TF_Graph methodsFor: 'accessing' stamp: 'gera 1/5/2017 20:20:02'!
newOperationIteratorContext
	^ ByteArray new: 8! !

!TF_Graph methodsFor: 'accessing' stamp: 'gera 1/5/2017 20:09:21'!
operationAt: contextULongLongPtr
	^ self library graph: self operationAt: contextULongLongPtr! !

!TF_Graph methodsFor: 'accessing' stamp: 'gera 1/19/2017 23:27:23'!
operationNamed: aString
	| answer |
	answer _ self library graph: self getOperationNamed: aString asAsciiZ.
	answer isNull ifTrue: [self error: 'Operation not found'].
	^ answer
! !

!TF_Graph methodsFor: 'accessing' stamp: 'gera 1/19/2017 11:10:01'!
operationsCount
	| answer |
	answer _ 0.
	self operationsDo: [:each | answer _ answer + 1].
	^ answer! !

!TF_Graph methodsFor: 'accessing' stamp: 'gera 1/5/2017 20:23:10'!
operationsDo: oneArgBlock
	| context operation |
	context _ self newOperationIteratorContext.
	[operation _ self operationAt: context.
		operation isNull] whileFalse: [oneArgBlock value: operation].
! !

!TF_Graph methodsFor: 'accessing' stamp: 'gera 1/5/2017 20:33:53'!
operationsSelect: oneArgBlock
	| answer |
	answer _ OrderedCollection new.
	
	self operationsSelect: oneArgBlock thenDo: [:op | answer add: op].
	
	^ answer asArray! !

!TF_Graph methodsFor: 'accessing' stamp: 'gera 1/5/2017 20:33:20'!
operationsSelect: conditionBlock thenDo: actionBlock
	| answer |
	answer _ OrderedCollection new.
	
	self operationsDo: [:op |
		(conditionBlock value: op)
			ifTrue: [actionBlock value: op]].
	
	^ answer asArray! !

!TF_Graph methodsFor: 'outputs' stamp: 'gera 1/3/2017 17:20:21'!
outputDimenssionsCount: aTF_Output
	| status answer |
	status _ TF_Status create.
	answer _ self library forGraph: self outputDims: aTF_Output status: status.
	status check.
	^answer! !

!TF_Graph methodsFor: 'operations' stamp: 'gera 1/6/2017 17:21:53'!
placeholder: nameString type: typeInteger
	^ self
		newOperation: 'Placeholder'
		named: nameString
		described: [:description |
			description at: 'dtype' putType: typeInteger]! !

!TF_Graph class methodsFor: 'instance creation' stamp: 'gera 1/20/2017 17:53:59'!
create
	^ TensorFlowCAPI current newGraph useFinalization! !

!TF_Graph class methodsFor: 'instance creation' stamp: 'gera 12/22/2016 02:18:26'!
fromFileNamed: aString
	^ self create importFileNamed: aString! !

!TF_Graph class methodsFor: 'instance creation' stamp: 'gera 12/22/2016 02:18:33'!
fromString: aString
	^ self create importString: aString! !

!TF_ImportGraphDefOptions methodsFor: 'release' stamp: 'gera 1/19/2017 09:46:56'!
delete
	self isNull ifFalse: [
		self library deleteImportGraphDefOptions: self].
	handle _ nil! !

!TF_ImportGraphDefOptions class methodsFor: 'instance creation' stamp: 'gera 12/22/2016 01:15:16'!
create
	^ TensorFlowCAPI current newImportGraphDefOptions! !

!TF_Operation methodsFor: 'ops binary' stamp: 'gera 1/20/2017 18:22:38'!
* aTF_Operation
	^ self binaryOp: 'MatMul' with: aTF_Operation! !

!TF_Operation methodsFor: 'ops binary' stamp: 'gera 1/21/2017 02:26:09'!
*\ aTF_Operation
	^ self
		binaryOp: 'MatMul'
		with: aTF_Operation
		described: [:description |
			description at: 'transpose_b' putBoolean: true]
! !

!TF_Operation methodsFor: 'ops binary' stamp: 'gera 1/19/2017 11:19:11'!
+ aTF_Operation
	^ self binaryOp: 'Add' with: aTF_Operation! !

!TF_Operation methodsFor: 'ops binary' stamp: 'gera 1/20/2017 18:20:07'!
@* aTF_Operation
	^ self binaryOp: 'Mul' with: aTF_Operation! !

!TF_Operation methodsFor: 'ops binary' stamp: 'gera 1/21/2017 02:26:18'!
\* aTF_Operation
	^ self
		binaryOp: 'MatMul'
		with: aTF_Operation
		described: [:description |
			description at: 'transpose_a' putBoolean: true]
! !

!TF_Operation methodsFor: 'ops binary' stamp: 'gera 1/19/2017 12:52:43'!
\\ aTF_Operation
	^ self binaryOp: 'Mod' with: aTF_Operation! !

!TF_Operation methodsFor: 'accessing' stamp: 'gera 12/22/2016 03:57:35'!
InputsCount
	^ self library operationNumInputs: self! !

!TF_Operation methodsFor: 'ops unary' stamp: 'gera 1/19/2017 12:55:41'!
abs
	^ self unaryOp: 'Abs'! !

!TF_Operation methodsFor: 'ops unary' stamp: 'gera 1/19/2017 12:58:32'!
arcCos
	^ self unaryOp: 'Acos'
	! !

!TF_Operation methodsFor: 'ops unary' stamp: 'gera 1/19/2017 13:03:06'!
arcSin
	^ self unaryOp: 'Asin'
	! !

!TF_Operation methodsFor: 'ops unary' stamp: 'gera 1/19/2017 13:04:36'!
arcTan
	^ self unaryOp: 'Atan'
	! !

!TF_Operation methodsFor: 'attributes' stamp: 'gera 1/12/2017 18:46:27'!
attrMetadata: nameString
	| status answer |
	status _ TF_Status create.
	answer _ self library operation: self getMetadataFor: nameString status: status.
	status check.
	^ answer! !

!TF_Operation methodsFor: 'ops binary' stamp: 'gera 1/21/2017 02:22:49'!
binaryOp: aString with: aTF_Operation
	^ self binaryOp: aString with: aTF_Operation described: [:nothing]
! !

!TF_Operation methodsFor: 'ops binary' stamp: 'gera 1/21/2017 02:20:55'!
binaryOp: aString with: aTF_Operation described: oneArgBlock
	| name |
	name _ aString,'_',graph operationsCount printString.
	
	^ graph newOperation: aString named: name described: [:description |
		description
			addInput: (self output: 0);
			addInput: (aTF_Operation output: 0).
		oneArgBlock value: description].
	! !

!TF_Operation methodsFor: 'attributes' stamp: 'gera 1/9/2017 19:44:59'!
boolAt: nameString
	| value status |
	status _ TF_Status create.
	value _ ByteArray new: 1.
	self library operation: self attr: nameString asAsciiZ getBool: value status: status.
	status check.
	^ (value booleanAt: 1) 
! !

!TF_Operation methodsFor: 'ops unary' stamp: 'gera 1/19/2017 13:08:18'!
cos
	^ self unaryOp: 'Cos'
	! !

!TF_Operation methodsFor: 'accessing' stamp: 'gera 12/22/2016 03:57:05'!
device
	^ self library operationDevice: self! !

!TF_Operation methodsFor: 'ops binary' stamp: 'gera 1/21/2017 03:37:51'!
findMinOn: aTF_Operation
	^ self binaryOp: 'ArgMin' with: aTF_Operation! !

!TF_Operation methodsFor: 'attributes' stamp: 'gera 1/10/2017 10:22:29'!
floatAt: nameString
	| value status |
	status _ TF_Status create.
	value _ ByteArray new: 8.
	self library operation: self attr: nameString asAsciiZ getFloat: value status: status.
	status check.
	^ value floatAt: 1
! !

!TF_Operation methodsFor: 'accessing' stamp: 'gera 1/19/2017 10:35:20'!
graph
	^ graph! !

!TF_Operation methodsFor: 'accessing' stamp: 'gera 1/19/2017 10:35:14'!
graph: aTF_Graph
	graph _ aTF_Graph! !

!TF_Operation methodsFor: 'accessing' stamp: 'gera 12/28/2016 01:39:51'!
input: indexInteger
	^ TF_Input onOperation: self index: indexInteger! !

!TF_Operation methodsFor: 'attributes' stamp: 'gera 1/7/2017 04:49:04'!
intAt: nameString
	| value status |
	status _ TF_Status create.
	value _ ByteArray new: 8.
	self library operation: self attr: nameString asAsciiZ getInt64: value status: status.
	status check.
	^ value unsignedLongLongAt: 1
! !

!TF_Operation methodsFor: 'accessing' stamp: 'gera 12/22/2016 03:56:55'!
name
	^ self library operationName: self! !

!TF_Operation methodsFor: 'ops unary' stamp: 'gera 1/19/2017 11:14:17'!
negated
	^ self unaryOp: 'Neg'! !

!TF_Operation methodsFor: 'accessing' stamp: 'gera 12/28/2016 01:35:51'!
output: indexInteger
	^ TF_Output onOperation: self index: indexInteger! !

!TF_Operation methodsFor: 'accessing' stamp: 'gera 12/22/2016 03:57:48'!
outputsCount
	^ self library operationNumOutputs: self! !

!TF_Operation methodsFor: 'printing' stamp: 'gera 1/13/2017 12:41:03'!
printOn: strm
	super printOn: strm.
	handle isNull ifFalse: [
		strm
			space;
			print: self type;
			space;
			print: self name]! !

!TF_Operation methodsFor: 'attributes' stamp: 'gera 1/13/2017 13:14:12'!
shapeAt: nameString
	| value status size answer |
	size _ (self attrMetadata: nameString) totalSize.
	status _ TF_Status create.
	value _ Int64Array externalNew: size.
	
	self library
		operation: self
		attr: nameString asAsciiZ
		getShape: value
		size: size
		status: status.
		
	answer _ (1 to: size) collect: [:i | value at: i].
	status check.
	^ answer! !

!TF_Operation methodsFor: 'ops unary' stamp: 'gera 1/19/2017 13:08:09'!
sin
	^ self unaryOp: 'Sin'
	! !

!TF_Operation methodsFor: 'attributes' stamp: 'gera 1/10/2017 09:03:57'!
stringAt: nameString
	| value status size |
	size _ 100.
	status _ TF_Status create.
	value _ ByteArray new: size.
	self library operation: self attr: nameString asAsciiZ getString: value size: size status: status.
	status check.
	^ (value upTo: 0) asString
! !

!TF_Operation methodsFor: 'ops binary' stamp: 'gera 1/21/2017 03:30:55'!
sumOn: aTF_Operation
	^ self binaryOp: 'Sum' with: aTF_Operation! !

!TF_Operation methodsFor: 'ops unary' stamp: 'gera 1/19/2017 13:08:05'!
tan
	^ self unaryOp: 'Tan'
	! !

!TF_Operation methodsFor: 'attributes' stamp: 'gera 1/7/2017 05:00:29'!
tensorAt: nameString
	| value status |
	status _ TF_Status create.
	value _ ByteArray new: ExternalAddress wordSize.
	self library operation: self attr: nameString asAsciiZ getTensor: value status: status.
	status check.
	^ TF_Tensor fromHandle: (value pointerAt: 1)
! !

!TF_Operation methodsFor: 'accessing' stamp: 'gera 12/22/2016 03:58:02'!
type
	^ self library operationOpType: self! !

!TF_Operation methodsFor: 'attributes' stamp: 'gera 1/7/2017 04:54:37'!
typeAt: nameString
	| value status |
	status _ TF_Status create.
	value _ ByteArray new: 8.
	self library operation: self attr: nameString asAsciiZ getType: value status: status.
	status check.
	^ value unsignedLongLongAt: 1
! !

!TF_Operation methodsFor: 'ops unary' stamp: 'gera 1/19/2017 11:18:57'!
unaryOp: aString
	| name |
	name _ aString,'_',graph operationsCount printString.
	
	^ graph newOperation: aString named: name described: [:description |
		description addInput: (self output: 0)].
	! !

!TF_OperationDescription methodsFor: 'accessing' stamp: 'jb 12/26/2016 22:35:53'!
add: input
	self library description: self add: input.! !

!TF_OperationDescription methodsFor: 'inputs' stamp: 'gera 12/28/2016 01:32:31'!
addInput: aTF_OutputOrInput
	^self library
		description: self
		addInput: aTF_OutputOrInput! !

!TF_OperationDescription methodsFor: 'inputs' stamp: 'gera 12/28/2016 01:40:51'!
addInputFromOutput: indexInteger of: aTF_Operation
	| input |
	input _ aTF_Operation input: indexInteger.
	[self library
		description: self
		addInput: input
	] ensure: [input free]! !

!TF_OperationDescription methodsFor: 'inputs' stamp: 'gera 1/13/2017 13:08:59'!
addInputs: anArrayOfTF_Output
	| inputs |
	inputs _ TF_OutputArray externalFromArray: anArrayOfTF_Output.
	self library
		description: self
		addInputs: inputs
		size: anArrayOfTF_Output size.! !

!TF_OperationDescription methodsFor: 'attributes' stamp: 'gera 1/9/2017 19:53:37'!
at: attribute putBoolean: value
	self library
		description: self
		set: attribute asAsciiZ
		toBool: value
		! !

!TF_OperationDescription methodsFor: 'attributes' stamp: 'gera 1/10/2017 10:35:19'!
at: attribute putFloat: value
	self library
		description: self
		set: attribute asAsciiZ
		toFloat: value asFloat
		! !

!TF_OperationDescription methodsFor: 'attributes' stamp: 'gera 1/7/2017 05:11:24'!
at: attribute putInt: value
	self library
		description: self
		set: attribute asAsciiZ
		toInt64: value! !

!TF_OperationDescription methodsFor: 'attributes' stamp: 'gera 1/13/2017 13:09:06'!
at: attribute putShape: anIntegerArray
	| status value |
	value _ Int64Array externalFromArray: anIntegerArray.
	status _ TF_Status create.
	self library
		description: self
		set: attribute asAsciiZ
		toShape: value
		size: anIntegerArray size.
	status check.! !

!TF_OperationDescription methodsFor: 'attributes' stamp: 'gera 1/10/2017 20:57:22'!
at: attribute putString: aString
	| status |
	status _ TF_Status create.
	self library
		description: self
		set: attribute asAsciiZ
		toString: aString
		size: aString size.
	status check
	! !

!TF_OperationDescription methodsFor: 'attributes' stamp: 'gera 1/7/2017 05:12:02'!
at: attribute putTensor: aTF_Tensor
	| status |
	status _ TF_Status create.
	self library
		description: self
		set: attribute asAsciiZ
		toTensor: aTF_Tensor
		status: status.
	status check
	! !

!TF_OperationDescription methodsFor: 'attributes' stamp: 'gera 1/7/2017 05:12:20'!
at: attribute putType: value
	self library
		description: self
		set: attribute asAsciiZ
		toType: value! !

!TF_OperationDescription methodsFor: 'accessing' stamp: 'gera 1/19/2017 23:33:39'!
finish
	| answer status |
	status _ TF_Status create.
	answer _ self library
		finishOperation: self
		status: status.
	handle _ nil.
	status check.
	^ answer.! !

!TF_Output methodsFor: 'accessing' stamp: 'gera 12/23/2016 10:17:53'!
index
	"This method was automatically generated. See TF_Output class>>fields."
	^handle unsignedLongLongAt: 9! !

!TF_Output methodsFor: 'accessing' stamp: 'gera 12/23/2016 10:17:53'!
index: anObject
	"This method was automatically generated. See TF_Output class>>fields."
	handle unsignedLongLongAt: 9 put: anObject! !

!TF_Output methodsFor: 'accessing' stamp: 'gera 12/23/2016 10:17:53'!
operation
	"This method was automatically generated. See TF_Output class>>fields."
	^TF_Operation fromHandle: (handle longPointerAt: 1)! !

!TF_Output methodsFor: 'accessing' stamp: 'gera 12/23/2016 10:17:53'!
operation: anObject
	"This method was automatically generated. See TF_Output class>>fields."
	handle longPointerAt: 1 put: anObject getHandle.! !

!TF_Output methodsFor: 'accessing' stamp: 'gera 12/23/2016 10:26:22'!
type
	^ self library operationOutputType: self! !

!TF_Output class methodsFor: 'field definition' stamp: 'gera 1/12/2017 18:46:11'!
fields
	"self initialize.
	self defineFields"
	^#(
		(operation 'TF_Operation*')
		(index 'ulonglong'))
	! !

!TF_Output class methodsFor: 'instance creation' stamp: 'gera 12/23/2016 10:40:36'!
onOperation: aTF_Operation index: anInteger
	^ self externalNew
		operation: aTF_Operation;
		index: anInteger! !

!TF_Session methodsFor: 'release' stamp: 'gera 1/3/2017 17:20:55'!
close
	| status |
	status _ TF_Status create.
	self library closeSession: self status: status.
	status check! !

!TF_Session methodsFor: 'release' stamp: 'gera 1/19/2017 09:47:09'!
delete
	| status |
	self ignoreFinalization.
	status _ TF_Status create.
	self isNull ifFalse: [
		self library
			deleteSession: self
			status: status].
	status check.
	handle _ nil.
! !

!TF_Session methodsFor: 'release' stamp: 'gera 1/13/2017 10:40:47'!
finalize
	[
	self isNull ifFalse: [ self close ].
	self delete ]
		on: Error
		do: [
			Transcript
				newLine;
				print: 'Error finalizing ';
				show: self ].! !

!TF_Session methodsFor: 'running' stamp: 'gera 1/3/2017 17:21:29'!
run
	| status |
	status _ TF_Status create.
	
	self library
		runSession: self options: nil
		inputs: nil values: nil count: 0
		outputs: nil values: nil count: 0
		targets: nil count: 0 metadata: nil status: status.
	
	status check! !

!TF_Session methodsFor: 'running' stamp: 'gera 12/29/2016 18:03:48'!
runOperation: aTF_Operation
	^self runOperations: (Array with: aTF_Operation)! !

!TF_Session methodsFor: 'running' stamp: 'gera 1/13/2017 13:09:21'!
runOperation: aTF_Operation input: inTF_OutputOrInput value: inTF_Tensor output: outTF_Output
	| inputs invalues operations outputs outvalues tensor |
	inputs _ Array with: inTF_OutputOrInput.
	invalues _ Array with: inTF_Tensor.
	outputs _ Array with: outTF_Output.
	operations _ Array with: aTF_Operation.
	
	outvalues _ self runOperations: operations inputs: inputs values: invalues outputs: outputs.
	tensor _ outvalues first.
	^ tensor
! !

!TF_Session methodsFor: 'running' stamp: 'gera 1/13/2017 13:09:51'!
runOperation: aTF_Operation output: aTF_Output
	| operations  answer outputs |
	operations _ TF_OperationPtrArray externalNew: 1.
	outputs _ TF_OutputArray externalNew: 1.
	outputs first: aTF_Output.
	
	operations first: aTF_Operation.	
	answer _ self runOperations: operations outputs: outputs size: 1.

	^ answer! !

!TF_Session methodsFor: 'running' stamp: 'gera 1/13/2017 13:10:00'!
runOperations: anArrayOfTF_Operations
	| status operations |
	status _ TF_Status create.
	operations _ TF_OperationPtrArray externalFromArray: anArrayOfTF_Operations.
	
	self library
		runSession: self options: nil
		inputs: nil values: nil count: 0
		outputs: nil values: nil count: 0
		targets: operations count: anArrayOfTF_Operations size
		metadata: nil status: status.
	status check
	! !

!TF_Session methodsFor: 'running' stamp: 'gera 1/13/2017 13:10:39'!
runOperations: anArrayOfTF_Operations inputs: inArrayOfTF_Outputs values: inArrayOfTF_Tensor outputs: outArrayOfTF_Outputs
	| operations inputs invalues outputs outvalues status |
	status _ TF_Status create.
	operations _ TF_OperationPtrArray externalFromArray: anArrayOfTF_Operations.
	inputs _ TF_OutputArray externalFromArray: inArrayOfTF_Outputs.
	invalues _ TF_TensorPtrArray externalFromArray: inArrayOfTF_Tensor.
	outputs _ TF_OutputArray externalFromArray: outArrayOfTF_Outputs.
	outvalues _ TF_TensorPtrArray new: outArrayOfTF_Outputs size.
	
	self library
		runSession: self options: nil
		inputs: inputs values: invalues count: inArrayOfTF_Outputs size
		outputs: outputs values: outvalues count: outArrayOfTF_Outputs size
		targets: operations count: anArrayOfTF_Operations size
		metadata: nil status: status.
		
	status check.
	^ outvalues
! !

!TF_Session methodsFor: 'running' stamp: 'gera 1/13/2017 13:10:35'!
runOperations: aTF_OperationArray outputs: aTF_OutputArray size: anInteger
	| status outvalues tensor |
	status _ TF_Status create.
	outvalues _ TF_TensorPtrArray new: anInteger.
	
	self library
		runSession: self options: nil
		inputs: nil values: nil count: 0
		outputs: aTF_OutputArray values: outvalues count: anInteger
		targets: aTF_OperationArray count: 1
		metadata: nil status: status.
		
	tensor _ outvalues first.
	status check.
	^ tensor! !

!TF_Session class methodsFor: 'instance creation' stamp: 'gera 1/7/2017 20:27:48'!
onGraph: aTF_Graph
	| options status answer |
	options _ TF_SessionOptions create.
	status _ TF_Status create.
	answer _ TensorFlowCAPI current newSessionGraph: aTF_Graph options: options status: status.
	options delete.
	status check.
	answer useFinalization.
	^ answer! !

!TF_SessionOptions methodsFor: 'initialize-release' stamp: 'gera 1/19/2017 09:47:17'!
delete
	self isNull ifFalse: [
		self library deleteSessionOptions: self].
	handle _ nil! !

!TF_SessionOptions methodsFor: 'initialize-release' stamp: 'gera 12/21/2016 23:37:00'!
target: aString
	" 'local' 'google.com:1234' '192.168.1.1:1234' 'local,example.com:1234' etc.
	are all valid target strings"
	self library sessionOptions: self setTarget: (self library externalizeString: aString).
! !

!TF_SessionOptions class methodsFor: 'instance creation' stamp: 'gera 12/21/2016 02:50:29'!
create
	^ TensorFlowCAPI current newSessionOptions! !

!TF_SessionOptions class methodsFor: 'instance creation' stamp: 'gera 12/21/2016 23:47:47'!
onTarget: aString
	^ self create target: aString! !

!TF_Status methodsFor: 'testing' stamp: 'gera 12/21/2016 11:05:49'!
check
	self isOk ifFalse: [
		Error signal: (self codeText, ': ', self message)]! !

!TF_Status methodsFor: 'accessing' stamp: 'gera 12/21/2016 09:59:56'!
code
	^self library getCode: self! !

!TF_Status methodsFor: 'accessing' stamp: 'jb 12/26/2016 10:29:57'!
code: aTF_Code message: aString
	| externalized |
	self assert: aTF_Code ~= 0.
	externalized _ self library externalizeString: aString.
	self library setStatus: self code: aTF_Code message: externalized
! !

!TF_Status methodsFor: 'accessing' stamp: 'gera 12/21/2016 11:03:40'!
codeText
	^ self codeTexts at: self code + 1! !

!TF_Status methodsFor: 'accessing' stamp: 'gera 12/21/2016 11:02:44'!
codeTexts
	"self new codeTexts"
	^ #(  
  'OK' "0"
  'CANCELLED' "1"
  'UNKNOWN' "2"
  'INVALID_ARGUMENT' "3"
  'DEADLINE_EXCEEDED' "4"
  'NOT_FOUND' "5"
  'ALREADY_EXISTS' "6"
  'PERMISSION_DENIED' "7"
  'RESOURCE_EXHAUSTED' "8"
  'FAILED_PRECONDITION' "9"
  'ABORTED' "10"
  'OUT_OF_RANGE' "11"
  'UNIMPLEMENTED' "12"
  'INTERNALt' "13"
  'UNAVAILABLE' "14"
  'DATA_LOSS' "15"
  'UNAUTHENTICATED' "16"
)! !

!TF_Status methodsFor: 'initialize-release' stamp: 'gera 1/19/2017 09:47:23'!
delete
	self ignoreFinalization.
	self isNull ifFalse: [
		self library deleteStatus: self].
	handle _ nil.! !

!TF_Status methodsFor: 'initialize-release' stamp: 'gera 1/13/2017 10:31:44'!
finalize
	handle isNull ifTrue: [^self].
	self delete! !

!TF_Status methodsFor: 'testing' stamp: 'gera 12/21/2016 10:48:36'!
isOk
	^self code == 0! !

!TF_Status methodsFor: 'accessing' stamp: 'gera 12/21/2016 23:50:21'!
message
	^ self library message: self! !

!TF_Status methodsFor: 'printing' stamp: 'gera 1/9/2017 11:15:35'!
printOn: strm
	super printOn: strm.
	self isNull ifFalse: [
		strm
			space;
			print: self code;
			space;
			print: self message]! !

!TF_Status class methodsFor: 'instance creation' stamp: 'gera 1/7/2017 20:20:47'!
create
	|answer |
	answer _ self persistent.
	answer useFinalization.
	^ answer! !

!TF_Status class methodsFor: 'instance creation' stamp: 'gera 1/3/2017 09:53:25'!
persistent
	^ TensorFlowCAPI current newStatus! !

!TF_Tensor methodsFor: 'converting' stamp: 'jb 1/15/2017 18:07:40'!
allFloats
	| total answer data byteSize |
	total _ self size.
	answer _ WriteStream on: (Array new: total).
	data _ self data getHandle.
	byteSize _ self class sizeForType: self class typeFloat.
	1 to: total do: [:i |
		| value |
		value _ data floatAt: i-1*byteSize+1.
		answer nextPut: value].
	
	^ answer contents! !

!TF_Tensor methodsFor: 'converting' stamp: 'jb 1/15/2017 18:02:38'!
allInt64s
	| total answer data byteSize |
	total _ self size.
	answer _ WriteStream on: (Array new: total).
	data _ self data getHandle.
	byteSize _ self class sizeForType: self class typeInt64.
	1 to: total do: [:i |
		| value |
		value _ data signedLongLongAt: i-1*byteSize+1..
		answer nextPut: value].
	
	^ answer contents! !

!TF_Tensor methodsFor: 'converting' stamp: 'gera 1/6/2017 14:56:06'!
allStrings
	| total answer bytes data |
	total _ self size.
	answer _ WriteStream on: (Array new: total).
	bytes _ self dataBytes.
	data _ self data getHandle + (8 * total).
	1 to: total do: [:i |
		| offset str |
		offset _ bytes unsignedLongLongAt: i - 1 * 8 + 1.
		str _ self library stringDecode: data + offset.
		answer nextPut: str].
	
	^ answer contents! !

!TF_Tensor methodsFor: 'accessing' stamp: 'gera 1/5/2017 23:35:00'!
byteSize
	^ self library tensorByteSize: self! !

!TF_Tensor methodsFor: 'accessing' stamp: 'gera 12/27/2016 09:18:01'!
data
	^self library tensorData: self.! !

!TF_Tensor methodsFor: 'accessing' stamp: 'gera 1/6/2017 11:10:43'!
dataBytes
	^self data getHandle structAt: 1 length: self byteSize! !

!TF_Tensor methodsFor: 'release' stamp: 'gera 1/19/2017 09:45:31'!
delete
	self ignoreFinalization.
	self isNull ifFalse: [
		self library deleteTensor: self].
	handle _ nil! !

!TF_Tensor methodsFor: 'accessing' stamp: 'gera 1/6/2017 12:41:46'!
dimensions
	| answer count |
	count _ self numberOfDimensions.
	answer _ WriteStream on: (Array new: count).
	1 to: count do: [:i |
		answer nextPut: (self sizeOn: i-1)].
	^ answer contents! !

!TF_Tensor methodsFor: 'release' stamp: 'gera 1/13/2017 11:15:56'!
finalize
	self delete! !

!TF_Tensor methodsFor: 'accessing' stamp: 'jb 12/26/2016 12:09:58'!
numberOfDimensions
	^self library tensorNumDimensions: self.! !

!TF_Tensor methodsFor: 'accessing' stamp: 'gera 1/6/2017 14:37:36'!
size
	^ self dimensions inject: 1 into: [:prev :next | prev * next].! !

!TF_Tensor methodsFor: 'accessing' stamp: 'jb 12/26/2016 12:10:24'!
sizeOn: dimension
	^self library tensor: self sizeOn: dimension.! !

!TF_Tensor methodsFor: 'accessing' stamp: 'gera 12/26/2016 10:50:33'!
type
	^self library tensorType: self.! !

!TF_Tensor class methodsFor: 'utils' stamp: 'gera 1/13/2017 19:55:11'!
dimensionsOf: aMultidimensionalTensor
	^ aMultidimensionalTensor isCollection
		ifTrue: [{aMultidimensionalTensor size}, (self dimensionsOf: aMultidimensionalTensor first)]
		ifFalse: [#()].! !

!TF_Tensor class methodsFor: 'utils' stamp: 'gera 1/17/2017 18:57:49'!
elementsOf: aMultidimensionalTensor do: oneArgBlock
	^ aMultidimensionalTensor isCollection
		ifTrue: [aMultidimensionalTensor do: [:each |
				self elementsOf: each do: oneArgBlock]]
		ifFalse: [oneArgBlock value: aMultidimensionalTensor]! !

!TF_Tensor class methodsFor: 'utils' stamp: 'gera 1/18/2017 00:21:04'!
floatArray: values into: anExternalAddressOrByteArray
	| size index |
	size _ self sizeForType: self typeFloat.
	
	index _ 0.
	self elementsOf: values do: [:value |
		anExternalAddressOrByteArray floatAt: index*size + 1 put: value.
		index _ index + 1].
! !

!TF_Tensor class methodsFor: 'instance creation' stamp: 'gera 1/19/2017 11:26:11'!
fromFloats: values
	| answer dimensions size count |
	size _ self sizeOf: values.
	dimensions _ self dimensionsOf: values.
	count _ dimensions inject: 1 into: [:a :b | a * b].
	count = size ifFalse: [
		self error: 'Inferred size and real size don''t match.'].
	
	answer _ self type: self typeFloat dimensions: dimensions.
	self floatArray: values into: answer data getHandle.
	^ answer! !

!TF_Tensor class methodsFor: 'instance creation' stamp: 'gera 12/30/2016 12:46:27'!
fromInt32: value
	| answer |
	answer _ self type: self typeInt32 dimensions: #().
	answer data getHandle signedLongAt: 1 put: value.
	^ answer! !

!TF_Tensor class methodsFor: 'instance creation' stamp: 'gera 12/29/2016 17:53:03'!
fromInt64: value
	| answer |
	answer _ self type: self typeInt64 dimensions: #().
	answer data getHandle signedLongLongAt: 1 put: value.
	^ answer! !

!TF_Tensor class methodsFor: 'instance creation' stamp: 'gera 1/13/2017 03:21:48'!
fromInt64Array: values
	| answer |
	answer _ self type: self typeInt64 dimensions: {values size}.
	self int64Array: values into: answer data getHandle.
	^ answer! !

!TF_Tensor class methodsFor: 'instance creation' stamp: 'gera 1/13/2017 03:57:18'!
fromString: aString
	| answer bytesize |
	bytesize _ TensorFlowCAPI current stringEncodedSize: aString.
	bytesize _ bytesize.
	answer _ self type: self typeString dimensions: #() bytesize: bytesize.
	
	TensorFlowCAPI current
		stringEncode: aString
		to: answer data getHandle
		size: bytesize.
	^ answer! !

!TF_Tensor class methodsFor: 'instance creation' stamp: 'gera 1/13/2017 04:04:42'!
fromStringArray: aStringArray
	| answer sizes offsets offsetSize bytesize |
	sizes _ aStringArray collect: [:each | self sizeOfString: each].
	offsetSize _ self sizeForType: self typeInt64.
	bytesize _ offsetSize * aStringArray size + sizes sum.
	offsets _ OrderedCollection new: sizes size.
	
	sizes inject: 0 into: [:prev :each |
		offsets add: prev.
		each + prev].
	
	answer _ self
		type: self typeString
		dimensions: {aStringArray size}
		bytesize: bytesize.
		
	self int64Array: offsets into: answer data getHandle.	
	
	aStringArray withIndexDo: [:each :index |
		| offset |
		offset _ offsetSize * aStringArray size + (offsets at: index).
		TensorFlowCAPI current
			stringEncode: each
			to: answer data getHandle + offset
			size: bytesize - offset].
	^ answer
! !

!TF_Tensor class methodsFor: 'utils' stamp: 'gera 1/18/2017 00:32:18'!
int64Array: values into: anExternalAddressOrByteArray
	| size index |

	size _ self sizeForType: self typeInt64.
	
	index _ 0.
	self elementsOf: values do: [:value |
		anExternalAddressOrByteArray signedLongLongAt: index*size + 1 put: value.
		index _ index + 1].
! !

!TF_Tensor class methodsFor: 'utils' stamp: 'gera 1/13/2017 19:45:52'!
rankOf: aMultidimensionalTensor
	^ aMultidimensionalTensor isCollection
		ifTrue: [1 + (self rankOf: aMultidimensionalTensor first)]
		ifFalse: [0].! !

!TF_Tensor class methodsFor: 'utils' stamp: 'jb 1/15/2017 17:55:27'!
sizeForType: anInteger
	anInteger = self typeFloat ifTrue: [^4].
	anInteger = self typeInt64 ifTrue: [^8].
	anInteger = self typeInt32 ifTrue: [^4].

"
8
		UInt8 4
		Int8 6
		QInt8 11
		QUInt8 12

16
		Int16 5
		BFloat16 14
		QInt16 15
		QUInt16 16
		UInt16 17
		Half 19

32
		Float 1
		Int32 3
		QInt32 13

64
		Double 2
		Complex64 8
		Int64 9

128
		Complex128 18

		String 7
		Resource 20).

		Bool 10
"


	^ self shouldBeImplemented.! !

!TF_Tensor class methodsFor: 'utils' stamp: 'gera 1/18/2017 00:02:29'!
sizeOf: aMultidimensionalTensor
	^ aMultidimensionalTensor isCollection
		ifTrue: [aMultidimensionalTensor sum: [:each | self sizeOf: each]]
		ifFalse: [1].! !

!TF_Tensor class methodsFor: 'utils' stamp: 'gera 1/13/2017 03:17:43'!
sizeOfString: aString.
	^ TensorFlowCAPI current stringEncodedSize: aString.! !

!TF_Tensor class methodsFor: 'instance creation' stamp: 'gera 1/13/2017 01:42:58'!
type: anInteger dimensions: anIntegerArray
	" I understand dimenssions are:
		#()		-> Scalar
		#(7)		-> Unidimensional array of 7 elements
		#(7 4)		-> 7x4 elements matrix
		#(2 5 9)	-> 2x5x9 elements cube
		etc.
	"
		
	| bytesize elementSize |
	elementSize _ self sizeForType: anInteger.
	bytesize _ anIntegerArray inject: elementSize into: [:prev :each | prev * each].
	^self type: anInteger dimensions: anIntegerArray bytesize: bytesize
! !

!TF_Tensor class methodsFor: 'instance creation' stamp: 'gera 1/13/2017 13:10:55'!
type: anInteger dimensions: anIntegerArray bytesize: bytesizeInteger
	" I understand dimenssions are:
		#()		-> Scalar
		#(7)		-> Unidimensional array of 7 elements
		#(7 4)		-> 7x4 elements matrix
		#(2 5 9)	-> 2x5x9 elements cube
		etc.
	"
		
	| externalized answer |
	externalized _ Int64Array externalFromArray: anIntegerArray.
		
	answer _ TensorFlowCAPI current
		allocateTensorType: anInteger
		dimensions: externalized
		dimensionsCount: anIntegerArray size
		length: bytesizeInteger.
	
	answer useFinalization.
	^ answer
! !

!TF_Tensor class methodsFor: 'types' stamp: 'gera 1/2/2017 19:43:54'!
typeBFloat16
	^14! !

!TF_Tensor class methodsFor: 'types' stamp: 'gera 1/2/2017 19:42:52'!
typeBool
	^10! !

!TF_Tensor class methodsFor: 'types' stamp: 'gera 1/2/2017 19:42:37'!
typeComplex128
	^18! !

!TF_Tensor class methodsFor: 'types' stamp: 'gera 1/2/2017 19:42:31'!
typeComplex64
	^8! !

!TF_Tensor class methodsFor: 'types' stamp: 'gera 1/2/2017 19:41:29'!
typeDouble
	^2! !

!TF_Tensor class methodsFor: 'types' stamp: 'gera 12/26/2016 09:41:58'!
typeFloat
	^1! !

!TF_Tensor class methodsFor: 'types' stamp: 'gera 1/2/2017 19:44:35'!
typeHalf
	^19! !

!TF_Tensor class methodsFor: 'types' stamp: 'gera 1/2/2017 19:41:54'!
typeInt16
	^5! !

!TF_Tensor class methodsFor: 'types' stamp: 'gera 12/26/2016 11:14:05'!
typeInt32
	^3! !

!TF_Tensor class methodsFor: 'types' stamp: 'gera 12/26/2016 09:42:07'!
typeInt64
	^9! !

!TF_Tensor class methodsFor: 'types' stamp: 'gera 1/2/2017 19:42:06'!
typeInt8
	^6! !

!TF_Tensor class methodsFor: 'types' stamp: 'gera 1/2/2017 19:43:14'!
typeQInt16
	^15! !

!TF_Tensor class methodsFor: 'types' stamp: 'gera 1/2/2017 19:43:30'!
typeQInt32
	^13! !

!TF_Tensor class methodsFor: 'types' stamp: 'gera 1/2/2017 19:43:00'!
typeQInt8
	^11! !

!TF_Tensor class methodsFor: 'types' stamp: 'gera 1/2/2017 19:44:08'!
typeQUInt16
	^16! !

!TF_Tensor class methodsFor: 'types' stamp: 'gera 1/2/2017 19:50:48'!
typeQUInt8
	^12! !

!TF_Tensor class methodsFor: 'types' stamp: 'gera 1/2/2017 19:44:40'!
typeResource
	^20! !

!TF_Tensor class methodsFor: 'types' stamp: 'gera 1/2/2017 19:42:12'!
typeString
	^7! !

!TF_Tensor class methodsFor: 'types' stamp: 'gera 1/2/2017 19:44:19'!
typeUInt16
	^17! !

!TF_Tensor class methodsFor: 'types' stamp: 'gera 1/2/2017 19:41:40'!
typeUInt8
	^4! !

!ExamplePlan methodsFor: 'initialization' stamp: 'JB 1/19/2017 10:23:20'!
initialize
	self initializeGraph! !

!ExamplePlan methodsFor: 'initialization' stamp: 'JB 1/19/2017 10:23:46'!
initializeGraph
	graph _ TF_Graph create useFinalization.
	! !

!ExamplePlan methodsFor: 'running' stamp: 'gera 1/21/2017 04:22:27'!
runOn: inputValues
	| session results |	
	session _ TF_Session onGraph: graph.
	
	results _ session
		runOperations: {operation}
		inputs: inputs
		values: inputValues
		outputs: outputs.
		
	^ results first
	! !

!ExampleNearestNeighborPlan methodsFor: 'initialization' stamp: 'JB 1/19/2017 10:34:53'!
initializeGraph
	| neg add abs axis axisValue distance testing training prediction |
	super initializeGraph.
	
	training _ graph placeholder: 'training' type: TF_Tensor typeFloat.
	testing _ graph placeholder: 'testing' type: TF_Tensor typeFloat.
	axisValue _ TF_Tensor fromInt32: 0.
	axis _ graph const: 'axis' value: axisValue.
	neg _ graph newOperation: 'Neg' named: 'neg' described: [:description |
		description addInput: (testing output: 0)].
	add _ graph add: 'add' described:[:description| 
		description addInput: (neg output: 0).
		description addInput: (training output: 0).
		].
	abs _ graph newOperation: 'Abs' named: 'abs' described: [:description|
		description addInput: (add output: 0)].
	distance _ graph newOperation: 'Sum' named: 'distance' described: [:description|
		description addInput: (abs output: 0).
		description addInput: (axis output: 0).].
	
	prediction _ graph newOperation: 'ArgMin' named: 'argmin' described:[:description|
		description addInput: (distance output: 0).
		description addInput: (axis output: 0)].
	operation _ prediction.
	
	outputs _ {operation output: 0}.
	inputs _ {training input: 0. testing input: 0}.! !

!ExampleNearestNeighborPlan methodsFor: 'evaluating' stamp: 'gera 1/21/2017 04:19:59'!
predict: covariatesTesting from: covariatesTraining
	| result |

 	result _ self runOn: {covariatesTraining. covariatesTesting}.
	^ result allInt64s + 1! !

!ExampleOLSPlan methodsFor: 'initialization' stamp: 'gera 1/21/2017 04:44:38'!
initializeGraph
	
	| x xtx y xty inverse result |
	super initializeGraph.
	
	x _ graph placeholder: 'x' type: TF_Tensor typeFloat.
	y _ graph placeholder: 'y' type: TF_Tensor typeFloat.
	
	xtx _ graph newOperation: 'MatMul' named:'xTx' described:[:description|
		description addInput: (x output: 0).
		description addInput: (x output: 0).
		description at: 'transpose_a' putBoolean: true.
		].
	
	inverse _ graph newOperation: 'MatrixInverse' named:'inv' described:[:description|
		description addInput: (xtx output: 0)].
	
	xty _ graph newOperation: 'MatMul' named:'xTy' described:[:description|
		description addInput: (x output: 0).
		description addInput: (y output: 0).
		description at: 'transpose_a' putBoolean: true.
		].
	
	result _ graph newOperation: 'MatMul' named:'result' described:[:description|
		description addInput: (inverse output: 0).
		description addInput: (xty output: 0).
		].
	
	operation  _ result.
	
	outputs _ {operation output: 0}.
	inputs _ {x input: 0. y input: 0}.! !

!NearestNeighbor methodsFor: 'private-csv' stamp: 'jb 1/15/2017 23:23:00'!
decodeCSV: csvLines graphRun: description
	|  graph output records session results values answer |
		
	graph _ self decodeCSVGraphDefaults: description.
	records _ (graph operationNamed: 'records') input: 0.
	output _ graph operationNamed: 'output'.
	values _ TF_Tensor fromStringArray: csvLines.
	
	session _ TF_Session onGraph: graph.
	results _ session
		runOperations: {output}
		inputs: {records}
		values: {values}
		outputs: {
			(output output: 0).
			(output output: 1).
			(output output: 2).
			(output output: 3).
			(output output: 4)}.
		
	graph delete.
	
	answer _ (1 to: 4) asOrderedCollection collect: [:i |
		(results at: i) allFloats].
	^ answer add: (results at: 5) allInt64s; yourself.! !

!NearestNeighbor methodsFor: 'private-csv' stamp: 'jb 1/15/2017 17:19:41'!
decodeCSVGraphDefaults: anArrayOfTF_Tensors
	| graph records defaults |
	
	graph _ TF_Graph create.
	records _ (graph placeholder: 'records' type: TF_Tensor typeString) output: 0.
	defaults _ Array new: anArrayOfTF_Tensors size.
	
	anArrayOfTF_Tensors withIndexDo: [:each :index |
		| one |
		one _ (graph const: 'default',index printString value: each) output: 0.
		defaults at: index put: one].
	graph newOperation: 'DecodeCSV' named: 'output' described: [:description |
		description addInput: records.
		description addInputs: defaults].
	
	^ graph! !

!NearestNeighbor methodsFor: 'initialization' stamp: 'JB 1/19/2017 10:46:56'!
initialize
	plan _ ExampleNearestNeighborPlan new.! !

!NearestNeighbor methodsFor: 'private-csv' stamp: 'gera 1/18/2017 01:00:47'!
irisDescription
	^{TF_Tensor fromFloats: #(-1.0).
		TF_Tensor fromFloats: #(-1.0).
		TF_Tensor fromFloats: #(-1.0).
		TF_Tensor fromFloats: #(-1.0).
		TF_Tensor fromInt64Array: #(-1)}! !

!NearestNeighbor methodsFor: 'private-csv' stamp: 'gera 1/16/2017 23:00:14'!
irisFile
	^ self class irisFile! !

!NearestNeighbor methodsFor: 'initialization' stamp: 'jb 1/15/2017 23:28:54'!
loadDataSet
	| dataSet description stream |
	stream _ self irisFile.
	stream nextLine.
	dataSet _ self testingAndTrainingFrom: stream.
	description _ self irisDescription.
	^ dataSet collect: [ :csv |
		self
			decodeCSV: csv
			graphRun: description ].! !

!NearestNeighbor methodsFor: 'running' stamp: 'JB 1/19/2017 10:49:44'!
run
	| covariatesTr covariatesTe predictions size dataset testing training |
	dataset _ self loadDataSet.
	training _ dataset first.
	testing _ dataset second.
	covariatesTr _  TF_Tensor fromFloats: training allButLast.
	size _ testing first size.
	predictions _ (1 to: size) collect: [ :i | | covariates index predicted results trueValue |
		covariates _ testing  collect:[:col| {col at: i}].
		covariatesTe _ TF_Tensor fromFloats: covariates allButLast.
		results _ plan predict: covariatesTe from: covariatesTr.
		index _ results first.
		trueValue _ covariates fifth first.
		predicted _ training fifth at: index.
		trueValue = predicted
			ifTrue: [ 1 ]
			ifFalse: [ 0 ]].
	^ predictions.! !

!NearestNeighbor methodsFor: 'sampling' stamp: 'jb 1/15/2017 19:13:00'!
testingAndTrainingFrom: stream
	| p random testing training lines |
	lines _ stream upToEnd lines.
	p _ 0.7.
	training _ OrderedCollection new.
	testing _ OrderedCollection new.
	random _ Random new.
	lines collect: [ :line | | collection |
		line isEmpty ifFalse: [
			collection _ random next < p
				ifTrue: [ training ]
				ifFalse: [ testing ].
			collection add: line ]].
	^ {training. testing}.! !

!NearestNeighbor class methodsFor: 'as yet unclassified' stamp: 'gera 1/16/2017 23:00:53'!
irisFile
	^ '../Machine-Learning/datasets/iris.csv' asFileEntry readStream.! !

!NearestNeighbor class methodsFor: 'instance creation' stamp: 'gera 1/16/2017 23:12:43'!
new
	^ super new initialize! !

!NearestNeighborWithOperations methodsFor: 'graph' stamp: 'gera 1/21/2017 03:37:29'!
initializeGraph
	| axis axisValue distance testing training |
	graph _ TF_Graph create useFinalization.
	
	training _ graph placeholder: 'training' type: TF_Tensor typeFloat.
	testing _ graph placeholder: 'testing' type: TF_Tensor typeFloat.
	axisValue _ TF_Tensor fromInt32: 0.
	axis _ graph const: 'axis' value: axisValue.
	
	distance _ (testing negated + training) abs sumOn: axis.
	prediction _ distance findMinOn: axis.
	
	outputs _ {prediction output: 0}.
	inputs _ {training input: 0. testing input: 0}.
! !

!OLSExample methodsFor: 'initialization' stamp: 'JB 1/19/2017 10:29:16'!
initialize
	plan _ ExampleOLSPlan new.! !

!OLSExample methodsFor: 'running' stamp: 'gera 1/21/2017 04:20:26'!
regress: x on: y
	| result |
	result _ plan runOn: {x. y}.
	^ result allFloats! !
TF_Structure initialize!
