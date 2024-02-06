'From Cuis6.0 [latest update: #6118] on 12 December 2023 at 12:28:07 pm'!
'Description Please enter a description for this package'!
!provides: 'TensorFlow-Kernel' 1 429!
!requires: 'FFI' 1 18 nil!
!requires: 'Statistics' 1 14 nil!
SystemOrganization addCategory: #'TensorFlow-Kernel'!


!classDefinition: #ExternalWordArrayTest category: #'TensorFlow-Kernel'!
TestCase subclass: #ExternalWordArrayTest
	instanceVariableNames: 'library'
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Kernel'!
!classDefinition: 'ExternalWordArrayTest class' category: #'TensorFlow-Kernel'!
ExternalWordArrayTest class
	instanceVariableNames: ''!

!classDefinition: #TensorFlowTestCase category: #'TensorFlow-Kernel'!
TestCase subclass: #TensorFlowTestCase
	instanceVariableNames: ''
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Kernel'!
!classDefinition: 'TensorFlowTestCase class' category: #'TensorFlow-Kernel'!
TensorFlowTestCase class
	instanceVariableNames: ''!

!classDefinition: #TensorFlowCAPISlowTests category: #'TensorFlow-Kernel'!
TensorFlowTestCase subclass: #TensorFlowCAPISlowTests
	instanceVariableNames: 'library'
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Kernel'!
!classDefinition: 'TensorFlowCAPISlowTests class' category: #'TensorFlow-Kernel'!
TensorFlowCAPISlowTests class
	instanceVariableNames: ''!

!classDefinition: #TensorFlowCAPITest category: #'TensorFlow-Kernel'!
TensorFlowTestCase subclass: #TensorFlowCAPITest
	instanceVariableNames: 'library'
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Kernel'!
!classDefinition: 'TensorFlowCAPITest class' category: #'TensorFlow-Kernel'!
TensorFlowCAPITest class
	instanceVariableNames: ''!

!classDefinition: #TensorFlowOperationsTest category: #'TensorFlow-Kernel'!
TensorFlowTestCase subclass: #TensorFlowOperationsTest
	instanceVariableNames: 'library'
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Kernel'!
!classDefinition: 'TensorFlowOperationsTest class' category: #'TensorFlow-Kernel'!
TensorFlowOperationsTest class
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

!classDefinition: #TFStructure category: #'TensorFlow-Kernel'!
ExternalStructure subclass: #TFStructure
	instanceVariableNames: ''
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Kernel'!
!classDefinition: 'TFStructure class' category: #'TensorFlow-Kernel'!
TFStructure class
	instanceVariableNames: ''!

!classDefinition: #ExternalWordArray category: #'TensorFlow-Kernel'!
TFStructure subclass: #ExternalWordArray
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

!classDefinition: #TFOperationPtrArray category: #'TensorFlow-Kernel'!
ExternalWordArray subclass: #TFOperationPtrArray
	instanceVariableNames: ''
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Kernel'!
!classDefinition: 'TFOperationPtrArray class' category: #'TensorFlow-Kernel'!
TFOperationPtrArray class
	instanceVariableNames: ''!

!classDefinition: #TFOutputArray category: #'TensorFlow-Kernel'!
ExternalWordArray subclass: #TFOutputArray
	instanceVariableNames: ''
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Kernel'!
!classDefinition: 'TFOutputArray class' category: #'TensorFlow-Kernel'!
TFOutputArray class
	instanceVariableNames: ''!

!classDefinition: #TFTensorPtrArray category: #'TensorFlow-Kernel'!
ExternalWordArray subclass: #TFTensorPtrArray
	instanceVariableNames: ''
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Kernel'!
!classDefinition: 'TFTensorPtrArray class' category: #'TensorFlow-Kernel'!
TFTensorPtrArray class
	instanceVariableNames: ''!

!classDefinition: #TFAttrMetadata category: #'TensorFlow-Kernel'!
TFStructure subclass: #TFAttrMetadata
	instanceVariableNames: ''
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Kernel'!
!classDefinition: 'TFAttrMetadata class' category: #'TensorFlow-Kernel'!
TFAttrMetadata class
	instanceVariableNames: ''!

!classDefinition: #TFBuffer category: #'TensorFlow-Kernel'!
TFStructure subclass: #TFBuffer
	instanceVariableNames: ''
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Kernel'!
!classDefinition: 'TFBuffer class' category: #'TensorFlow-Kernel'!
TFBuffer class
	instanceVariableNames: ''!

!classDefinition: #TFGraph category: #'TensorFlow-Kernel'!
TFStructure subclass: #TFGraph
	instanceVariableNames: 'context'
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Kernel'!
!classDefinition: 'TFGraph class' category: #'TensorFlow-Kernel'!
TFGraph class
	instanceVariableNames: ''!

!classDefinition: #TFImportGraphDefOptions category: #'TensorFlow-Kernel'!
TFStructure subclass: #TFImportGraphDefOptions
	instanceVariableNames: ''
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Kernel'!
!classDefinition: 'TFImportGraphDefOptions class' category: #'TensorFlow-Kernel'!
TFImportGraphDefOptions class
	instanceVariableNames: ''!

!classDefinition: #TFOperation category: #'TensorFlow-Kernel'!
TFStructure subclass: #TFOperation
	instanceVariableNames: 'graph output'
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Kernel'!
!classDefinition: 'TFOperation class' category: #'TensorFlow-Kernel'!
TFOperation class
	instanceVariableNames: ''!

!classDefinition: #TFOperationDescription category: #'TensorFlow-Kernel'!
TFStructure subclass: #TFOperationDescription
	instanceVariableNames: ''
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Kernel'!
!classDefinition: 'TFOperationDescription class' category: #'TensorFlow-Kernel'!
TFOperationDescription class
	instanceVariableNames: ''!

!classDefinition: #TFOutput category: #'TensorFlow-Kernel'!
TFStructure subclass: #TFOutput
	instanceVariableNames: ''
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Kernel'!
!classDefinition: 'TFOutput class' category: #'TensorFlow-Kernel'!
TFOutput class
	instanceVariableNames: ''!

!classDefinition: #TFInput category: #'TensorFlow-Kernel'!
TFOutput subclass: #TFInput
	instanceVariableNames: ''
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Kernel'!
!classDefinition: 'TFInput class' category: #'TensorFlow-Kernel'!
TFInput class
	instanceVariableNames: ''!

!classDefinition: #TFSession category: #'TensorFlow-Kernel'!
TFStructure subclass: #TFSession
	instanceVariableNames: ''
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Kernel'!
!classDefinition: 'TFSession class' category: #'TensorFlow-Kernel'!
TFSession class
	instanceVariableNames: ''!

!classDefinition: #TFSessionOptions category: #'TensorFlow-Kernel'!
TFStructure subclass: #TFSessionOptions
	instanceVariableNames: ''
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Kernel'!
!classDefinition: 'TFSessionOptions class' category: #'TensorFlow-Kernel'!
TFSessionOptions class
	instanceVariableNames: ''!

!classDefinition: #TFStatus category: #'TensorFlow-Kernel'!
TFStructure subclass: #TFStatus
	instanceVariableNames: ''
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Kernel'!
!classDefinition: 'TFStatus class' category: #'TensorFlow-Kernel'!
TFStatus class
	instanceVariableNames: ''!

!classDefinition: #TFTensor category: #'TensorFlow-Kernel'!
TFStructure subclass: #TFTensor
	instanceVariableNames: ''
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Kernel'!
!classDefinition: 'TFTensor class' category: #'TensorFlow-Kernel'!
TFTensor class
	instanceVariableNames: ''!


!ExternalWordArrayTest methodsFor: 'testing' stamp: 'gera 12/29/2016 14:51:01'!
testArray: anExternalWordArraySubclass sampleElement: anObject
	| array size  |
	size := 4.
	array := anExternalWordArraySubclass new: 4.
	self assert: size * Smalltalk wordSize = array getHandle size.
	self assert: (array first isKindOf: anObject class).
	1 to: size do: [:index |
		self assert: ((array at: index) isKindOf: anObject class)].
! !

!ExternalWordArrayTest methodsFor: 'testing' stamp: 'gera 1/13/2017 12:17:48'!
testArrayAtPutFor: anExternalWordArraySubclass elementSample: anObject
	| array size one  |
	size := 4.
	array := anExternalWordArraySubclass new: 4.
	one := array first.
	
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

!ExternalWordArrayTest methodsFor: 'testing' stamp: 'gera 7/3/2019 03:06:39'!
testExternalFromArrayOperation
	| op1 op2 array |
	op1 := TFOperation fromHandle: (ExternalAddress new + 11111).
	op2 := TFOperation fromHandle: (ExternalAddress new + 22222).
	array := TFOperationPtrArray externalFromArray: (Array with: op1 with: op2).

	self assert: array first getHandle = op1 getHandle.
	self assert: (array at: 2) getHandle = op2 getHandle.
! !

!ExternalWordArrayTest methodsFor: 'testing' stamp: 'gera 7/3/2019 03:30:56'!
testExternalFromArrayOutput
	| o1 o2 array |
	o1 := TFOutput new index: 16r40302010.
	o2 := TFOutput new index: 16r76543210.
	array := TFOutputArray externalFromArray: (Array with: o1 with: o2).

	self assert: array first index = o1 index.
	self assert: (array at: 2) index = o2 index.
! !

!ExternalWordArrayTest methodsFor: 'testing' stamp: 'gera 7/3/2019 03:10:27'!
testExternalFromArrayTensor
	| t1 t2 array |
	t1 := TFTensor fromInt64: 16r8070605040302010.
	t2 := TFTensor fromInt64: 16rFEDCBA9876543210.
	array := TFTensorPtrArray externalFromArray: (Array with: t1 with: t2).
	
	self assert: array first getHandle = t1 getHandle.
	self assert: (array at: 2) getHandle = t2 getHandle.
	! !

!ExternalWordArrayTest methodsFor: 'testing' stamp: 'gera 7/3/2019 03:06:39'!
testOperationArrayAtPut
	| graph sample |
	graph := TFGraph create.
	sample := graph placeholder: 'var' type: TFTensor typeInt64.
	
	self testArrayAtPutFor: TFOperationPtrArray elementSample: sample.
! !

!ExternalWordArrayTest methodsFor: 'testing' stamp: 'gera 7/3/2019 03:06:39'!
testOperationArrayElementsOfRightClass
	^ self testArray: TFOperationPtrArray sampleElement: TFOperation new.
! !

!ExternalWordArrayTest methodsFor: 'testing' stamp: 'gera 7/3/2019 03:30:56'!
testOutputArrayAtPut
	| array size one other operation |
	size := 4.
	array := TFOutputArray externalNew: size.
	operation := TFOperation fromHandle: ExternalAddress new + 16rF1F2F3F4.
	
	one := TFOutput new.
	one index: 16r75F6F7F8.
	one operation: operation.

	other := array first.
	array first: one.

	self deny: one getHandle = other getHandle.	
	
	self assert: one index = other index.
	self assert: one operation = other operation.
	
	operation := TFOperation fromHandle: ExternalAddress new + 16r12345678.
	one index: 16r75F6F7F8.
	one operation: operation.

	1 to: size do: [:index |
		other := array at: index.
		array at: index put: one.

		self deny: one getHandle = other getHandle.	
		self assert: one index = other index.
		self assert: one operation = other operation.
	]! !

!ExternalWordArrayTest methodsFor: 'testing' stamp: 'gera 7/3/2019 03:30:56'!
testOutputArrayElementsAreAliases
	| array size one other |
	size := 4.
	array := TFOutputArray externalNew: size.

	one := array first.
	other := array at: 1.
	
	one index: 12345678.
	self assert: one index = 12345678.
	self assert: other index = 12345678.

	other index: 77665544.
	self assert: one index = 77665544.
	self assert: other index = 77665544.

	1 to: size do: [:index |
		one := array at: index.
		other := array at: index.

		one index: 12348765.
		self assert: one index = 12348765.
		self assert: other index = 12348765.

		other index: 77222244.
		self assert: one index = 77222244.
		self assert: other index = 77222244].
! !

!ExternalWordArrayTest methodsFor: 'testing' stamp: 'gera 7/3/2019 03:30:56'!
testOutputArrayElementsOfRightClass
	| array size  |
	size := 4.
	array := TFOutputArray externalNew: size.
	self assert: (array first isKindOf: TFOutput).
	1 to: size do: [:index |
		self assert: ((array at: index) isKindOf: TFOutput)].
! !

!ExternalWordArrayTest methodsFor: 'testing' stamp: 'gera 7/3/2019 03:10:27'!
testTensorArrayAtPut
	| sample |
	sample := TFTensor fromInt64: 16r12341234.
	
	self testArrayAtPutFor: TFTensorPtrArray elementSample: sample.! !

!ExternalWordArrayTest methodsFor: 'testing' stamp: 'gera 7/3/2019 03:10:27'!
testTensorArrayElementsOfRightClass
	^ self testArray: TFTensorPtrArray sampleElement: TFTensor new.! !

!ExternalWordArrayTest methodsFor: 'testing' stamp: 'gera 12/25/2016 20:54:13'!
testUIn64ArrayfromArray
	self testUIn64ArrayfromArray: #(1).
	self testUIn64ArrayfromArray: #(16r12345678 2 16r76543210 4 16rDEADBEAF 6 16rFFFFFFFF).
! !

!ExternalWordArrayTest methodsFor: 'testing' stamp: 'gera 1/13/2017 13:08:30'!
testUIn64ArrayfromArray: template
	| array |
	array := Int64Array externalFromArray: template.
	
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

!TensorFlowTestCase methodsFor: 'asserting' stamp: 'gera 6/29/2019 13:16:30'!
assert: aNumber closeTo: anotherNumber
	^ self
		assert: (self number: aNumber isCloseTo: anotherNumber)
		description: (self comparingStringBetween: aNumber and: anotherNumber)
! !

!TensorFlowTestCase methodsFor: 'asserting' stamp: 'gera 6/29/2019 13:12:25'!
assertAll: expectedArray closeTo: actualArray
	expectedArray with: actualArray do: [:expected :actual |
		self assert: expected closeTo: actual]! !

!TensorFlowTestCase methodsFor: 'asserting' stamp: 'gera 6/29/2019 13:15:53'!
comparingStringDifferentBetween: expected and: actual
	^ String streamContents: [:stream |
		stream
			nextPutAll: 'Expected something different to ';
			nextPutAll: (expected printStringLimitedTo: 50);
			nextPutAll: ' but was ';
			nextPutAll: (actual printStringLimitedTo: 50);
			nextPutAll: '.'
		]! !

!TensorFlowTestCase methodsFor: 'asserting' stamp: 'gera 6/29/2019 13:18:11'!
deny: aNumber closeTo: anotherNumber
	^ self
		deny: (self number: aNumber isCloseTo: anotherNumber)
		description: (self comparingStringDifferentBetween: aNumber and: anotherNumber)
! !

!TensorFlowTestCase methodsFor: 'asserting' stamp: 'gera 6/29/2019 13:21:52'!
number: aNumber isCloseTo: anotherNumber
 	"are these two numbers close?"
	self = 0.0 ifTrue: [^anotherNumber abs < 0.0001].
	anotherNumber = 0 ifTrue: [^aNumber abs < 0.0001].
	^aNumber = anotherNumber asFloat
		or: [(aNumber - anotherNumber) abs / (aNumber abs max: anotherNumber abs) < 0.0001]! !

!TensorFlowCAPISlowTests methodsFor: 'testing tensor' stamp: 'gera 1/29/2017 02:22:17'!
mulGraphTwoInputsInt64ConstTensorDeleted
	^ TensorFlowCAPITest new mulGraphTwoInputsInt64ConstTensorDeleted! !

!TensorFlowCAPISlowTests methodsFor: 'testing tensor' stamp: 'gera 7/3/2019 03:35:50'!
testRunGraphMulTwoInputsConstTensorDeleted
	| graph inputs inputValues mul output session results |
	graph := self mulGraphTwoInputsInt64ConstTensorDeleted.
	
	inputs := Array
		with: ((graph operationNamed: 'in1') input: 0)
		with: ((graph operationNamed: 'in2') input: 0).
		
	inputValues := Array
		with: (TFTensor fromInt64: 16r23)
		with: (TFTensor fromInt64: 16r24).

	Smalltalk garbageCollect.
	(TFTensor fromInt64: 16r1234123412341234) useFinalization.
	Smalltalk garbageCollect.
	
	mul := graph operationNamed: 'mul2'.
	output := mul output: 0.
	session := TFSession on: graph.
	results := session
		runOperations: (Array with: mul)
		inputs: inputs
		values: inputValues
		outputs: (Array with: output).
	
	self deny: results isNull.
	self deny: results first isNull.
	self deny: results first data isNull.
	self
		assert: (results first data getHandle signedLongLongAt: 1)
		equals: ((16r2121212121212121 * 16r23 * 16r24) bitAnd: 16rFFFFFFFFFFFFFFFF).

	graph delete.
	results first delete.
! !

!TensorFlowCAPISlowTests methodsFor: 'testing tensor' stamp: 'gera 1/29/2017 02:21:07'!
testRunGraphMulTwoInputsConstTensorDeletedManyTimes
	20 timesRepeat: [self testRunGraphMulTwoInputsConstTensorDeleted]! !

!TensorFlowCAPISlowTests methodsFor: 'testing tensor' stamp: 'gera 6/30/2019 00:45:26'!
testTensorFinalizationReleasesExternalMemory
	" WeakArray restartFinalizationProcess "
	| handles template tries total |
	total := 0.
	tries := 20.
	handles := Set new.

	template := {String new: 10. String new: 100. String new: 1000}.
	
	20 timesRepeat: [
		total := total + tries.
		handles addAll: ((1 to: tries) collect: [:i |
			Smalltalk garbageCollect.
			(TFTensor fromStringArray: template) getHandle]).
		handles size < total ifTrue: [
			^ self]
	].

	self assert: (handles size) < total.
! !

!TensorFlowCAPISlowTests methodsFor: 'testing graph' stamp: 'gera 7/3/2019 00:34:29'!
testGraphFinalizationReleasesExternalMemory
	" WeakArray restartFinalizationProcess "
	| tries total handles |
	total := 0.
	tries := 10.
	handles := Set new.

	20 timesRepeat: [
		total := total + tries.
		handles addAll: ((1 to: tries) collect: [:i |
			Smalltalk garbageCollect.
			TFGraph create useFinalization getHandle]).
		handles size < total ifTrue: [
			^ self]
	].
	
	self assert: (handles size) < total.
! !

!TensorFlowCAPISlowTests methodsFor: 'testing session' stamp: 'gera 7/3/2019 03:35:50'!
testSessionFinalizationReleasesExternalMemory 
	" WeakArray restartFinalizationProcess "
	| tries total handles graph |
	total := 0.
	tries := 20.
	graph := TFGraph create.
	handles := Set new.
	
	20 timesRepeat: [
		total := total + tries.
		handles addAll: ((1 to: tries) collect: [:i |
			Smalltalk garbageCollect.
			(TFSession on: graph) getHandle]).
		handles size < total ifTrue: [
			^ self]].
	
	self assert: (handles size) < total.
! !

!TensorFlowCAPISlowTests methodsFor: 'testing options' stamp: 'gera 7/3/2019 03:35:40'!
testSessionOptionFinalizationReleasesExternalMemory 
	" WeakArray restartFinalizationProcess "
	| tries total handles |
	total := 0.
	tries := 10.
	handles := Set new.
	
	20 timesRepeat: [
		total := total + tries.
		handles addAll: ((1 to: tries) collect: [:i |
			Smalltalk garbageCollect.
			(TFSessionOptions create) getHandle]).
		handles size < total ifTrue: [
			^ self]].
	
	self assert: (handles size) < total.
! !

!TensorFlowCAPISlowTests methodsFor: 'testing status' stamp: 'gera 7/3/2019 03:33:19'!
testStatusFinalizationReleasesExternalMemory
	" WeakArray restartFinalizationProcess "
	| handles |
	handles := (1 to: 11) collect: [:i |
		Smalltalk garbageCollect.
		TFStatus create getHandle].

	self assert: (handles asSet size) < 11.! !

!TensorFlowCAPITest methodsFor: 'graphs' stamp: 'gera 7/3/2019 00:34:29'!
addGraphTwoInputsInt64
	| graph in1 in2  |
	graph := TFGraph create.
	
	in1 := graph placeholder: 'in1' type: TFTensor typeInt64.
	in2 := graph placeholder: 'in2' type: TFTensor typeInt64.
	graph add: 'add' described: [:description |
		description addInput: (in1 output: 0).
		description addInput: (in2 output: 0)].
	^ graph! !

!TensorFlowCAPITest methodsFor: 'graphs' stamp: 'gera 7/3/2019 00:34:29'!
asStringGraphType: type
	| graph in |
	graph := TFGraph create.
	in := graph placeholder: 'in' type: type.
	graph asString: 'out' described: [:description |
		description addInput: (in output: 0)].
	^ graph! !

!TensorFlowCAPITest methodsFor: 'graphs' stamp: 'gera 7/3/2019 00:34:29'!
concatGraphInputList
	| graph in1 in2 concat dimension dimensionValue inputs |
	graph := TFGraph create.
	dimensionValue := TFTensor fromInt32: 0.
	dimension := graph const: 'const' value: dimensionValue.
	
	in1 := graph placeholder: 'in1' type: TFTensor typeInt64.
	in2 := graph placeholder: 'in2' type: TFTensor typeInt64.
	inputs := Array with: (in1 output: 0) with: (in2 output: 0).
	concat := graph concat: 'concat' described: [:description |
		description addInput: (dimension output: 0).
		description addInputs: inputs.
		description at: 'N' putInt: 2.
		description at: 'T' putType: TFTensor typeInt64.
	].
	concat.
	^ graph! !

!TensorFlowCAPITest methodsFor: 'graphs' stamp: 'gera 7/3/2019 00:34:29'!
concatGraphInputListNoSizeNoType
	| graph in1 in2 concat dimension dimensionValue inputs |
	graph := TFGraph create.
	dimensionValue := TFTensor fromInt32: 0.
	dimension := graph const: 'const' value: dimensionValue.
	
	in1 := graph placeholder: 'in1' type: TFTensor typeInt64.
	in2 := graph placeholder: 'in2' type: TFTensor typeInt64.
	inputs := Array with: (in1 output: 0) with: (in2 output: 0).
	concat := graph concat: 'concat' described: [:description |
		description addInput: (dimension output: 0).
		description addInputs: inputs.
	].
	concat.
	^ graph! !

!TensorFlowCAPITest methodsFor: 'graphs' stamp: 'gera 7/3/2019 00:34:29'!
concatGraphInputListWrongSize
	| graph in1 in2 concat dimension dimensionValue inputs |
	graph := TFGraph create.
	dimensionValue := TFTensor fromInt32: 0.
	dimension := graph const: 'const' value: dimensionValue.
	
	in1 := graph placeholder: 'in1' type: TFTensor typeInt64.
	in2 := graph placeholder: 'in2' type: TFTensor typeInt64.
	inputs := Array with: (in1 output: 0) with: (in2 output: 0).
	concat := graph concat: 'concat' described: [:description |
		description addInput: (dimension output: 0).
		description addInputs: inputs.
		description at: 'N' putInt: 0.
	].
	concat.
	^ graph! !

!TensorFlowCAPITest methodsFor: 'graphs' stamp: 'gera 7/3/2019 00:34:29'!
concatGraphInputListWrongType
	| graph in1 in2 concat dimension dimensionValue inputs |
	graph := TFGraph create.
	dimensionValue := TFTensor fromInt32: 0.
	dimension := graph const: 'const' value: dimensionValue.
	
	in1 := graph placeholder: 'in1' type: TFTensor typeInt64.
	in2 := graph placeholder: 'in2' type: TFTensor typeInt64.
	inputs := Array with: (in1 output: 0) with: (in2 output: 0).
	concat := graph concat: 'concat' described: [:description |
		description addInput: (dimension output: 0).
		description addInputs: inputs.
		description at: 'T' putType: TFTensor typeInt32.
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

!TensorFlowCAPITest methodsFor: 'graphs' stamp: 'gera 7/3/2019 00:34:29'!
constant2x2FloatGraphFromDef
	^ TFGraph fromString: self constant2x2FloatGraphDef! !

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

!TensorFlowCAPITest methodsFor: 'graphs' stamp: 'gera 7/3/2019 00:34:29'!
constantFloatGraphFromDef
	^ TFGraph fromString: self constantFloatGraphDef! !

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

!TensorFlowCAPITest methodsFor: 'graphs' stamp: 'gera 7/3/2019 00:34:29'!
constantInt32GraphFromDef
	^ TFGraph fromString: self constantInt32GraphDef! !

!TensorFlowCAPITest methodsFor: 'graphs' stamp: 'gera 7/3/2019 00:34:29'!
constantInt64Graph
	| graph operation constant |
	graph := TFGraph create.
	constant := TFTensor fromInt64: 16r4242424242424242.
	
	self shouldnt: [
		operation := graph const: 'a' value: constant
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

!TensorFlowCAPITest methodsFor: 'graphs' stamp: 'gera 7/3/2019 00:34:29'!
constantInt64GraphFromDef
	^ TFGraph fromString: self constantInt64GraphDef! !

!TensorFlowCAPITest methodsFor: 'graphs' stamp: 'gera 7/3/2019 00:34:29'!
decodeCSVGraphDefaults: anArrayOfTFTensors
	| graph records defaults |
	
	graph := TFGraph create.
	records := (graph placeholder: 'records' type: TFTensor typeString) output: 0.
	defaults := Array new: anArrayOfTFTensors size.
	
	anArrayOfTFTensors withIndexDo: [:each :index |
		| one |
		one := (graph const: 'default',index printString value: each) output: 0.
		defaults at: index put: one].
	graph newOperation: 'DecodeCSV' named: 'output' described: [:description |
		description addInput: records.
		description addInputs: defaults].
	
	^ graph! !

!TensorFlowCAPITest methodsFor: 'graphs' stamp: 'gera 7/3/2019 00:34:29'!
emptyGraph
	^ TFGraph fromString: self emptyGraphDef.
	! !

!TensorFlowCAPITest methodsFor: 'graphs' stamp: 'gera 12/22/2016 01:51:30'!
emptyGraphDef
	" This GraphDef corresponds to an Empty Graph (no operations), saved as ProtoBuf "
	^ #[16r0A 16r0C 16r0A 16r04 16r69 16r6E 16r69 16r74 16r12 16r04 16r4E 16r6F 16r4F 16r70 16r22 16r02 16r08 16r0F] asString! !

!TensorFlowCAPITest methodsFor: 'graphs' stamp: 'gera 1/6/2017 10:48:25'!
floatAsStringGraph
	| graph const |
	graph := self constantFloatGraphFromDef.
	const := graph operationNamed: 'a'.
	graph asString: 'output' described: [:description |
		description addInput: (const output: 0)].
	^ graph! !

!TensorFlowCAPITest methodsFor: 'graphs' stamp: 'gera 7/3/2019 00:34:29'!
mulGraphOneInputInt64
	| graph constant const in |
	graph := TFGraph create.
	constant := TFTensor fromInt64: 16r0606060606060606.
	
	in := graph placeholder: 'in' type: constant type.
	const := graph const: 'const' value: constant.
	graph mul: 'mul' described: [:description |
		description addInput: (in output: 0).
		description addInput: (const output: 0)].
	
	^ graph! !

!TensorFlowCAPITest methodsFor: 'graphs' stamp: 'gera 7/3/2019 00:34:29'!
mulGraphTwoInputsInt64
	| graph constant const in1 in2 mul1 |
	graph := TFGraph create.
	constant := TFTensor fromInt64: 16r0101010101010101.
	
	in1 := graph placeholder: 'in1' type: constant type.
	in2 := graph placeholder: 'in2' type: constant type.
	const := graph const: 'const' value: constant.
	mul1 := graph mul: 'mul1' described: [:description |
		description addInput: (const output: 0).
		description addInput: (in1 output: 0)].
	graph mul: 'mul2' described: [:description |
		description addInput: (mul1 output: 0).
		description addInput: (in2 output: 0)].
	
	^ graph! !

!TensorFlowCAPITest methodsFor: 'graphs' stamp: 'gera 7/3/2019 00:34:29'!
mulGraphTwoInputsInt64ConstTensorDeleted
	| graph constant const in1 in2 mul1 |
	graph := TFGraph create.
	constant := TFTensor fromInt64: 16r2121212121212121.
	
	in1 := graph placeholder: 'in1' type: constant type.
	in2 := graph placeholder: 'in2' type: constant type.
	const := graph const: 'const' value: constant.

	constant delete.	
	constant := TFTensor fromInt64: 16r2222222222222222.
	constant delete.	

	mul1 := graph mul: 'mul1' described: [:description |
		description addInput: (const output: 0).
		description addInput: (in1 output: 0)].

	graph mul: 'mul2' described: [:description |
		description addInput: (mul1 output: 0).
		description addInput: (in2 output: 0)].
	
	^ graph! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 6/30/2019 00:45:28'!
assertElementsOf: tensorArray are: allElementsArray
	self assert: allElementsArray equals: (TFTensor elementsOf: tensorArray)! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 6/30/2019 00:45:27'!
assertRankOf: aMultidimensionalTensor is: anInteger
	| rank |
	rank := TFTensor rankOf: aMultidimensionalTensor.
	self
		assert: rank = anInteger
		description: 'The rank is ', rank printString, ' and should have been ', anInteger printString! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 6/30/2019 00:45:25'!
assertShapeOf: aMultidimensionalTensor is: anArray
	| shape |
	shape := TFTensor shapeOf: aMultidimensionalTensor.
	self
		assert: shape = anArray
		description: 'The shape is ', shape printString, ' and should have been ', anArray printString! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 6/30/2019 00:45:28'!
assertSizeOf: aMultidimensionalTensor is: anInteger
	| size |
	size := TFTensor sizeOf: aMultidimensionalTensor.
	self
		assert: size = anInteger
		description: 'The size is ', size printString, ' and should have been ', anInteger printString! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 2/15/2017 01:00:06'!
assertTensor: aTFTensor elementsEquals: tensorArray
	self assert: aTFTensor allElements equals: tensorArray! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 2/15/2017 01:27:15'!
assertTensor: aTFTensor streamEquals: tensorArray
	| strm |
	strm := aTFTensor asStream.
	tensorArray do: [:each |
		self assert: each equals: strm next].
! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 7/3/2019 03:35:50'!
get2x2FloatFromGraphDef
	| graph session const result |
	graph := self constant2x2FloatGraphFromDef.
	
	const := (graph operationNamed: 'a') output: 0.
	session := TFSession on: graph.
	result := session runOutput: const.
	
	^ result! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 2/15/2017 01:44:48'!
testArrayFromStream
	| t template array |
	t := 1.0 asTensor.
	template := #(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17) readStream.
	
	array := t arrayFromStream: template reset shape: #(10).
	self assert: #(1 2 3 4 5 6 7 8 9 10) equals: array.
	
	array := t arrayFromStream: template reset shape: #(2 8).
	self assert: #((1 2 3 4 5 6 7 8) (9 10 11 12 13 14 15 16)) equals: array.
	
	array := t arrayFromStream: template reset shape: #(2 4 2).
	self assert: #(((1 2) (3 4) (5 6) (7 8)) ((9 10) (11 12) (13 14) (15 16))) equals: array.
	! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'JB 2/15/2017 23:54:30'!
testAsBooleanTensor
	self testAsBooleanTensor: true shape: #().
	self testAsBooleanTensor: #(true false true false) shape: #(4).
	self testAsBooleanTensor: #((true false true false) (false true false true)) shape: #(2 4).
! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 6/30/2019 00:45:26'!
testAsBooleanTensor: tensorArray shape: shapeArray
	| tensor index bools |
	tensor := tensorArray asBooleanTensor.
	
	self assert: tensor shape equals: shapeArray.
	
	index := 1.
	bools := tensor allElements.
	TFTensor elementsOf: tensorArray do: [:each |
		self assert: (bools at: index) equals: each.
		index := index + 1].

! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 1/30/2017 18:09:11'!
testAsFloatTensor
	self testAsFloatTensor: 1 shape: #().
	self testAsFloatTensor: #(1 2 3 4) shape: #(4).
	self testAsFloatTensor: #((1 2 3 4) (3.14 1.71 2.12 -7.8)) shape: #(2 4).
! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 6/30/2019 00:45:28'!
testAsFloatTensor: tensorArray shape: shapeArray
	| tensor index floats |
	tensor := tensorArray asFloatTensor.
	
	index := 1.
	floats := tensor allFloats.
	TFTensor elementsOf: tensorArray do: [:each |
		self assert: (floats at: index) closeTo: each.
		index := index + 1].

	self assert: tensor shape equals: shapeArray! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 1/31/2017 16:50:15'!
testAsInt32Tensor
	self testAsInt32Tensor: 1 shape: #().
	self testAsInt32Tensor: #(1 2 3 4) shape: #(4).
	self testAsInt32Tensor: #((1 2 3 4) (-314 171 -212 -78)) shape: #(2 4).
! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 6/30/2019 00:45:26'!
testAsInt32Tensor: tensorArray shape: shapeArray
	| tensor index ints |
	tensor := tensorArray asInt32Tensor.
	
	self assert: tensor shape equals: shapeArray.
	
	index := 1.
	ints := tensor allInt32s.
	TFTensor elementsOf: tensorArray do: [:each |
		self assert: (ints at: index) equals: each.
		index := index + 1].

! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 2/15/2017 23:11:03'!
testAsInt64Tensor
	self testAsInt64Tensor: 1 shape: #().
	self testAsInt64Tensor: #(1 2 3 4) shape: #(4).
	self testAsInt64Tensor: #((1 2 3 4) (-314 171 -212 -78)) shape: #(2 4).
! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 6/30/2019 00:45:26'!
testAsInt64Tensor: tensorArray shape: shapeArray
	| tensor index ints |
	tensor := tensorArray asInt64Tensor.
	
	self assert: tensor shape equals: shapeArray.
	
	index := 1.
	ints := tensor allInt64s.
	TFTensor elementsOf: tensorArray do: [:each |
		self assert: (ints at: index) equals: each.
		index := index + 1].

! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 6/30/2019 00:45:27'!
testElementsOf: tensorArray sum: aNumber
	| sum |
	sum := 0.
	TFTensor elementsOf: tensorArray do: [:each |
		sum := sum + each].
	
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

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 1/18/2017 15:19:54'!
testGet2x2FloatFromGraphDef
	| templates consts |
	templates := #(-1.1 -2.1 -1.2 -2.2).
	consts := self get2x2FloatFromGraphDef allFloats.
	
	templates with: consts do: [:temp :const |
		self assert: temp closeTo: const]! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 1/13/2017 19:50:59'!
testRanks
	self assertRankOf: -13123213 is: 0.
	self assertRankOf: #(123 123 123 123) is: 1.
	self assertRankOf: #((1 2 3) (4 5 6) (7 8 9)) is: 2.
	self assertRankOf: #(((1) (2) (3)) ((4) (5) (6)) ((7) (8) (9))) is: 3.! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 1/13/2017 19:56:50'!
testShape
	self assertShapeOf: -13123213 is: #().
	self assertShapeOf: #(123 123 123 123) is: #(4).
	self assertShapeOf: #((1 2 3) (4 5 6) (7 8 9) (10 11 12)) is: #(4 3).
	self assertShapeOf: #(((1) (2) (3)) ((4) (5) (6)) ((7) (8) (9))) is: #(3 3 1).! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 1/18/2017 00:05:31'!
testSizes
	self assertSizeOf: -13123213 is: 1.
	self assertSizeOf: #(123 123 123 123) is: 4.
	self assertSizeOf: #((1 2 3) (4 5 6) (7 8 9) (10 11 12)) is: 4*3.
	self assertSizeOf: #(((1) (2) (3)) ((4) (5) (6)) ((7) (8) (9))) is: 3*3*1.! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 2/15/2017 01:11:29'!
testTensorAllElements
	self assertTensor: -13123213 asInt32Tensor elementsEquals: #(-13123213).
	self assertTensor: #(123 123 123 123) asInt32Tensor elementsEquals: #(123 123 123 123).
	self assertTensor: #((1 2 3) (4 5 6) (7 8 9) (10 11 12)) asFloatTensor elementsEquals: #(1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0).
	self
		assertTensor: #(((1 100) (2 100) (3 100)) ((4 100) (5 100) (6 100)) ((7 100) (8 100) (9 100))) asFloatTensor
		elementsEquals: #(1.0 100.0 2.0 100.0 3.0 100.0 4.0 100.0 5.0 100.0 6.0 100.0 7.0 100.0 8.0 100.0 9.0 100.0).! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 1/13/2017 10:17:38'!
testTensorAllStrings
	| result strings expected |
	
	expected := #('0.420000').
	result := self runFloatAsStringGraph.
	strings := result allStrings.
	
	self assert: strings equals: expected.
	
	result delete.
! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 7/3/2019 03:35:50'!
testTensorArrayNumbersAt
	| graph inputValues inputs mul numbers output results session |
	graph := self mulGraphTwoInputsInt64.
	
	inputs := Array
		with: ((graph operationNamed: 'in1') input: 0)
		with: ((graph operationNamed: 'in2') input: 0).
	inputValues := Array
		with: (TFTensor fromInt64: 6)
		with: (TFTensor fromInt64: 11).
	mul := graph operationNamed: 'mul2'.
	output := mul output: 0.
	session := TFSession on: graph.
	results := session
		runOperations: (Array with: mul)
		inputs: inputs
		values: inputValues
		outputs: (Array with: output).
		
	numbers := results numbersAt: 1.
	self assert: numbers equals: 16r4242424242424242
	! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 6/30/2019 00:45:25'!
testTensorAsNumbers
	| tensor template array |
	template := #(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16).
	
	tensor := TFTensor fromFloats: template shape: #(16).
	array := tensor asNumbers.
	self assert: template equals: array.
	
	tensor := TFTensor fromFloats: template shape: #(2 8).
	array := tensor asNumbers.
	self assert: #((1 2 3 4 5 6 7 8) (9 10 11 12 13 14 15 16)) equals: array.
	
	tensor := TFTensor fromFloats: template shape: #(2 4 2).
	array := tensor asNumbers.
	self assert: #(((1 2) (3 4) (5 6) (7 8)) ((9 10) (11 12) (13 14) (15 16))) equals: array.
! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 6/30/2019 00:45:26'!
testTensorAsNumbersFloats
	| tensor template array |
	template := #(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16).
	
	tensor := TFTensor fromFloats: template shape: #(16).
	array := tensor asNumbers.
	self assert: #(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16) equals: array.
	
	tensor := TFTensor fromFloats: template shape: #(2 8).
	array := tensor asNumbers.
	self assert: #((1 2 3 4 5 6 7 8) (9 10 11 12 13 14 15 16)) equals: array.
	
	tensor := TFTensor fromFloats: template shape: #(2 4 2).
	array := tensor asNumbers.
	self assert: #(((1 2) (3 4) (5 6) (7 8)) ((9 10) (11 12) (13 14) (15 16))) equals: array.
	! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 6/30/2019 00:45:27'!
testTensorAsNumbersInt32
	| tensor template array |
	template := #(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16).
	
	tensor := TFTensor fromInt32s: template shape: #(16).
	array := tensor asNumbers.
	self assert: #(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16) equals: array.
	
	tensor := TFTensor fromInt32s: template shape: #(2 8).
	array := tensor asNumbers.
	self assert: #((1 2 3 4 5 6 7 8) (9 10 11 12 13 14 15 16)) equals: array.
	
	tensor := TFTensor fromInt32s: template shape: #(2 4 2).
	array := tensor asNumbers.
	self assert: #(((1 2) (3 4) (5 6) (7 8)) ((9 10) (11 12) (13 14) (15 16))) equals: array.
	! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 2/15/2017 08:05:47'!
testTensorAsNumbersRank0
	self assert: 1 equals: 1 asInt32Tensor asNumbers.
	self assert: 1.0 equals: 1.0 asTensor asNumbers.
! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 7/3/2019 03:35:50'!
testTensorAsOperation
	| graph a b result |
	graph := TFGraph create.
	a := graph const: 3.14 asTensor.
	b := a + 1.234 asTensor.
	
	result := (TFSession on: graph) runOutput: b output.
	
	self assert: 3.14+1.234 closeTo: result asNumbers.! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 2/15/2017 01:28:38'!
testTensorAsStream
	self assertTensor: -13123213 asInt32Tensor streamEquals: #(-13123213).
	self assertTensor: #(123 123 123 123) asInt32Tensor streamEquals: #(123 123 123 123).
	self assertTensor: #((1 2 3) (4 5 6) (7 8 9) (10 11 12)) asFloatTensor streamEquals: #(1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0).
	self
		assertTensor: #(((1 100) (2 100) (3 100)) ((4 100) (5 100) (6 100)) ((7 100) (8 100) (9 100))) asFloatTensor
		streamEquals: #(1.0 100.0 2.0 100.0 3.0 100.0 4.0 100.0 5.0 100.0 6.0 100.0 7.0 100.0 8.0 100.0 9.0 100.0).! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 6/30/2019 00:45:25'!
testTensorByteSize
	| tensor |
	tensor := TFTensor
		type: TFTensor typeInt64
		shape: #(2 3).

	self assert: tensor byteSize equals: (8*2*3)! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 6/30/2019 00:45:27'!
testTensorData
	| tensor |
	tensor := TFTensor
		type: TFTensor typeInt64
		shape: #(2 3).
	self assert: tensor rank equals: 2.
	self deny: tensor data getHandle asInteger = 0.
	self deny: tensor data getHandle isNil.
! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 2/18/2017 17:04:47'!
testTensorElementsOf
	self
		assertElementsOf: -13123213
		are: #(-13123213).
	self
		assertElementsOf: #(123 123 123 123)
		are: #(123 123 123 123).
	self
		assertElementsOf: #((1 2 3) (4 5 6) (7 8 9) (10 11 12))
		are: (1 to: 12) asArray.
	self
		assertElementsOf: #(((1) (2) (3)) ((4) (5) (6)) ((7) (8) (9)))
		are: (1 to: 9) asArray.
	self
		assertElementsOf: #(((1 100) (2 100) (3 100)) ((4 100) (5 100) (6 100)) ((7 100) (8 100) (9 100)))
		are: #(1 100 2 100 3 100 4 100 5 100 6 100 7 100 8 100 9 100)
		
! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 6/30/2019 00:45:27'!
testTensorFromDoublesOutOfRange
	| tensor template values |
	template := #(1.23456 0.0 -1.234567 1.0e39 1.0e-50 1.0e309 1.0e-324) copy.
	tensor := TFTensor fromDoubles: template.
	
	template at: 6 put: Float infinity.
	
	values := tensor allElements.
		
	self assert: tensor shape equals: #(7).
	self assert: tensor size equals: 7.
	self assert: tensor byteSize equals: (7*8).
	template with: values do: [:expected :actual |
		self assert: expected closeTo: actual].
	self assert: 0.0 equals: values last.
! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 6/30/2019 00:45:26'!
testTensorFromDoublesOutOfRangeForFloats
	| tensor template values |
	template := #(1.23456 0.0 -1.234567 3.402824e38 1.175494351e-46 1.0e39 1.0e-50) copy.
	tensor := TFTensor fromDoubles: template.
	
	values := tensor allElements.
		
	self assert: tensor shape equals: #(7).
	self assert: tensor size equals: 7.
	self assert: tensor byteSize equals: (7*8).
	template with: values do: [:expected :actual |
		self assert: expected closeTo: actual].
	self assert: 0.0 ~= values last.
	self assert: 0.0 ~= (values at: 5).! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 6/30/2019 00:45:26'!
testTensorFromDoublesShape
	| tensor template values |
	template := #(1.23456 0.0 -1.234567 1.1).
	tensor := TFTensor fromDoubles: template shape: #(2 2).
	values := tensor allElements.
		
	self assert: tensor shape equals: #(2 2).
	self assert: tensor size equals: 4.
	self assert: tensor byteSize equals: (tensor size*8).
	template with: values do: [:templ :actual |
		self assert: templ closeTo: actual]
! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 6/30/2019 00:45:26'!
testTensorFromFloats
	| tensor template values |
	template := #(1.23456 0.0 -1.234567 3.402823466e38 1.175494351e-38).
	tensor := TFTensor fromFloats: template.
	values := tensor allFloats.
		
	self assert: tensor shape equals: #(5).
	self assert: tensor size equals: 5.
	self assert: tensor byteSize equals: (5*4).
	template @ values do: [:point |
		self assert: point x closeTo: point y]
! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 6/30/2019 00:45:25'!
testTensorFromFloats2x2
	| tensor template values |
	template := #(
		(-1.1 -2.1)
		(-1.2 -2.2)).
		
	tensor := TFTensor fromFloats: template.
	values := tensor allFloats.
		
	self assert: tensor shape equals: #(2 2).
	self assert: tensor size equals: 4.
	self assert: tensor byteSize equals: (4*4).
		
	#(-1.1 -2.1 -1.2 -2.2) with: values do: [:reference :value |
		self assert: reference closeTo: value]
! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 6/30/2019 00:45:27'!
testTensorFromFloats2x2SameElementsOrder
	| tensor template values constTensor consts |
	template := #(
		(-1.1 -2.1)
		(-1.2 -2.2)).
		
	tensor := TFTensor fromFloats: template.
	values := tensor allFloats.
	
	constTensor := self get2x2FloatFromGraphDef.
	consts := constTensor allFloats.
	
	consts with: values do: [:const :value |
		self assert: const closeTo: value].! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 6/30/2019 00:45:26'!
testTensorFromFloatsOutOfRange
	| tensor template values |
	template := #(1.23456 0.0 -1.234567 3.402824e38 1.175494351e-46 1.0e39 1.0e-50) copy.
	tensor := TFTensor fromFloats: template.
	
	template at: 4 put: Float infinity.
	template at: 6 put: Float infinity.
	
	values := tensor allElements.
		
	self assert: tensor shape equals: #(7).
	self assert: tensor size equals: 7.
	self assert: tensor byteSize equals: (7*4).
	template with: values do: [:expected :actual |
		self assert: expected closeTo: actual].
	self assert: 0.0 equals: values last.
	self assert: 0.0 equals: (values at: 5).! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 6/30/2019 00:45:28'!
testTensorFromFloatsScalar
	| tensor template values |
	template := 3.141516.
	tensor := TFTensor fromFloats: template.
	values := tensor allFloats.
		
	self assert: tensor shape equals: #().
	self assert: tensor size equals: 1.
	self assert: tensor byteSize equals: (1*4).
	
	self assert: template closeTo: values first.! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 6/30/2019 00:45:27'!
testTensorFromFloatsShape
	| tensor template values |
	template := #(1.23456 0.0 -1.234567 1.1).
	tensor := TFTensor fromFloats: template shape: #(2 2).
	values := tensor allFloats.
		
	self assert: tensor shape equals: #(2 2).
	self assert: tensor size equals: 4.
	self assert: tensor byteSize equals: (tensor size*4).
	template with: values do: [:templ :actual |
		self assert: templ closeTo: actual]
! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 6/30/2019 00:45:27'!
testTensorFromFloatsShapeUndefinedSize
	| template |
	template := #(1.23456 0.0 -1.234567 1.1).
	
	self
		should: [TFTensor fromFloats: template shape: #(2 2 -1)]
		raiseError: 'Inferred size and real size don''t match.'.! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 6/30/2019 00:45:28'!
testTensorFromInt32
	| tensor template values |
	template := -1123123123.
	tensor := TFTensor fromInt32: template.
	values := tensor allInt32s.
		
	self assert: tensor shape equals: #().
	self assert: tensor size equals: 1.
	self assert: tensor byteSize equals: 4.
	self assert: values equals: {template}.! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 6/30/2019 00:45:26'!
testTensorFromInt32Shape
	| tensor template values |
	template := #(123456 0 -1234567 11).
	tensor := TFTensor fromInt32s: template shape: #(2 2).
	values := tensor allFloats.
		
	self assert: tensor type equals: TFTensor typeInt32.
	self assert: tensor shape equals: #(2 2).
	self assert: tensor size equals: 4.
	self assert: tensor byteSize equals: (tensor size*4).
	template with: values do: [:templ :actual |
		self assert: templ closeTo: actual]
! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 6/30/2019 00:45:26'!
testTensorFromInt32s
	| tensor template values |
	template := #(0 -1 1 -2 2 32768 65536 -1123123123).
	tensor := TFTensor fromInt32s: template.
	values := tensor allInt32s.
		
	self assert: tensor shape equals: {template size}.
	self assert: tensor size equals: template size.
	self assert: tensor byteSize equals: (template size*4).
	self assert: values equals: template.! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 6/30/2019 00:45:25'!
testTensorFromInt64Shape
	| tensor template values |
	template := #(123456 0 -1234567 11).
	tensor := TFTensor fromInt64s: template shape: #(2 2).
	values := tensor allFloats.
		
	self assert: tensor type equals: TFTensor typeInt64.
	self assert: tensor shape equals: #(2 2).
	self assert: tensor size equals: 4.
	self assert: tensor byteSize equals: (tensor size*8).
	template with: values do: [:templ :actual |
		self assert: templ closeTo: actual]
! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 6/30/2019 00:45:25'!
testTensorFromInt64s
	| tensor template values |
	template := #(16r1234567812345678 0 -12345678910111213).
	tensor := TFTensor fromInt64s: template.
	values := tensor allInt64s.
	
	self assert: tensor shape equals: #(3).
	self assert: tensor size equals: 3.
	self assert: tensor byteSize equals: (3*8).
	self assert: template equals: values.
! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 6/30/2019 00:45:27'!
testTensorNewScalar
	| tensor |
	tensor := TFTensor type: TFTensor typeInt64 shape: #().
	tensor ignoreFinalization.
	
	self deny: tensor isNull.
	tensor delete.
	self assert: tensor isNull.! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 12/26/2016 14:44:54'!
testTensorShape0D
	^ self testTensorShape: #()! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 12/26/2016 14:45:28'!
testTensorShape10D
	^ self testTensorShape: #(1 2 3 4 5 6 7 8 9 10)! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 12/26/2016 14:44:50'!
testTensorShape1D
	^ self testTensorShape: #(7)! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 12/26/2016 14:44:40'!
testTensorShape2D
	^ self testTensorShape: #(1 4)! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 6/30/2019 00:45:26'!
testTensorShape: anArray
	| tensor shape size |
	tensor := TFTensor
		type: TFTensor typeInt64
		shape: anArray.
	self assert: tensor rank equals: anArray size.
	anArray withIndexDo: [:each :index |
		self assert: (tensor sizeOn: index - 1) equals: (anArray at: index)].
	shape := tensor shape.
	size := anArray isEmpty
		ifTrue: [1]
		ifFalse: [anArray product].
	self assert: shape equals: anArray.
	self assert: tensor size equals: size.
! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 6/30/2019 00:45:26'!
testTensorType
	| tensor |
	tensor := TFTensor type: TFTensor typeInt64 shape: #().
	self assert: tensor type equals: tensor class typeInt64.

	tensor := TFTensor type: TFTensor typeFloat shape: #().
	self assert: tensor type equals: tensor class typeFloat.
! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 6/30/2019 00:45:28'!
testTensorTypes
	| types |
	types := #(
		Float 1
		Double 2
		Int32 3
		UInt8 4
		Int16 5
		Int8 6
		String 7
		Complex64 8
		Int64 9
		Boolean 10
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
		self assert: (TFTensor perform: (#type, name) asSymbol) equals: value]! !

!TensorFlowCAPITest methodsFor: 'testing strings' stamp: 'gera 6/30/2019 00:45:28'!
assertTensorFromStrings: strings shape: shape
	| tensor |
	tensor := TFTensor fromStrings: strings shape: shape.

	self assert: shape equals: tensor shape.	
	self assert: strings equals: tensor allStrings.

	! !

!TensorFlowCAPITest methodsFor: 'testing strings' stamp: 'gera 1/31/2017 00:47:19'!
testStringAsTensor
	| tensor template |
	template := 'hola manola'.
	
	tensor := template asTensor.
	
	self assert: tensor dataBytes first equals: template size.
	self assert: tensor dataBytes allButFirst asString equals: template.! !

!TensorFlowCAPITest methodsFor: 'testing strings' stamp: 'jmv 12/8/2023 09:36:12'!
testStringDecode
	| size string encoded decoded status |
	" This test assumes the internal representation of a TensorFlow string. May fail if they change it "
	
	string := 'a ver como queda este string encodeado?'.
	
	status := TFStatus create.
	size := library stringEncodedSize: string.
	encoded := ExternalAddress gcallocate: size + 20.
	encoded ffiUint8At: size + 1 put: $@ asciiValue.
	library stringEncode: string len: string size destination: encoded len: size + 20 status: status.
	status check.

	decoded := library stringDecode: encoded.

	self deny: decoded = (string, '@').
	self assert: decoded equals: string.
! !

!TensorFlowCAPITest methodsFor: 'testing strings' stamp: 'jmv 12/8/2023 09:34:02'!
testStringEncode
	| size string encoded encodedSize |
	" This test assumes the internal representation of a TensorFlow string. May fail if they change it "
	
	string := 'a ver como queda este string encodeado?'.
	size := library stringEncodedSize: string.
	encoded := ByteArray new: size+20.
	encodedSize := library stringEncode: string to: encoded.
	
	self assert: encodedSize equals: size.
	self assert: (encoded ffiUint8At: 1) equals: string size.
	self assert: (encoded structAt: 2 length: string size) asString equals: string.
! !

!TensorFlowCAPITest methodsFor: 'testing strings' stamp: 'gera 7/3/2019 03:33:19'!
testStringEncodeExternal
	| size string encoded encodedSize status |
	" This test assumes the internal representation of a TensorFlow string. May fail if they change it "
	
	status := TFStatus create.
	string := 'a ver como queda este string encodeado?'.
	
	size := library stringEncodedSize: string.
	encoded := ExternalAddress gcallocate: size + 20.
	encodedSize := library stringEncode: string len: string size destination: encoded len: size + 20 status: status.

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

!TensorFlowCAPITest methodsFor: 'testing strings' stamp: 'gera 6/30/2019 00:45:25'!
testTensorFromString
	| tensor template |
	template := 'hola manola'.
	
	tensor := TFTensor fromString: template.
	
	self assert: tensor dataBytes first equals: template size.
	self assert: tensor dataBytes allButFirst asString equals: template.! !

!TensorFlowCAPITest methodsFor: 'testing strings' stamp: 'gera 6/30/2019 00:45:26'!
testTensorFromStringArray
	| tensor template |
	template := #('hola manola' 'te traje una lola' 'pamela' 'que pandulce!!').
	
	tensor := TFTensor fromStringArray: template.
	
	self assert: tensor allStrings equals: template.! !

!TensorFlowCAPITest methodsFor: 'testing strings' stamp: 'gera 6/30/2019 00:45:27'!
testTensorFromStrings
	| tensor template flatten |
	template := #(('hola manola' 'te traje una lola') ('pamela' 'que pandulce!!') ('habia una vez' 'truz')).
	flatten := TFTensor elementsOf: template.
	tensor := TFTensor fromStrings: template.

	self assert: #(3 2) equals: tensor shape.	
	self assert: flatten equals: tensor allStrings.

	! !

!TensorFlowCAPITest methodsFor: 'testing strings' stamp: 'gera 2/18/2017 17:28:29'!
testTensorFromStringsShape
	| template |
	template := #('hola manola' 'te traje una lola' 'pamela' 'que pandulce!!' 'habia una vez' 'truz').
	
	self assertTensorFromStrings: template shape: #(6).
	self assertTensorFromStrings: template shape: #(3 2).
	self assertTensorFromStrings: template shape: #(1 1 6 1 1).
	self assertTensorFromStrings: #('hola como estas?') shape: #().
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 7/3/2019 03:35:50'!
runFloatAsStringGraph
	| session graph output result |
	
	graph := self floatAsStringGraph.
	session := TFSession on: graph.
	output := graph operationNamed: 'output'.
	result := session
		runOperation: output
		output: (output output: 0).
	
	graph delete.
	^ result
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 7/3/2019 03:35:50'!
testAddControlInput
	| graph in op result |
	graph := TFGraph create.
	in := graph const: 'const' value: (TFTensor fromInt64: 12345678).
	op := graph newOperation: 'Mul' named: 'out' described: [:description |
		description
			addInput: (in output: 0);
			addInput: (in output: 0);
			addControlInput: (in output: 0)].
	
	result := (TFSession on: graph) runOutput: (op output: 0).
	
	self assert: 12345678*12345678 equals: result allInt64s first.
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 7/3/2019 00:34:29'!
testAllInitializers
	| graph pisTensor initializers |
	graph := TFGraph create.
	pisTensor := TFTensor fromFloats: #(3.14 3.1415 3.141516).
	graph variable: 'var1' initialValue: pisTensor.
	graph variable: 'var2' initialValue: pisTensor.
	graph variable: 'var3' initialValue: pisTensor.

	initializers := graph allInitializers.
	
	self assert: initializers size = 3.
	self assert: 'var1_initializer' equals: initializers first name.
	self assert: 'var2_initializer' equals: initializers second name.
	self assert: 'var3_initializer' equals: initializers third name.
	! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 7/3/2019 00:34:29'!
testAllVariables
	| graph pisTensor var1 vars var2 var3 |
	graph := TFGraph create.
	pisTensor := TFTensor fromFloats: #(3.14 3.1415 3.141516).
	var1 := graph variable: 'var1' initialValue: pisTensor.
	var2 := graph variable: 'var2' initialValue: pisTensor.
	var3 := graph variable: 'var3' initialValue: pisTensor.

	vars := graph allVariables.
	
	self assert: vars size = 3.
	self assert: vars first = var1.
	self assert: vars second = var2.
	self assert: vars third = var3.
	! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 7/3/2019 03:35:50'!
testAsStringGraphRunOn: tensor
	| graph session in out result |
	graph := self asStringGraphType: tensor type.
	session := TFSession on: graph.
	
	in := graph operationNamed: 'in'.
	out := graph operationNamed: 'out'.
	
	result := session
		runOperation: out
		input: (in input: 0)
		value: tensor
		output: (out output: 0).

	graph delete.

	^ result! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 7/3/2019 00:34:29'!
testAttrGetBoolFalse
	| graph in op |
	graph := TFGraph create.
	in := graph placeholder: 'in' type: TFTensor typeDouble.
	op := graph asString: 'out' described: [:description |
		description addInput: (in output: 0)].
	
	self assert: (op boolAt: 'scientific') equals: false.
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 7/3/2019 00:34:29'!
testAttrGetBoolTrue
	| graph in op |
	graph := TFGraph create.
	in := graph placeholder: 'in' type: TFTensor typeDouble.
	op := graph newOperation: 'QuantizeAndDequantize' named: 'out' described: [:description |
		description addInput: (in output: 0)].
	
	self assert: (op boolAt: 'signed_input') equals: true.
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 7/3/2019 00:34:29'!
testAttrGetFloat
	| graph in op |
	graph := TFGraph create.
	in := graph placeholder: 'in' type: TFTensor typeFloat.
	op := graph
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
	graph := self concatGraphInputList.
	op := graph operationNamed: 'concat'.
	
	self assert: (op intAt: 'N') equals: 2.
	
	graph delete! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 7/3/2019 00:34:29'!
testAttrGetShape
	| graph op |
	graph := TFGraph create.
	op := graph placeholder: 'in' type: TFTensor typeDouble.
	
	self assert: (op shapeAt: 'shape') equals: #().
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 7/3/2019 00:34:29'!
testAttrGetString
	| graph in op |
	graph := TFGraph create.
	in := graph placeholder: 'in' type: TFTensor typeDouble.
	op := graph newOperation: 'AsString' named: 'out' described: [:description |
		description addInput: (in output: 0)].
	
	self assert: (op stringAt: 'fill') equals: ''.
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 7/3/2019 00:34:29'!
testAttrGetStrings
	| graph template in op strings |
	graph := TFGraph create.
	template := #('hola' 'como' 'estas?').
	in := graph const: 'in' value: (TFTensor fromFloats: 1).
	op := graph newOperation: 'DebugIdentity' named: 'out' described: [:description |
			description at: 'debug_urls' putStrings: template.
			description addInput: (in output: 0)].
	
	strings := op stringsAt: 'debug_urls'.
	
	self assert: template equals: strings.! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 6/30/2019 00:45:27'!
testAttrGetTensor
	| op graph tensor |
	graph := self constantInt64Graph.
	op := graph operationNamed: 'a'.

	tensor := op tensorAt: 'value'.
	
	self assert: tensor type equals: TFTensor typeInt64.
	self assert: tensor shape equals: #().
	self assert: tensor allInt64s equals: #(16r4242424242424242).
	
	graph delete! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 6/30/2019 00:45:26'!
testAttrGetType
	| op graph |
	graph := self concatGraphInputList.
	op := graph operationNamed: 'concat'.
	
	self assert: (op typeAt: 'T') equals: TFTensor typeInt64.
	
	graph delete! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 7/3/2019 00:34:29'!
testAttrSetBoolFalse
	| graph in op |
	graph := TFGraph create.
	in := graph placeholder: 'in' type: TFTensor typeDouble.
	op := graph
		newOperation: 'QuantizeAndDequantize'
		named: 'out'
		described: [:description |
			description at: 'signed_input' putBoolean: false.
			description addInput: (in output: 0)].
	
	self assert: (op boolAt: 'signed_input') equals: false.
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 7/3/2019 00:34:29'!
testAttrSetBoolTrue
	| graph in op |
	graph := TFGraph create.
	in := graph placeholder: 'in' type: TFTensor typeDouble.
	op := graph asString: 'out' described: [:description |
		description at: 'scientific' putBoolean: true.
		description addInput: (in output: 0)].
	
	self assert: (op boolAt: 'scientific') equals: true.
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 7/3/2019 00:34:29'!
testAttrSetFloat
	| graph in op min max |
	min := -1234.5678e10.
	max := 12345678e-10 asFraction.
	graph := TFGraph create.
	in := graph placeholder: 'in' type: TFTensor typeFloat.
	op := graph
		newOperation: 'FakeQuantWithMinMaxArgs'
		named: 'out'
		described: [:description |
			description at: 'min' putFloat: min.
			description at: 'max' putFloat: max.
			description addInput: (in output: 0)].
	
	self assert: (op floatAt: 'min') closeTo: min.
	self assert: (op floatAt: 'max') closeTo: max.
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/11/2017 23:38:13'!
testAttrSetShape
	self testAttrSetShape: #().
	self testAttrSetShape: #(16r1234567890ABCDEF).
	self testAttrSetShape: #(1 2 3 4).
	self testAttrSetShape: (1 to: 200) asArray.! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 7/3/2019 00:34:29'!
testAttrSetShape: anIntegerArray
	| graph op |
	graph := TFGraph create.
	op := graph
		newOperation: 'Placeholder'
		named: 'const'
		described: [:description |
			description at: 'shape' putShape: anIntegerArray.
			description at: 'dtype' putType: TFTensor typeInt64].
		
	self assert: (op shapeAt: 'shape') equals: anIntegerArray.
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/24/2017 18:57:10'!
testAttrSetString
	self testAttrSetString: '1'.
	self testAttrSetString: '12'.
	self testAttrSetString: '1234'.
	self testAttrSetString: '1234567'.
	self testAttrSetString: '12345678'.
	self testAttrSetString: '123456789'.
	self testAttrSetString: ((ByteArray new: 100) atAllPut: 65) asString.! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 7/3/2019 00:34:29'!
testAttrSetString: aString
	| graph in op copy |
	graph := TFGraph create.
	copy := aString asByteArray.
	
	in := graph placeholder: 'in' type: TFTensor typeDouble.
	op := graph newOperation: 'AsString' named: 'out' described: [:description |
		description at: 'fill' putString: copy.
		description addInput: (in output: 0)].
	
	copy at: 1 put: 65.	"Change Smalltalk String to see if TensorFlow makes a copy"	
	
	self assert: (op stringAt: 'fill') equals: aString.
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/25/2017 10:58:07'!
testAttrSetStrings
	self testAttrSetStrings: #('file://tmp/TFDebug.log').
	self testAttrSetStrings: #('file://tmp/TFDebug.log' 'file://tmp/TFDebug.2.log').! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 7/3/2019 03:35:50'!
testAttrSetStrings: aAnArrayOfStrings
	| graph template in op |
	graph := TFGraph create.
	template := #((1 2 3) (4 5 6) (7 8 9)).
	in := graph const: 'in' value: (TFTensor fromFloats: template).
	op := graph newOperation: 'DebugIdentity' named: 'out' described: [:description |
		description at: 'debug_urls' putStrings: aAnArrayOfStrings.
		description addInput: (in output: 0)].
	
	(TFSession on: graph)
		runOutput: (op output: 0).

! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 7/3/2019 00:34:29'!
testAttrSetStringsInvalid
	| graph template in notAList |
	graph := TFGraph create.
	template := #((1 2 3) (4 5 6) (7 8 9)).
	in := graph const: 'in' value: (TFTensor fromFloats: template).
	
	notAList := 'INVALID_ARGUMENT: AttrValue had value with type ''list(string)'' when ''string'' expected
	 for attr ''tensor_name''
	; NodeDef: out = DebugIdentity[T=DT_FLOAT, _class=[], debug_urls=[], tensor_name=["hola", "como", "estas?"]](in); Op<name=DebugIdentity; signature=input:T -> output:T; attr=T:type; attr=tensor_name:string,default=""; attr=debug_urls:list(string),default=[]; allows_uninitialized_input=true>'.

	self
		should: [
			graph newOperation: 'DebugIdentity' named: 'out' described: [:description |
				description at: 'tensor_name' putStrings: #('hola' 'como' 'estas?').
				description addInput: (in output: 0)]]
		raiseError: notAList.	! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/29/2017 02:53:17'!
testConcatGraphInputList
	| wrongSize wrongType |
	wrongSize := 'INVALID_ARGUMENT: Inconsistent values for attr ''N'' 2 vs. 0 while building NodeDef ''concat'' using Op<name=Concat; signature=concat_dim:int32, values:N*T -> output:T; attr=N:int,min=2; attr=T:type>'.
	wrongType := 'INVALID_ARGUMENT: Inconsistent values for attr ''T'' DT_INT64 vs. DT_INT32 while building NodeDef ''concat'' using Op<name=Concat; signature=concat_dim:int32, values:N*T -> output:T; attr=N:int,min=2; attr=T:type>'.
	
	self concatGraphInputListNoSizeNoType.
	self concatGraphInputList.
	self
		should: [self concatGraphInputListWrongSize]
		raiseError: wrongSize.
		
	self
		should: [self concatGraphInputListWrongType]
		raiseError: wrongType.
	! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/5/2017 23:56:22'!
testCreateGraphAddTwoInputs
	| graph |
	graph := self addGraphTwoInputsInt64.
	graph delete.! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/13/2017 10:10:32'!
testCreateGraphMulOneInput
	| graph input mul |
	graph := self mulGraphOneInputInt64.
	
	self shouldnt: [input := graph operationNamed: 'in'] raise: Error.
	self shouldnt: [mul := graph operationNamed: 'mul'] raise: Error.
	
	self assert: input name equals: 'in'.
	self assert: mul name equals: 'mul'.

	graph delete.
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 6/30/2019 00:45:26'!
testDecodeCSVGraphCreate
	| defaults |
	defaults := {
		TFTensor fromInt64s: #(-1).
		TFTensor fromInt64s: #(-1).
		TFTensor fromInt64s: #(-1).
		TFTensor fromInt64s: #(-1)}.
		
	self decodeCSVGraphDefaults: defaults.! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 7/3/2019 03:35:50'!
testDecodeCSVGraphRunCSV: csvLines
	| defaults graph output records session results values |
	defaults := {
		TFTensor fromInt64s: #(-1).
		TFTensor fromInt64s: #(-1).
		TFTensor fromInt64s: #(-1).
		TFTensor fromInt64s: #(-1)}.
		
	graph := self decodeCSVGraphDefaults: defaults.
	records := (graph operationNamed: 'records') input: 0.
	output := graph operationNamed: 'output'.
	values := TFTensor fromStringArray: csvLines.
	
	session := TFSession on: graph.
	results := session
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
	cols := self testDecodeCSVGraphRunCSV:
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
	cols := self testDecodeCSVGraphRunCSV: '11111111111,22222222,33333333,44444444' lines.
	
	self assert: cols first equals: #(11111111111).
	self assert: cols second equals: #(22222222).
	self assert: cols third equals: #(33333333).
	self assert: cols fourth equals: #(44444444).
	! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 7/3/2019 03:35:50'!
testDescriptionDevice
	| graph in op expected |
	graph := TFGraph create.
	in := graph const: 'const' value: (TFTensor fromInt64: 12345678).
	op := graph newOperation: 'Mul' named: 'out' described: [:description |
		description
			device: 'anInvalidDevice';
			addInput: (in output: 0);
			addInput: (in output: 0)].
	
	expected :=  'INVALID_ARGUMENT: Malformed device specification ''anInvalidDevice''
	 [[Node: out = Mul[T=DT_INT64, _class=[], _device="anInvalidDevice"](const, const)]]'.

	self
		should: [(TFSession on: graph) runOutput: (op output: 0)]
		raiseError: expected.! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/6/2017 10:43:49'!
testFloatAsStringGraphCreate
	self floatAsStringGraph
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/13/2017 10:11:07'!
testFloatAsStringGraphRun
	| result str expected |
	
	expected := '0.420000'.
	result := self runFloatAsStringGraph.
	
	self deny: result isNull.
	self deny: result data isNull.
	str := result dataBytes.
	
	self assert: 8+1+ expected size equals: str size.
	self assert: (str unsignedLongLongAt: 1) equals: 0.
	self assert: (str at: 9) equals: expected size.
	self assert: (str copyFrom: 10 count: expected size) asString equals: expected.
	
	result delete.
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 7/3/2019 00:34:29'!
testGetAttrMetadataBoolean
	| graph in op template metadata |
	template := '1234567890abc'.
	graph := TFGraph create.
	in := graph placeholder: 'in' type: TFTensor typeDouble.
	op := graph newOperation: 'AsString' named: 'out' described: [:description |
		description at: 'fill' putString: template.
		description addInput: (in output: 0)].
		
	metadata := op attrMetadata: 'scientific'.
	self assert: metadata isBoolean.
	self assert: metadata isList equals: false.
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 7/3/2019 00:34:29'!
testGetAttrMetadataFloat
	| graph in op metadata |
	graph := TFGraph create.
	in := graph placeholder: 'in' type: TFTensor typeFloat.
	op := graph
		newOperation: 'FakeQuantWithMinMaxArgs'
		named: 'out'
		described: [:description |
			description addInput: (in output: 0)].
	
	metadata := op attrMetadata: 'min'.
	self assert: metadata isFloat.
	self assert: metadata isList equals: false.! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 7/3/2019 00:34:29'!
testGetAttrMetadataInt
	| graph in op template metadata |
	template := '1234567890abc'.
	graph := TFGraph create.
	in := graph placeholder: 'in' type: TFTensor typeDouble.
	op := graph newOperation: 'AsString' named: 'out' described: [:description |
		description at: 'fill' putString: template.
		description addInput: (in output: 0)].
		
	metadata := op attrMetadata: 'precision'.
	self assert: metadata isInt.
	self assert: metadata isList equals: false.
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 7/3/2019 00:34:29'!
testGetAttrMetadataShape
	| graph op template metadata |
	template := #(1 2 3 4 5).
	graph := TFGraph create.
	op := graph 
		newOperation: 'Placeholder'
		named: 'const'
		described: [:description |
			description at: 'shape' putShape: template.
			description at: 'dtype' putType: TFTensor typeInt64].
		
	metadata := op attrMetadata: 'shape'.
	self assert: metadata isShape.
	self assert: metadata isList equals: false.
	self assert: metadata totalSize equals: template size.
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 7/3/2019 00:34:29'!
testGetAttrMetadataString
	| graph in op template metadata |
	template := '1234567890abc'.
	graph := TFGraph create.
	in := graph placeholder: 'in' type: TFTensor typeDouble.
	op := graph newOperation: 'AsString' named: 'out' described: [:description |
		description at: 'fill' putString: template.
		description addInput: (in output: 0)].
		
	self assert: (op stringAt: 'fill') equals: template.
	
	metadata := op attrMetadata: 'fill'.
	self assert: metadata isString.
	self assert: metadata isList equals: false.
	self assert: metadata totalSize equals: template size.
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 7/3/2019 00:34:29'!
testGetAttrMetadataTensor
	| graph op template metadata |
	template := #(1 2 3 4 5).
	graph := TFGraph create.
	op := graph const: 'const' value: (TFTensor fromInt64s: template).
		
	metadata := op attrMetadata: 'value'.
	self assert: metadata isTensor.
	self assert: metadata isList equals: false.
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 7/3/2019 00:34:29'!
testGetAttrMetadataType
	| graph op template metadata |
	template := #(1 2 3 4 5).
	graph := TFGraph create.
	op := graph 
		newOperation: 'Placeholder'
		named: 'const'
		described: [:description |
			description at: 'shape' putShape: template.
			description at: 'dtype' putType: TFTensor typeInt64].
		
	metadata := op attrMetadata: 'dtype'.
	self assert: metadata isType.
	self assert: metadata isList equals: false.
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/13/2017 10:12:06'!
testGetOperationOnConstantGraph
	| graph op |
	graph := self constantFloatGraphFromDef.
	self shouldnt: [op := graph operationNamed: 'a'] raise: Error.
	self assert: op name equals: 'a'.
	self assert: op type equals: 'Const'.
	self assert: op InputsCount equals: 0.
	self assert: op outputsCount equals: 1.! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/29/2017 02:58:40'!
testGetOperationOnEmptyGraph
	| graph |
	graph := self emptyGraph.
	
	self
		should: [graph operationNamed: 'something']
		raiseError: 'Operation not found'.! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/13/2017 10:12:12'!
testGraph: aTFGraph outputType: anInteger
	| operation output |
	operation := aTFGraph operationNamed: 'a'.
	output := operation output: 0.
	self assert: output type equals: anInteger.
	output free.! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/13/2017 10:12:23'!
testGraphCreationConst
	| graph operation |
	graph := self constantInt64Graph.
	
	operation := graph operationNamed: 'a'.
	self assert: operation type equals: 'Const'.
	self assert: operation name equals: 'a'.
	self assert: operation InputsCount equals: 0.
	self assert: operation outputsCount equals: 1.
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 6/30/2019 00:45:28'!
testInt64AsStringGraph
	| result tensor |
	tensor := TFTensor fromInt64: 101010101.
	result := self testAsStringGraphRunOn: tensor.
	
	self assert: result allStrings first equals: '101010101'.

	result delete.! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 6/30/2019 00:45:25'!
testInt64rrayAsStringGraph
	| result tensor strings template |
	
	template := #(101010101 -123321 1 2 3 4).
	tensor := TFTensor fromInt64s: template.
	result := self testAsStringGraphRunOn: tensor.
	strings := result allStrings.
	
	strings withIndexDo: [:value :index |
		self assert: value equals: (template at: index) asString].
	
	result delete.! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 7/3/2019 00:34:29'!
testNewOperationDescription
	| graph description |
	graph := TFGraph create.
	description := graph newOperationDescription: 'Const' named: 'first_operation'.
	self deny: description isNull.
	self should: [description finish] raise: Error description: 'This should have complained of missing attributes'.! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 7/3/2019 00:34:29'!
testNewOperationMul
	| graph operation a b |
	graph := TFGraph create.
	a := graph placeholder: 'a' type: TFTensor typeInt64.
	b := graph placeholder: 'b' type: TFTensor typeInt64.
	
	operation := graph
		mul: 'aMultiplication'
		described: [:description |
			description addInputFromOutput: 0 of: a.
			description addInputFromOutput: 0 of: b].

	self assert: operation type equals: 'Mul'.
	self assert: operation name equals: 'aMultiplication'.
	self assert: operation InputsCount equals: 2.
	self assert: operation outputsCount equals: 1.
	
	operation := graph operationNamed: 'aMultiplication'.
	self assert: operation type equals: 'Mul'.
	self assert: operation name equals: 'aMultiplication'.
	self assert: operation InputsCount equals: 2.
	self assert: operation outputsCount equals: 1.
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 7/3/2019 00:34:29'!
testNewOperationPlaceholder
	| graph operation |
	graph := TFGraph create.
	self shouldnt: [
		operation := graph placeholder: 'aPlaceholder' type: TFTensor typeInt64.
	] raise: Error.
	
	self assert: operation type equals: 'Placeholder'.
	self assert: operation name equals: 'aPlaceholder'.
	self assert: operation InputsCount equals: 0.
	self assert: operation outputsCount equals: 1.
	
	operation := graph operationNamed: 'aPlaceholder'.
	self assert: operation type equals: 'Placeholder'.
	self assert: operation name equals: 'aPlaceholder'.
	self assert: operation InputsCount equals: 0.
	self assert: operation outputsCount equals: 1.
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 7/3/2019 00:34:29'!
testNewOperationPlaceholderNoType
	| graph noType |
	noType := 'INVALID_ARGUMENT: NodeDef missing attr ''dtype'' from Op<name=Placeholder; signature= -> output:dtype; attr=dtype:type; attr=shape:shape,default=[]>; NodeDef: placeholder = Placeholder[_class=[], shape=[]]()'.
	graph := TFGraph create.
	self
		should: [(graph newOperationDescription: 'Placeholder' named: 'placeholder') finish]
		raiseError: noType.! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 7/3/2019 03:35:50'!
testNewVariableForTensor
	| graph var assign result session pisTensor pis |
	graph := TFGraph create.
	pisTensor := TFTensor fromFloats: #(3.14 3.1415 3.141516).
	var := graph variable: 'var' forTensor: pisTensor.
	pis := graph const: 'pis' value: pisTensor.
	assign := graph newOperation: 'Assign' named: 'assign' described: [:description |
		description
			addInput: (var output: 0);
			addInput: (pis output: 0)].
	
	session := TFSession on: graph.
	
	session runOutput: (assign output: 0).
	result := session runOutput: (var output: 0).

	self assert: result allFloats equals: pisTensor allFloats! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 7/3/2019 03:35:50'!
testNewVariableInitialValue
	| graph var assign result session pisTensor |
	graph := TFGraph create.
	pisTensor := TFTensor fromFloats: #(3.14 3.1415 3.141516).
	var := graph variable: 'var' initialValue: pisTensor.
	assign := graph operationNamed: 'var_initializer'.
	
	session := TFSession on: graph.
	
	session runOutput: (assign output: 0).
	result := session runOutput: (var output: 0).

	self assert: result allFloats equals: pisTensor allFloats! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 7/3/2019 03:35:50'!
testNewVariableInitialValueAutomaticInitialization
	| graph var result session pisTensor |
	graph := TFGraph create.
	pisTensor := TFTensor fromFloats: #(3.14 3.1415 3.141516).
	var := graph variable: 'var' initialValue: pisTensor.
	
	session := TFSession on: graph.
	
	graph initializeOn: session.
	result := session runOutput: (var output: 0).

	self assert: result allFloats equals: pisTensor allFloats! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 7/3/2019 03:35:50'!
testNewVariableInitialization
	| graph var assign pi result session |
	graph := TFGraph create.
	var := graph variable: 'var' type: TFTensor typeFloat shape: #().
	pi := graph const: 'pi' value: (TFTensor fromFloats: 3.14).
	assign := graph newOperation: 'Assign' named: 'assign' described: [:description |
		description
			addInput: (var output: 0);
			addInput: (pi output: 0)].
	
	session := TFSession on: graph.
	
	session runOutput: (assign output: 0).
	result := session runOutput: (var output: 0).

	self assert: result allFloats first closeTo: 3.14! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 7/3/2019 03:35:50'!
testNewVariableInitializationNodeNotRun
	| graph var pi lastError |
	graph := TFGraph create.
	var := graph variable: 'var' type: TFTensor typeFloat shape: #().
	pi := graph const: 'pi' value: (TFTensor fromFloats: 3.14).
	graph newOperation: 'Assign' named: 'assign' described: [:description |
		description
			addInput: (var output: 0);
			addInput: (pi output: 0)].
	
	[(TFSession on: graph)
		runOutput: (var output: 0)] ifError: [:description :receiver | lastError := description].
	
	self
		assert: 'Error: FAILED_PRECONDITION: Attempting to use uninitialized value var'
		equals: lastError lines first! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 7/3/2019 00:34:29'!
testNewVariableNoAttributes
	| graph error |
	graph := TFGraph create.
	error :=  'INVALID_ARGUMENT: NodeDef missing attrs ''shape'', ''dtype'' from Op<name=Variable; signature= -> ref:Ref(dtype); attr=shape:shape; attr=dtype:type; attr=container:string,default=""; attr=shared_name:string,default=""; is_stateful=true>; NodeDef: var = Variable[_class=[], container="", shared_name=""]()'.
	self
		should: [graph newOperation: 'Variable' named: 'var']
		raiseError: error.! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 7/3/2019 03:35:50'!
testNewVariableNoInitialization
	| graph var expectedError lastError |
	graph := TFGraph create.
	var := graph variable: 'var' type: TFTensor typeFloat shape: #().
	[(TFSession on: graph)
		runOutput: (var output: 0)] ifError: [:description :receiver | lastError := description].

	expectedError := 'Error: FAILED_PRECONDITION: Attempting to use uninitialized value var'.
	
	self
		assert: expectedError equals: lastError lines first
		! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 7/3/2019 00:34:29'!
testOperationAsOperationDifferentGraph
	| const graph1 graph2 |
	graph1 := TFGraph create.
	graph2 := TFGraph create.
	const := graph1 const: 1.0 asTensor.
	self should: [const asOperationOn: graph2] raiseError: 'Can''t move an operation to another Graph'.
	
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 7/3/2019 00:34:29'!
testOperationAsOperationOk
	| const1 const2 graph |
	graph := TFGraph create.
	const1 := graph const: 1.0 asTensor.
	const2 := const1 asOperationOn: graph.
	
	self assert: const1 == const2! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 7/3/2019 00:34:29'!
testOperationEquals
	| graph in1 |
	graph := TFGraph create.
	in1 := graph placeholder: 'in1' type: TFTensor typeFloat.
	
	self assert: in1 = in1.
	self deny: in1 = 'in1'.! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 6/30/2019 00:45:28'!
testOperationOutputTypeFloat
	^self testGraph: self constantFloatGraphFromDef outputType: TFTensor typeFloat! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 6/30/2019 00:45:26'!
testOperationOutputTypeInt32
	^self testGraph: self constantInt32GraphFromDef outputType: TFTensor typeInt32! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 6/30/2019 00:45:25'!
testOperationOutputTypeInt64
	^self testGraph: self constantInt64GraphFromDef outputType: TFTensor typeInt64! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 7/3/2019 03:35:50'!
testPlaceholderType: type
	| graph var session result tensor abs |
	graph := TFGraph create.
	tensor := TFTensor type: type shape: #().
	var := graph placeholder: 'var' type: type.
	abs := graph newOperation: 'Abs' named: 'abs' described: [:description |
		description addInput: (var output: 0)].
	session := TFSession on: graph.
	
	result := session
		runOperation: abs
		input: (var input: 0)
		value: tensor
		output: (abs output: 0).

	result delete.
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 6/30/2019 00:45:26'!
testPlaceholderTypes
	self testPlaceholderType: TFTensor typeInt64.
	self testPlaceholderType: TFTensor typeInt32.
	self testPlaceholderType: TFTensor typeFloat.
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 1/13/2017 10:15:18'!
testPrintOn
	| graph printString |
	graph := self addGraphTwoInputsInt64.

	printString := (graph operationNamed: 'in1') printString substrings.
	self assert: printString second equals: 'TFOperation'.
	self assert: printString third equals: '@'.
	self assert: (printString fourth beginsWith: '16r').
	self assert: printString fifth equals: '''Placeholder'''.
	self assert: printString last equals: '''in1'''.

	printString := (graph operationNamed: 'add') printString substrings.
	self assert: printString second equals: 'TFOperation'.
	self assert: printString third equals: '@'.
	self assert: (printString fourth beginsWith: '16r').
	self assert: printString fifth equals: '''Add'''.
	self assert: printString last equals: '''add'''.
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 7/3/2019 00:34:29'!
testRankOfOutput
	| graph template const rank |
	graph := TFGraph create.
	template := TFTensor fromFloats: #(
		((1) (2) (3)) 
		((4) (5) (6))
	).
	const := graph const: 'const' value: template.
	rank := graph rankOf: (const output: 0).
	
	self assert: template shape size equals: rank.
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 7/3/2019 03:35:50'!
testRunGraphAddTwoInputs
	| graph inputs inputValues add output session results |
	graph := self addGraphTwoInputsInt64.
	
	inputs := Array
		with: ((graph operationNamed: 'in1') input: 0)
		with: ((graph operationNamed: 'in2') input: 0).
	inputValues := Array
		with: (TFTensor fromInt64: 16r2021222021222021)
		with: (TFTensor fromInt64: 16r2221202221202221).
	add := graph operationNamed: 'add'.
	output := add output: 0.
	session := TFSession on: graph.
	results := session
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

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 7/3/2019 03:35:50'!
testRunOperationArray
	| graph operation session |
	graph := self constantFloatGraphFromDef.
	session := TFSession on: graph.
	operation := graph operationNamed: 'a'.
	
	self shouldnt: [session runOperations: (Array with: operation)] raise: Error.

	graph delete.! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 7/3/2019 00:34:29'!
testShapeOfInput
	| graph template const shape same |
	graph := TFGraph create.
	template := TFTensor fromFloats: #(
		((1) (2) (3)) 
		((4) (5) (6))
	).
	const := graph const: 'const' value: template.
	same := const identity.
	shape := graph shapeOf: (same input: 0).
	
	self assert: template shape equals: shape.
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 7/3/2019 00:34:29'!
testShapeOfOutput
	| graph template const shape |
	graph := TFGraph create.
	template := TFTensor fromFloats: #(
		((1) (2) (3)) 
		((4) (5) (6))
	).
	const := graph const: 'const' value: template.
	shape := graph shapeOf: (const output: 0).
	
	self assert: template shape equals: shape.
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 7/3/2019 00:34:29'!
testShapeOfOutputSet
	| graph var shape output |
	graph := TFGraph create.
	
	var := graph newOperation: 'Placeholder' named: 'var' described: [:description |
		description
			at: 'dtype' putType: TFTensor typeInt64;
			at: 'shape' putShape: #(3 -1 -1)].
	
	output := var output: 0.
	
	shape := graph shapeOf: output.
	self assert: shape equals: #(3 -1 -1).

	graph shapeOf: output set: #(-1 3 -1).

	shape := graph shapeOf: output.
	self assert: shape equals: #(3 3 -1).
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 7/3/2019 00:34:29'!
testShapeOfOutputSetInvalid
	| graph template const output |
	graph := TFGraph create.
	template := TFTensor fromFloats: #(
		((1) (2) (3)) 
		((4) (5) (6))
	).
	const := graph const: 'const' value: template.
	output := const output: 0.
	
	self
		should: [graph shapeOf: output set: #(1 2 3)] 
		raiseError: 'INVALID_ARGUMENT: Dimension 0 in both shapes must be equal, but are 2 and 1'.
! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 7/3/2019 00:34:29'!
testShapeOfOutputSetInvalidRank
	| graph template const output |
	graph := TFGraph create.
	template := TFTensor fromFloats: #(
		((1) (2) (3)) 
		((4) (5) (6))
	).
	const := graph const: 'const' value: template.
	output := const output: 0.
	
	self
		should: [graph shapeOf: output set: #(1 2 3 -1)]
		raiseError: 'INVALID_ARGUMENT: Shapes must be equal rank, but are 3 and 4'.! !

!TensorFlowCAPITest methodsFor: 'initialization' stamp: 'gera 12/21/2016 02:11:17'!
setUp
	library := TensorFlowCAPI current! !

!TensorFlowCAPITest methodsFor: 'initialization' stamp: 'gera 1/29/2017 02:51:28'!
should: aBlock raiseError: aString
	| message |
	message := 'No Error was signaled'.
	aBlock ifError: [:description :receiver | message := description].
	self assert: 'Error: ',aString equals: message.! !

!TensorFlowCAPITest methodsFor: 'testing session' stamp: 'gera 7/3/2019 03:35:50'!
tesCloseSessionTwiceNoCrash
	| graph session |
	graph := TFGraph create.
	session := TFSession on: graph.
	self deny: session isNull.
	session ignoreFinalization.
	session close.
	session close.
	session delete.
	self assert: session isNull.! !

!TensorFlowCAPITest methodsFor: 'testing session' stamp: 'gera 7/3/2019 03:35:50'!
testCloseSession
	| graph session |
	graph := TFGraph create.
	session := TFSession on: graph.
	self deny: session isNull.
	session ignoreFinalization.
	self shouldnt: [session close] raise: Error.
	self shouldnt: [session delete] raise: Error.
	self assert: session isNull.! !

!TensorFlowCAPITest methodsFor: 'testing session' stamp: 'gera 7/3/2019 03:35:50'!
testNewSession
	| graph session |
	graph := TFGraph create.
	session := TFSession on: graph.
	session ignoreFinalization.
	self deny: session isNull.
	session delete.
	self assert: session isNull.! !

!TensorFlowCAPITest methodsFor: 'testing session' stamp: 'gera 7/3/2019 03:35:50'!
testRunGraphMulOneInput
	| graph input inputValue result mul output session |
	graph := self mulGraphOneInputInt64.
	
	input := (graph operationNamed: 'in') input: 0.
	inputValue := TFTensor fromInt64: 11.
	mul := graph operationNamed: 'mul'.
	output := mul output: 0.
	session := TFSession on: graph.
	result := session runOperation: mul input: input value: inputValue output: output.
	
	self deny: result isNull.
	self deny: result data isNull.
	self assert: (result data getHandle signedLongLongAt: 1) equals: 16r4242424242424242.

	result delete.
	input free.
	output free.
	graph delete.! !

!TensorFlowCAPITest methodsFor: 'testing session' stamp: 'gera 7/3/2019 03:35:50'!
testRunGraphMulTwoInputs
	| graph inputs inputValues mul output session results |
	graph := self mulGraphTwoInputsInt64.
	
	inputs := Array
		with: ((graph operationNamed: 'in1') input: 0)
		with: ((graph operationNamed: 'in2') input: 0).
	inputValues := Array
		with: (TFTensor fromInt64: 6)
		with: (TFTensor fromInt64: 11).
	mul := graph operationNamed: 'mul2'.
	output := mul output: 0.
	session := TFSession on: graph.
	results := session
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

!TensorFlowCAPITest methodsFor: 'testing session' stamp: 'gera 7/3/2019 03:35:50'!
testRunGraphMulTwoInputsRunInputsOutputs
	| graph inputs inputValues mul output session results |
	graph := self mulGraphTwoInputsInt64.
	
	inputs := Array
		with: ((graph operationNamed: 'in1') input: 0)
		with: ((graph operationNamed: 'in2') input: 0).
	inputValues := Array
		with: (TFTensor fromInt64: 6)
		with: (TFTensor fromInt64: 11).
	mul := graph operationNamed: 'mul2'.
	output := mul output: 0.
	session := TFSession on: graph.
	
	results := session
		runInputs: inputs
		values: inputValues
		outputs: {output}.
	
	self deny: results isNull.
	self deny: results first isNull.
	self deny: results first data isNull.
	self assert: (results first data getHandle signedLongLongAt: 1) equals: 16r4242424242424242.

	graph delete.
	results first delete.
! !

!TensorFlowCAPITest methodsFor: 'testing session' stamp: 'gera 7/3/2019 03:35:50'!
testSessionOnEmptyGraph
	| session |
	session := TFSession on: self emptyGraph.
	self
		should: [session run]
		raiseError: 'INVALID_ARGUMENT: Must specify at least one target to fetch or execute.'.! !

!TensorFlowCAPITest methodsFor: 'testing session' stamp: 'gera 1/22/2017 16:28:12'!
testSessionRunOutput
	^ self testSessionRunOutputOnGraph: self constantInt64GraphFromDef.
! !

!TensorFlowCAPITest methodsFor: 'testing session' stamp: 'gera 7/3/2019 03:35:50'!
testSessionRunOutputOnGraph: graph
	| operation session output tensor |
	session := TFSession on: graph.
	operation := graph operationNamed: 'a'.
	output := operation output: 0.
	
	tensor := session runOutput: output.
	
	self deny: tensor isNull.
	self deny: tensor data isNull.
	self assert: (tensor data getHandle signedLongLongAt: 1) equals: 16r4242424242424242.

	output free.
	graph delete.! !

!TensorFlowCAPITest methodsFor: 'testing session' stamp: 'gera 7/3/2019 03:35:50'!
testSessionRunTarget
	| graph operation session |
	graph := self constantFloatGraphFromDef.
	session := TFSession on: graph.
	operation := graph operationNamed: 'a'.
	
	self shouldnt: [session runOperation: operation] raise: Error.

	graph delete.! !

!TensorFlowCAPITest methodsFor: 'testing session' stamp: 'gera 1/6/2017 00:06:16'!
testSessionRunTargetOutput
	^ self testSessionRunTargetOutputOnGraph: self constantInt64GraphFromDef.
! !

!TensorFlowCAPITest methodsFor: 'testing session' stamp: 'gera 7/3/2019 03:35:50'!
testSessionRunTargetOutputOnGraph: graph
	| operation session output tensor |
	session := TFSession on: graph.
	operation := graph operationNamed: 'a'.
	output := operation output: 0.
	
	tensor := session runOperation: operation output: output.
	
	self deny: tensor isNull.
	self deny: tensor data isNull.
	self assert: (tensor data getHandle signedLongLongAt: 1) equals: 16r4242424242424242.

	output free.
	graph delete.! !

!TensorFlowCAPITest methodsFor: 'testing graph' stamp: 'gera 7/3/2019 00:34:29'!
testAllOperations
	| graph pisTensor operations names |
	graph := TFGraph create.
	pisTensor := TFTensor fromFloats: #(3.14 3.1415 3.141516).
	graph variable: 'var1' initialValue: pisTensor.
	graph variable: 'var2' initialValue: pisTensor.
	graph variable: 'var3' initialValue: pisTensor.

	operations := graph allOperations.
	
	self assert: operations size = 9.
	names := #(
		'var1' 'var1_initialValue' 'var1_initializer'
		'var2' 'var2_initialValue' 'var2_initializer'
		'var3' 'var3_initialValue' 'var3_initializer').
		
	names with: operations do: [:name :op |
		self assert: name equals: op name].
	! !

!TensorFlowCAPITest methodsFor: 'testing graph' stamp: 'gera 7/3/2019 00:34:29'!
testGraphDefinition
	| definition operations |
	definition := self mulGraphTwoInputsInt64 definition.
	operations := (TFGraph fromString: definition) allInputs.
		
	self assert: operations size equals: 2.
	self assert: operations first name equals: 'in2'.
	self assert: operations second name equals: 'in1'.
! !

!TensorFlowCAPITest methodsFor: 'testing graph' stamp: 'gera 2/15/2017 01:40:03'!
testGraphDeletionDoesntBreakOperations
	| graph in1 in2 add |
	self assert: false description: 'This test actually fails, and by failing it corrupts external memory and leads to a crash. This means when a TFGraph is deleted, all the TFOperations composing it are also deleted, hence pointers held to them (from Smalltalk or otherwise) become invalid'.
	
	graph := self addGraphTwoInputsInt64.
	graph ignoreFinalization.
	
	in1 := graph operationNamed: 'in1'.
	in2 := graph operationNamed: 'in2'.
	add := graph operationNamed: 'add'.

	self assert: in1 name equals: 'in1'.
	self assert: in2 name equals: 'in2'.
	self assert: add name equals: 'add'.
	
	graph delete.
	
	self assert: in1 name equals: 'in1'.
	self assert: in2 name equals: 'in2'.
	self assert: add name equals: 'add'.
	
	graph := self mulGraphTwoInputsInt64.
	graph ignoreFinalization.
	graph delete.

	self assert: in1 name equals: 'in1'.
	self assert: in2 name equals: 'in2'.
	self assert: add name equals: 'add'.! !

!TensorFlowCAPITest methodsFor: 'testing graph' stamp: 'gera 7/3/2019 03:35:50'!
testGraphDeletionDoesntBreakSessions
	| graph inputs inputValues add output session results |
	graph := self addGraphTwoInputsInt64.
	graph ignoreFinalization.
	
	inputs := Array
		with: ((graph operationNamed: 'in1') input: 0)
		with: ((graph operationNamed: 'in2') input: 0).
	inputValues := Array
		with: (TFTensor fromInt64: 16r2021222021222021)
		with: (TFTensor fromInt64: 16r2221202221202221).
	add := graph operationNamed: 'add'.
	output := add output: 0.
	session := TFSession on: graph.
	graph delete.

	graph := self addGraphTwoInputsInt64.
	graph ignoreFinalization.
	graph delete.
		
	results := session
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

!TensorFlowCAPITest methodsFor: 'testing graph' stamp: 'gera 7/3/2019 03:35:50'!
testGraphFromBlockIdentity
	| graph output inputs results |
	graph := TFGraph fromBlock: [:a | a ].
	
	inputs := graph allInputs collect: [:placeholder | placeholder input: 0].
	output := graph operationNamed: 'output'.
	
	results := (TFSession on: graph)
		runInputs: inputs
		values: {TFTensor fromFloats: 3.1415}
		outputs: {output output: 0}.
		
	self assert: results first allFloats first closeTo: 3.1415! !

!TensorFlowCAPITest methodsFor: 'testing graph' stamp: 'gera 7/3/2019 03:35:50'!
testGraphFromBlockIdentityInstance
	| graph output inputs results |
	graph := TFGraph create.
	output := graph fromBlock: [:a | a alias: 'a_1'].	
	inputs := graph allInputs collect: [:input | input input: 0].
	
	results := (TFSession on: graph)
		runInputs: inputs
		values: {TFTensor fromFloats: 3.1415}
		outputs: {output output: 0}.
		
	self assert: results first allFloats first closeTo: 3.1415! !

!TensorFlowCAPITest methodsFor: 'testing graph' stamp: 'gera 7/3/2019 03:35:50'!
testGraphFromBlockSimple
	| graph output inputs results |
	graph := TFGraph fromBlock: [:a :b | a + b].
	
	inputs := graph allInputs collect: [:placeholder | placeholder input: 0].
	output := graph operationNamed: 'output'.
	
	results := (TFSession on: graph)
		runInputs: inputs
		values: {TFTensor fromFloats: 3.1415. TFTensor fromFloats: 1.2345.}
		outputs: {output output: 0}.
		
	self assert: results first allFloats first closeTo: 3.1415 + 1.2345! !

!TensorFlowCAPITest methodsFor: 'testing graph' stamp: 'gera 7/3/2019 03:35:50'!
testGraphFromBlockSimpleInstance
	| graph output inputs results |
	graph := TFGraph create.
	output := graph fromBlock: [:a :b | a + b].	
	inputs := graph allInputs collect: [:placeholder | placeholder input: 0].
	
	results := (TFSession on: graph)
		runInputs: inputs
		values: {TFTensor fromFloats: 3.1415. TFTensor fromFloats: 1.2345.}
		outputs: {output output: 0}.
		
	self assert: results first allFloats first closeTo: 3.1415 + 1.2345! !

!TensorFlowCAPITest methodsFor: 'testing graph' stamp: 'gera 1/20/2017 18:04:24'!
testGraphNotFinalizedWhenHeldByOperations
	| graph in1 in2 add |
	graph := self addGraphTwoInputsInt64.
	graph useFinalization.
	
	in1 := graph operationNamed: 'in1'.
	in2 := graph operationNamed: 'in2'.
	add := graph operationNamed: 'add'.

	self assert: in1 name equals: 'in1'.
	self assert: in2 name equals: 'in2'.
	self assert: add name equals: 'add'.
		
	graph := nil.
	Smalltalk garbageCollect.
	
	self assert: in1 name equals: 'in1'.
	self assert: in2 name equals: 'in2'.
	self assert: add name equals: 'add'.
	
	graph := self mulGraphTwoInputsInt64.
	graph delete.

	self assert: in1 name equals: 'in1'.
	self assert: in2 name equals: 'in2'.
	self assert: add name equals: 'add'.! !

!TensorFlowCAPITest methodsFor: 'testing graph' stamp: 'gera 1/13/2017 10:13:07'!
testGraphOperationAt
	| graph operation context |
	graph := self mulGraphTwoInputsInt64.
	context := graph newOperationIteratorContext.
	
	operation := graph operationAt: context.
	self assert: operation name equals: 'in1'.
	self assert: operation type equals: 'Placeholder'.

	operation := graph operationAt: context.
	self assert: operation name equals: 'in2'.
	self assert: operation type equals: 'Placeholder'.

	operation := graph operationAt: context.
	self assert: operation name equals: 'const'.
	self assert: operation type equals: 'Const'.

	operation := graph operationAt: context.
	self assert: operation name equals: 'mul1'.
	self assert: operation type equals: 'Mul'.

	operation := graph operationAt: context.
	self assert: operation name equals: 'mul2'.
	self assert: operation type equals: 'Mul'.

	operation := graph operationAt: context.
	self assert: operation isNull.
! !

!TensorFlowCAPITest methodsFor: 'testing graph' stamp: 'gera 1/19/2017 11:11:05'!
testGraphOperationsCount
	| graph |
	graph := self mulGraphTwoInputsInt64.
	
	self assert: graph operationsCount equals: 5.! !

!TensorFlowCAPITest methodsFor: 'testing graph' stamp: 'gera 1/13/2017 10:13:21'!
testGraphOperationsDo
	| graph operations |
	graph := self mulGraphTwoInputsInt64.
	operations := OrderedCollection new.
	
	graph operationsDo: [:op |
		operations add: op name].
	
	self assert: operations size equals: 5.
	self assert: operations first equals: 'in1'.
	self assert: operations second equals: 'in2'.
	self assert: operations third equals: 'const'.
	self assert: operations fourth equals: 'mul1'.
	self assert: operations fifth equals: 'mul2'.
! !

!TensorFlowCAPITest methodsFor: 'testing graph' stamp: 'gera 1/29/2017 03:41:15'!
testGraphOperationsSelect
	| operations |
	operations := self mulGraphTwoInputsInt64 allInputs.
		
	self assert: operations size equals: 2.
	self assert: operations first name equals: 'in1'.
	self assert: operations second name equals: 'in2'.
! !

!TensorFlowCAPITest methodsFor: 'testing graph' stamp: 'gera 1/13/2017 10:13:43'!
testGraphOperationsSelectEmpty
	| graph operations |
	graph := self mulGraphTwoInputsInt64.
	
	operations := graph operationsSelect: [:op | false].
	
	self assert: operations size equals: 0.
! !

!TensorFlowCAPITest methodsFor: 'testing graph' stamp: 'gera 7/3/2019 00:34:29'!
testGraphRunInputsOutputs
	| graph output result input |
	graph := TFGraph create.
	
	output := graph fromBlock: [:a |
		input := a.
		a @* TFTensor pi].

	result := graph
		runInputs: {input input: 0}
		values: {7.23 asTensor}
		outputs: {output output}.
	
	self assert: Float pi * 7.23 closeTo: result first asNumbers.! !

!TensorFlowCAPITest methodsFor: 'testing graph' stamp: 'gera 7/3/2019 00:34:29'!
testGraphRunOutput
	| graph output result |
	graph := TFGraph create.
	output := graph const: TFTensor pi.
	result := graph runOutput: output output.
	
	self assert: Float pi closeTo: result asNumbers.! !

!TensorFlowCAPITest methodsFor: 'testing graph' stamp: 'gera 7/3/2019 00:34:29'!
testGraphRunOutputs
	| graph output1 output2 results |
	graph := TFGraph create.
	output1 := graph const: TFTensor pi.
	output2 := output1 @/ 2.0 asTensor.

	results := graph runOutputs: {output1 output. output2 output}.
	
	self assert: Float pi closeTo: results first asNumbers.
	self assert: Float pi / 2 closeTo: results second asNumbers.! !

!TensorFlowCAPITest methodsFor: 'testing graph' stamp: 'gera 7/3/2019 00:34:29'!
testImportBad
	| graph buffer |
	graph := TFGraph create.
	buffer := TFBuffer fromString: 'ouch'.
	
	self
		should: [graph import: buffer]
		raiseError: 'INVALID_ARGUMENT: Invalid GraphDef'.
	
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

!TensorFlowCAPITest methodsFor: 'testing graph' stamp: 'gera 7/3/2019 03:35:50'!
testInitializeOn
	| graph session |
	graph := TFGraph create.
	session := TFSession on: graph.
	self shouldnt: [graph initializeOn: session] raise: Error.! !

!TensorFlowCAPITest methodsFor: 'testing graph' stamp: 'gera 7/3/2019 00:34:29'!
testNewGraph
	| graph |
	graph := TFGraph create.
	self deny: graph isNull.
	graph delete.
	self assert: graph isNull.! !

!TensorFlowCAPITest methodsFor: 'testing graph' stamp: 'gera 1/20/2017 18:07:50'!
testOutputDims
	| graph operation output |
	graph := self constantInt64GraphFromDef.
	operation := graph operationNamed: 'a'.
	output := operation output: 0.
	self assert: (graph outputDimenssionsCount: output) equals: 0.
	output free.! !

!TensorFlowCAPITest methodsFor: 'testing graph' stamp: 'gera 7/3/2019 03:35:50'!
testSessionDeletionDoesntDeleteGraphs
	| session graph allocatedObjects |
	graph := TFGraph create.
	session := TFSession on: graph.
	session ignoreFinalization.
	session close.
	session delete.

	" Allocate some external objects using the library, if the graph was released, we expect its space to be reused "
	allocatedObjects := OrderedCollection new.
	10 timesRepeat: [allocatedObjects add: (TFStatus create)].
	
	self
		shouldnt: [graph placeholder: 'a' type: TFTensor typeInt64]
		raise: Error
		description: 'The FFI call would crash if the graph was released by deleting the session'.
! !

!TensorFlowCAPITest methodsFor: 'testing graph' stamp: 'gera 2/9/2017 10:50:39'!
testWriteDefTo
	| graph strm |
	graph := self constantInt64GraphFromDef.
	strm := WriteStream on: String new.
	graph writeDefTo: strm.
	
	self assert: self constantInt64GraphDef size equals: strm contents size.! !

!TensorFlowCAPITest methodsFor: 'testing graph' stamp: 'gera 2/9/2017 10:50:45'!
testWriteDefToFileNamed
	| graph filename filedata |
	filename := 'temporaryGraph.pb'.
	graph := self constantInt64GraphFromDef.
	graph writeDefToFileNamed: filename.

     filedata := filename asFileEntry readStream upToEnd.
	filename asFileEntry delete.

	self assert: self constantInt64GraphDef size equals: filedata size.! !

!TensorFlowCAPITest methodsFor: 'testing buffer' stamp: 'gera 7/3/2019 00:28:37'!
testBufferDataBytes
	| buffer string data |
	string := ' hola manola'.
	buffer := TFBuffer fromString: string.

	data := buffer dataBytes.
	self assert: string equals: data asString.
	
	buffer delete.
! !

!TensorFlowCAPITest methodsFor: 'testing buffer' stamp: 'gera 7/3/2019 00:28:37'!
testBufferNoNeedExternalize
	| buffer string data |
	string := ' hola manola'.
	buffer := TFBuffer fromString: string.
	
	string := string copy.
	Smalltalk garbageCollect.
	
	data := buffer dataBytes.
	self assert: string equals: data asString.
	
	buffer delete.
! !

!TensorFlowCAPITest methodsFor: 'testing buffer' stamp: 'gera 7/3/2019 00:28:37'!
testNewBufferFromFileNamed
	| buffer string data temporaryFile |
	string := ' hola manola'.
	temporaryFile := 'temporaryFile.txt'.
	temporaryFile asFileEntry writeStream nextPutAll: string; close.

	buffer := TFBuffer fromFileNamed: temporaryFile.
	temporaryFile asFileEntry delete.
	
	self deny: buffer isNull.
	self assert: buffer length equals: string size.
	
	data := buffer data fromCString first: string size.
	self assert: string equals: data.
	
	buffer delete.
	self assert: buffer isNull.
! !

!TensorFlowCAPITest methodsFor: 'testing buffer' stamp: 'gera 7/3/2019 00:28:37'!
testNewBufferFromString
	| buffer string data |
	string := ' hola manola'.
	buffer := TFBuffer fromString: string.
	self deny: buffer isNull.
	self assert: buffer length equals: string size.
	
	data := buffer data fromCString first: string size.
	self assert: string equals: data.
	
	buffer delete.
	self assert: buffer isNull.! !

!TensorFlowCAPITest methodsFor: 'testing options' stamp: 'jmv 12/8/2023 09:33:37'!
testExternalizeString
	| original copy |
	original := 'hola manola'.
	copy := library externalizeString: original.
	original withIndexDo: [:each :index |
		self assert: each asciiValue equals: (copy ffiUint8At: index)].
	self assert: (copy ffiUint8At: original size + 1) equals: 0.
	copy free.! !

!TensorFlowCAPITest methodsFor: 'testing options' stamp: 'gera 7/3/2019 01:08:38'!
testNewImportGraphDefOptions
	| options |
	options := TFImportGraphDefOptions create.
	self deny: options isNull.
	options delete.
	self assert: options isNull.! !

!TensorFlowCAPITest methodsFor: 'testing options' stamp: 'gera 7/3/2019 03:35:40'!
testNewSessionOptions
	| options |
	options := TFSessionOptions create.
	self deny: options isNull.
	options delete.
	self assert: options isNull.! !

!TensorFlowCAPITest methodsFor: 'testing options' stamp: 'gera 7/3/2019 03:35:40'!
testSessionOptionsFromProtoBufEmpty
	TFSessionOptions fromProtoBuf: ''.! !

!TensorFlowCAPITest methodsFor: 'testing options' stamp: 'gera 7/3/2019 03:35:40'!
testSessionOptionsFromProtoBufInvalid
	self
		should: [TFSessionOptions fromProtoBuf: '.']
		raiseError: 'INVALID_ARGUMENT: Unparseable ConfigProto'.! !

!TensorFlowCAPITest methodsFor: 'testing options' stamp: 'gera 7/3/2019 03:35:40'!
testSessionOptionsFromProtoBufValid
	"
	In [241]: tf.ConfigProto(allow_soft_placement=True, log_device_placement=True).SerializeToString()
	Out[241]: b'8\x01@\x01'
	"
	| config |
	
	config := #[16r38 1 16r40 1].
	TFSessionOptions fromProtoBuf: config.
	! !

!TensorFlowCAPITest methodsFor: 'testing options' stamp: 'gera 12/28/2016 00:12:49'!
testSessionOptionsSetTarget
	| options |
	options := library newSessionOptions.
	self deny: options isNull.
	options target: 'local'.
	options delete.
	self assert: options isNull.! !

!TensorFlowCAPITest methodsFor: 'testing library' stamp: 'gera 1/3/2017 09:37:31'!
testGetAllOps
	| ops |
	ops := library getAllOps.
	self assert: (ops data fromCString includesSubString: 'tensor').
	ops delete! !

!TensorFlowCAPITest methodsFor: 'testing library' stamp: 'gera 2/13/2017 15:59:41'!
testVersion
	| version |
	version := library version.
	self assert: (#('0.11.head' '0.12.0' '0.12.head' '1.0.0-alpha' '1.0.0') includes: version)! !

!TensorFlowCAPITest methodsFor: 'testing status' stamp: 'gera 7/3/2019 03:33:19'!
testNewStatus
	| status |
	status := TFStatus create.
	status ignoreFinalization.
	
	self deny: status isNull.
	status delete.
	self assert: status isNull.! !

!TensorFlowCAPITest methodsFor: 'testing status' stamp: 'gera 7/3/2019 03:33:19'!
testStatusCodes
	| status  msg |
	status := TFStatus create.
	
	self assert: status isOk.
	self assert: status codeText equals: 'OK'.
	self shouldnt: [status check] raise: Error.
	
	msg := 'You cancelled it!!'.
	status code: 1 message: msg.
	self assert: status codeText equals: 'CANCELLED'.
	
	self
		should: [status check]
		raiseError: 'CANCELLED: ', msg! !

!TensorFlowCAPITest methodsFor: 'testing status' stamp: 'gera 7/3/2019 03:33:19'!
testStatusGetMessage
	| status message |
	status := TFStatus create.
	
	status code: 1 message: 'All is one'.
	message := status message.
	self assert: message equals: 'All is one'.

	status code: 7 message: 'Something is very seven'.
	message := status message.
	self assert: message equals: 'Something is very seven'.
! !

!TensorFlowCAPITest methodsFor: 'testing status' stamp: 'gera 7/3/2019 03:33:19'!
testStatusSetGetCode
	| status code |
	status := TFStatus create.
	status code: 1 message: ''.
	code := status code.
	self assert: code equals: 1.

	status code: 2 message: ''.
	code := status code.
	self assert: code equals: 2.
! !

!TensorFlowOperationsTest methodsFor: 'other operations' stamp: 'gera 7/3/2019 03:35:50'!
assert: integers sizeOn: dimension is: expected
	| graph session const result size |
	graph := TFGraph create.
	const := graph const: integers asInt32Tensor.
	
	size := const sizeOn: dimension.

	session := TFSession on: graph.
	result := session runOutput: size output.
	
	self assert: expected equals: result asNumbers.! !

!TensorFlowOperationsTest methodsFor: 'other operations' stamp: 'gera 7/3/2019 03:35:50'!
assertSlice: integers from: begin size: size is: expected
	| graph session const result slice |
	graph := TFGraph create.
	const := graph const: integers asInt32Tensor.
	
	slice := const sliceFrom: begin asInt32Tensor size: size asInt32Tensor.

	session := TFSession on: graph.
	result := session runOutput: slice output.
	
	self assert: expected equals: result asNumbers.! !

!TensorFlowOperationsTest methodsFor: 'other operations' stamp: 'gera 7/3/2019 03:35:50'!
testOperationDescentRate
	| graph var output pi result session rate |
	graph := TFGraph create.
	var := graph variable: 'var' initialValue: 1.0 asTensor.
	pi := graph const: 'pi' value: (TFTensor pi).
	rate := graph const: 'rate' value: 0.7 asTensor.
	output := var descent: pi rate: rate.
	
	session := TFSession on: graph.
	graph initializeOn: session.
	result := session runOutput: (output output: 0).
	
	self assert: 1-(Float pi * 0.7) closeTo: result allFloats first! !

!TensorFlowOperationsTest methodsFor: 'other operations' stamp: 'gera 7/3/2019 03:35:50'!
testOperationDescentRateConstants
	| graph var output pi result session rate |
	graph := TFGraph create.
	var := graph variable: 'var' initialValue: 1.0 asTensor.
	pi := Float pi asTensor.
	rate := 0.7 asTensor.
	output := var descent: pi rate: rate.
	
	session := TFSession on: graph.
	graph initializeOn: session.
	result := session runOutput: (output output: 0).
	
	self assert: 1-(Float pi * 0.7) closeTo: result allFloats first! !

!TensorFlowOperationsTest methodsFor: 'other operations' stamp: 'gera 2/19/2017 15:11:43'!
testOperationSizeOn
	| t3x1x4 |
	t3x1x4 := #(	"shape: (3 1 3)"
		((1 2 3 0))
		((4 5 6 0))
		((7 8 9 0))
	).
	self assert: #(1 2 3 4 5 6 7) sizeOn: 0 is: #(7).
	self assert: t3x1x4 sizeOn: 0 is: #(3).
	self assert: t3x1x4 sizeOn: 1 is: #(1).
	self assert: t3x1x4 sizeOn: 2 is: #(4).
 
! !

!TensorFlowOperationsTest methodsFor: 'other operations' stamp: 'gera 2/19/2017 12:43:03'!
testOperationSlice
	| m3x3 |
	m3x3 := #(	"shape: (3 1 3)"
		((1 2 3))
		((4 5 6))
		((7 8 9))
	).
	self assertSlice: #(1 2 3 4 5 6 7) from: #(0) size: #(1) is: #(1).
	self assertSlice: m3x3 from: #(0 0 0) size: #(1 1 1) is: #(((1))).
	self assertSlice: m3x3 from: #(1 0 1) size: #(1 1 1) is: #(((5))).
	self assertSlice: m3x3 from: #(1 0 1) size: #(1 -1 1) is: #(((5))).
	self assertSlice: m3x3 from: #(1 0 1) size: #(1 1 -1) is: #(((5 6))).
	self assertSlice: m3x3 from: #(1 0 1) size: #(1 -1 -1) is: #(((5 6))).
	self assertSlice: m3x3 from: #(1 0 1) size: #(-1 1 -1) is: #(((5 6)) ((8 9))).
! !

!TensorFlowOperationsTest methodsFor: 'other operations' stamp: 'gera 7/3/2019 03:35:50'!
testOperationSliceSimple
	| graph session const result slice |
	graph := TFGraph create.
	const := graph const: #(1 2 3 4 5 6 7) asInt32Tensor.
	slice := const sliceFrom: #(0) asInt32Tensor size: #(1) asInt32Tensor.

	session := TFSession on: graph.
	result := session runOutput: slice output.
	
	self assert: #(1) equals: result asNumbers.! !

!TensorFlowOperationsTest methodsFor: 'private' stamp: 'JB 2/7/2017 11:13:02'!
percentile: percent from: collection
	| position sorted |
	position := (collection size * percent) ceiling.
	sorted := collection sorted: nil.
	^ sorted at: position
	! !

!TensorFlowOperationsTest methodsFor: 'initialization' stamp: 'gera 1/20/2017 18:23:31'!
setUp
	library := TensorFlowCAPI current! !

!TensorFlowOperationsTest methodsFor: 'initialization' stamp: 'gera 1/31/2017 02:00:03'!
should: aBlock raiseError: aString
	| message |
	message := 'No Error was signaled'.
	aBlock ifError: [:description :receiver | message := description].
	self assert: 'Error: ',aString equals: message.! !

!TensorFlowOperationsTest methodsFor: 'binary operations' stamp: 'gera 7/3/2019 03:35:50'!
testGraphTruncatedNormal
	| graph session result random values std theoreticalDecile expected |	
	graph := TFGraph create.
	
	random := graph truncatedNormalRandomShaped: #(100 100 10).

	session := TFSession on: graph.
	result := session runOutput: (random output: 0).
	
	values := result allFloats.
	theoreticalDecile := -1.1840324666939051.
	std := values standardDeviation.
	self assert: result shape equals: #(100 100 10).
	self assert: values size equals: 100*100*10.
	self assert: values min + 2 < 0.01.
	self assert: 2 - values max < 0.01.
	self assert: values mean abs < (0.01*std).
	expected :=  { theoreticalDecile. 
							theoreticalDecile negated}.
							
	#(0.1 0.9) with: expected do:[:p :e| | observed | 
		observed := self percentile: p from: values.
		self assert: (observed - e) abs < 0.2].
	! !

!TensorFlowOperationsTest methodsFor: 'binary operations' stamp: 'gera 7/3/2019 03:35:50'!
testGraphTruncatedNormalStddev
	| graph session result random values sigma twoSigma std theoreticalDecile expected |	
	graph := TFGraph create.
	
	sigma := 3.14.
	twoSigma := 2 * sigma.
	random := graph truncatedNormalRandomShaped: #(100 100 10) stddev: sigma.

	session := TFSession on: graph.
	result := session runOutput: (random output: 0).
	values := result allFloats.
	
	std := values standardDeviation.
	theoreticalDecile := -1.1840324666939051.
		
	self assert: result shape equals: #(100 100 10).
	self assert: values size equals: 100*100*10.
	self assert: twoSigma - (values min abs)  < 0.1.
	self assert: twoSigma - values max < 0.1.
	self assert: values mean abs < (0.01 * std).
	expected :=  { theoreticalDecile * sigma. 
							theoreticalDecile negated * sigma.
							-2.
							2}.
							
	#(0.1 0.9 0.25 0.75) with: expected do:[:p :e| | observed | 
		observed := self percentile: p from: values.
		self assert: (observed - e) abs < 0.2].
	! !

!TensorFlowOperationsTest methodsFor: 'binary operations' stamp: 'gera 7/3/2019 03:35:50'!
testGraphUnnamedConst
	| graph session result const |	
	graph := TFGraph create.
	
	const := graph const: #(1 2 3 4) asFloatTensor.

	session := TFSession on: graph.
	result := session runOutput: (const output: 0).
	
	self assert: result shape equals: #(4).
	self assert: result allFloats equals: #(1 2 3 4).! !

!TensorFlowOperationsTest methodsFor: 'binary operations' stamp: 'gera 7/3/2019 03:35:50'!
testGraphVariableFromTruncatedNormalStddev
	| graph session result1 result2 var random values1 values2 init sigma std twoSigma |	
	graph := TFGraph create.
	
	sigma := 3.14.
	twoSigma := 2 * sigma.
	random := graph truncatedNormalRandomShaped: #(100 100 10) stddev: sigma.
	var := graph variable: 'var' initialValueFrom: random.
	
	session := TFSession on: graph.
	init := graph operationNamed: 'var_initializer'.
	result1 := session runOutput: (init output: 0).
	result2 := session runOutput: (var output: 0).

	self assert: result1 shape equals: #(100 100 10).
	
	values1 := result1 allFloats.
	std := values1 standardDeviation.
	self assert: values1 size equals: 100*100*10.	
	self assert: twoSigma - (values1 min abs)  < 0.1.
	self assert: twoSigma - values1 max < 0.1.
	self assert: values1 mean abs < (0.01 * std).	
	
	self assert: result1 shape equals: result2 shape.
	values2 := result2 allFloats.
	self assert: values1 equals: values2! !

!TensorFlowOperationsTest methodsFor: 'binary operations' stamp: 'gera 7/3/2019 03:35:50'!
testGraphZeros
	| graph session result zeros values |	
	graph := TFGraph create.
	
	zeros := graph zerosShaped: #(100 100 10).

	session := TFSession on: graph.
	result := session runOutput: (zeros output: 0).
	
	values := result allFloats.
	self assert: #(100 100 10) equals: result shape.
	self assert: 100*100*10 equals: values size.
	self assert: 0 equals: values min.
	self assert: 0 equals: values max.
! !

!TensorFlowOperationsTest methodsFor: 'binary operations' stamp: 'gera 7/3/2019 03:35:50'!
testOperationArgMax
	| graph a session const result output0 output1 axis0 axis1 |
	
	a := TFTensor fromFloats: #(
		(1.1 1.2)
		(2.1 2.2)
		(3.1 0.2)
	).
	
	graph := TFGraph create.
	
	const := graph const: 'a' value: a.
	axis0 := graph const: 'axis0' value: (TFTensor fromInt32: 0).
	axis1 := graph const: 'axis1' value: (TFTensor fromInt32: 1).
	
	output0 := const findMaxOn: axis0.
	output1 := const findMaxOn: axis1.
	
	session := TFSession on: graph.
	result := session runOutput: (output0 output: 0).
	
	self assert: result shape equals: #(2).
	result allInt64s with: #(2 1) do: [:r :t |
		self assert: r equals: t].
	
	result := session runOutput: (output1 output: 0).
	
	self assert: result shape equals: #(3).
	result allInt64s with: #(1 1 0) do: [:r :t |
		self assert: r equals: t].! !

!TensorFlowOperationsTest methodsFor: 'binary operations' stamp: 'gera 7/3/2019 03:35:50'!
testOperationArgMin
	| graph a session const result output0 output1 axis0 axis1 |
	
	a := TFTensor fromFloats: #(
		(1.1 1.2)
		(2.1 2.2)
		(3.1 0.2)
	).
	
	graph := TFGraph create.
	
	const := graph const: 'a' value: a.
	axis0 := graph const: 'axis0' value: (TFTensor fromInt32: 0).
	axis1 := graph const: 'axis1' value: (TFTensor fromInt32: 1).
	
	output0 := const findMinOn: axis0.
	output1 := const findMinOn: axis1.
	
	session := TFSession on: graph.
	result := session runOutput: (output0 output: 0).
	
	self assert: result shape equals: #(2).
	result allInt64s with: #(0 2) do: [:r :t |
		self assert: r equals: t].
	
	result := session runOutput: (output1 output: 0).
	
	self assert: result shape equals: #(3).
	result allInt64s with: #(0 0 1) do: [:r :t |
		self assert: r equals: t].! !

!TensorFlowOperationsTest methodsFor: 'binary operations' stamp: 'gera 7/3/2019 03:35:50'!
testOperationArgMinTwoOutputs
	| graph a session const results first second output0 output1 axis0 axis1 |
	
	a := TFTensor fromFloats: #(
		(1.1 1.2)
		(2.1 2.2)
		(3.1 0.2)
	).
	
	graph := TFGraph create.
	
	const := graph const: 'a' value: a.
	axis0 := graph const: 'axis0' value: (TFTensor fromInt32: 0).
	axis1 := graph const: 'axis1' value: (TFTensor fromInt32: 1).
	
	output0 := (const findMinOn: axis0) output: 0.
	output1 := (const findMinOn: axis1) output: 0.
	
	session := TFSession on: graph.
	results := session
		runOutputs: {output0. output1}.
	
	first := results at: 1.
	second := results at: 2.
	
	self assert: first shape equals: #(2).
	first allInt64s with: #(0 2) do: [:r :t |
		self assert: r equals: t].
	
	self assert: second shape equals: #(3).
	second allInt64s with: #(0 0 1) do: [:r :t |
		self assert: r equals: t].! !

!TensorFlowOperationsTest methodsFor: 'binary operations' stamp: 'gera 7/3/2019 03:35:50'!
testOperationAssignSub
	| graph var output pi result session |
	graph := TFGraph create.
	var := graph variable: 'var' initialValue: 1.0 asTensor.
	pi := graph const: 'pi' value: TFTensor pi.
	output := var -= pi.
	
	session := TFSession on: graph.
	graph initializeOn: session.
	result := session runOutput: (output output: 0).
	
	self assert: 1-Float pi closeTo: result allFloats first! !

!TensorFlowOperationsTest methodsFor: 'binary operations' stamp: 'gera 7/3/2019 00:34:29'!
testOperationCastTo
	| graph template cast input result |
	
	template := #((1.1 1.2) (2.1 2.2)).	
	graph := TFGraph create.
	
	input := graph const: template asFloatTensor.
	cast := input castTo: TFTensor typeInt32.

	result := graph runOutput: cast output.

	self assert: template truncated equals: result asNumbers.! !

!TensorFlowOperationsTest methodsFor: 'binary operations' stamp: 'gera 7/3/2019 03:35:50'!
testOperationDiv
	| graph a b session constA constB result output template |
	
	a := TFTensor fromFloats: #((1.1 1.2) (2.1 2.2)).
	b := TFTensor fromFloats: #((2.0 4.0) (8.0 16.0)).
	
	graph := TFGraph create.
	
	constA := graph const: 'a' value: a.
	constB := graph const: 'b' value: b.
	
	output := constA @/ constB.
	template := a allFloats / b allFloats.
	
	session := TFSession on: graph.
	result := session runOutput: (output output: 0).
	
	self assert: result shape equals: a shape.
	result allFloats with: template do: [:r :t |
		self assert: r closeTo: t].! !

!TensorFlowOperationsTest methodsFor: 'binary operations' stamp: 'gera 7/3/2019 03:35:50'!
testOperationDivScalar
	| graph a b session constA constB result div template |
	
	a := TFTensor fromFloats: #((1.1 1.2) (2.1 2.2)).
	b := TFTensor fromFloats: 2.0.
	
	graph := TFGraph create.
	
	constA := graph const: 'a' value: a.
	constB := graph const: 'b' value: b.
	
	div := constA @/ constB.
	template := a allFloats / 2.0.
	
	session := TFSession on: graph.
	result := session runOutput: (div output: 0).
	
	self assert: result shape equals: a shape.
	result allFloats with: template do: [:r :t |
		self assert: r closeTo: t].! !

!TensorFlowOperationsTest methodsFor: 'binary operations' stamp: 'gera 7/3/2019 03:35:50'!
testOperationGreater
	| graph a b session constA constB result  template comparisons |
	
	a := TFTensor fromFloats: #((1.1 1.2) (3.14 2.2)).
	b := TFTensor fromFloats: #((3.14 3.14) (2.1 3.14)).
	
	graph := TFGraph create.
	
	constA := graph const: 'a' value: a.
	constB := graph const: 'b' value: b.
	
	comparisons := constA > constB.
	template := a allFloats with: b allFloats collect:[:x :y| x > y].
	
	session := TFSession on: graph.
	result := session runOutput: (comparisons output: 0).
	
	self assert: result shape equals: a shape.
	result allElements with: template do: [:r :t |
		self assert: r equals: t].! !

!TensorFlowOperationsTest methodsFor: 'binary operations' stamp: 'gera 7/3/2019 03:35:50'!
testOperationMatMul
	| graph a b session constA constB result output template |
	
	a := TFTensor fromFloats: #((1.1 1.2) (2.1 2.2)).
	b := TFTensor fromFloats: #((2.0 4.0) (8.0 16.0)).
	
	graph := TFGraph create.
	
	constA := graph const: 'a' value: a.
	constB := graph const: 'b' value: b.

	output := constA * constB.
	template := {
		1.1*2.0+(1.2*8.0).		1.1*4.0+(1.2*16.0).
		2.1*2.0+(2.2*8.0).		2.1*4.0+(2.2*16.0).
	}.

	session := TFSession on: graph.
	result := session runOutput: (output output: 0).

	self assert: result shape equals: a shape.
	result allFloats with: template do: [:r :t |
		self assert: r closeTo: t].! !

!TensorFlowOperationsTest methodsFor: 'binary operations' stamp: 'gera 7/3/2019 03:35:50'!
testOperationMatMul1x1
	| graph a b session constA constB output result |
	
	a := #((1.1)) asFloatTensor.
	b := #((3.14)) asFloatTensor.
	
	graph := TFGraph create.
	
	constA := graph const: 'a' value: a.
	constB := graph const: 'b' value: b.

	output := constA * constB.

	session := TFSession on: graph.
	result := session runOutput: (output output: 0).

	self assert: 1.1 * 3.14 closeTo: result allFloats first! !

!TensorFlowOperationsTest methodsFor: 'binary operations' stamp: 'gera 7/3/2019 03:35:50'!
testOperationMatMul1x1Wrong
	| graph a b session constA constB output |
	
	a := #(1.1) asFloatTensor.
	b := #(2.02) asFloatTensor.
	
	graph := TFGraph create.
	
	constA := graph const: 'a' value: a.
	constB := graph const: 'b' value: b.

	output := constA * constB.

	session := TFSession on: graph.
	self
		should: [session runOutput: (output output: 0)]
		raiseError: 'INVALID_ARGUMENT: In[0] is not a matrix
	 [[Node: MatMul_2 = MatMul[T=DT_FLOAT, _class=[], transpose_a=false, transpose_b=false, _device="/job:localhost/replica:0/task:0/cpu:0"](a, b)]]'
! !

!TensorFlowOperationsTest methodsFor: 'binary operations' stamp: 'gera 7/3/2019 03:35:50'!
testOperationMatMulABTransposed
	| graph a b session constA constB result output template |
	
	a := TFTensor fromFloats: #((1.1 1.2) (2.1 2.2)).
	b := TFTensor fromFloats: #((2.0 4.0) (8.0 16.0)).
	
	graph := TFGraph create.
	
	constA := graph const: 'a' value: a.
	constB := graph const: 'b' value: b.
	
	output := constA \*\ constB.
	template := {
		1.1*2.0+(2.1*4.0).		1.1*8.0+(2.1*16.0).
		1.2*2.0+(2.2*4.0).		1.2*8.0+(2.2*16.0).
	}.

	session := TFSession on: graph.
	result := session runOutput: (output output: 0).
	
	self assert: result shape equals: a shape.
	result allFloats with: template do: [:r :t |
		self assert: r closeTo: t].! !

!TensorFlowOperationsTest methodsFor: 'binary operations' stamp: 'gera 7/3/2019 03:35:50'!
testOperationMatMulATransposed
	| graph a b session constA constB result output template |
	
	a := TFTensor fromFloats: #((1.1 1.2) (2.1 2.2)).
	b := TFTensor fromFloats: #((2.0 4.0) (8.0 16.0)).
	
	graph := TFGraph create.
	
	constA := graph const: 'a' value: a.
	constB := graph const: 'b' value: b.
	
	output := constA \* constB.
	template := {
		1.1*2.0+(2.1*8.0).		1.1*4.0+(2.1*16.0).
		1.2*2.0+(2.2*8.0).		1.2*4.0+(2.2*16.0).
	}.

	session := TFSession on: graph.
	result := session runOutput: (output output: 0).
	
	self assert: result shape equals: a shape.
	result allFloats with: template do: [:r :t |
		self assert: r closeTo: t].! !

!TensorFlowOperationsTest methodsFor: 'binary operations' stamp: 'gera 7/3/2019 03:35:50'!
testOperationMatMulBTransposed
	| graph a b session constA constB result output template |
	
	a := TFTensor fromFloats: #((1.1 1.2) (2.1 2.2)).
	b := TFTensor fromFloats: #((2.0 4.0) (8.0 16.0)).
	
	graph := TFGraph create.
	
	constA := graph const: 'a' value: a.
	constB := graph const: 'b' value: b.
	
	output := constA *\ constB.
	template := {
		1.1*2.0+(1.2*4.0).		1.1*8.0+(1.2*16.0).
		2.1*2.0+(2.2*4.0).		2.1*8.0+(2.2*16.0).
	}.

	session := TFSession on: graph.
	result := session runOutput: (output output: 0).
	
	self assert: result shape equals: a shape.
	result allFloats with: template do: [:r :t |
		self assert: r closeTo: t].! !

!TensorFlowOperationsTest methodsFor: 'binary operations' stamp: 'gera 7/3/2019 00:34:29'!
testOperationMatMulWrongType
	| graph a b constA constB |
	
	a := TFTensor fromFloats: #((1.1 1.2) (2.1 2.2)).
	b := #((2 4) (8 16)) asInt32Tensor.
	
	graph := TFGraph create.
	
	constA := graph const: 'a' value: a.
	constB := graph const: 'b' value: b.

	self
		should: [constA * constB]
		raiseError: 'INVALID_ARGUMENT: Inconsistent values for attr ''T'' DT_FLOAT vs. DT_INT32 while building NodeDef ''MatMul_2'' using Op<name=MatMul; signature=a:T, b:T -> product:T; attr=transpose_a:bool,default=false; attr=transpose_b:bool,default=false; attr=T:type,allowed=[DT_HALF, DT_FLOAT, DT_DOUBLE, DT_INT32, DT_COMPLEX64, DT_COMPLEX128]>'.! !

!TensorFlowOperationsTest methodsFor: 'binary operations' stamp: 'gera 7/3/2019 03:35:50'!
testOperationMeanOn
	| graph a session mean0 mean01 mean1 results meanNone |
	
	graph := TFGraph create.
	a := graph const: #(
		(1.1 1.2)
		(2.1 2.2)
		(3.1 0.2)
	) asFloatTensor.
		
	mean0 := a meanOn: #(0) asInt32Tensor.
	mean1 := a meanOn: #(1) asInt32Tensor.
	mean01 := a meanOn: #(0 1) asInt32Tensor.
	meanNone := a meanOn: #() asInt32Tensor.
		
	session := TFSession on: graph.
	results := session runOutputs: {mean0 output: 0. mean1 output: 0. mean01 output: 0. meanNone output: 0}." mean01bis output: 0}."
	
	self assert: #(2) equals: results first shape.
	self assert: #(3) equals: results second shape.
	self assert: #() equals: results third shape.
	self assert: #(3 2) equals: results fourth shape.

	self assertAll: #(2.1 1.2) closeTo: results first allFloats.
	self assertAll: #(1.15 2.15 1.65) closeTo: results second allFloats.
	self assertAll: #(1.65) closeTo: results third allFloats.
	self
		assertAll: #(
			1.1 1.2
			2.1 2.2
			3.1 0.2)
		closeTo: results fourth allFloats.
	! !

!TensorFlowOperationsTest methodsFor: 'binary operations' stamp: 'gera 7/3/2019 03:35:50'!
testOperationMinus
	| graph a b session constA constB result sum template |
	
	a := TFTensor fromFloats: #((1.1 1.2) (2.1 2.2)).
	b := TFTensor fromFloats: #((3.14 3.14) (3.14 3.14)).
	
	graph := TFGraph create.
	
	constA := graph const: 'a' value: a.
	constB := graph const: 'b' value: b.
	
	sum := constA - constB.
	
	session := TFSession on: graph.
	result := session runOutput: (sum output: 0).
	
	template := a allFloats - b allFloats.

	self assert: result shape equals: a shape.
	result allFloats with: template do: [:r :t |
		self assert: r closeTo: t].! !

!TensorFlowOperationsTest methodsFor: 'binary operations' stamp: 'gera 7/3/2019 03:35:50'!
testOperationMod
	| graph a b session constA constB result output template |
	
	a := TFTensor fromFloats: #((10.0 11.0) (12.0 13.0)).
	b := TFTensor fromFloats: #((3.0 2.0) (7.0 5.0)).
	
	graph := TFGraph create.
	
	constA := graph const: 'a' value: a.
	constB := graph const: 'b' value: b.
	
	output := constA \\ constB.
	template := a allFloats \\ b allFloats.
	
	session := TFSession on: graph.
	result := session runOutput: (output output: 0).
	
	self assert: result shape equals: a shape.
	result allFloats with: template do: [:r :t |
		self assert: r closeTo: t].! !

!TensorFlowOperationsTest methodsFor: 'binary operations' stamp: 'gera 7/3/2019 03:35:50'!
testOperationMul
	| graph a b session constA constB result output template |
	
	a := TFTensor fromFloats: #((1.1 1.2) (2.1 2.2)).
	b := TFTensor fromFloats: #((2.0 4.0) (8.0 16.0)).
	
	graph := TFGraph create.
	
	constA := graph const: 'a' value: a.
	constB := graph const: 'b' value: b.
	
	output := constA @* constB.
	template := a allFloats * b allFloats.
	
	session := TFSession on: graph.
	result := session runOutput: (output output: 0).
	
	self assert: result shape equals: a shape.
	result allFloats with: template do: [:r :t |
		self assert: r closeTo: t].! !

!TensorFlowOperationsTest methodsFor: 'binary operations' stamp: 'gera 7/3/2019 03:35:50'!
testOperationMulScalar
	| graph a b session constA constB result mul template |
	
	a := TFTensor fromFloats: #((1.1 1.2) (2.1 2.2)).
	b := TFTensor fromFloats: 2.0.
	
	graph := TFGraph create.
	
	constA := graph const: 'a' value: a.
	constB := graph const: 'b' value: b.
	
	mul := constA @* constB.
	template := a allFloats * 2.0.
	
	session := TFSession on: graph.
	result := session runOutput: (mul output: 0).
	
	self assert: result shape equals: a shape.
	result allFloats with: template do: [:r :t |
		self assert: r closeTo: t].! !

!TensorFlowOperationsTest methodsFor: 'binary operations' stamp: 'gera 7/3/2019 03:35:50'!
testOperationPlus
	| graph a b session constA constB result sum template |
	
	a := TFTensor fromFloats: #((1.1 1.2) (2.1 2.2)).
	b := TFTensor fromFloats: #((3.14 3.14) (3.14 3.14)).
	
	graph := TFGraph create.
	
	constA := graph const: 'a' value: a.
	constB := graph const: 'b' value: b.
	
	sum := constA + constB.
	template := a allFloats + b allFloats.
	
	session := TFSession on: graph.
	result := session runOutput: (sum output: 0).
	
	self assert: result shape equals: a shape.
	result allFloats with: template do: [:r :t |
		self assert: r closeTo: t].! !

!TensorFlowOperationsTest methodsFor: 'binary operations' stamp: 'gera 7/3/2019 03:35:50'!
testOperationPlusNegated
	| graph a b session constA constB result sum template |
	
	a := TFTensor fromFloats: #((1.1 1.2) (2.1 2.2)).
	b := TFTensor fromFloats: #((3.14 3.14) (3.14 3.14)).
	
	graph := TFGraph create.
	
	constA := graph const: 'a' value: a.
	constB := graph const: 'b' value: b.
	
	sum := constA + constB negated.
	
	session := TFSession on: graph.
	result := session runOutput: (sum output: 0).
	
	template := a allFloats - b allFloats.

	self assert: result shape equals: a shape.
	result allFloats with: template do: [:r :t |
		self assert: r closeTo: t].! !

!TensorFlowOperationsTest methodsFor: 'binary operations' stamp: 'gera 7/3/2019 03:35:50'!
testOperationPlusNegatedNegated
	| graph a b session constA constB result sum template |
	
	a := TFTensor fromFloats: #((1.1 1.2) (2.1 2.2)).
	b := TFTensor fromFloats: #((3.14 3.14) (3.14 3.14)).
	
	graph := TFGraph create.
	
	constA := graph const: 'a' value: a.
	constB := graph const: 'b' value: b.
	
	sum := (constA + constB negated) negated.
	template := (a allFloats + b allFloats negated) negated.
	
	session := TFSession on: graph.
	result := session runOutput: (sum output: 0).
	
	self assert: result shape equals: a shape.
	result allFloats with: template do: [:r :t |
		self assert: r closeTo: t].! !

!TensorFlowOperationsTest methodsFor: 'binary operations' stamp: 'gera 7/3/2019 03:35:50'!
testOperationPlusNegatedNegatedPlus
	| graph a b session constA constB result sum template |
	
	a := TFTensor fromFloats: #((1.1 1.2) (2.1 2.2)).
	b := TFTensor fromFloats: #((3.14 3.14) (3.14 3.14)).
	
	graph := TFGraph create.
	
	constA := graph const: 'a' value: a.
	constB := graph const: 'b' value: b.
	
	sum := (constA + constB negated) negated + constB negated.
	template := (a allFloats + b allFloats negated) negated + b allFloats negated.
	
	session := TFSession on: graph.
	result := session runOutput: (sum output: 0).
	
	self assert: result shape equals: a shape.
	
	result allFloats with: a allFloats negated do: [:r :t |
		self assert: r closeTo: t].
	
	result allFloats with: template do: [:r :t |
		self assert: r closeTo: t].! !

!TensorFlowOperationsTest methodsFor: 'binary operations' stamp: 'gera 7/3/2019 03:35:50'!
testOperationSparseSoftmaxCrossEntropyWithLogits
	| graph label prediction results session xentropy gradient lastVector loss|
	
	graph := TFGraph create.
	prediction := graph const: #(
		(0.1 0.2 0.3 0.9 0.0 0.5 0.4)
		(0.1 0.2 0.1 0.1 0.8 0.1 0.1)
		(0 0 0 0 0 0 1)
	) asFloatTensor.
	label := graph const: #(3 4 6) asInt32Tensor.
	
	xentropy := prediction sparseSoftmaxCrossEntropyWithLogits: label.
	
	session := TFSession on: graph.
	results := session runOutputs: {xentropy output: 0. xentropy output: 1}.
	
	loss := results first.
	self assert: #(3) equals: loss shape.
	self assert: (loss allFloats first between: 1 and: 1.5).
	self assert: (loss allFloats second between: 1 and: 1.5).
	self assert: (loss allFloats third between: 1 and: 1.2).
	
	gradient := results second.
	self assert: #(3 7) equals: gradient shape.
	lastVector := gradient allFloats last: 7.
	self assert: lastVector last < 0.
	lastVector allButLastDo: [:x | self assert: x > 0] 
! !

!TensorFlowOperationsTest methodsFor: 'binary operations' stamp: 'gera 7/3/2019 03:35:50'!
testOperationSparseSoftmaxCrossEntropyWithLogitsUseOutput
	| graph label prediction results session xentropy gradient lastVector loss|
	
	graph := TFGraph create.
	prediction := graph const: #(
		(0.1 0.2 0.3 0.9 0.0 0.5 0.4)
		(0.1 0.2 0.1 0.1 0.8 0.1 0.1)
		(0 0 0 0 0 0 1)
	) asFloatTensor.
	label := graph const: #(3 4 6) asInt32Tensor.
	
	xentropy := prediction sparseSoftmaxCrossEntropyWithLogits: label.
	
	session := TFSession on: graph.
	results := session runOutputs: {xentropy output. (xentropy useOutput: 1) output}.
	
	loss := results first.
	self assert: #(3) equals: loss shape.
	self assert: (loss allFloats first between: 1 and: 1.5).
	self assert: (loss allFloats second between: 1 and: 1.5).
	self assert: (loss allFloats third between: 1 and: 1.2).
	
	gradient := results second.
	self assert: #(3 7) equals: gradient shape.
	lastVector := gradient allFloats last: 7.
	self assert: lastVector last < 0.
	lastVector allButLastDo: [:x | self assert: x > 0] 
! !

!TensorFlowOperationsTest methodsFor: 'binary operations' stamp: 'gera 7/3/2019 03:35:50'!
testOperationSub
	| graph a b session constA constB result sub template |
	
	a := TFTensor fromFloats: #((1.1 1.2) (2.1 2.2)).
	b := TFTensor fromFloats: #((3.14 3.14) (3.14 3.14)).
	
	graph := TFGraph create.
	
	constA := graph const: 'a' value: a.
	constB := graph const: 'b' value: b.
	
	sub := constA - constB.
	
	session := TFSession on: graph.
	result := session runOutput: (sub output: 0).
	
	template := a allFloats - b allFloats.

	self assert: result shape equals: a shape.
	result allFloats with: template do: [:r :t |
		self assert: r closeTo: t].! !

!TensorFlowOperationsTest methodsFor: 'binary operations' stamp: 'gera 7/3/2019 03:35:50'!
testOperationSum
	| graph a session const result output0 output1 axis0 axis1 template sumOn1 sumOn0 |
	
	template := #(
		(1.1 1.2)
		(2.1 2.2)
		(3.1 0.2)
	).
	
	sumOn0 := template sum.
	sumOn1 := template collect: [:line | line sum].
	
	a := TFTensor fromFloats: template.
	
	graph := TFGraph create.
	
	const := graph const: 'a' value: a.
	axis0 := graph const: 'axis0' value: (TFTensor fromInt32: 0).
	axis1 := graph const: 'axis1' value: (TFTensor fromInt32: 1).
	
	output0 := const sumOn: axis0.
	output1 := const sumOn: axis1.
	
	session := TFSession on: graph.
	result := session
		runOperation: output0
		output: (output0 output: 0).
	
	self assert: result shape equals: #(2).
	result allFloats with: sumOn0 do: [:r :t |
		self assert: r closeTo: t].
	
	result := session
		runOperation: output1
		output: (output1 output: 0).
	
	self assert: result shape equals: #(3).
	result allFloats with: sumOn1 do: [:r :t |
		self assert: r closeTo: t].! !

!TensorFlowOperationsTest methodsFor: 'binary operations' stamp: 'gera 7/3/2019 03:35:50'!
testOperationSumOnOutputs
	| graph a session const results first second output0 output1 axis0 axis1 template sumOn1 sumOn0 |
	
	template := #(
		(1.1 1.2)
		(2.1 2.2)
		(3.1 0.2)
	).
	
	sumOn0 := template sum.
	sumOn1 := template collect: [:line | line sum].
	
	a := TFTensor fromFloats: template.
	
	graph := TFGraph create.
	
	const := graph const: 'a' value: a.
	axis0 := graph const: 'axis0' value: (TFTensor fromInt32: 0).
	axis1 := graph const: 'axis1' value: (TFTensor fromInt32: 1).
	
	output0 := const sumOn: axis0.
	output1 := const sumOn: axis1.
	
	session := TFSession on: graph.
	results := session runOutputs: {output0 output: 0. output1 output: 0}.
	first := results at: 1.
	second := results at: 2.
	
	self assert: first shape equals: #(2).
	first allFloats with: sumOn0 do: [:r :t |
		self assert: r closeTo: t].
	
	self assert: second shape equals: #(3).
	second allFloats with: sumOn1 do: [:r :t |
		self assert: r closeTo: t].! !

!TensorFlowOperationsTest methodsFor: 'unary operations' stamp: 'gera 7/3/2019 03:35:50'!
testOperationAbs
	| graph template session const result output |
	
	template := TFTensor fromFloats: #((-1.1 1.2) (-2.1 -2.2)).
	graph := TFGraph create.
	
	const := graph const: 'const' value: template.
	output := const abs output: 0.
	
	session := TFSession on: graph.
	result := session runOutput: output.
	
	self assert: result shape equals: template shape.
	self assert: result allFloats equals: template allFloats abs.
	
	output free.! !

!TensorFlowOperationsTest methodsFor: 'unary operations' stamp: 'gera 7/3/2019 03:35:50'!
testOperationAcos
	| graph template session const result output intput |
	
	template := #(1.1 0.2 2.34 0.717273).
	intput := TFTensor fromFloats: template cos.
	graph := TFGraph create.
	
	const := graph const: 'const' value: intput.
	output := const arcCos output: 0.
	
	session := TFSession on: graph.
	result := session runOutput: output.
	
	self assert: result shape equals: intput shape.
	result allFloats with: template do: [:res :temp |
		self assert: res closeTo: temp].
	
	output free.! !

!TensorFlowOperationsTest methodsFor: 'unary operations' stamp: 'gera 7/3/2019 03:35:50'!
testOperationAsin
	| graph template session const result output intput |
	
	template := #(1.1 0.2 -1.34 0.717273).
	intput := TFTensor fromFloats: template sin.
	graph := TFGraph create.
	
	const := graph const: 'const' value: intput.
	output := const arcSin output: 0.
	
	session := TFSession on: graph.
	result := session runOutput: output.
	
	self assert: result shape equals: intput shape.
	result allFloats with: template do: [:res :temp |
		self assert: res closeTo: temp].
	
	output free.! !

!TensorFlowOperationsTest methodsFor: 'unary operations' stamp: 'gera 7/3/2019 03:35:50'!
testOperationAtan
	| graph template session const result output intput |
	
	template := #(1.1 0.2 -1.34 0.717273).
	intput := TFTensor fromFloats: template tan.
	graph := TFGraph create.
	
	const := graph const: 'const' value: intput.
	output := const arcTan output: 0.
	
	session := TFSession on: graph.
	result := session runOutput: output.
	
	self assert: result shape equals: intput shape.
	result allFloats with: template do: [:res :temp |
		self assert: res closeTo: temp].
	
	output free.! !

!TensorFlowOperationsTest methodsFor: 'unary operations' stamp: 'gera 7/3/2019 03:35:50'!
testOperationCos
	| graph template session const result output intput |

	template := #(0.1 0.0 -0.94 0.717273).
	intput := TFTensor fromFloats: template arcCos.
	graph := TFGraph create.
	
	const := graph const: 'const' value: intput.
	output := const cos output: 0.
	
	session := TFSession on: graph.
	result := session runOutput: output.
	
	self assert: result shape equals: intput shape.
	result allFloats with: template do: [:res :temp |
		self assert: res closeTo: temp].
	
	output free.! !

!TensorFlowOperationsTest methodsFor: 'unary operations' stamp: 'gera 7/3/2019 03:35:50'!
testOperationExp
	| graph inputs template session const result output |
	
	template := #(-1.1 1.2).
	
	inputs := TFTensor fromFloats: template.
	graph := TFGraph create.
	
	const := graph const: 'const' value: inputs.
	output := const exp output: 0.
	
	session := TFSession on: graph.
	result := session runOutput: output.
	
	self assert: result shape equals: inputs shape.
	result allFloats with: template exp do: [:res :temp |
		self assert: res closeTo: temp].
	
	output free.! !

!TensorFlowOperationsTest methodsFor: 'unary operations' stamp: 'gera 7/3/2019 03:35:50'!
testOperationIAlias
	| graph template const result same1 same2 |
	graph := TFGraph create.
	template := TFTensor fromFloats: #(
		((1) (2) (3)) 
		((4) (5) (6))
	).
	const := graph const: 'const' value: template.
	same1 := const alias: 'another_name'.
	same2 := graph operationNamed: 'another_name'.
	
	result := (TFSession on: graph) runOutput: (same1 output: 0).
	
	self assert: template shape equals: result shape.
	self assert: #(1 2 3 4 5 6) equals: result allFloats.
	
	result := (TFSession on: graph) runOutput: (same2 output: 0).
	
	self assert: template shape equals: result shape.
	self assert: #(1 2 3 4 5 6) equals: result allFloats.
	
! !

!TensorFlowOperationsTest methodsFor: 'unary operations' stamp: 'gera 7/3/2019 03:35:50'!
testOperationIdentity
	| graph template const result same |
	graph := TFGraph create.
	template := TFTensor fromFloats: #(
		((1) (2) (3)) 
		((4) (5) (6))
	).
	const := graph const: 'const' value: template.
	same := const identity output: 0.
	
	result := (TFSession on: graph) runOutput: same.
	
	self assert: template shape equals: result shape.
	self assert: #(1 2 3 4 5 6) equals: result allFloats.! !

!TensorFlowOperationsTest methodsFor: 'unary operations' stamp: 'gera 7/3/2019 03:35:50'!
testOperationMatrixInverse
	| graph template session const result identity inverse |
	
	template := TFTensor fromFloats: #((-1.1 1.2) (-2.1 -2.2)).
	graph := TFGraph create.
	
	const := graph const: 'const' value: template.
	inverse := const inverse.
	identity := const * inverse output: 0.
	
	session := TFSession on: graph.
	result := session runOutput: identity.
	
	self assert: result shape equals: template shape.
	result allFloats with: #(1.0 0.0 0.0 1.0) do: [:real :expected |
		self assert: real closeTo: expected].
	
	identity free.! !

!TensorFlowOperationsTest methodsFor: 'unary operations' stamp: 'gera 7/3/2019 03:35:50'!
testOperationNegated
	| graph template session const result negated |
	
	template := TFTensor fromFloats: #((1.1 1.2) (2.1 2.2)).
	graph := TFGraph create.
	
	const := graph const: 'const' value: template.
	negated := const negated output: 0.
	
	session := TFSession on: graph.
	result := session runOutput: negated.
	
	self assert: result shape equals: template shape.
	self assert: result allFloats equals: template allFloats negated.
	
	negated free.! !

!TensorFlowOperationsTest methodsFor: 'unary operations' stamp: 'gera 7/3/2019 03:35:50'!
testOperationRelu
	| graph template session const result output |
	
	template := #(
		(-1.1	1.2)
		(-2.1	2.2)
		(0		-0)) asFloatTensor.
		
	graph := TFGraph create.
	
	const := graph const: 'const' value: template.
	output := const rectified output: 0.
	
	session := TFSession on: graph.
	result := session runOutput: output.
	
	self assert: result shape equals: template shape.
	self assert: result allFloats equals: (template allFloats collect: [:each | each max: 0]).
! !

!TensorFlowOperationsTest methodsFor: 'unary operations' stamp: 'gera 7/3/2019 03:35:50'!
testOperationReluGrad
	| graph template session result previousGradient rectifiedGrad expected previousGradientTemplate |
	
	template := #(
		(-1.1	1.2)
		(-2.1	2.2)
		(0		-0)) asFloatTensor.
		

	graph := TFGraph create.
	
	previousGradientTemplate := #(
		(0	1)
		(1	0)
		(1		1)) asFloatTensor.
		
	previousGradient := previousGradientTemplate asOperationOn: graph.
	rectifiedGrad := previousGradient timesRectifiedGradOf: template.
	
	session := TFSession on: graph.
	result := session runOutput: rectifiedGrad output.
	
	self assert: result shape equals: template shape.
	expected := template allFloats with: previousGradientTemplate allFloats collect:[:x :y| (x sign max: 0) * y].
	self assert: result allFloats equals: expected.
! !

!TensorFlowOperationsTest methodsFor: 'unary operations' stamp: 'gera 7/3/2019 03:35:50'!
testOperationShape
	| graph template session const result shape |
	
	template := TFTensor fromFloats: #(((((-1.1 1.2 1)) ((-2.1 -2.2 0))))).
	graph := TFGraph create.
	
	const := graph const: 'const' value: template.
	shape := const shape output.
	
	session := TFSession on: graph.
	result := session runOutput: shape.
	
	self assert: {template rank} equals: result shape.
	self assert: #(1 1 2 1 3) equals: result asNumbers
! !

!TensorFlowOperationsTest methodsFor: 'unary operations' stamp: 'gera 7/3/2019 03:35:50'!
testOperationSigmoid
	| graph inputs template session const result output transformed |
	
	template := #(-1.1 1.2 0 4).
	
	inputs := TFTensor fromFloats: template.
	graph := TFGraph create.
	
	const := graph const: 'const' value: inputs.
	output := const sigmoid output: 0.
	
	session := TFSession on: graph.
	result := session runOutput: output.
	
	transformed := template collect:[:x| (x negated exp + 1) reciprocal].
	self assert: result shape equals: inputs shape.
	result allFloats with: transformed do: [:res :temp |
		self assert: res closeTo: temp].
	
	output free.! !

!TensorFlowOperationsTest methodsFor: 'unary operations' stamp: 'gera 7/3/2019 03:35:50'!
testOperationSin
	| graph template session const result output intput |

	template := #(0.1 0.0 -0.94 0.717273).
	intput := TFTensor fromFloats: template arcSin.
	graph := TFGraph create.
	
	const := graph const: 'const' value: intput.
	output := const sin output: 0.
	
	session := TFSession on: graph.
	result := session runOutput: output.
	
	self assert: result shape equals: intput shape.
	result allFloats with: template do: [:res :temp |
		self assert: res closeTo: temp].
	
	output free.! !

!TensorFlowOperationsTest methodsFor: 'unary operations' stamp: 'gera 7/3/2019 03:35:50'!
testOperationSquared
	| graph template session const result output |
	
	template := TFTensor fromFloats: #((-1.1 1.2) (-2.1 -2.2)).
	graph := TFGraph create.
	
	const := graph const: 'const' value: template.
	output := const squared output: 0.
	
	session := TFSession on: graph.
	result := session runOutput: output.
	
	self assert: result shape equals: template shape.
	result allFloats with: template allFloats squared do: [:res :temp |
		self assert: res closeTo: temp].
	
	output free.! !

!TensorFlowOperationsTest methodsFor: 'unary operations' stamp: 'gera 7/3/2019 03:35:50'!
testOperationTan
	| graph template session const result output intput |

	template := #(0.1 0.0 -0.94 0.717273).
	intput := TFTensor fromFloats: template arcTan.
	graph := TFGraph create.
	
	const := graph const: 'const' value: intput.
	output := const tan output: 0.
	
	session := TFSession on: graph.
	result := session runOutput: output.
	
	self assert: result shape equals: intput shape.
	result allFloats with: template do: [:res :temp |
		self assert: res closeTo: temp].
	
	output free.! !

!TensorFlowOperationsTest methodsFor: 'unary operations' stamp: 'gera 7/3/2019 03:35:50'!
testSoftmax
	| graph template session const result output denominator |
	
	template := TFTensor fromFloats: {{0. 3  ln}}.
	graph := TFGraph create.
	
	const := graph const: 'const' value: template.
	output := const softmax output.
	
	session := TFSession on: graph.
	result := session runOutput: output.

	self assert: result shape equals: template shape.
	
	denominator := template allFloats exp sum. 
	result allFloats with: template allFloats exp / denominator do: [:res :temp |
		| delta |
		delta := (res  - temp) abs.
		self assert: delta < 0.001].
	
	output free.
! !

!TensorFlowCAPI methodsFor: 'tensor' stamp: 'gera 12/27/2016 11:56:22'!
allocateTensorType: anInteger shape: aLonnLongArray rank: dimCount length: len
	<cdecl: TFTensor* 'TF_AllocateTensor' (ulong UInt64Array* ulonglong ulonglong)>
	^ self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'tensor' stamp: 'gera 12/23/2016 20:54:17'!
deleteTensor: aTFTensor
	<cdecl: void 'TF_DeleteTensor' (TFTensor*)>
	^ self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'tensor' stamp: 'jb 12/26/2016 12:15:05'!
tensor: aTFTensor sizeOn: dimension
	<cdecl: longlong 'TF_Dim' (TFTensor* ulonglong)>
	^ self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'tensor' stamp: 'gera 1/5/2017 23:36:32'!
tensorByteSize: aTFTensor
	<cdecl: ulonglong 'TF_TensorByteSize' (TFTensor*)>
	^ self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'tensor' stamp: 'gera 12/27/2016 09:17:49'!
tensorData: aTFTensor
	<cdecl: void* 'TF_TensorData' (TFTensor*)>
	^ self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'tensor' stamp: 'jb 12/26/2016 12:04:40'!
tensorRank: aTFTensor
	<cdecl: longlong 'TF_NumDims' (TFTensor*)>
	^ self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'tensor' stamp: 'gera 12/23/2016 20:54:40'!
tensorType: aTFTensor
	<cdecl: ulonglong 'TF_TensorType' (TFTensor*)>
	^ self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'session' stamp: 'gera 12/21/2016 23:26:55'!
closeSession: aTFSession status: aTFStatus
	<cdecl: void 'TF_CloseSession' (TFSession* TFStatus*)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'session' stamp: 'gera 12/21/2016 23:23:33'!
deleteSession: aTFSession status: aTFStatus
	<cdecl: void 'TF_DeleteSession' (TFSession* TFStatus*)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'session' stamp: 'gera 12/21/2016 23:21:09'!
newSession: aTFGraph options:  aTFSessionOptions status: aTFStatus
	<cdecl: TFSession* 'TF_NewSession' (TFGraph* TFSessionOptions* TFStatus*)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'session' stamp: 'gera 7/6/2019 10:41:22'!
runSession: aTFSession options: runOptionsTFBufferOrNil
	inputs: inTFOutputArray values: inTFTensorArray count: inCount
	outputs: outTFOutputArray values: outTFTensorArrayPtr count: outCount
	targets: targetTFOperationArrayOrNil count: targetCount
	metadata: metadataTFBuffer
	status: aTFStatus
	
	<cdecl: void 'TF_SessionRun' (
		TFSession* TFBuffer*
		TFOutputArray* TFTensorPtrArray* long
		TFOutputArray* TFTensorPtrArray* long
		TFOperationPtrArray* long	
		TFBuffer*
		TFStatus*)>
	^ self externalCallFailed
! !

!TensorFlowCAPI methodsFor: 'buffer' stamp: 'gera 12/22/2016 00:38:51'!
deleteBuffer: aTFBuffer
	<cdecl: void 'TF_DeleteBuffer' (TFBuffer*)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'buffer' stamp: 'gera 12/22/2016 00:38:18'!
newBufferFromString: aString size: anInteger
	<cdecl: TFBuffer* 'TF_NewBufferFromString' (char* ulong)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'graph' stamp: 'gera 12/21/2016 22:32:51'!
deleteGraph: aTFGraph
	<cdecl: void 'TF_DeleteGraph' (TFGraph*)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'graph' stamp: 'gera 7/3/2019 00:47:14'!
graph: aTFGraph getOperationNamed: aString
	| answer |
	answer := self graph: aTFGraph getOperationNamedAsVoid: aString.
	answer := TFOperation fromHandle: answer getHandle.
	answer graph: aTFGraph.
	^ answer! !

!TensorFlowCAPI methodsFor: 'graph' stamp: 'gera 1/19/2017 23:12:26'!
graph: aTFGraph getOperationNamedAsVoid: aString
	<cdecl: void* 'TF_GraphOperationByName' (TFGraph* char*)>
	^ self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'graph' stamp: 'gera 1/23/2017 11:31:00'!
graph: aTFGraph getRankOf:  aTFOutputOrInput status: status
	<cdecl: ulonglong 'TF_GraphGetTensorNumDims' (TFGraph* TFOutput TFStatus*)>
	^ self externalCallFailed
! !

!TensorFlowCAPI methodsFor: 'graph' stamp: 'gera 1/23/2017 11:23:19'!
graph: aTFGraph getShapeOf:  aTFOutputOrInput into: anInt64Array size: anInteger status: status
	<cdecl: ulonglong 'TF_GraphGetTensorShape' (TFGraph* TFOutput Int64Array* ulonglong TFStatus*)>
	^ self externalCallFailed
! !

!TensorFlowCAPI methodsFor: 'graph' stamp: 'gera 1/23/2017 15:12:33'!
graph: aTFGraph setShapeOf:  aTFOutputOrInput to: anInt64Array size: anInteger status: status
	<cdecl: void 'TF_GraphSetTensorShape' (TFGraph* TFOutput Int64Array* ulonglong TFStatus*)>
	^ self externalCallFailed
! !

!TensorFlowCAPI methodsFor: 'graph' stamp: 'gera 1/5/2017 20:47:43'!
graph: aTFGraph toGraphDef: aTFBuffer status: aTFStatus
	<cdecl: void 'TF_GraphToGraphDef' (TFGraph* TFBuffer* TFStatus*)>
	^ self externalCallFailed
! !

!TensorFlowCAPI methodsFor: 'graph' stamp: 'gera 12/22/2016 01:22:22'!
importGraphDefInto: aTFGraph from: aTFBuffer options: aTFImportGraphDefOptions status: aTFStatus
	<cdecl: void 'TF_GraphImportGraphDef' (TFGraph* TFBuffer* TFImportGraphDefOptions* TFStatus*)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'graph' stamp: 'gera 7/3/2019 00:34:29'!
newGraph
	| answer |
	answer := self newGraphAsVoid.
	answer := TFGraph fromHandle: answer getHandle.
	^ answer! !

!TensorFlowCAPI methodsFor: 'graph' stamp: 'gera 2/9/2017 12:58:18'!
newGraphAsVoid
	<cdecl: void* 'TF_NewGraph' (void)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'options' stamp: 'gera 12/22/2016 01:16:37'!
deleteImportGraphDefOptions: aTFImportGraphDefOptions
	<cdecl: void 'TF_DeleteImportGraphDefOptions' (TFImportGraphDefOptions*)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'options' stamp: 'GerardoRicharte 12/18/2016 21:41'!
deleteSessionOptions: aTFSessionOptions
	<cdecl: void 'TF_DeleteSessionOptions' (TFSessionOptions*)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'options' stamp: 'gera 12/22/2016 01:17:45'!
newImportGraphDefOptions
	<cdecl: TFImportGraphDefOptions* 'TF_NewImportGraphDefOptions' (void)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'options' stamp: 'GerardoRicharte 12/18/2016 21:35'!
newSessionOptions
	<cdecl: TFSessionOptions* 'TF_NewSessionOptions' (void)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'options' stamp: 'gera 12/21/2016 23:47:01'!
sessionOptions: aTFSessionOptions setConfig: aString configSize: anInteger status: aTFStatus
	" not sure how to use this. Best information found in http://devdocs.io/tensorflow~python/contrib.learn#RunConfig"
	<cdecl: void 'TF_SetConfig' (TFSessionOptions* void* ulong TFStatus*)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'options' stamp: 'GerardoRicharte 12/19/2016 10:58'!
sessionOptions: aTFSessionOptions setTarget: aString
	<cdecl: void 'TF_SetTarget' (TFSessionOptions* char*)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'status' stamp: 'gera 12/21/2016 02:57:15'!
deleteStatus: aTFStatus
	<cdecl: void 'TF_DeleteStatus' (TFStatus*)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'status' stamp: 'gera 12/21/2016 09:59:30'!
getCode: aTFStatus
	<cdecl: ulong 'TF_GetCode' (TFStatus*)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'status' stamp: 'gera 12/21/2016 10:14:23'!
message: aTFStatus
	<cdecl: char* 'TF_Message' (TFStatus*)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'status' stamp: 'gera 12/21/2016 02:56:25'!
newStatus
	<cdecl: TFStatus* 'TF_NewStatus' (void)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'status' stamp: 'gera 12/21/2016 09:54:03'!
setStatus: aTFStatus code: anInteger message: anExternalString
	<cdecl: void 'TF_SetStatus' (TFStatus* ulong char*)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'operation description' stamp: 'gera 1/22/2017 20:32:21'!
description: aTFOperationDescription addControlInput: aTFOutput
	<cdecl: void 'TF_AddControlInput' (TFOperationDescription* TFOutput)>
	^ self externalCallFailed.! !

!TensorFlowCAPI methodsFor: 'operation description' stamp: 'gera 12/28/2016 01:59:08'!
description: aTFOperationDescription addInput: aTFOutput
	<cdecl: void 'TF_AddInput' (TFOperationDescription* TFOutput)>
	^ self externalCallFailed.! !

!TensorFlowCAPI methodsFor: 'operation description' stamp: 'gera 1/2/2017 12:36:00'!
description: aTFOperationDescription addInputs: aTFOutputArray size: anInteger
	<cdecl: void 'TF_AddInputList' (TFOperationDescription* TFOutputArray* longlong)>
	^ self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'operation description' stamp: 'gera 1/9/2017 19:53:21'!
description: aTFOperationDescription set: aString toBool: anInt
	<cdecl: void 'TF_SetAttrBool' (TFOperationDescription* char* byte)>
	^ self externalCallFailed.! !

!TensorFlowCAPI methodsFor: 'operation description' stamp: 'gera 1/10/2017 10:32:25'!
description: aTFOperationDescription set: aString toFloat: valueFloat
	<cdecl: void 'TF_SetAttrFloat' (TFOperationDescription* char* float)>
	^ self externalCallFailed.! !

!TensorFlowCAPI methodsFor: 'operation description' stamp: 'gera 12/30/2016 12:52:47'!
description: aTFOperationDescription set: aString toInt64: anInt
	<cdecl: void 'TF_SetAttrInt' (TFOperationDescription* char* ulonglong)>
	^ self externalCallFailed.! !

!TensorFlowCAPI methodsFor: 'operation description' stamp: 'gera 1/10/2017 20:51:14'!
description: aTFOperationDescription set: aString toShape: anInt64Array size: size
	<cdecl: void 'TF_SetAttrShape' (TFOperationDescription* char* Int64Array* ulonglong)>
	^ self externalCallFailed.! !

!TensorFlowCAPI methodsFor: 'operation description' stamp: 'gera 1/23/2017 17:18:33'!
description: aTFOperationDescription set: nameString toString: valueString size: size
	<cdecl: void 'TF_SetAttrString' (TFOperationDescription* char* char* ulonglong)>
	^ self externalCallFailed.! !

!TensorFlowCAPI methodsFor: 'operation description' stamp: 'gera 1/24/2017 01:24:05'!
description: aTFOperationDescription set: aString toStrings: anArrayOfString sizes: sizes count: count
	<cdecl: void 'TF_SetAttrStringList' (TFOperationDescription* char* void* Int64Array* longlong)>
	^ self externalCallFailed.! !

!TensorFlowCAPI methodsFor: 'operation description' stamp: 'gera 12/28/2016 01:02:26'!
description: aTFOperationDescription set: aString toTensor: aTFTensor status: aTFStatus
	<cdecl: void 'TF_SetAttrTensor' (TFOperationDescription* char* TFTensor* TFStatus*)>
	^ self externalCallFailed.! !

!TensorFlowCAPI methodsFor: 'operation description' stamp: 'gera 12/28/2016 00:44:16'!
description: aTFOperationDescription set: aString toType: anInt
	<cdecl: void 'TF_SetAttrType' (TFOperationDescription* char* ulonglong)>
	^ self externalCallFailed.! !

!TensorFlowCAPI methodsFor: 'operation description' stamp: 'gera 1/22/2017 17:57:36'!
description: aTFOperationDescription setDevice: aString
	<cdecl: void 'TF_SetDevice' (TFOperationDescription* char*)>
	^ self externalCallFailed.! !

!TensorFlowCAPI methodsFor: 'operation description' stamp: 'gera 7/3/2019 00:47:14'!
finishOperation: aTFOperationDescription status: aTFStatus
	| answer |
	answer := self finishOperationAsVoid: aTFOperationDescription status: aTFStatus.
	^ TFOperation fromHandle: answer getHandle.! !

!TensorFlowCAPI methodsFor: 'operation description' stamp: 'gera 1/19/2017 10:33:08'!
finishOperationAsVoid: aTFOperationDescription status: aTFStatus
	<cdecl: void* 'TF_FinishOperation' (TFOperationDescription* TFStatus*)>
	^ self externalCallFailed.! !

!TensorFlowCAPI methodsFor: 'utils' stamp: 'jmv 12/8/2023 09:36:19'!
externalizeString: aString
	| answer |
	answer := ExternalAddress allocate: aString size + 1.
	answer ffiUint8At: aString size + 1 put: 0.
	aString withIndexDo: [:char :index |
		answer ffiUint8At:  index put: char asciiValue].
	^ answer! !

!TensorFlowCAPI methodsFor: 'utils' stamp: 'gera 12/29/2016 23:03:34'!
getAllOps
	<cdecl: TFBuffer* 'TF_GetAllOpList' (void)>
	^ self externalCallFailed.! !

!TensorFlowCAPI methodsFor: 'utils' stamp: 'jb 12/26/2016 11:48:09'!
version
	<cdecl: char* 'TF_Version' (void)>
	^ self externalCallFailed.! !

!TensorFlowCAPI methodsFor: 'operation' stamp: 'gera 12/26/2016 13:13:55'!
forGraph: aTFGraph outputDims: aTFOutput status: aTFStatus
	<cdecl: ulonglong 'TF_GraphGetTensorNumDims' (TFGraph* TFOutput TFStatus*)>
	^ self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'operation' stamp: 'gera 7/3/2019 00:47:14'!
graph: aTFGraph operationAt: contextULongLongPtr
	| answer |
	answer := self graph: aTFGraph operationAtAsVoid: contextULongLongPtr.
	answer := TFOperation fromHandle: answer getHandle.
	answer graph: aTFGraph.
	^ answer! !

!TensorFlowCAPI methodsFor: 'operation' stamp: 'gera 1/19/2017 23:28:23'!
graph: aTFGraph operationAtAsVoid: contextULongLongPtr
	<cdecl: void* 'TF_GraphNextOperation' (TFGraph* ulonglong*)>
	^ self externalCallFailed
! !

!TensorFlowCAPI methodsFor: 'operation' stamp: 'gera 1/29/2017 02:33:28'!
newOperationDescriptionOn: graph type: type named: name
	<cdecl: TFOperationDescription* 'TF_NewOperation' (TFGraph* char* char*)>
	^ self externalCallFailed
	! !

!TensorFlowCAPI methodsFor: 'operation' stamp: 'gera 1/9/2017 19:45:05'!
operation: aTFOperation attr: nameZString getBool: valueBoolPtr status: status
	<cdecl: void 'TF_OperationGetAttrBool' (TFOperation* char* ulonglong* TFStatus*)>
	^ self externalCallFailed

! !

!TensorFlowCAPI methodsFor: 'operation' stamp: 'gera 1/10/2017 10:21:45'!
operation: aTFOperation attr: nameZString getFloat: valueFloatPtr status: status
	<cdecl: void 'TF_OperationGetAttrFloat' (TFOperation* char* float* TFStatus*)>
	^ self externalCallFailed

! !

!TensorFlowCAPI methodsFor: 'operation' stamp: 'gera 1/7/2017 04:24:06'!
operation: aTFOperation attr: nameZString getInt64: valueLongPtr status: status
	<cdecl: void 'TF_OperationGetAttrInt' (TFOperation* char* ulonglong* TFStatus*)>
	^ self externalCallFailed

! !

!TensorFlowCAPI methodsFor: 'operation' stamp: 'gera 1/10/2017 16:39:17'!
operation: aTFOperation attr: nameZString getShape: int64array size: maxSize status: status
	<cdecl: void 'TF_OperationGetAttrShape' (TFOperation* char* Int64Array* ulonglong TFStatus*)>
	^ self externalCallFailed

! !

!TensorFlowCAPI methodsFor: 'operation' stamp: 'gera 1/10/2017 09:00:54'!
operation: aTFOperation attr: nameZString getString: valueString size: maxSize status: status
	<cdecl: void 'TF_OperationGetAttrString' (TFOperation* char* void* ulonglong TFStatus*)>
	^ self externalCallFailed

! !

!TensorFlowCAPI methodsFor: 'operation' stamp: 'gera 1/26/2017 22:36:48'!
operation: aTFOperation attr: nameZString getStrings: valueStringArray sizes: sizesArray maxCount: maxCount storage: aBytaArray size: storageSize status: status
	<cdecl: void 'TF_OperationGetAttrStringList' (TFOperation* char* void* Int64Array* ulonglong void* size_t TFStatus*)>
	^ self externalCallFailed

! !

!TensorFlowCAPI methodsFor: 'operation' stamp: 'gera 1/7/2017 04:59:08'!
operation: aTFOperation attr: nameZString getTensor: valueLongPtr status: status
	<cdecl: void 'TF_OperationGetAttrTensor' (TFOperation* char* void* TFStatus*)>
	^ self externalCallFailed

! !

!TensorFlowCAPI methodsFor: 'operation' stamp: 'gera 1/7/2017 04:54:15'!
operation: aTFOperation attr: nameZString getType: valueLongPtr status: status
	<cdecl: void 'TF_OperationGetAttrType' (TFOperation* char* ulonglong* TFStatus*)>
	^ self externalCallFailed

! !

!TensorFlowCAPI methodsFor: 'operation' stamp: 'gera 1/11/2017 22:34:01'!
operation: aTFOperation getMetadataFor: nameZString status: status
	<cdecl: TFAttrMetadata 'TF_OperationGetAttrMetadata' (TFOperation* char* TFStatus*)>
	^ self externalCallFailed

! !

!TensorFlowCAPI methodsFor: 'operation' stamp: 'gera 12/22/2016 03:54:06'!
operationDevice: aTFOperation
	<cdecl: char* 'TF_OperationDevice' (TFOperation*)>
	^ self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'operation' stamp: 'gera 12/22/2016 03:53:36'!
operationName: aTFOperation
	<cdecl: char* 'TF_OperationName' (TFOperation*)>
	^ self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'operation' stamp: 'gera 12/22/2016 03:55:39'!
operationNumInputs: aTFOperation
	<cdecl: long 'TF_OperationNumInputs' (TFOperation*)>
	^ self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'operation' stamp: 'gera 12/22/2016 03:54:41'!
operationNumOutputs: aTFOperation
	<cdecl: long 'TF_OperationNumOutputs' (TFOperation*)>
	^ self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'operation' stamp: 'gera 12/22/2016 03:53:56'!
operationOpType: aTFOperation
	<cdecl: char* 'TF_OperationOpType' (TFOperation*)>
	^ self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'operation' stamp: 'gera 12/23/2016 10:45:09'!
operationOutputType: aTFOutput
	<cdecl: ulonglong 'TF_OperationOutputType' (TFOutput)>
	^ self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'strings' stamp: 'gera 1/3/2017 09:43:38'!
primStringEncodedSize: anInteger
	<cdecl: ulonglong 'TF_StringEncodedSize' (ulonglong)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'strings' stamp: 'gera 7/3/2019 03:33:19'!
stringDecode: src
	| destination status answer dstSize |
	destination := ByteArray new: 8.
	dstSize := ByteArray new: 8.
	status := TFStatus create.
	answer := self stringDecode: src len: src size destination: destination len: dstSize status: status.
	status check.
	dstSize := dstSize unsignedLongLongAt: 1.
	destination := destination pointerAt: 1.
	answer := (destination structAt: 1 length: dstSize) asString.
	^ answer! !

!TensorFlowCAPI methodsFor: 'strings' stamp: 'gera 1/5/2017 03:11:10'!
stringDecode: srcString len: srcLen destination: dstPointer len: dstLenPointer status: status
	<cdecl: ulonglong 'TF_StringDecode' (char* ulonglong char* ulonglong* TFStatus)>
	^ self externalCallFailed
! !

!TensorFlowCAPI methodsFor: 'strings' stamp: 'gera 1/4/2017 19:02:15'!
stringEncode: srcString len: srcLen destination: dstString len: dstLen status: status
	<cdecl: ulonglong 'TF_StringEncode' (char* ulonglong char* ulonglong TFStatus)>
	^ self externalCallFailed
! !

!TensorFlowCAPI methodsFor: 'strings' stamp: 'gera 1/13/2017 01:50:35'!
stringEncode: src to: dst
	^ self stringEncode: src to: dst size: dst size

! !

!TensorFlowCAPI methodsFor: 'strings' stamp: 'gera 7/3/2019 03:33:19'!
stringEncode: src to: dst size: dstSize
	| status answer |
	status := TFStatus create.
	answer := self stringEncode: src len: src size destination: dst len: dstSize status: status.
	status check.
	^ answer
! !

!TensorFlowCAPI methodsFor: 'strings' stamp: 'gera 1/3/2017 09:40:22'!
stringEncodedSize: aString
	^self primStringEncodedSize: aString size! !

!TensorFlowCAPI class methodsFor: 'as yet unclassified' stamp: 'jb 12/26/2016 10:19:17'!
current
	^ current ifNil: [current := self new forceLoading].
! !

!TensorFlowCAPI class methodsFor: 'as yet unclassified' stamp: 'GerardoRicharte 12/18/2016 12:18'!
moduleName
	^ 'libtensorflow'! !

!TFStructure methodsFor: 'printing' stamp: 'gera 1/3/2017 10:26:05'!
printOn: strm
	super printOn: strm.
	strm space; print: handle.! !

!TFStructure methodsFor: 'testing' stamp: 'gera 7/3/2019 03:42:08'!
= aTFStructure
	^ self class = aTFStructure class
		and: [handle = aTFStructure getHandle]! !

!TFStructure methodsFor: 'private' stamp: 'gera 12/21/2016 02:52:42'!
library
	^ TensorFlowCAPI current! !

!TFStructure class methodsFor: 'field definition' stamp: 'gera 12/22/2016 00:24:05'!
fields
	"self initialize"
	^#()! !

!TFStructure class methodsFor: 'field definition' stamp: 'gera 12/22/2016 00:33:15'!
pointerSize
	" Fixed for 64 bit platforms. TensorFlow is only available in 64 bit platforms "
	self assert: 8 == Smalltalk wordSize.
	^ Smalltalk wordSize! !

!TFStructure class methodsFor: 'class initialization' stamp: 'gera 6/30/2019 00:14:46'!
initialize
	super initialize.
	self == TFStructure ifTrue: [
		self allSubclassesDo: [:cls | cls initialize].
		self allSubclassesDo: [:cls | cls initialize].
	]! !

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

!ExternalWordArray methodsFor: 'accessing' stamp: 'gera 2/2/2017 02:25:59'!
fourth
	^ self at: 4! !

!ExternalWordArray methodsFor: 'accessing' stamp: 'gera 2/2/2017 02:25:49'!
second
	^ self at: 2! !

!ExternalWordArray methodsFor: 'accessing' stamp: 'gera 12/23/2016 11:10:15'!
template
	"This method was automatically generated. See PointerArray class>>fields."
	^ExternalData fromHandle: (handle longPointerAt: 1) type: ExternalType void asPointerType! !

!ExternalWordArray methodsFor: 'accessing' stamp: 'gera 12/23/2016 11:10:15'!
template: anObject
	"This method was automatically generated. See PointerArray class>>fields."
	handle longPointerAt: 1 put: anObject getHandle.! !

!ExternalWordArray methodsFor: 'accessing' stamp: 'gera 2/2/2017 02:25:53'!
third
	^ self at: 3! !

!ExternalWordArray methodsFor: 'private' stamp: 'gera 12/25/2016 13:44:02'!
elementAtOffset: byteOffset
	^self elementClass fromHandle: (handle longPointerAt: byteOffset)
	! !

!ExternalWordArray methodsFor: 'private' stamp: 'gera 12/25/2016 14:14:46'!
elementAtOffset: byteOffset put: anExternalStructure
	handle longPointerAt: byteOffset put: anExternalStructure getHandle.! !

!ExternalWordArray class methodsFor: 'instance creation' stamp: 'gera 12/29/2016 18:35:06'!
externalFromArray: anArray
	| answer |
	answer := self externalNew: anArray size.
	anArray withIndexDo: [:each :index |
		answer at: index put: each].
	^ answer! !

!ExternalWordArray class methodsFor: 'instance creation' stamp: 'gera 12/23/2016 11:15:00'!
externalNew
	^ self shouldNotImplement! !

!ExternalWordArray class methodsFor: 'instance creation' stamp: 'gera 1/13/2017 13:11:46'!
externalNew: count
	| address answer |
	address := ExternalAddress gcallocate: self byteSize * count.
	answer := self fromHandle: address.
	^ answer! !

!ExternalWordArray class methodsFor: 'instance creation' stamp: 'gera 12/23/2016 11:14:45'!
new
	^ self shouldNotImplement! !

!ExternalWordArray class methodsFor: 'instance creation' stamp: 'gera 12/22/2016 20:34:52'!
new: count
	^self fromHandle: (ByteArray new: self byteSize * count)! !

!ExternalWordArray class methodsFor: 'field definition' stamp: 'gera 12/23/2016 11:10:13'!
fields
	"self initialize."
	^#((template 'void*'))! !

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

!TFOperationPtrArray methodsFor: 'as yet unclassified' stamp: 'gera 7/3/2019 00:47:14'!
elementClass
	^TFOperation! !

!TFOutputArray methodsFor: 'accessing' stamp: 'gera 12/29/2016 09:44:00'!
elementAtOffset: byteOffset
	^self elementClass fromHandle: (handle + (byteOffset - 1))
	
! !

!TFOutputArray methodsFor: 'accessing' stamp: 'gera 7/3/2019 03:26:12'!
elementAtOffset: byteOffset put: anExternalStructure
	handle structAt: byteOffset put: anExternalStructure getHandle length: TFOutput byteSize.! !

!TFOutputArray methodsFor: 'accessing' stamp: 'gera 7/3/2019 03:26:12'!
elementClass
	^TFOutput! !

!TFOutputArray methodsFor: 'accessing' stamp: 'gera 7/3/2019 03:26:12'!
template
	"This method was automatically generated. See TFOutputArray class>>fields."
	^TFOutput fromHandle: (handle structAt: 1 length: 16)! !

!TFOutputArray methodsFor: 'accessing' stamp: 'JB 4/18/2018 21:59:06'!
template: anObject
	"This method was automatically generated. See TFOutputArray class>>fields."
	handle structAt: 1 put: anObject getHandle length: 16.! !

!TFOutputArray class methodsFor: 'as yet unclassified' stamp: 'gera 12/29/2016 09:39:03'!
fields
	"self initialize"
	^#((template 'TFOutput'))! !

!TFOutputArray class methodsFor: 'as yet unclassified' stamp: 'gera 12/29/2016 09:39:40'!
new: unused
	^ self shouldNotImplement! !

!TFTensorPtrArray methodsFor: 'accessing' stamp: 'gera 6/30/2019 00:45:26'!
elementClass
	^TFTensor! !

!TFTensorPtrArray methodsFor: 'accessing' stamp: 'JB 2/23/2017 11:14:36'!
numbersAt: index
	| tensor |
	tensor := self at: index.
	^ tensor asNumbers! !

!TFAttrMetadata methodsFor: 'types' stamp: 'gera 1/11/2017 23:12:07'!
isBoolean
	^ self type = 3! !

!TFAttrMetadata methodsFor: 'types' stamp: 'gera 1/11/2017 23:11:55'!
isFloat
	^ self type = 2! !

!TFAttrMetadata methodsFor: 'types' stamp: 'gera 1/11/2017 23:15:35'!
isFunc
	^ self type = 8! !

!TFAttrMetadata methodsFor: 'types' stamp: 'gera 1/11/2017 23:11:31'!
isInt
	^ self type = 1! !

!TFAttrMetadata methodsFor: 'types' stamp: 'gera 1/11/2017 23:12:54'!
isPlaceholder
	^ self type = 7! !

!TFAttrMetadata methodsFor: 'types' stamp: 'gera 1/11/2017 23:12:27'!
isShape
	^ self type = 5! !

!TFAttrMetadata methodsFor: 'types' stamp: 'gera 1/11/2017 23:11:09'!
isString
	^ self type = 0
! !

!TFAttrMetadata methodsFor: 'types' stamp: 'gera 1/11/2017 23:12:41'!
isTensor
	^ self type = 6! !

!TFAttrMetadata methodsFor: 'types' stamp: 'gera 1/11/2017 23:12:17'!
isType
	^ self type = 4! !

!TFAttrMetadata methodsFor: 'accessing' stamp: 'gera 1/11/2017 22:41:49'!
isList
	"This method was automatically generated. See TFAttrMetadata class>>fields."
	^self _isList \\ 256 ~= 0.! !

!TFAttrMetadata methodsFor: 'accessing' stamp: 'gera 1/11/2017 22:43:36'!
listSize
	"This method was automatically generated. See TFAttrMetadata class>>fields."
	^handle signedLongLongAt: 9! !

!TFAttrMetadata methodsFor: 'accessing' stamp: 'gera 1/11/2017 22:43:40'!
totalSize
	"This method was automatically generated. See TFAttrMetadata class>>fields."
	^handle signedLongLongAt: 25! !

!TFAttrMetadata methodsFor: 'accessing' stamp: 'gera 1/11/2017 22:42:04'!
type
	"This method was automatically generated. See TFAttrMetadata class>>fields."
	^self _type \\ 256! !

!TFAttrMetadata methodsFor: 'accessing' stamp: 'gera 1/11/2017 22:40:36'!
_isList
	"This method was automatically generated. See TFAttrMetadata class>>fields."
	^handle unsignedLongLongAt: 1! !

!TFAttrMetadata methodsFor: 'accessing' stamp: 'gera 1/11/2017 22:40:36'!
_type
	"This method was automatically generated. See TFAttrMetadata class>>fields."
	^handle unsignedLongLongAt: 17! !

!TFAttrMetadata class methodsFor: 'field definition' stamp: 'gera 1/11/2017 22:43:56'!
fields
	"self initialize"
	^#(
		(_isList 'ulonglong')
		(listSize 'longlong')
		(_type 'ulonglong')
		(totalSize 'longlong'))
	! !

!TFBuffer methodsFor: 'accessing' stamp: 'JB 1/18/2017 23:01:10'!
data
	"This method was automatically generated. See TFBuffer class>>fields."
	^ExternalData fromHandle: (handle longPointerAt: 1) type: ExternalType void asPointerType! !

!TFBuffer methodsFor: 'accessing' stamp: 'gera 12/22/2016 00:33:21'!
data: anObject
	"This method was automatically generated. See TFBuffer class>>fields."
	handle longPointerAt: 1 put: anObject getHandle.! !

!TFBuffer methodsFor: 'accessing' stamp: 'gera 1/5/2017 20:43:10'!
dataBytes
	^ self data getHandle structAt: 1 length: self length! !

!TFBuffer methodsFor: 'accessing' stamp: 'gera 12/22/2016 00:34:10'!
deallocator
	"This method was automatically generated. See TFBuffer class>>fields."
	^ExternalData fromHandle: (handle longPointerAt: 17) type: ExternalType void asPointerType! !

!TFBuffer methodsFor: 'accessing' stamp: 'gera 12/22/2016 00:34:10'!
deallocator: anObject
	"This method was automatically generated. See TFBuffer class>>fields."
	handle longPointerAt: 17 put: anObject getHandle.! !

!TFBuffer methodsFor: 'accessing' stamp: 'gera 12/22/2016 00:34:10'!
length
	"This method was automatically generated. See TFBuffer class>>fields."
	^handle unsignedLongLongAt: 9! !

!TFBuffer methodsFor: 'accessing' stamp: 'gera 12/22/2016 00:34:10'!
length: anObject
	"This method was automatically generated. See TFBuffer class>>fields."
	handle unsignedLongLongAt: 9 put: anObject! !

!TFBuffer methodsFor: 'release' stamp: 'gera 1/19/2017 09:46:20'!
delete
	self isNull ifFalse: [
		self library deleteBuffer: self].
	handle := nil! !

!TFBuffer class methodsFor: 'field definition' stamp: 'gera 1/12/2017 18:45:49'!
fields
	"self initialize.
	self defineFields"
	^#(
		(data 'void*')
		(length 'ulonglong')
		(deallocator 'void*'))
	! !

!TFBuffer class methodsFor: 'instance creation' stamp: 'gera 12/22/2016 01:05:44'!
fromFileNamed: aString
	| fileContents |
	fileContents := aString asFileEntry readStream contents.
	^ self fromString: fileContents! !

!TFBuffer class methodsFor: 'instance creation' stamp: 'gera 1/10/2017 21:00:42'!
fromString: aString
	| answer |
	answer := TensorFlowCAPI current newBufferFromString: aString size: aString size.
	^ answer! !

!TFGraph methodsFor: 'accessing' stamp: 'gera 1/29/2017 08:15:34'!
allInitializers
	^ self operationsSelect: [:op | op type = 'Assign' and: [op name endsWith: '_initializer']]! !

!TFGraph methodsFor: 'accessing' stamp: 'gera 1/29/2017 03:30:15'!
allInputs
	^ self operationsSelect: [:op | op type = 'Placeholder']! !

!TFGraph methodsFor: 'accessing' stamp: 'gera 1/29/2017 05:03:52'!
allOperations
	^ self operationsSelect: [:op | true]! !

!TFGraph methodsFor: 'accessing' stamp: 'gera 1/29/2017 03:34:38'!
allVariables
	^ self operationsSelect: [:op | op type = 'Variable']! !

!TFGraph methodsFor: 'accessing' stamp: 'gera 7/3/2019 03:33:19'!
definition
	| buffer status |
	status := TFStatus create.
	buffer := TFBuffer new.
	self library graph: self toGraphDef: buffer status: status.
	status check.
	^ buffer dataBytes asString! !

!TFGraph methodsFor: 'accessing' stamp: 'gera 1/5/2017 20:20:02'!
newOperationIteratorContext
	^ ByteArray new: 8! !

!TFGraph methodsFor: 'accessing' stamp: 'gera 1/5/2017 20:09:21'!
operationAt: contextULongLongPtr
	^ self library graph: self operationAt: contextULongLongPtr! !

!TFGraph methodsFor: 'accessing' stamp: 'gera 1/19/2017 23:27:23'!
operationNamed: aString
	| answer |
	answer := self library graph: self getOperationNamed: aString asAsciiZ.
	answer isNull ifTrue: [self error: 'Operation not found'].
	^ answer
! !

!TFGraph methodsFor: 'accessing' stamp: 'gera 1/19/2017 11:10:01'!
operationsCount
	| answer |
	answer := 0.
	self operationsDo: [:each | answer := answer + 1].
	^ answer! !

!TFGraph methodsFor: 'accessing' stamp: 'gera 2/25/2017 11:46:38'!
operationsDo: oneArgBlock
	| iterator operation |
	iterator := self newOperationIteratorContext.
	[operation := self operationAt: iterator.
		operation isNull] whileFalse: [oneArgBlock value: operation].
! !

!TFGraph methodsFor: 'accessing' stamp: 'gera 1/5/2017 20:33:53'!
operationsSelect: oneArgBlock
	| answer |
	answer := OrderedCollection new.
	
	self operationsSelect: oneArgBlock thenDo: [:op | answer add: op].
	
	^ answer asArray! !

!TFGraph methodsFor: 'accessing' stamp: 'gera 1/5/2017 20:33:20'!
operationsSelect: conditionBlock thenDo: actionBlock
	| answer |
	answer := OrderedCollection new.
	
	self operationsDo: [:op |
		(conditionBlock value: op)
			ifTrue: [actionBlock value: op]].
	
	^ answer asArray! !

!TFGraph methodsFor: 'operations' stamp: 'gera 1/5/2017 05:21:30'!
add: nameString described: aBlock
	^self newOperation: 'Add' named: nameString described: aBlock! !

!TFGraph methodsFor: 'operations' stamp: 'gera 1/5/2017 23:55:21'!
asString: nameString described: aBlock
	^self newOperation: 'AsString' named: nameString described: aBlock! !

!TFGraph methodsFor: 'operations' stamp: 'gera 12/30/2016 12:16:54'!
concat: nameString described: aBlock
	^self newOperation: 'Concat' named: nameString described: aBlock! !

!TFGraph methodsFor: 'operations' stamp: 'gera 12/28/2016 02:11:44'!
mul: nameString described: aBlock
	^self newOperation: 'Mul' named: nameString described: aBlock! !

!TFGraph methodsFor: 'operations' stamp: 'gera 1/29/2017 02:35:44'!
newOperation: typeString named: aString
	^ self newOperation: typeString named: aString described: [:unused | ]! !

!TFGraph methodsFor: 'operations' stamp: 'gera 1/29/2017 02:32:14'!
newOperation: typeString named: aString described: aBlock
	| description answer |
	description := self newOperationDescription: typeString named: aString.
	aBlock value: description.
	answer := description finish.
	answer graph: self.
	^ answer! !

!TFGraph methodsFor: 'operations' stamp: 'gera 1/29/2017 02:33:39'!
newOperationDescription: typeString named: aString
	^ self library newOperationDescriptionOn: self type: typeString named: aString
! !

!TFGraph methodsFor: 'root operations' stamp: 'gera 1/31/2017 15:07:44'!
const: aTFTensor
	| name |
	name := self nameFor: 'constant'.
	^ self const: name value: aTFTensor! !

!TFGraph methodsFor: 'root operations' stamp: 'gera 12/28/2016 02:07:25'!
const: nameString value: aTFTensor
	^ self
		newOperation: 'Const'
		named: nameString
		described: [:description |
			description at: 'dtype' putType: aTFTensor type.
			description at: 'value' putTensor: aTFTensor]! !

!TFGraph methodsFor: 'root operations' stamp: 'gera 2/2/2017 19:45:59'!
inputType: typeInteger
	^ self
		newOperation: 'Placeholder'
		named: (self nameFor: 'input')
		described: [:description |
			description at: 'dtype' putType: typeInteger]! !

!TFGraph methodsFor: 'root operations' stamp: 'gera 2/10/2017 11:28:09'!
nameFor: namePrefix
	^ context, namePrefix,'_',self operationsCount printString.! !

!TFGraph methodsFor: 'root operations' stamp: 'gera 1/6/2017 17:21:53'!
placeholder: nameString type: typeInteger
	^ self
		newOperation: 'Placeholder'
		named: nameString
		described: [:description |
			description at: 'dtype' putType: typeInteger]! !

!TFGraph methodsFor: 'root operations' stamp: 'gera 6/30/2019 00:45:27'!
truncatedNormalRandomShaped: shapeArray
	| shape |
	shape := self const: shapeArray asInt32Tensor.
	^ shape unaryOp: 'TruncatedNormal' described: [:description |
		description at: 'dtype' putType: TFTensor typeFloat]! !

!TFGraph methodsFor: 'root operations' stamp: 'gera 1/31/2017 17:44:21'!
truncatedNormalRandomShaped: shapeArray stddev: aNumber
	| random |
	random := self truncatedNormalRandomShaped: shapeArray.
	^ random @* (self const: aNumber asTensor)! !

!TFGraph methodsFor: 'root operations' stamp: 'gera 1/29/2017 03:23:14'!
variable: nameString forTensor: aTFTensor
	^ self variable: nameString type: aTFTensor type shape: aTFTensor shape! !

!TFGraph methodsFor: 'root operations' stamp: 'gera 2/1/2017 01:02:23'!
variable: nameString initialValue: aTFTensor
	| const var |
	var := self variable: nameString forTensor: aTFTensor.
	const := self const: nameString, '_initialValue' value: aTFTensor.
	var assign: const.
	^ var! !

!TFGraph methodsFor: 'root operations' stamp: 'gera 2/1/2017 01:22:52'!
variable: nameString initialValueFrom: aTFOperation
	| output var shape |
	output := aTFOperation output: 0.
	shape := self shapeOf: output.
	var := self variable: nameString type: output type shape: shape..
	var assign: aTFOperation.
	^ var! !

!TFGraph methodsFor: 'root operations' stamp: 'gera 1/29/2017 03:04:54'!
variable: nameString type: typeInteger shape: anArray
	^ self
		newOperation: 'Variable'
		named: nameString
		described: [:description |
			description
				at: 'dtype' putType: typeInteger;
				at: 'shape' putShape: anArray]! !

!TFGraph methodsFor: 'root operations' stamp: 'gera 2/17/2017 00:47:45'!
zerosShaped: shapeArray
	| shape |
	shape := self const: shapeArray asInt32Tensor.
	^ shape binaryOp: 'Fill' with: 0.0 asTensor! !

!TFGraph methodsFor: 'initialize-release' stamp: 'gera 1/19/2017 09:45:55'!
delete
	self ignoreFinalization.
	self isNull ifFalse: [
		self library deleteGraph: self].
	handle := nil! !

!TFGraph methodsFor: 'initialize-release' stamp: 'gera 1/13/2017 10:32:27'!
finalize
	self delete! !

!TFGraph methodsFor: 'initialize-release' stamp: 'gera 6/30/2019 00:45:28'!
fromBlock: aBlockClosure
	| types |
	types := Array new: aBlockClosure argumentCount.
	types atAllPut: TFTensor typeFloat.
	^ self fromBlock: aBlockClosure inputTypes: types! !

!TFGraph methodsFor: 'initialize-release' stamp: 'gera 2/2/2017 19:46:21'!
fromBlock: aBlockClosure inputTypes: anArray
	| inputs index |
	index := 0.
	inputs := (1 to: aBlockClosure argumentCount) collect: [:each |
		index := index + 1.
		self inputType: (anArray at: index)].
	^ aBlockClosure valueWithArguments: inputs.
! !

!TFGraph methodsFor: 'initialize-release' stamp: 'gera 2/10/2017 15:26:14'!
fromBlock: aBlockClosure inputTypes: anArray named: nameString
	| previousContext answer |
	previousContext := context.
	context := context, nameString, '/'.
	answer := self fromBlock: aBlockClosure inputTypes: anArray.
	context := previousContext.
	^ answer! !

!TFGraph methodsFor: 'initialize-release' stamp: 'gera 6/30/2019 00:45:27'!
fromBlock: aBlockClosure named: nameString
	| types |
	types := Array new: aBlockClosure argumentCount.
	types atAllPut: TFTensor typeFloat.
	^ self fromBlock: aBlockClosure inputTypes: types named: nameString! !

!TFGraph methodsFor: 'initialize-release' stamp: 'gera 7/3/2019 03:33:19'!
import: aTFBuffer
	| options status |
	
	status := TFStatus create.
	options := TFImportGraphDefOptions create.
	
	self library importGraphDefInto: self from: aTFBuffer options: options status: status.
	
	options delete.
	status check.! !

!TFGraph methodsFor: 'initialize-release' stamp: 'gera 7/3/2019 00:28:37'!
importFileNamed: aString
	| buffer |
	buffer := TFBuffer fromFileNamed: aString.
	[self import: buffer] ensure: [buffer delete].! !

!TFGraph methodsFor: 'initialize-release' stamp: 'gera 7/3/2019 00:28:37'!
importString: aString
	| buffer |
	buffer := TFBuffer fromString: aString.
	[self import: buffer] ensure: [buffer delete].! !

!TFGraph methodsFor: 'initialize-release' stamp: 'gera 2/10/2017 11:26:27'!
initialize
	context := ''.! !

!TFGraph methodsFor: 'running' stamp: 'gera 1/29/2017 04:32:04'!
initializeOn: aTFSession
	| initializers |
	initializers := self allInitializers collect: [:each | each output: 0].
	initializers size > 0 ifTrue: [
		aTFSession runOutputs: initializers]
! !

!TFGraph methodsFor: 'running' stamp: 'gera 7/3/2019 03:35:50'!
runInputs: inArrayOfTFOutputs values: inArrayOfTFTensor outputs: outArrayOfTFOutputs
	| session |
	session := TFSession on: self.
	self initializeOn: session.
	^ session runInputs: inArrayOfTFOutputs values: inArrayOfTFTensor outputs: outArrayOfTFOutputs
! !

!TFGraph methodsFor: 'running' stamp: 'gera 7/3/2019 03:35:50'!
runOutput: aTFOutput
	| session |
	session := TFSession on: self.
	self initializeOn: session.
	^ session runOutput: aTFOutput! !

!TFGraph methodsFor: 'running' stamp: 'gera 7/3/2019 03:35:50'!
runOutputs: anArrayOfTFOutputs
	| session |
	session := TFSession on: self.
	self initializeOn: session.
	^ session runOutputs: anArrayOfTFOutputs! !

!TFGraph methodsFor: 'outputs' stamp: 'gera 7/3/2019 03:33:19'!
outputDimenssionsCount: aTFOutput
	| status answer |
	status := TFStatus create.
	answer := self library forGraph: self outputDims: aTFOutput status: status.
	status check.
	^answer! !

!TFGraph methodsFor: 'outputs' stamp: 'gera 7/3/2019 03:33:19'!
rankOf: aTFOutputOrInput
	| status answer |
	status := TFStatus create.
	answer := self library graph: self getRankOf:  aTFOutputOrInput status: status.
	status check.
	^ answer! !

!TFGraph methodsFor: 'outputs' stamp: 'gera 7/3/2019 03:33:19'!
shapeOf: aTFOutputOrInput
	| status value size answer |
	size := self rankOf: aTFOutputOrInput.
	value := Int64Array externalNew: size.
	status := TFStatus create.
	
	self library
		graph: self
		getShapeOf: aTFOutputOrInput
		into: value
		size: size
		status: status.
	status check.
	
	answer := (1 to: size) collect: [:i | value at: i].
	^ answer! !

!TFGraph methodsFor: 'outputs' stamp: 'gera 7/3/2019 03:33:19'!
shapeOf: aTFOutputOrInput set: shape
	| status value |
	value := Int64Array externalFromArray: shape.
	status := TFStatus create.
	
	self library
		graph: self
		setShapeOf: aTFOutputOrInput
		to: value
		size: shape size
		status: status.
	status check.! !

!TFGraph methodsFor: 'debugging' stamp: 'gera 2/9/2017 09:39:58'!
writeDefTo: strm
	strm nextPutAll: self definition! !

!TFGraph methodsFor: 'debugging' stamp: 'gera 2/9/2017 10:50:05'!
writeDefToFileNamed: filename
	filename asFileEntry writeStreamDo: [:strm |
		self writeDefTo: strm]! !

!TFGraph class methodsFor: 'instance creation' stamp: 'gera 2/10/2017 11:26:01'!
create
	^ TensorFlowCAPI current newGraph initialize useFinalization! !

!TFGraph class methodsFor: 'instance creation' stamp: 'gera 2/2/2017 01:55:59'!
fromBlock: aBlockClosure
	| output graph |
	graph := self create.
	output := graph fromBlock: aBlockClosure .
	output alias: 'output'.
	^ graph! !

!TFGraph class methodsFor: 'instance creation' stamp: 'gera 12/22/2016 02:18:26'!
fromFileNamed: aString
	^ self create importFileNamed: aString! !

!TFGraph class methodsFor: 'instance creation' stamp: 'gera 12/22/2016 02:18:33'!
fromString: aString
	^ self create importString: aString! !

!TFImportGraphDefOptions methodsFor: 'release' stamp: 'gera 1/19/2017 09:46:56'!
delete
	self isNull ifFalse: [
		self library deleteImportGraphDefOptions: self].
	handle := nil! !

!TFImportGraphDefOptions class methodsFor: 'instance creation' stamp: 'gera 12/22/2016 01:15:16'!
create
	^ TensorFlowCAPI current newImportGraphDefOptions! !

!TFOperation methodsFor: 'accessing' stamp: 'gera 12/22/2016 03:57:35'!
InputsCount
	^ self library operationNumInputs: self! !

!TFOperation methodsFor: 'accessing' stamp: 'gera 12/22/2016 03:57:05'!
device
	^ self library operationDevice: self! !

!TFOperation methodsFor: 'accessing' stamp: 'gera 1/19/2017 10:35:20'!
graph
	^ graph! !

!TFOperation methodsFor: 'accessing' stamp: 'gera 1/19/2017 10:35:14'!
graph: aTFGraph
	graph := aTFGraph! !

!TFOperation methodsFor: 'accessing' stamp: 'gera 7/3/2019 03:26:18'!
input: indexInteger
	^ TFInput onOperation: self index: indexInteger! !

!TFOperation methodsFor: 'accessing' stamp: 'gera 12/22/2016 03:56:55'!
name
	^ self library operationName: self! !

!TFOperation methodsFor: 'accessing' stamp: 'gera 7/3/2019 03:26:12'!
output: indexInteger
	^ TFOutput onOperation: self index: indexInteger! !

!TFOperation methodsFor: 'accessing' stamp: 'gera 12/22/2016 03:57:48'!
outputsCount
	^ self library operationNumOutputs: self! !

!TFOperation methodsFor: 'accessing' stamp: 'gera 12/22/2016 03:58:02'!
type
	^ self library operationOpType: self! !

!TFOperation methodsFor: 'accessing' stamp: 'JB 2/9/2017 20:22:33'!
useOutput: anInteger
	output := anInteger! !

!TFOperation methodsFor: 'printing' stamp: 'gera 1/13/2017 12:41:03'!
printOn: strm
	super printOn: strm.
	handle isNull ifFalse: [
		strm
			space;
			print: self type;
			space;
			print: self name]! !

!TFOperation methodsFor: 'ops binary' stamp: 'gera 1/20/2017 18:22:38'!
* aTFOperation
	^ self binaryOp: 'MatMul' with: aTFOperation! !

!TFOperation methodsFor: 'ops binary' stamp: 'gera 1/21/2017 02:26:09'!
*\ aTFOperation
	^ self
		binaryOp: 'MatMul'
		with: aTFOperation
		described: [:description |
			description at: 'transpose_b' putBoolean: true]
! !

!TFOperation methodsFor: 'ops binary' stamp: 'gera 1/19/2017 11:19:11'!
+ aTFOperation
	^ self binaryOp: 'Add' with: aTFOperation! !

!TFOperation methodsFor: 'ops binary' stamp: 'gera 1/29/2017 19:47:10'!
- aTFOperation
	^ self binaryOp: 'Sub' with: aTFOperation! !

!TFOperation methodsFor: 'ops binary' stamp: 'gera 1/29/2017 15:21:14'!
-= aTFOperation
	^ self binaryOp: 'AssignSub' with: aTFOperation! !

!TFOperation methodsFor: 'ops binary' stamp: 'JB 2/15/2017 23:27:00'!
> aTFOperation
	^ self binaryOp: 'Greater' with: aTFOperation! !

!TFOperation methodsFor: 'ops binary' stamp: 'gera 1/20/2017 18:20:07'!
@* aTFOperation
	^ self binaryOp: 'Mul' with: aTFOperation! !

!TFOperation methodsFor: 'ops binary' stamp: 'gera 2/19/2017 15:19:17'!
@/ aTFOperation
	^ self binaryOp: 'Div' with: aTFOperation
	! !

!TFOperation methodsFor: 'ops binary' stamp: 'gera 1/21/2017 02:26:18'!
\* aTFOperation
	^ self
		binaryOp: 'MatMul'
		with: aTFOperation
		described: [:description |
			description at: 'transpose_a' putBoolean: true]
! !

!TFOperation methodsFor: 'ops binary' stamp: 'JB 2/17/2017 20:01:18'!
\*\ aTFOperation
	^ self
		binaryOp: 'MatMul'
		with: aTFOperation
		described: [:description |
			description at: 'transpose_a' putBoolean: true.
			description at: 'transpose_b' putBoolean: true]
! !

!TFOperation methodsFor: 'ops binary' stamp: 'gera 1/19/2017 12:52:43'!
\\ aTFOperation
	^ self binaryOp: 'Mod' with: aTFOperation! !

!TFOperation methodsFor: 'ops binary' stamp: 'gera 2/1/2017 01:02:16'!
assign: aTFOperation
	^ self
		binaryOp: 'Assign'
		with: aTFOperation
		named: self name, '_initializer'! !

!TFOperation methodsFor: 'ops binary' stamp: 'gera 1/21/2017 02:22:49'!
binaryOp: aString with: aTFOperation
	^ self binaryOp: aString with: aTFOperation described: [:nothing]
! !

!TFOperation methodsFor: 'ops binary' stamp: 'gera 1/29/2017 15:14:52'!
binaryOp: aString with: aTFOperation described: oneArgBlock
	| name |
	name := self nameFor: aString.
	^ self binaryOp: aString with: aTFOperation named: name described: oneArgBlock! !

!TFOperation methodsFor: 'ops binary' stamp: 'gera 1/29/2017 04:54:08'!
binaryOp: aString with: aTFOperation named: name
	^ self binaryOp: aString with: aTFOperation named: name described: [:nothing]
! !

!TFOperation methodsFor: 'ops binary' stamp: 'gera 2/19/2017 11:26:18'!
binaryOp: aString with: aTFOperation named: name described: oneArgBlock
	^ self op: aString withAll: {aTFOperation} named: name described: oneArgBlock! !

!TFOperation methodsFor: 'ops binary' stamp: 'gera 2/19/2017 15:40:22'!
castTo: typeInteger
	^ self unaryOp: 'Cast' described: [:description |
		description at: 'DstT' putType: typeInteger]! !

!TFOperation methodsFor: 'ops binary' stamp: 'JB 2/11/2017 13:23:34'!
findMaxOn: aTFOperation
	^ self binaryOp: 'ArgMax' with: aTFOperation! !

!TFOperation methodsFor: 'ops binary' stamp: 'gera 1/21/2017 03:37:51'!
findMinOn: aTFOperation
	^ self binaryOp: 'ArgMin' with: aTFOperation! !

!TFOperation methodsFor: 'ops binary' stamp: 'gera 2/17/2017 00:45:25'!
meanOn:  shapeTensorOrOperation
	^ self binaryOp: 'Mean' with: shapeTensorOrOperation! !

!TFOperation methodsFor: 'ops binary' stamp: 'gera 2/19/2017 15:14:55'!
sizeOn: dimensionInteger
	^ self shape sliceFrom: {dimensionInteger} asInt32Tensor size: #(1) asInt32Tensor.! !

!TFOperation methodsFor: 'ops binary' stamp: 'gera 2/2/2017 02:09:13'!
sparseSoftmaxCrossEntropyWithLogits: aTFOperation
	^ self
		binaryOp: 'SparseSoftmaxCrossEntropyWithLogits'
		with: aTFOperation
		named: (self nameFor: 'SparseSoftmaxCrossEntropyWithLogits')! !

!TFOperation methodsFor: 'ops binary' stamp: 'gera 1/21/2017 03:30:55'!
sumOn: aTFOperation
	^ self binaryOp: 'Sum' with: aTFOperation! !

!TFOperation methodsFor: 'ops binary' stamp: 'JB 2/17/2017 11:38:27'!
timesRectifiedGradOf: aTFOperation
	^ self binaryOp: 'ReluGrad' with: aTFOperation! !

!TFOperation methodsFor: 'ops unary' stamp: 'gera 1/19/2017 12:55:41'!
abs
	^ self unaryOp: 'Abs'! !

!TFOperation methodsFor: 'ops unary' stamp: 'gera 1/29/2017 11:59:40'!
alias: nameString
	^ self unaryOp: 'Identity' named: nameString! !

!TFOperation methodsFor: 'ops unary' stamp: 'gera 1/19/2017 12:58:32'!
arcCos
	^ self unaryOp: 'Acos'
	! !

!TFOperation methodsFor: 'ops unary' stamp: 'gera 1/19/2017 13:03:06'!
arcSin
	^ self unaryOp: 'Asin'
	! !

!TFOperation methodsFor: 'ops unary' stamp: 'gera 1/19/2017 13:04:36'!
arcTan
	^ self unaryOp: 'Atan'
	! !

!TFOperation methodsFor: 'ops unary' stamp: 'gera 1/19/2017 13:08:18'!
cos
	^ self unaryOp: 'Cos'
	! !

!TFOperation methodsFor: 'ops unary' stamp: 'JB 1/23/2017 10:39:24'!
exp
	^ self unaryOp: 'Exp'
	! !

!TFOperation methodsFor: 'ops unary' stamp: 'gera 1/23/2017 09:52:07'!
identity
	^ self unaryOp: 'Identity'
	! !

!TFOperation methodsFor: 'ops unary' stamp: 'gera 1/21/2017 05:15:34'!
inverse
	^ self unaryOp: 'MatrixInverse'! !

!TFOperation methodsFor: 'ops unary' stamp: 'gera 1/19/2017 11:14:17'!
negated
	^ self unaryOp: 'Neg'! !

!TFOperation methodsFor: 'ops unary' stamp: 'gera 1/31/2017 01:33:48'!
rectified
	^ self unaryOp: 'Relu'! !

!TFOperation methodsFor: 'ops unary' stamp: 'gera 2/18/2017 19:14:52'!
shape
	^ self unaryOp: 'Shape'
	! !

!TFOperation methodsFor: 'ops unary' stamp: 'Javier Burroni 1/24/2017 21:36:48'!
sigmoid
	^ self unaryOp: 'Sigmoid'! !

!TFOperation methodsFor: 'ops unary' stamp: 'gera 1/19/2017 13:08:09'!
sin
	^ self unaryOp: 'Sin'
	! !

!TFOperation methodsFor: 'ops unary' stamp: 'JB 2/7/2017 11:32:45'!
softmax
	^ self unaryOp: 'Softmax'
! !

!TFOperation methodsFor: 'ops unary' stamp: 'JB 1/25/2017 11:50:16'!
squared
	^ self @* self
	! !

!TFOperation methodsFor: 'ops unary' stamp: 'gera 1/19/2017 13:08:05'!
tan
	^ self unaryOp: 'Tan'
	! !

!TFOperation methodsFor: 'ops unary' stamp: 'gera 1/29/2017 15:14:27'!
unaryOp: aString
	| name |
	name := self nameFor: aString.
	^ self unaryOp: aString named: name! !

!TFOperation methodsFor: 'ops unary' stamp: 'gera 1/31/2017 16:53:48'!
unaryOp: aString described: oneArgBlock
	| name |
	name := self nameFor: aString.
	^ self unaryOp: aString named: name described: oneArgBlock! !

!TFOperation methodsFor: 'ops unary' stamp: 'gera 1/31/2017 16:58:06'!
unaryOp: aString named: name
	^ self unaryOp: aString named: name described: [:description | ].
	! !

!TFOperation methodsFor: 'ops unary' stamp: 'gera 2/19/2017 11:26:59'!
unaryOp: aString named: name described: oneArgBlock
	^ self op: aString withAll: {} named: name described: oneArgBlock! !

!TFOperation methodsFor: 'converting' stamp: 'gera 2/16/2017 22:57:20'!
asOperationOn: aTFGraph
	graph == aTFGraph
		ifTrue: [^ self]
		ifFalse: [^ self error: 'Can''t move an operation to another Graph']! !

!TFOperation methodsFor: 'attributes' stamp: 'gera 7/3/2019 03:33:19'!
attrMetadata: nameString
	| status answer |
	status := TFStatus create.
	answer := self library operation: self getMetadataFor: nameString asAsciiZ status: status.
	status check.
	^ answer! !

!TFOperation methodsFor: 'attributes' stamp: 'gera 7/3/2019 03:33:19'!
boolAt: nameString
	| value status |
	status := TFStatus create.
	value := ByteArray new: 1.
	self library operation: self attr: nameString asAsciiZ getBool: value status: status.
	status check.
	^ (value booleanAt: 1) 
! !

!TFOperation methodsFor: 'attributes' stamp: 'gera 7/3/2019 03:33:19'!
floatAt: nameString
	| value status |
	status := TFStatus create.
	value := ByteArray new: 8.
	self library operation: self attr: nameString asAsciiZ getFloat: value status: status.
	status check.
	^ value floatAt: 1
! !

!TFOperation methodsFor: 'attributes' stamp: 'gera 7/3/2019 03:33:19'!
intAt: nameString
	| value status |
	status := TFStatus create.
	value := ByteArray new: 8.
	self library operation: self attr: nameString asAsciiZ getInt64: value status: status.
	status check.
	^ value unsignedLongLongAt: 1
! !

!TFOperation methodsFor: 'attributes' stamp: 'gera 7/3/2019 03:33:19'!
shapeAt: nameString
	| value status size answer |
	size := (self attrMetadata: nameString) totalSize.
	status := TFStatus create.
	value := Int64Array externalNew: size.
	
	self library
		operation: self
		attr: nameString asAsciiZ
		getShape: value
		size: size
		status: status.
	status check.

	answer := (1 to: size) collect: [:i | value at: i].
	^ answer! !

!TFOperation methodsFor: 'attributes' stamp: 'gera 7/3/2019 03:33:19'!
stringAt: nameString
	| metadata value status |
	metadata := self attrMetadata: nameString.
	status := TFStatus create.
	value := ByteArray new: metadata totalSize .
	self library
		operation: self
		attr: nameString asAsciiZ
		getString: value
		size: metadata totalSize 
		status: status.
	status check.
	^ (value upTo: 0) asString
! !

!TFOperation methodsFor: 'attributes' stamp: 'gera 7/3/2019 03:33:19'!
stringsAt: nameString
	| status pointers sizes spaceRequired storage metadata valuesCount |
	metadata := self attrMetadata: nameString.
	spaceRequired := metadata totalSize.
	valuesCount := metadata listSize.
	pointers := ByteArray new: Smalltalk wordSize * valuesCount.
	sizes := (Int64Array externalNew: valuesCount) useFinalization.
	storage := ExternalAddress gcallocate: spaceRequired.
	status := TFStatus create.
	self library
		operation: self
		attr: nameString asAsciiZ 
		getStrings: pointers
		sizes: sizes
		maxCount: valuesCount
		storage: storage
		size: spaceRequired
		status: status.
	status check.
	^ (1 to: valuesCount) collect: [:i |
		| one |
		one := pointers pointerAt: i-1*Smalltalk wordSize+1.
		one := one structAt: 1 length: (sizes at: i).
		one asString]! !

!TFOperation methodsFor: 'attributes' stamp: 'gera 7/3/2019 03:33:19'!
tensorAt: nameString
	| value status |
	status := TFStatus create.
	value := ByteArray new: ExternalAddress wordSize.
	self library operation: self attr: nameString asAsciiZ getTensor: value status: status.
	status check.
	^ TFTensor fromHandle: (value pointerAt: 1)
! !

!TFOperation methodsFor: 'attributes' stamp: 'gera 7/3/2019 03:33:19'!
typeAt: nameString
	| value status |
	status := TFStatus create.
	value := ByteArray new: 8.
	self library operation: self attr: nameString asAsciiZ getType: value status: status.
	status check.
	^ value unsignedLongLongAt: 1
! !

!TFOperation methodsFor: 'ops other' stamp: 'gera 2/19/2017 11:37:15'!
descent: delta rate: learningRate
	^ self op: 'ApplyGradientDescent' withAll: {learningRate. delta}! !

!TFOperation methodsFor: 'ops other' stamp: 'gera 1/30/2017 23:39:24'!
nameFor: namePrefix
	^ graph nameFor: namePrefix! !

!TFOperation methodsFor: 'ops other' stamp: 'gera 2/19/2017 11:32:50'!
op: aString withAll: aTFOperationArray
	^ self op: aString withAll: aTFOperationArray described: [:nothing]
! !

!TFOperation methodsFor: 'ops other' stamp: 'gera 2/19/2017 11:32:46'!
op: aString withAll: aTFOperationArray described: oneArgBlock
	| name |
	name := self nameFor: aString.
	^ self op: aString withAll: aTFOperationArray named: name described: oneArgBlock! !

!TFOperation methodsFor: 'ops other' stamp: 'gera 2/19/2017 11:33:17'!
op: aString withAll: aTFOperationArray named: name
	^ self op: aString withAll: aTFOperationArray named: name described: [:nothing]
! !

!TFOperation methodsFor: 'ops other' stamp: 'gera 2/25/2017 11:47:40'!
op: aString withAll: aTFOperationArray named: name described: oneArgBlock
	^ graph newOperation: aString named: name described: [:description |
		description addInput: self output.
		aTFOperationArray do: [:each |
			| input |
			input := (each asOperationOn: graph) output.
			description addInput: input].
		oneArgBlock value: description].
	! !

!TFOperation methodsFor: 'ops other' stamp: 'gera 2/19/2017 11:39:00'!
sliceFrom: begin size: size
	^ self op: 'Slice' withAll: {begin. size}! !

!TFOperation methodsFor: 'initialization' stamp: 'gera 2/12/2017 20:10:23'!
output
	output ifNil: [output := 0].
	^ self output: output	! !

!TFOperationDescription methodsFor: 'accessing' stamp: 'jb 12/26/2016 22:35:53'!
add: input
	self library description: self add: input.! !

!TFOperationDescription methodsFor: 'accessing' stamp: 'gera 7/3/2019 03:33:19'!
finish
	| answer status |
	status := TFStatus create.
	answer := self library
		finishOperation: self
		status: status.
	handle := nil.
	status check.
	^ answer.! !

!TFOperationDescription methodsFor: 'inputs' stamp: 'gera 1/22/2017 20:37:18'!
addControlInput: aTFOutputOrInput
	^self library
		description: self
		addControlInput: aTFOutputOrInput! !

!TFOperationDescription methodsFor: 'inputs' stamp: 'gera 12/28/2016 01:32:31'!
addInput: aTFOutputOrInput
	^self library
		description: self
		addInput: aTFOutputOrInput! !

!TFOperationDescription methodsFor: 'inputs' stamp: 'gera 12/28/2016 01:40:51'!
addInputFromOutput: indexInteger of: aTFOperation
	| input |
	input := aTFOperation input: indexInteger.
	[self library
		description: self
		addInput: input
	] ensure: [input free]! !

!TFOperationDescription methodsFor: 'inputs' stamp: 'gera 7/3/2019 03:30:56'!
addInputs: anArrayOfTFOutput
	| inputs |
	inputs := TFOutputArray externalFromArray: anArrayOfTFOutput.
	self library
		description: self
		addInputs: inputs
		size: anArrayOfTFOutput size.! !

!TFOperationDescription methodsFor: 'attributes' stamp: 'gera 1/9/2017 19:53:37'!
at: attribute putBoolean: value
	self library
		description: self
		set: attribute asAsciiZ
		toBool: value
		! !

!TFOperationDescription methodsFor: 'attributes' stamp: 'gera 1/10/2017 10:35:19'!
at: attribute putFloat: value
	self library
		description: self
		set: attribute asAsciiZ
		toFloat: value asFloat
		! !

!TFOperationDescription methodsFor: 'attributes' stamp: 'gera 1/7/2017 05:11:24'!
at: attribute putInt: value
	self library
		description: self
		set: attribute asAsciiZ
		toInt64: value! !

!TFOperationDescription methodsFor: 'attributes' stamp: 'gera 7/3/2019 03:33:19'!
at: attribute putShape: anIntegerArray
	| status value |
	value := Int64Array externalFromArray: anIntegerArray.
	status := TFStatus create.
	self library
		description: self
		set: attribute asAsciiZ
		toShape: value
		size: anIntegerArray size.
	status check.! !

!TFOperationDescription methodsFor: 'attributes' stamp: 'gera 7/3/2019 03:33:19'!
at: attribute putString: aString
	| status |
	status := TFStatus create.
	self library
		description: self
		set: attribute asAsciiZ
		toString: aString
		size: aString size.
	status check
	! !

!TFOperationDescription methodsFor: 'attributes' stamp: 'gera 7/3/2019 03:33:19'!
at: attribute putStrings: anArrayOfStrings
	| status sizes strings pointers |
	sizes := anArrayOfStrings collect: [:str | str size].
	sizes := Int64Array externalFromArray: sizes.
	sizes useFinalization.
	strings := anArrayOfStrings collect: [:each | (self library externalizeString: each) useFinalization].
	pointers := ByteArray new: Smalltalk wordSize * strings size.
	strings withIndexDo: [:each :index |
		pointers pointerAt: index-1*Smalltalk wordSize+1 put: each].
	 
	status := TFStatus create.
	self library
		description: self
		set: attribute asAsciiZ
		toStrings: pointers
		sizes: sizes
		count: anArrayOfStrings size.
	status check
	! !

!TFOperationDescription methodsFor: 'attributes' stamp: 'gera 7/3/2019 03:33:19'!
at: attribute putTensor: aTFTensor
	| status |
	status := TFStatus create.
	self library
		description: self
		set: attribute asAsciiZ
		toTensor: aTFTensor
		status: status.
	status check
	! !

!TFOperationDescription methodsFor: 'attributes' stamp: 'gera 1/7/2017 05:12:20'!
at: attribute putType: value
	self library
		description: self
		set: attribute asAsciiZ
		toType: value! !

!TFOperationDescription methodsFor: 'attributes' stamp: 'gera 1/22/2017 17:58:17'!
device: aString
	^self library description: self setDevice: aString! !

!TFOutput methodsFor: 'accessing' stamp: 'JB 4/18/2018 20:13:17'!
index
	"This method was automatically generated. See TFOutput class>>fields."
	^handle signedLongAt: 9! !

!TFOutput methodsFor: 'accessing' stamp: 'JB 4/18/2018 19:55:17'!
index: anObject
	"This method was automatically generated. See TFOutput class>>fields."
	handle signedLongAt: 9 put: anObject! !

!TFOutput methodsFor: 'accessing' stamp: 'gera 7/3/2019 00:47:14'!
operation
	"This method was automatically generated. See TFOutput class>>fields."
	^TFOperation fromHandle: (handle longPointerAt: 1)! !

!TFOutput methodsFor: 'accessing' stamp: 'gera 12/23/2016 10:17:53'!
operation: anObject
	"This method was automatically generated. See TFOutput class>>fields."
	handle longPointerAt: 1 put: anObject getHandle.! !

!TFOutput methodsFor: 'accessing' stamp: 'gera 7/3/2019 00:47:14'!
operationOn: aTFGraph
	| answer |
	answer := TFOperation fromHandle: (handle longPointerAt: 1).
	answer graph: aTFGraph.
	^ answer! !

!TFOutput methodsFor: 'accessing' stamp: 'JB 4/18/2018 21:58:27'!
pad
	"This method was automatically generated. See TFOutput class>>fields."
	^handle signedLongAt: 13! !

!TFOutput methodsFor: 'accessing' stamp: 'JB 4/18/2018 21:58:27'!
pad: anObject
	"This method was automatically generated. See TFOutput class>>fields."
	handle signedLongAt: 13 put: anObject! !

!TFOutput methodsFor: 'accessing' stamp: 'gera 12/23/2016 10:26:22'!
type
	^ self library operationOutputType: self! !

!TFOutput class methodsFor: 'field definition' stamp: 'JB 4/18/2018 21:58:22'!
fields
	"self initialize.
	self defineFields"
	^#(
		(operation 'TFOperation*')
		(index 'long')
		(pad 'long'))
	! !

!TFOutput class methodsFor: 'instance creation' stamp: 'gera 12/23/2016 10:40:36'!
onOperation: aTFOperation index: anInteger
	^ self externalNew
		operation: aTFOperation;
		index: anInteger! !

!TFSession methodsFor: 'release' stamp: 'gera 7/3/2019 03:33:19'!
close
	| status |
	status := TFStatus create.
	self library closeSession: self status: status.
	status check! !

!TFSession methodsFor: 'release' stamp: 'gera 7/3/2019 03:33:19'!
delete
	| status |
	self ignoreFinalization.
	self isNull ifFalse: [
		self close.
		status := TFStatus create.
		self library
			deleteSession: self
			status: status.
		status check.
	].
	handle := nil.
! !

!TFSession methodsFor: 'release' stamp: 'gera 1/21/2017 17:08:40'!
finalize
	[self delete] on: Error do: [
		Transcript
			newLine;
			print: 'Error finalizing ';
			show: self
	].! !

!TFSession methodsFor: 'running' stamp: 'gera 7/3/2019 03:33:19'!
run
	| status |
	status := TFStatus create.
	
	self library
		runSession: self options: nil
		inputs: nil values: nil count: 0
		outputs: nil values: nil count: 0
		targets: nil count: 0 metadata: nil status: status.
	
	status check! !

!TFSession methodsFor: 'running' stamp: 'gera 7/3/2019 03:33:19'!
runInputs: inArrayOfTFOutputs values: inArrayOfTFTensor outputs: outArrayOfTFOutputs
	| inputs invalues outputs outvalues status |
	status := TFStatus create.
	inputs := TFOutputArray externalFromArray: inArrayOfTFOutputs.
	invalues := TFTensorPtrArray externalFromArray: inArrayOfTFTensor.
	outputs := TFOutputArray externalFromArray: outArrayOfTFOutputs.
	outvalues := TFTensorPtrArray new: outArrayOfTFOutputs size.
	
	self library
		runSession: self options: nil
		inputs: inputs values: invalues count: inArrayOfTFOutputs size
		outputs: outputs values: outvalues count: outArrayOfTFOutputs size
		targets: nil count: 0
		metadata: nil status: status.
		
	status check.
	^ outvalues
! !

!TFSession methodsFor: 'running' stamp: 'gera 12/29/2016 18:03:48'!
runOperation: aTFOperation
	^self runOperations: (Array with: aTFOperation)! !

!TFSession methodsFor: 'running' stamp: 'gera 1/13/2017 13:09:21'!
runOperation: aTFOperation input: inTFOutputOrInput value: inTFTensor output: outTFOutput
	| inputs invalues operations outputs outvalues tensor |
	inputs := Array with: inTFOutputOrInput.
	invalues := Array with: inTFTensor.
	outputs := Array with: outTFOutput.
	operations := Array with: aTFOperation.
	
	outvalues := self runOperations: operations inputs: inputs values: invalues outputs: outputs.
	tensor := outvalues first.
	^ tensor
! !

!TFSession methodsFor: 'running' stamp: 'gera 7/3/2019 03:30:56'!
runOperation: aTFOperation output: aTFOutput
	| operations  answer outputs |
	operations := TFOperationPtrArray externalNew: 1.
	outputs := TFOutputArray externalNew: 1.
	outputs first: aTFOutput.
	
	operations first: aTFOperation.	
	answer := self runOperations: operations outputs: outputs size: 1.

	^ answer first! !

!TFSession methodsFor: 'running' stamp: 'gera 7/3/2019 03:33:19'!
runOperations: anArrayOfTFOperations
	| status operations |
	status := TFStatus create.
	operations := TFOperationPtrArray externalFromArray: anArrayOfTFOperations.
	
	self library
		runSession: self options: nil
		inputs: nil values: nil count: 0
		outputs: nil values: nil count: 0
		targets: operations count: anArrayOfTFOperations size
		metadata: nil status: status.
	status check
	! !

!TFSession methodsFor: 'running' stamp: 'gera 7/3/2019 03:33:19'!
runOperations: anArrayOfTFOperations inputs: inArrayOfTFOutputs values: inArrayOfTFTensor outputs: outArrayOfTFOutputs
	| operations inputs invalues outputs outvalues status |
	status := TFStatus create.
	operations := TFOperationPtrArray externalFromArray: anArrayOfTFOperations.
	inputs := TFOutputArray externalFromArray: inArrayOfTFOutputs.
	invalues := TFTensorPtrArray externalFromArray: inArrayOfTFTensor.
	outputs := TFOutputArray externalFromArray: outArrayOfTFOutputs.
	outvalues := TFTensorPtrArray new: outArrayOfTFOutputs size.
	
	self library
		runSession: self options: nil
		inputs: inputs values: invalues count: inArrayOfTFOutputs size
		outputs: outputs values: outvalues count: outArrayOfTFOutputs size
		targets: operations count: anArrayOfTFOperations size
		metadata: nil status: status.
		
	status check.
	^ outvalues
! !

!TFSession methodsFor: 'running' stamp: 'gera 7/3/2019 03:33:19'!
runOperations: aTFOperationArray outputs: aTFOutputArray size: anInteger
	| status outvalues |
	status := TFStatus create.
	outvalues := TFTensorPtrArray new: anInteger.
	
	self library
		runSession: self options: nil
		inputs: nil values: nil count: 0
		outputs: aTFOutputArray values: outvalues count: anInteger
		targets: aTFOperationArray count: 1
		metadata: nil status: status.
		
	status check.
	^ outvalues! !

!TFSession methodsFor: 'running' stamp: 'gera 1/21/2017 22:00:03'!
runOutput: aTFOutput
	| results |
	results := self runOutputs: {aTFOutput}.
	^ results first! !

!TFSession methodsFor: 'running' stamp: 'gera 7/3/2019 03:30:56'!
runOutputs: anArrayOfTFOutputs
	| outputs |
	outputs := TFOutputArray externalFromArray: anArrayOfTFOutputs .
	
	^ self runOutputs: outputs size: anArrayOfTFOutputs size! !

!TFSession methodsFor: 'running' stamp: 'gera 7/3/2019 03:33:19'!
runOutputs: aTFOutputArray size: anInteger
	| status outvalues |
	status := TFStatus create.
	outvalues := TFTensorPtrArray new: anInteger.
	
	self library
		runSession: self options: nil
		inputs: nil values: nil count: 0
		outputs: aTFOutputArray values: outvalues count: anInteger
		targets: nil count: 0
		metadata: nil status: status.
		
	status check.
	^ outvalues! !

!TFSession class methodsFor: 'instance creation' stamp: 'gera 7/3/2019 03:35:40'!
on: aTFGraph
	| options status answer |
	options := TFSessionOptions create.
	status := TFStatus create.
	answer := TensorFlowCAPI current newSession: aTFGraph options: options status: status.
	status check.
	answer useFinalization.
	^ answer! !

!TFSessionOptions methodsFor: 'initialize-release' stamp: 'gera 7/3/2019 03:33:19'!
config: aString
	| status |
	status := TFStatus create.
	self library sessionOptions: self setConfig: aString configSize: aString size status: status.
	status check.! !

!TFSessionOptions methodsFor: 'initialize-release' stamp: 'gera 1/21/2017 17:12:10'!
delete
	self ignoreFinalization.
	self isNull ifFalse: [
		self library deleteSessionOptions: self].
	handle := nil! !

!TFSessionOptions methodsFor: 'initialize-release' stamp: 'gera 1/21/2017 17:12:20'!
finalize
	self delete! !

!TFSessionOptions methodsFor: 'initialize-release' stamp: 'gera 12/21/2016 23:37:00'!
target: aString
	" 'local' 'google.com:1234' '192.168.1.1:1234' 'local,example.com:1234' etc.
	are all valid target strings"
	self library sessionOptions: self setTarget: (self library externalizeString: aString).
! !

!TFSessionOptions class methodsFor: 'instance creation' stamp: 'gera 1/21/2017 17:12:26'!
create
	^ TensorFlowCAPI current newSessionOptions useFinalization! !

!TFSessionOptions class methodsFor: 'instance creation' stamp: 'gera 1/21/2017 17:32:21'!
fromProtoBuf: aString
	| answer |
	answer := self create.
	answer config: aString.
	^ answer! !

!TFSessionOptions class methodsFor: 'instance creation' stamp: 'gera 12/21/2016 23:47:47'!
onTarget: aString
	^ self create target: aString! !

!TFStatus methodsFor: 'printing' stamp: 'gera 1/9/2017 11:15:35'!
printOn: strm
	super printOn: strm.
	self isNull ifFalse: [
		strm
			space;
			print: self code;
			space;
			print: self message]! !

!TFStatus methodsFor: 'testing' stamp: 'gera 12/21/2016 11:05:49'!
check
	self isOk ifFalse: [
		Error signal: (self codeText, ': ', self message)]! !

!TFStatus methodsFor: 'testing' stamp: 'gera 12/21/2016 10:48:36'!
isOk
	^self code == 0! !

!TFStatus methodsFor: 'accessing' stamp: 'gera 12/21/2016 09:59:56'!
code
	^self library getCode: self! !

!TFStatus methodsFor: 'accessing' stamp: 'jb 12/26/2016 10:29:57'!
code: aTF_Code message: aString
	| externalized |
	self assert: aTF_Code ~= 0.
	externalized := self library externalizeString: aString.
	self library setStatus: self code: aTF_Code message: externalized
! !

!TFStatus methodsFor: 'accessing' stamp: 'gera 12/21/2016 11:03:40'!
codeText
	^ self codeTexts at: self code + 1! !

!TFStatus methodsFor: 'accessing' stamp: 'gera 12/21/2016 11:02:44'!
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

!TFStatus methodsFor: 'accessing' stamp: 'gera 12/21/2016 23:50:21'!
message
	^ self library message: self! !

!TFStatus methodsFor: 'initialize-release' stamp: 'gera 1/19/2017 09:47:23'!
delete
	self ignoreFinalization.
	self isNull ifFalse: [
		self library deleteStatus: self].
	handle := nil.! !

!TFStatus methodsFor: 'initialize-release' stamp: 'gera 1/21/2017 17:09:37'!
finalize
	self delete! !

!TFStatus class methodsFor: 'instance creation' stamp: 'gera 1/7/2017 20:20:47'!
create
	|answer |
	answer := self persistent.
	answer useFinalization.
	^ answer! !

!TFStatus class methodsFor: 'instance creation' stamp: 'gera 1/3/2017 09:53:25'!
persistent
	^ TensorFlowCAPI current newStatus! !

!TFTensor methodsFor: 'accessing' stamp: 'gera 2/13/2017 23:34:13'!
byteSize
	^ self library tensorByteSize: self! !

!TFTensor methodsFor: 'accessing' stamp: 'gera 12/27/2016 09:18:01'!
data
	^self library tensorData: self.! !

!TFTensor methodsFor: 'accessing' stamp: 'gera 2/13/2017 23:34:30'!
dataBytes
	^self data getHandle structAt: 1 length: self byteSize! !

!TFTensor methodsFor: 'accessing' stamp: 'gera 2/13/2017 23:32:07'!
elementSize
	^ self class sizeForType: self type! !

!TFTensor methodsFor: 'accessing' stamp: 'jb 12/26/2016 12:09:58'!
rank
	^self library tensorRank: self.! !

!TFTensor methodsFor: 'accessing' stamp: 'gera 1/6/2017 12:41:46'!
shape
	| answer count |
	count := self rank.
	answer := WriteStream on: (Array new: count).
	1 to: count do: [:i |
		answer nextPut: (self sizeOn: i-1)].
	^ answer contents! !

!TFTensor methodsFor: 'accessing' stamp: 'gera 1/6/2017 14:37:36'!
size
	^ self shape inject: 1 into: [:prev :next | prev * next].! !

!TFTensor methodsFor: 'accessing' stamp: 'jb 12/26/2016 12:10:24'!
sizeOn: dimension
	^self library tensor: self sizeOn: dimension.! !

!TFTensor methodsFor: 'accessing' stamp: 'gera 12/26/2016 10:50:33'!
type
	^self library tensorType: self.! !

!TFTensor methodsFor: 'converting' stamp: 'gera 2/14/2017 16:12:18'!
allElements
	^ self asStream contents! !

!TFTensor methodsFor: 'converting' stamp: 'gera 2/13/2017 23:59:20'!
allFloats
	^ self allElements! !

!TFTensor methodsFor: 'converting' stamp: 'gera 2/13/2017 23:59:27'!
allInt32s
	^ self allElements! !

!TFTensor methodsFor: 'converting' stamp: 'gera 2/13/2017 23:59:32'!
allInt64s
	^ self allElements! !

!TFTensor methodsFor: 'converting' stamp: 'gera 1/6/2017 14:56:06'!
allStrings
	| total answer bytes data |
	total := self size.
	answer := WriteStream on: (Array new: total).
	bytes := self dataBytes.
	data := self data getHandle + (8 * total).
	1 to: total do: [:i |
		| offset str |
		offset := bytes unsignedLongLongAt: i - 1 * 8 + 1.
		str := self library stringDecode: data + offset.
		answer nextPut: str].
	
	^ answer contents! !

!TFTensor methodsFor: 'converting' stamp: 'gera 2/15/2017 08:03:16'!
arrayFromStream: strm shape: shape
	^ shape isEmpty
		ifTrue: [strm next]
		ifFalse: [
			| first tail |
			first := shape first.
			tail := shape allButFirst.
			Array
				streamContents: [:answer |
					first timesRepeat: [
						| next |
						next := self arrayFromStream: strm shape: tail.
						answer nextPut: next]]
				estimatedSize: first]
				
! !

!TFTensor methodsFor: 'converting' stamp: 'gera 2/15/2017 08:04:19'!
asNumbers
	^ self arrayFromStream: self asStream shape: self shape! !

!TFTensor methodsFor: 'converting' stamp: 'gera 2/17/2017 00:29:42'!
asOperationOn: aTFGraph
	^ aTFGraph const: self! !

!TFTensor methodsFor: 'converting' stamp: 'gera 2/15/2017 01:29:29'!
asStream
	| answer |
	answer := ReadWriteStream on: (Array new: self size).
	self elementsDo: [:each | answer nextPut: each].
	^ answer reset! !

!TFTensor methodsFor: 'iterating' stamp: 'gera 2/13/2017 23:35:27'!
dataAndOffsetsCollect: twoArgsBlock thenDo: oneArgBlock
	| data elementSize |
	elementSize := self elementSize.
	data := self data getHandle.
	1 to: self size do: [:i |
		| value |
		value := twoArgsBlock value: data value: i-1*elementSize+1.
		oneArgBlock value: value].
! !

!TFTensor methodsFor: 'iterating' stamp: 'gera 2/15/2017 17:01:13'!
elementsDo: oneArgBlock
	self
		dataAndOffsetsCollect: self getterBlock
		thenDo: oneArgBlock.
! !

!TFTensor methodsFor: 'iterating' stamp: 'gera 2/13/2017 18:53:45'!
floatsDo: oneArgBlock
	self
		dataAndOffsetsCollect: [:data :offset | data floatAt: offset]
		thenDo: oneArgBlock.
! !

!TFTensor methodsFor: 'iterating' stamp: 'gera 2/15/2017 17:01:04'!
getterBlock
	^ self class getterBlockFor: self type! !

!TFTensor methodsFor: 'iterating' stamp: 'gera 2/13/2017 18:54:54'!
int32sDo: oneArgBlock
	self
		dataAndOffsetsCollect: [:data :offset | data signedLongAt: offset]
		thenDo: oneArgBlock.
! !

!TFTensor methodsFor: 'iterating' stamp: 'gera 2/13/2017 23:36:22'!
int64sDo: oneArgBlock
	self
		dataAndOffsetsCollect: [:data :offset | data signedLongLongAt: offset]
		thenDo: oneArgBlock.
! !

!TFTensor methodsFor: 'release' stamp: 'gera 1/19/2017 09:45:31'!
delete
	self ignoreFinalization.
	self isNull ifFalse: [
		self library deleteTensor: self].
	handle := nil! !

!TFTensor methodsFor: 'release' stamp: 'gera 1/13/2017 11:15:56'!
finalize
	self delete! !

!TFTensor class methodsFor: 'utils' stamp: 'gera 2/15/2017 21:15:40'!
array: values type: type into: anExternalAddressOrByteArray
	| size index setter |
	size := self sizeForType: type.
	setter := self setterBlockFor: type.
	index := 0.
	self elementsOf: values do: [:value |
		setter value: anExternalAddressOrByteArray value: index*size + 1 value: value.
		index := index + 1].
! !

!TFTensor class methodsFor: 'utils' stamp: 'gera 2/18/2017 17:01:13'!
elementsOf: aMultidimensionalTensor
	| answer |
	answer := (Array new: (self sizeOf: aMultidimensionalTensor)) writeStream.
	self elementsOf: aMultidimensionalTensor do: [:each | answer nextPut: each].
	^ answer contents! !

!TFTensor class methodsFor: 'utils' stamp: 'gera 2/18/2017 16:55:42'!
elementsOf: tensorArray do: oneArgBlock
	^ (tensorArray isCollection and: [tensorArray isString not])
		ifTrue: [tensorArray do: [:each |
				self elementsOf: each do: oneArgBlock]]
		ifFalse: [oneArgBlock value: tensorArray]! !

!TFTensor class methodsFor: 'utils' stamp: 'gera 2/15/2017 22:29:58'!
int64Array: values into: anExternalAddressOrByteArray
	self array: values type: self typeInt64 into: anExternalAddressOrByteArray.
! !

!TFTensor class methodsFor: 'utils' stamp: 'gera 1/13/2017 19:45:52'!
rankOf: aMultidimensionalTensor
	^ aMultidimensionalTensor isCollection
		ifTrue: [1 + (self rankOf: aMultidimensionalTensor first)]
		ifFalse: [0].! !

!TFTensor class methodsFor: 'utils' stamp: 'gera 2/3/2017 12:09:32'!
shapeOf: aMultidimensionalTensor
	^ aMultidimensionalTensor isCollection
		ifTrue: [aMultidimensionalTensor isEmpty
			ifTrue: [#(0)]
			ifFalse: [{aMultidimensionalTensor size}, (self shapeOf: aMultidimensionalTensor first)]]
		ifFalse: [#()].! !

!TFTensor class methodsFor: 'utils' stamp: 'gera 2/3/2017 12:06:44'!
sizeOf: aMultidimensionalTensor
	^ aMultidimensionalTensor isCollection
		ifTrue: [
			aMultidimensionalTensor isEmpty
				ifTrue: [0]
				ifFalse: [aMultidimensionalTensor sum: [:each | self sizeOf: each]]]
		ifFalse: [1].! !

!TFTensor class methodsFor: 'utils' stamp: 'gera 1/13/2017 03:17:43'!
sizeOfString: aString.
	^ TensorFlowCAPI current stringEncodedSize: aString.! !

!TFTensor class methodsFor: 'instance creation' stamp: 'gera 2/19/2017 19:51:36'!
fromBooleans: values
	^ self fromNumbers: values type: self typeBoolean
! !

!TFTensor class methodsFor: 'instance creation' stamp: 'gera 2/19/2017 19:51:38'!
fromBools: values shape: shape
	^ self fromNumbers: values type: self typeBoolean shape: shape
! !

!TFTensor class methodsFor: 'instance creation' stamp: 'gera 2/17/2017 09:45:34'!
fromDoubles: values
	^ self fromNumbers: values type: self typeDouble
! !

!TFTensor class methodsFor: 'instance creation' stamp: 'gera 2/17/2017 09:27:42'!
fromDoubles: values shape: shape
	^ self fromNumbers: values type: self typeDouble shape: shape
! !

!TFTensor class methodsFor: 'instance creation' stamp: 'gera 2/15/2017 22:35:31'!
fromFloats: values
	^ self fromNumbers: values type: self typeFloat
! !

!TFTensor class methodsFor: 'instance creation' stamp: 'gera 2/15/2017 22:38:57'!
fromFloats: values shape: shape
	^ self fromNumbers: values type: self typeFloat shape: shape
! !

!TFTensor class methodsFor: 'instance creation' stamp: 'gera 12/30/2016 12:46:27'!
fromInt32: value
	| answer |
	answer := self type: self typeInt32 shape: #().
	answer data getHandle signedLongAt: 1 put: value.
	^ answer! !

!TFTensor class methodsFor: 'instance creation' stamp: 'gera 2/15/2017 22:37:53'!
fromInt32s: values
	^ self fromNumbers: values type: self typeInt32.
! !

!TFTensor class methodsFor: 'instance creation' stamp: 'gera 2/15/2017 22:39:40'!
fromInt32s: values shape: shape
	^ self fromNumbers: values type: self typeInt32 shape: shape
! !

!TFTensor class methodsFor: 'instance creation' stamp: 'gera 12/29/2016 17:53:03'!
fromInt64: value
	| answer |
	answer := self type: self typeInt64 shape: #().
	answer data getHandle signedLongLongAt: 1 put: value.
	^ answer! !

!TFTensor class methodsFor: 'instance creation' stamp: 'gera 2/15/2017 22:41:15'!
fromInt64s: values
	^ self fromNumbers: values type: self typeInt64! !

!TFTensor class methodsFor: 'instance creation' stamp: 'gera 2/15/2017 22:44:54'!
fromInt64s: values shape: shape
	^ self fromNumbers: values type: self typeInt64 shape: shape
! !

!TFTensor class methodsFor: 'instance creation' stamp: 'gera 2/15/2017 22:36:51'!
fromNumbers: values type: type
	| shape |
	shape := self shapeOf: values.
	^ self fromNumbers: values type: type shape: shape! !

!TFTensor class methodsFor: 'instance creation' stamp: 'gera 2/15/2017 22:36:20'!
fromNumbers: values type: type shape: shape
	| answer size count |
	size := self sizeOf: values.
	count := shape inject: 1 into: [:a :b | a * b].
	count = size ifFalse: [
		self error: 'Inferred size and real size don''t match.'].
		
	answer := self type: type shape: shape.
	self array: values type: type into: answer data getHandle.
	^ answer! !

!TFTensor class methodsFor: 'instance creation' stamp: 'gera 1/13/2017 03:57:18'!
fromString: aString
	| answer bytesize |
	bytesize := TensorFlowCAPI current stringEncodedSize: aString.
	bytesize := bytesize.
	answer := self type: self typeString shape: #() bytesize: bytesize.
	
	TensorFlowCAPI current
		stringEncode: aString
		to: answer data getHandle
		size: bytesize.
	^ answer! !

!TFTensor class methodsFor: 'instance creation' stamp: 'gera 2/18/2017 17:23:33'!
fromStringArray: aStringArray
	^ self fromStrings: aStringArray shape: {aStringArray size}! !

!TFTensor class methodsFor: 'instance creation' stamp: 'gera 2/18/2017 16:51:06'!
fromStringArray: aStringArray shape: shape
	| answer sizes offsets offsetSize bytesize |
	sizes := aStringArray collect: [:each | self sizeOfString: each].
	offsetSize := self sizeForType: self typeInt64.
	bytesize := offsetSize * aStringArray size + sizes sum.
	offsets := OrderedCollection new.
	
	sizes inject: 0 into: [:prev :each |
		offsets add: prev.
		each + prev].
	
	answer := self
		type: self typeString
		shape: shape
		bytesize: bytesize.
		
	self int64Array: offsets into: answer data getHandle.	
	
	aStringArray withIndexDo: [:each :index |
		| offset |
		offset := offsetSize * aStringArray size + (offsets at: index).
		TensorFlowCAPI current
			stringEncode: each
			to: answer data getHandle + offset
			size: bytesize - offset].
	^ answer
! !

!TFTensor class methodsFor: 'instance creation' stamp: 'gera 6/30/2019 00:45:26'!
fromStrings: aStringArray
	^ self fromStrings: aStringArray shape: (TFTensor shapeOf: aStringArray) allButLast.
! !

!TFTensor class methodsFor: 'instance creation' stamp: 'gera 2/18/2017 17:20:48'!
fromStrings: strings shape: shape
	| flatten |
	flatten := self elementsOf: strings.
	^ self fromStringArray:  flatten shape: shape.
! !

!TFTensor class methodsFor: 'instance creation' stamp: 'gera 1/29/2017 23:22:39'!
pi
	^ Float pi asTensor! !

!TFTensor class methodsFor: 'instance creation' stamp: 'gera 1/23/2017 16:11:51'!
type: anInteger shape: anIntegerArray
	" I understand dimenssions are:
		#()		-> Scalar
		#(7)		-> Unidimensional array of 7 elements
		#(7 4)		-> 7x4 elements matrix
		#(2 5 9)	-> 2x5x9 elements cube
		etc.
	"
		
	| bytesize elementSize |
	elementSize := self sizeForType: anInteger.
	bytesize := anIntegerArray inject: elementSize into: [:prev :each | prev * each].
	^self type: anInteger shape: anIntegerArray bytesize: bytesize
! !

!TFTensor class methodsFor: 'instance creation' stamp: 'gera 1/13/2017 13:10:55'!
type: anInteger shape: anIntegerArray bytesize: bytesizeInteger
	" I understand dimenssions are:
		#()		-> Scalar
		#(7)		-> Unidimensional array of 7 elements
		#(7 4)		-> 7x4 elements matrix
		#(2 5 9)	-> 2x5x9 elements cube
		etc.
	"
		
	| externalized answer |
	externalized := Int64Array externalFromArray: anIntegerArray.
		
	answer := TensorFlowCAPI current
		allocateTensorType: anInteger
		shape: externalized
		rank: anIntegerArray size
		length: bytesizeInteger.
	
	answer useFinalization.
	^ answer
! !

!TFTensor class methodsFor: 'types' stamp: 'gera 6/30/2019 00:45:27'!
getterBlockFor: type
	TFTensor typeInt32 = type ifTrue: [^ [:data :offset | data signedLongAt: offset]].
	TFTensor typeInt64 = type ifTrue: [^ [:data :offset | data signedLongLongAt: offset]].
	TFTensor typeFloat = type ifTrue: [^ [:data :offset | data floatAt: offset]].
	TFTensor typeDouble = type ifTrue: [^ [:data :offset | data doubleAt: offset]].
	TFTensor typeBoolean = type ifTrue: [^ [:data :offset | data booleanAt: offset]].
	^ self shouldBeImplemented ! !

!TFTensor class methodsFor: 'types' stamp: 'gera 6/30/2019 00:45:28'!
setterBlockFor: type
	TFTensor typeInt32 = type ifTrue: [^ [:data :offset :value | data signedLongAt: offset put: value]].
	TFTensor typeInt64 = type ifTrue: [^ [:data :offset :value | data signedLongLongAt: offset put: value]].
	TFTensor typeFloat = type ifTrue: [^ [:data :offset :value | data floatAt: offset put: value]].
	TFTensor typeDouble = type ifTrue: [^ [:data :offset :value | data doubleAt: offset put: value]].
	TFTensor typeBoolean = type ifTrue: [^ [:data :offset :value | data booleanAt: offset put: value]].
	^ self shouldBeImplemented ! !

!TFTensor class methodsFor: 'types' stamp: 'gera 2/19/2017 19:51:47'!
sizeForType: anInteger
	self typeFloat = anInteger ifTrue: [^4].
	self typeInt64 = anInteger ifTrue: [^8].
	self typeInt32 = anInteger ifTrue: [^4].
	self typeDouble = anInteger ifTrue: [^8].
	self typeBoolean = anInteger ifTrue: [^1].

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

!TFTensor class methodsFor: 'types' stamp: 'gera 1/2/2017 19:43:54'!
typeBFloat16
	^14! !

!TFTensor class methodsFor: 'types' stamp: 'gera 2/19/2017 19:51:24'!
typeBoolean
	^10! !

!TFTensor class methodsFor: 'types' stamp: 'gera 1/2/2017 19:42:37'!
typeComplex128
	^18! !

!TFTensor class methodsFor: 'types' stamp: 'gera 1/2/2017 19:42:31'!
typeComplex64
	^8! !

!TFTensor class methodsFor: 'types' stamp: 'gera 1/2/2017 19:41:29'!
typeDouble
	^2! !

!TFTensor class methodsFor: 'types' stamp: 'gera 12/26/2016 09:41:58'!
typeFloat
	^1! !

!TFTensor class methodsFor: 'types' stamp: 'gera 1/2/2017 19:44:35'!
typeHalf
	^19! !

!TFTensor class methodsFor: 'types' stamp: 'gera 1/2/2017 19:41:54'!
typeInt16
	^5! !

!TFTensor class methodsFor: 'types' stamp: 'gera 12/26/2016 11:14:05'!
typeInt32
	^3! !

!TFTensor class methodsFor: 'types' stamp: 'gera 12/26/2016 09:42:07'!
typeInt64
	^9! !

!TFTensor class methodsFor: 'types' stamp: 'gera 1/2/2017 19:42:06'!
typeInt8
	^6! !

!TFTensor class methodsFor: 'types' stamp: 'gera 1/2/2017 19:43:14'!
typeQInt16
	^15! !

!TFTensor class methodsFor: 'types' stamp: 'gera 1/2/2017 19:43:30'!
typeQInt32
	^13! !

!TFTensor class methodsFor: 'types' stamp: 'gera 1/2/2017 19:43:00'!
typeQInt8
	^11! !

!TFTensor class methodsFor: 'types' stamp: 'gera 1/2/2017 19:44:08'!
typeQUInt16
	^16! !

!TFTensor class methodsFor: 'types' stamp: 'gera 1/2/2017 19:50:48'!
typeQUInt8
	^12! !

!TFTensor class methodsFor: 'types' stamp: 'gera 1/2/2017 19:44:40'!
typeResource
	^20! !

!TFTensor class methodsFor: 'types' stamp: 'gera 1/2/2017 19:42:12'!
typeString
	^7! !

!TFTensor class methodsFor: 'types' stamp: 'gera 1/2/2017 19:44:19'!
typeUInt16
	^17! !

!TFTensor class methodsFor: 'types' stamp: 'gera 1/2/2017 19:41:40'!
typeUInt8
	^4! !

!Object methodsFor: '*TensorFlow-Kernel' stamp: 'gera 1/7/2017 20:18:45'!
ignoreFinalization
	self finalizationRegistry remove: self ifAbsent: [].
! !

!Object methodsFor: '*TensorFlow-Kernel' stamp: 'gera 1/7/2017 20:18:16'!
useFinalization
	self finalizationRegistry add: self! !

!Boolean methodsFor: '*TensorFlow-Kernel' stamp: 'gera 6/30/2019 00:45:27'!
asBooleanTensor
	^ TFTensor fromBooleans: self! !

!Number methodsFor: '*TensorFlow-Kernel' stamp: 'gera 6/30/2019 00:45:26'!
asFloatTensor
	^ TFTensor fromFloats: self! !

!Number methodsFor: '*TensorFlow-Kernel' stamp: 'gera 6/30/2019 00:45:25'!
asInt32Tensor
	^ TFTensor fromInt32s: self! !

!Number methodsFor: '*TensorFlow-Kernel' stamp: 'gera 6/30/2019 00:45:27'!
asInt64Tensor
	^ TFTensor fromInt64s: self! !

!Float methodsFor: '*TensorFlow-Kernel' stamp: 'gera 6/30/2019 00:45:26'!
asTensor
	^ TFTensor fromFloats: self! !

!Fraction methodsFor: '*TensorFlow-Kernel' stamp: 'gera 6/30/2019 00:45:27'!
asTensor
	^ TFTensor fromFloats: self! !

!SequenceableCollection methodsFor: '*TensorFlow-Kernel' stamp: 'gera 6/30/2019 00:45:26'!
asBooleanTensor
	^ TFTensor fromBooleans: self! !

!SequenceableCollection methodsFor: '*TensorFlow-Kernel' stamp: 'gera 6/30/2019 00:45:27'!
asFloatTensor
	^ TFTensor fromFloats: self! !

!SequenceableCollection methodsFor: '*TensorFlow-Kernel' stamp: 'gera 6/30/2019 00:45:28'!
asInt32Tensor
	^ TFTensor fromInt32s: self! !

!SequenceableCollection methodsFor: '*TensorFlow-Kernel' stamp: 'gera 6/30/2019 00:45:26'!
asInt64Tensor
	^ TFTensor fromInt64s: self! !

!String methodsFor: '*TensorFlow-Kernel' stamp: 'gera 1/7/2017 04:33:10'!
asAsciiZ
	^self, Character null asString! !

!String methodsFor: '*TensorFlow-Kernel' stamp: 'gera 6/30/2019 00:45:26'!
asTensor
	^ TFTensor fromString: self! !
TFStructure initialize!
