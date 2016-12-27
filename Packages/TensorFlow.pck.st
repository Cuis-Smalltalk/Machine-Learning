'From Cuis 5.0 of 7 November 2016 [latest update: #3007] on 27 December 2016 at 6:31:20 pm'!
'Description Please enter a description for this package'!
!provides: 'TensorFlow' 1 46!
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

!classDefinition: #TF_OperationArray category: #'TensorFlow-Kernel'!
ExternalWordArray subclass: #TF_OperationArray
	instanceVariableNames: ''
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Kernel'!
!classDefinition: 'TF_OperationArray class' category: #'TensorFlow-Kernel'!
TF_OperationArray class
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

!classDefinition: #TF_TensorArray category: #'TensorFlow-Kernel'!
ExternalWordArray subclass: #TF_TensorArray
	instanceVariableNames: ''
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Kernel'!
!classDefinition: 'TF_TensorArray class' category: #'TensorFlow-Kernel'!
TF_TensorArray class
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
	instanceVariableNames: ''
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Kernel'!
!classDefinition: 'TF_Operation class' category: #'TensorFlow-Kernel'!
TF_Operation class
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


!ExternalWordArrayTest methodsFor: 'testing' stamp: 'gera 12/25/2016 19:05:51'!
testArray: anExternalArraySubclass sampleElement: anObject
	| array size  |
	size _ 4.
	array _ anExternalArraySubclass new: 4.
	self assert: size * Smalltalk wordSize = array getHandle size.
	self assert: (array first isKindOf: anObject class).
	1 to: size do: [:index |
		self assert: ((array at: index) isKindOf: anObject class)].
! !

!ExternalWordArrayTest methodsFor: 'testing' stamp: 'gera 12/25/2016 19:14:38'!
testArrayAtPutFor: anExternalArraySubclass elementSample: anObject
	| array size one  |
	size _ 4.
	array _ anExternalArraySubclass new: 4.
	one _ array first.
	
	array first: anObject.
	self assert: array first = anObject.
	self assert: (array first isKindOf: anObject class).
	array first: one.
	self assert: array first = one.
		
	1 to: size do: [:index |
		array at: index put: anObject.
		self assert: (array at: index) = anObject.
		self assert: ((array at: index) isKindOf: anObject class).
		array at: index put: one.
		self assert: (array at: index) = one.
	]
	! !

!ExternalWordArrayTest methodsFor: 'testing' stamp: 'gera 12/25/2016 19:12:06'!
testOperationArrayAtPut
	| sample |
	sample _ TF_Operation externalNew.
	
	[self testArrayAtPutFor: TF_OperationArray elementSample: sample]
		ensure: [sample free]! !

!ExternalWordArrayTest methodsFor: 'testing' stamp: 'gera 12/25/2016 19:06:54'!
testOperationArrayElementsOfRightClass
	^ self testArray: TF_OperationArray sampleElement: TF_Operation new.
! !

!ExternalWordArrayTest methodsFor: 'testing' stamp: 'gera 12/25/2016 19:12:29'!
testOutputArrayAtPut
	| sample |
	sample _ TF_Output externalNew.
	
	[self testArrayAtPutFor: TF_OutputArray elementSample: sample]
		ensure: [sample free]! !

!ExternalWordArrayTest methodsFor: 'testing' stamp: 'gera 12/25/2016 19:06:44'!
testOutputArrayElementsOfRightClass
	^ self testArray: TF_OutputArray sampleElement: TF_Output new.! !

!ExternalWordArrayTest methodsFor: 'testing' stamp: 'gera 12/27/2016 11:30:34'!
testTensorArrayAtPut
	| sample |
	sample _ TF_Tensor externalNew.
	
	[self testArrayAtPutFor: TF_TensorArray elementSample: sample]
		ensure: [sample free]! !

!ExternalWordArrayTest methodsFor: 'testing' stamp: 'gera 12/27/2016 11:30:55'!
testTensorArrayElementsOfRightClass
	^ self testArray: TF_TensorArray sampleElement: TF_Tensor new.! !

!ExternalWordArrayTest methodsFor: 'testing' stamp: 'gera 12/25/2016 20:54:13'!
testUIn64ArrayfromArray
	self testUIn64ArrayfromArray: #(1).
	self testUIn64ArrayfromArray: #(16r12345678 2 16r76543210 4 16rDEADBEAF 6 16rFFFFFFFF).
! !

!ExternalWordArrayTest methodsFor: 'testing' stamp: 'gera 12/27/2016 18:11:45'!
testUIn64ArrayfromArray: template
	| array |
	array _ Int64Array externalFromArray: template.
	
	template withIndexDo: [:each :index |
		self assert: ((template at: index) = (array at: index))
	].

	array free.! !

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

!TensorFlowCAPITest methodsFor: 'testing graph' stamp: 'gera 12/26/2016 10:59:08'!
constantFloatGraph
	^ TF_Graph fromString: self constantFloatGraphDef! !

!TensorFlowCAPITest methodsFor: 'testing graph' stamp: 'gera 12/26/2016 10:58:56'!
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

!TensorFlowCAPITest methodsFor: 'testing graph' stamp: 'gera 12/26/2016 11:13:24'!
constantInt32Graph
	^ TF_Graph fromString: self constantInt32GraphDef! !

!TensorFlowCAPITest methodsFor: 'testing graph' stamp: 'gera 12/26/2016 11:13:29'!
constantInt32GraphDef
	" This GraphDef corresponds to simple Graph, defined as
	
		tf.constant(0.42, name='a')
		
	 saved as ProtoBuf "
	^ #[
 16r0A 16r2B 16r0A 16r01 16r61 16r12 16r05 16r43 16r6F 16r6E 16r73 16r74 16r2A 16r12 16r0A 16r05
 16r76 16r61 16r6C 16r75 16r65 16r12 16r09 16r42 16r07 16r08 16r03 16r12 16r00 16r3A 16r01 16r2A
 16r2A 16r0B 16r0A 16r05 16r64 16r74 16r79 16r70 16r65 16r12 16r02 16r30 16r03 16r0A 16r0C 16r0A
 16r04 16r69 16r6E 16r69 16r74 16r12 16r04 16r4E 16r6F 16r4F 16r70 16r22 16r02 16r08 16r0F] asString! !

!TensorFlowCAPITest methodsFor: 'testing graph' stamp: 'gera 12/26/2016 12:23:16'!
constantInt64Graph
	^ TF_Graph fromString: self constantInt64GraphDef! !

!TensorFlowCAPITest methodsFor: 'testing graph' stamp: 'gera 12/26/2016 12:23:03'!
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

!TensorFlowCAPITest methodsFor: 'testing graph' stamp: 'gera 12/22/2016 02:20:25'!
emptyGraph
	^ TF_Graph fromString: self emptyGraphDef.
	! !

!TensorFlowCAPITest methodsFor: 'testing graph' stamp: 'gera 12/22/2016 01:51:30'!
emptyGraphDef
	" This GraphDef corresponds to an Empty Graph (no operations), saved as ProtoBuf "
	^ #[16r0A 16r0C 16r0A 16r04 16r69 16r6E 16r69 16r74 16r12 16r04 16r4E 16r6F 16r4F 16r70 16r22 16r02 16r08 16r0F] asString! !

!TensorFlowCAPITest methodsFor: 'initialization' stamp: 'gera 12/21/2016 02:11:17'!
setUp
	library _ TensorFlowCAPI current! !

!TensorFlowCAPITest methodsFor: 'testing session' stamp: 'gera 12/21/2016 23:26:08'!
testCloseSession
	| graph session |
	graph _ TF_Graph create.
	session _ TF_Session onGraph: graph.
	self deny: session getHandle isNil.
	self deny: session getHandle = 0.
	self shouldnt: [session close] raise: Error.
	self shouldnt: [session delete] raise: Error.
	self assert: session getHandle isNil.! !

!TensorFlowCAPITest methodsFor: 'testing options' stamp: 'gera 12/22/2016 00:58:57'!
testExternalizeString
	| original copy |
	original _ 'hola manola'.
	copy _ library externalizeString: original.
	original withIndexDo: [:each :index |
		self assert: each asciiValue = (copy byteAt: index)].
	self assert: (copy byteAt: original size + 1) = 0.
	copy free.! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 12/26/2016 10:59:21'!
testGetOperationOnConstantGraph
	| graph op |
	graph _ self constantFloatGraph.
	self shouldnt: [op _ graph operationNamed: 'a'] raise: Error.
	self assert: op name = 'a'.
	self assert: op type = 'Const'.
	self assert: op InputsCount = 0.
	self assert: op outputsCount = 1.! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 12/22/2016 03:44:19'!
testGetOperationOnEmptyGraph
	| graph |
	graph _ self emptyGraph.
	self should: [graph operationNamed: 'something'] raise: Error! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 12/26/2016 11:10:00'!
testGraph: aTF_Graph outputType: anInteger
	| operation output |
	operation _ aTF_Graph operationNamed: 'a'.
	output _ TF_Output onOperation: operation index: 0.
	self assert: output type == anInteger.
	output free.! !

!TensorFlowCAPITest methodsFor: 'testing graph' stamp: 'gera 12/22/2016 01:30:28'!
testImportBad
	| graph buffer |
	graph _ TF_Graph create.
	buffer _ TF_Buffer fromString: 'ouch'.
	
	self should: [graph import: buffer] raise: Error.
	
	buffer delete.
	graph delete.
	! !

!TensorFlowCAPITest methodsFor: 'testing graph' stamp: 'gera 12/26/2016 12:23:48'!
testImportConstantGraph
	| graph |

	self shouldnt: [
		graph _ self constantFloatGraph.
		graph delete.
		graph _ self constantInt32Graph.
		graph delete.
		graph _ self constantInt64Graph.
		graph delete.
		] raise: Error.
	! !

!TensorFlowCAPITest methodsFor: 'testing graph' stamp: 'gera 12/22/2016 02:19:54'!
testImportEmpty
	| graph |

	self shouldnt: [
		graph _ self emptyGraph.
		graph delete] raise: Error.
	! !

!TensorFlowCAPITest methodsFor: 'testing buffer' stamp: 'gera 12/22/2016 01:10:48'!
testNewBufferFromFileNamed
	| buffer string data temporaryFile |
	string _ ' hola manola'.
	temporaryFile _ 'temporaryFile.txt'.
	temporaryFile asFileEntry writeStream nextPutAll: string; close.

	buffer _ TF_Buffer fromFileNamed: temporaryFile.
	temporaryFile asFileEntry delete.
	
	self deny: buffer getHandle isNil.
	self deny: buffer getHandle = 0.
	self assert: buffer length == string size.
	
	data _ buffer data fromCString first: string size.
	self assert: string = data.
	
	buffer delete.
	self assert: buffer getHandle isNil.
! !

!TensorFlowCAPITest methodsFor: 'testing buffer' stamp: 'gera 12/22/2016 01:01:59'!
testNewBufferFromString
	| buffer string data |
	string _ ' hola manola'.
	buffer _ TF_Buffer fromString: string.
	self deny: buffer getHandle isNil.
	self deny: buffer getHandle = 0.
	self assert: buffer length == string size.
	
	data _ buffer data fromCString first: string size.
	self assert: string = data.
	
	buffer delete.
	self assert: buffer getHandle isNil.! !

!TensorFlowCAPITest methodsFor: 'testing graph' stamp: 'gera 12/21/2016 22:35:14'!
testNewGraph
	| graph |
	graph _ TF_Graph create.
	self deny: graph getHandle isNil.
	self deny: graph getHandle = 0.
	graph delete.
	self assert: graph getHandle isNil.! !

!TensorFlowCAPITest methodsFor: 'testing options' stamp: 'gera 12/22/2016 01:18:12'!
testNewImportGraphDefOptions
	| options |
	options _ TF_ImportGraphDefOptions create.
	self deny: options getHandle isNil.
	self deny: options getHandle = 0.
	options delete.
	self assert: options getHandle isNil.! !

!TensorFlowCAPITest methodsFor: 'testing session' stamp: 'gera 12/21/2016 23:06:41'!
testNewSession
	| graph session |
	graph _ TF_Graph create.
	session _ TF_Session onGraph: graph.
	self deny: session getHandle isNil.
	self deny: session getHandle = 0.
	session delete.
	self assert: session getHandle isNil.! !

!TensorFlowCAPITest methodsFor: 'testing options' stamp: 'gera 12/21/2016 02:58:13'!
testNewSessionOptions
	| options |
	options _ TF_SessionOptions create.
	self deny: options getHandle isNil.
	self deny: options getHandle = 0.
	options delete.
	self assert: options getHandle isNil.! !

!TensorFlowCAPITest methodsFor: 'testing status' stamp: 'gera 12/21/2016 02:58:04'!
testNewStatus
	| status |
	status _ TF_Status create.
	self deny: status getHandle isNil.
	self deny: status getHandle = 0.
	status delete.
	self assert: status getHandle isNil.! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 12/26/2016 11:10:55'!
testOperationOutputTypeFloat
	^self testGraph: self constantFloatGraph outputType: TF_Tensor typeFloat! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 12/26/2016 11:13:10'!
testOperationOutputTypeInt32
	^self testGraph: self constantInt32Graph outputType: TF_Tensor typeInt32! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 12/26/2016 12:24:05'!
testOperationOutputTypeInt64
	^self testGraph: self constantInt64Graph outputType: TF_Tensor typeInt64! !

!TensorFlowCAPITest methodsFor: 'testing graph' stamp: 'gera 12/26/2016 14:26:05'!
testOutputDims
	| graph operation output |
	graph _ self constantInt64Graph.
	operation _ graph operationNamed: 'a'.
	output _ TF_Output onOperation: operation index: 0.
	self assert: (graph outputDimenssionsCount: output) = 0.
	graph delete.
	output free.! !

!TensorFlowCAPITest methodsFor: 'testing operation' stamp: 'gera 12/26/2016 10:59:53'!
testRunOperationArray
	| graph operation session operations |
	graph _ self constantFloatGraph.
	session _ TF_Session onGraph: graph.
	operation _ graph operationNamed: 'a'.
	operations _ TF_OperationArray externalNew: 1.
	operations first: operation.
	self shouldnt: [session runOperations: operations] raise: Error.
	operations free.
	session close.
	session delete.! !

!TensorFlowCAPITest methodsFor: 'testing session' stamp: 'gera 12/23/2016 11:22:27'!
testSessionOnEmptyGraph
	| session |
	session _ TF_Session onGraph: self emptyGraph.
	self should: [session run] raise: Error.
	session close.
	session delete.! !

!TensorFlowCAPITest methodsFor: 'testing options' stamp: 'gera 12/21/2016 02:49:10'!
testSessionOptionsSetTarget
	| options |
	options _ library newSessionOptions.
	self deny: options getHandle isNil.
	self deny: options getHandle = 0.
	options target: 'local'.
	options delete.
	self assert: options getHandle isNil.! !

!TensorFlowCAPITest methodsFor: 'testing session' stamp: 'gera 12/26/2016 10:59:57'!
testSessionRunTarget
	| graph operation session |
	graph _ self constantFloatGraph.
	session _ TF_Session onGraph: graph.
	operation _ graph operationNamed: 'a'.
	self shouldnt: [session runOperation: operation] raise: Error.
	session close.
	session delete.! !

!TensorFlowCAPITest methodsFor: 'testing session' stamp: 'gera 12/27/2016 12:27:06'!
testSessionRunTargetOutput
	| graph operation session output tensor |
	graph _ self constantInt64Graph.
	session _ TF_Session onGraph: graph.
	operation _ graph operationNamed: 'a'.
	output _ TF_Output onOperation: operation index: 0.
	self shouldnt: [
		tensor _ session runOperation: operation output: output] raise: Error.
	self deny: tensor getHandle isNil.
	self deny: tensor getHandle = 0.
	self deny: tensor data getHandle isNil.
	self deny: tensor data getHandle = 0.
	self assert: (tensor data getHandle signedLongLongAt: 1) = 16r4242424242424242.
	session close.
	session delete.
	tensor delete.
	output free.! !

!TensorFlowCAPITest methodsFor: 'testing status' stamp: 'gera 12/21/2016 11:09:31'!
testStatusCodes
	| status  |
	status _ TF_Status create.
	
	self assert: status isOk.
	self assert: status codeText = 'OK'.
	self shouldnt: [status check] raise: Error.
	
	status code: 1 message: 'You cancelled it!!'.
	self assert: status codeText = 'CANCELLED'.
	self should: [status check] raise: Error.

	status delete.
	self assert: status getHandle isNil.! !

!TensorFlowCAPITest methodsFor: 'testing status' stamp: 'jb 12/26/2016 10:25:46'!
testStatusGetMessage
	| status message |
	status _ TF_Status create.
	
	status code: 1 message: 'All is one'.
	message _ status message.
	self assert: message = 'All is one'.

	status code: 7 message: 'Something is very seven'.
	message _ status message.
	self assert: message = 'Something is very seven'.

	status delete.
	self assert: status getHandle isNil.! !

!TensorFlowCAPITest methodsFor: 'testing status' stamp: 'jb 12/26/2016 10:26:17'!
testStatusSetGetCode
	| status code |
	status _ TF_Status create.
	status code: 1 message: ''.
	code _ status code.
	self assert: code = 1.

	status code: 2 message: ''.
	code _ status code.
	self assert: code = 2.

	status delete.
	self assert: status getHandle isNil.! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 12/27/2016 11:36:05'!
testTensorData
	self fail! !

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

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 12/27/2016 11:55:49'!
testTensorDimensions: anArray
	| tensor |
	tensor _ TF_Tensor
		type: TF_Tensor typeInt64
		dimensions: anArray.
	self assert: tensor numberOfDimensions = anArray size.
	anArray withIndexDo: [:each :index |
		self assert: (tensor sizeOn: index - 1) = (anArray at: index)].
	tensor delete.! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 12/27/2016 11:55:52'!
testTensorNewScalar
	| tensor |
	tensor _ TF_Tensor type: TF_Tensor typeInt64 dimensions: #().
	self deny: tensor getHandle isNil.
	self deny: tensor getHandle = 0.
	tensor delete.
	self assert: tensor getHandle isNil.! !

!TensorFlowCAPITest methodsFor: 'testing tensor' stamp: 'gera 12/27/2016 11:55:58'!
testTensorType
	| tensor |
	tensor _ TF_Tensor type: TF_Tensor typeInt64 dimensions: #().
	self assert: tensor type == tensor class typeInt64.
	tensor delete.

	tensor _ TF_Tensor type: TF_Tensor typeFloat dimensions: #().
	self assert: tensor type == tensor class typeFloat.
	tensor delete.
! !

!TensorFlowCAPITest methodsFor: 'testing library' stamp: 'jb 12/26/2016 11:46:26'!
testVersion
	| version |
	version _ TensorFlowCAPI current version.
	self assert: version =  '0.12.0'! !

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

!TensorFlowCAPI methodsFor: 'utils' stamp: 'gera 12/21/2016 09:56:40'!
externalizeString: aString
	| answer |
	answer := ExternalAddress allocate: aString size + 1.
	answer byteAt: aString size + 1 put: 0.
	aString withIndexDo: [:char :index |
		answer byteAt:  index put: char asciiValue].
	^ answer! !

!TensorFlowCAPI methodsFor: 'operation' stamp: 'gera 12/26/2016 13:13:55'!
forGraph: aTF_Graph outputDims: aTF_Output status: aTF_Status
	<cdecl: ulonglong 'TF_GraphGetTensorNumDims' (TF_Graph* TF_Output TF_Status*)>
	^ self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'status' stamp: 'gera 12/21/2016 09:59:30'!
getCode: aTF_Status
	<cdecl: ulong 'TF_GetCode' (TF_Status*)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'graph' stamp: 'gera 12/22/2016 03:36:16'!
graph: aTF_Graph getOperationNamed: aString
	<cdecl: TF_Operation* 'TF_GraphOperationByName' (TF_Graph* char*)>
	^ self externalCallFailed! !

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

!TensorFlowCAPI methodsFor: 'session' stamp: 'gera 12/27/2016 09:13:16'!
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
		TF_Output*				"inputs"
		TF_TensorArray*			"inputs' values"
		long						"inputs count"
		TF_Output*				"outputs"
		TF_TensorArray*			"outputs values (results)"
		long						"outputs count"
		TF_OperationArray*		"target operations array, may be nil"
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

!TensorFlowCAPI methodsFor: 'tensor' stamp: 'jb 12/26/2016 12:15:05'!
tensor: aTF_Tensor sizeOn: dimension
	<cdecl: longlong 'TF_Dim' (TF_Tensor* ulonglong)>
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

!TF_Structure class methodsFor: 'field definition' stamp: 'gera 12/22/2016 00:24:05'!
fields
	"self initialize"
	^#()! !

!TF_Structure class methodsFor: 'class initialization' stamp: 'gera 12/21/2016 02:54:06'!
initialize
	super initialize.
	self defineFields! !

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

!ExternalWordArray class methodsFor: 'instance creation' stamp: 'gera 12/23/2016 11:15:00'!
externalNew
	^ self shouldNotImplement! !

!ExternalWordArray class methodsFor: 'instance creation' stamp: 'gera 12/23/2016 11:14:10'!
externalNew: count
	^self fromHandle: (ExternalAddress allocate: self byteSize * count)! !

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

!Int64Array class methodsFor: 'instance creation' stamp: 'gera 12/26/2016 09:10:31'!
externalFromArray: anIntegerArray
	| answer |
	answer _ self externalNew: anIntegerArray size.
	anIntegerArray withIndexDo: [:each :index |
		answer at: index put: each].
	^ answer! !

!TF_OperationArray methodsFor: 'as yet unclassified' stamp: 'gera 12/25/2016 03:50:55'!
elementClass
	^TF_Operation! !

!TF_OutputArray methodsFor: 'as yet unclassified' stamp: 'gera 12/25/2016 03:50:48'!
elementClass
	^TF_Output! !

!TF_TensorArray methodsFor: 'as yet unclassified' stamp: 'gera 12/27/2016 09:09:07'!
elementClass
	^TF_Tensor! !

!TF_Buffer methodsFor: 'accessing' stamp: 'gera 12/22/2016 00:49:03'!
data
	"This method was automatically generated. See TF_Buffer class>>fields."
	^ExternalData fromHandle: (handle longPointerAt: 1) type: ExternalType char asPointerType! !

!TF_Buffer methodsFor: 'accessing' stamp: 'gera 12/22/2016 00:33:21'!
data: anObject
	"This method was automatically generated. See TF_Buffer class>>fields."
	handle longPointerAt: 1 put: anObject getHandle.! !

!TF_Buffer methodsFor: 'accessing' stamp: 'gera 12/22/2016 00:34:10'!
deallocator
	"This method was automatically generated. See TF_Buffer class>>fields."
	^ExternalData fromHandle: (handle longPointerAt: 17) type: ExternalType void asPointerType! !

!TF_Buffer methodsFor: 'accessing' stamp: 'gera 12/22/2016 00:34:10'!
deallocator: anObject
	"This method was automatically generated. See TF_Buffer class>>fields."
	handle longPointerAt: 17 put: anObject getHandle.! !

!TF_Buffer methodsFor: 'release' stamp: 'gera 12/22/2016 00:41:49'!
delete
	self library deleteBuffer: self.
	handle _ nil! !

!TF_Buffer methodsFor: 'accessing' stamp: 'gera 12/22/2016 00:34:10'!
length
	"This method was automatically generated. See TF_Buffer class>>fields."
	^handle unsignedLongLongAt: 9! !

!TF_Buffer methodsFor: 'accessing' stamp: 'gera 12/22/2016 00:34:10'!
length: anObject
	"This method was automatically generated. See TF_Buffer class>>fields."
	handle unsignedLongLongAt: 9 put: anObject! !

!TF_Buffer class methodsFor: 'field definition' stamp: 'gera 12/22/2016 00:51:23'!
fields
	"self initialize"
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

!TF_Buffer class methodsFor: 'instance creation' stamp: 'gera 12/22/2016 01:02:39'!
fromString: aString
	| externalized answer |
	externalized _ TensorFlowCAPI current externalizeString: aString.
	answer _ TensorFlowCAPI current newBufferFromString: externalized size: aString size.
	externalized free.
	^ answer! !

!TF_Graph methodsFor: 'initialize-release' stamp: 'gera 12/22/2016 00:41:55'!
delete
	self library deleteGraph: self.
	handle _ nil! !

!TF_Graph methodsFor: 'initialize-release' stamp: 'gera 12/22/2016 01:25:56'!
import: aTF_Buffer
	| options status |
	
	status _ TF_Status create.
	options _ TF_ImportGraphDefOptions create.
	
	self library importGraphDefInto: self from: aTF_Buffer options: options status: status.
	
	options delete.
	[status check] ensure: [status delete].! !

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

!TF_Graph methodsFor: 'accessing' stamp: 'gera 12/22/2016 03:44:06'!
operationNamed: aString
	| externalized answer |
	externalized _ self library externalizeString: aString.
	answer _ self library graph: self getOperationNamed: externalized.
	externalized free.
	answer isNull ifTrue: [self error: 'Operation not found'].
	^ answer
! !

!TF_Graph methodsFor: 'outputs' stamp: 'gera 12/26/2016 12:55:11'!
outputDimenssionsCount: aTF_Output
	| status answer |
	status _ TF_Status create.
	answer _ self library forGraph: self outputDims: aTF_Output status: status.
	[status check] ensure: [status delete].
	^answer! !

!TF_Graph class methodsFor: 'instance creation' stamp: 'gera 12/21/2016 22:31:20'!
create
	^ TensorFlowCAPI current newGraph! !

!TF_Graph class methodsFor: 'instance creation' stamp: 'gera 12/22/2016 02:18:26'!
fromFileNamed: aString
	^ self create importFileNamed: aString! !

!TF_Graph class methodsFor: 'instance creation' stamp: 'gera 12/22/2016 02:18:33'!
fromString: aString
	^ self create importString: aString! !

!TF_ImportGraphDefOptions methodsFor: 'release' stamp: 'gera 12/22/2016 01:14:43'!
delete
	self library deleteImportGraphDefOptions: self.
	handle _ nil! !

!TF_ImportGraphDefOptions class methodsFor: 'instance creation' stamp: 'gera 12/22/2016 01:15:16'!
create
	^ TensorFlowCAPI current newImportGraphDefOptions! !

!TF_Operation methodsFor: 'accessing' stamp: 'gera 12/22/2016 03:57:35'!
InputsCount
	^ self library operationNumInputs: self! !

!TF_Operation methodsFor: 'accessing' stamp: 'gera 12/22/2016 03:57:05'!
device
	^ self library operationDevice: self! !

!TF_Operation methodsFor: 'accessing' stamp: 'gera 12/22/2016 03:56:55'!
name
	^ self library operationName: self! !

!TF_Operation methodsFor: 'accessing' stamp: 'gera 12/22/2016 03:57:48'!
outputsCount
	^ self library operationNumOutputs: self! !

!TF_Operation methodsFor: 'accessing' stamp: 'gera 12/22/2016 03:58:02'!
type
	^ self library operationOpType: self! !

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

!TF_Output class methodsFor: 'field definition' stamp: 'gera 12/23/2016 10:17:51'!
fields
	"self initialize"
	^#(
		(operation 'TF_Operation*')
		(index 'ulonglong'))
	! !

!TF_Output class methodsFor: 'instance creation' stamp: 'gera 12/23/2016 10:40:36'!
onOperation: aTF_Operation index: anInteger
	^ self externalNew
		operation: aTF_Operation;
		index: anInteger! !

!TF_Session methodsFor: 'release' stamp: 'gera 12/21/2016 23:28:28'!
close
	| status |
	status _ TF_Status create.
	self library closeSession: self status: status.
	[status check] ensure: [status delete].! !

!TF_Session methodsFor: 'release' stamp: 'gera 12/22/2016 00:41:59'!
delete
	| status |
	status _ TF_Status create.
	self library deleteSession: self status: status.
	[status check] ensure: [status delete].
	handle _ nil! !

!TF_Session methodsFor: 'running' stamp: 'jb 12/26/2016 11:39:16'!
run
	| status |
	status _ TF_Status create.
	
	[self library
		runSession: self options: nil inputs: nil values: nil count: 0 outputs: nil values: nil count: 0 targets: nil count: 0 metadata: nil status: status.
	
	status check] ensure: [status delete].! !

!TF_Session methodsFor: 'running' stamp: 'jb 12/26/2016 11:38:43'!
runOperation: aTF_Operation
	| operations |
	operations _ TF_OperationArray externalNew: 1.
	operations first: aTF_Operation.	
	[self runOperations: operations] ensure:[operations free]! !

!TF_Session methodsFor: 'running' stamp: 'gera 12/27/2016 18:31:13'!
runOperation: aTF_Operation output: aTF_Output
	| operations  answer |
	operations _ TF_OperationArray externalNew: 1.
	[	operations first: aTF_Operation.	
		answer _ self runOperations: operations outputs: aTF_Output size: 1
	] ensure:[operations free].
	^ answer! !

!TF_Session methodsFor: 'running' stamp: 'jb 12/26/2016 11:39:02'!
runOperations: aTF_OperationArray
	| status |
	status _ TF_Status create.
	
	[self library
		runSession: self options: nil
		inputs: nil values: nil count: 0
		outputs: nil values: nil count: 0
		targets: aTF_OperationArray count: 1
		metadata: nil status: status.
	
	status check] ensure: [status delete].! !

!TF_Session methodsFor: 'running' stamp: 'gera 12/27/2016 18:30:29'!
runOperations: aTF_OperationArray outputs: aTF_Output size: anInteger
	| status outvalues tensor |
	self assert: anInteger = 1.
	status _ TF_Status create.
	outvalues _ TF_TensorArray new: 1.
	[self library
		runSession: self options: nil
		inputs: nil values: nil count: 0
		outputs: aTF_Output values: outvalues count: anInteger
		targets: aTF_OperationArray count: 1
		metadata: nil status: status.
	tensor _ outvalues first.
	status check] ensure: [
		status delete.
		outvalues free].
	^ tensor! !

!TF_Session class methodsFor: 'instance creation' stamp: 'gera 12/21/2016 23:28:37'!
onGraph: aTF_Graph
	| options status answer |
	options _ TF_SessionOptions create.
	status _ TF_Status create.
	answer _ TensorFlowCAPI current newSessionGraph: aTF_Graph options: options status: status.
	options delete.
	[status check] ensure: [status delete].
	^ answer! !

!TF_SessionOptions methodsFor: 'initialize-release' stamp: 'gera 12/22/2016 00:42:04'!
delete
	self library deleteSessionOptions: self.
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

!TF_Status methodsFor: 'initialize-release' stamp: 'gera 12/22/2016 00:42:12'!
delete
	self library deleteStatus: self.
	handle _ nil! !

!TF_Status methodsFor: 'testing' stamp: 'gera 12/21/2016 10:48:36'!
isOk
	^self code == 0! !

!TF_Status methodsFor: 'accessing' stamp: 'gera 12/21/2016 23:50:21'!
message
	^ self library message: self! !

!TF_Status class methodsFor: 'instance creation' stamp: 'gera 12/21/2016 02:55:36'!
create
	^ TensorFlowCAPI current newStatus! !

!TF_Tensor methodsFor: 'release' stamp: 'gera 12/27/2016 09:18:01'!
data
	^self library tensorData: self.! !

!TF_Tensor methodsFor: 'release' stamp: 'gera 12/26/2016 10:44:32'!
delete
	self library deleteTensor: self.
	handle _ nil! !

!TF_Tensor methodsFor: 'release' stamp: 'jb 12/26/2016 12:09:58'!
numberOfDimensions
	^self library tensorNumDimensions: self.! !

!TF_Tensor methodsFor: 'release' stamp: 'jb 12/26/2016 12:10:24'!
sizeOn: dimension
	^self library tensor: self sizeOn: dimension.! !

!TF_Tensor methodsFor: 'release' stamp: 'gera 12/26/2016 10:50:33'!
type
	^self library tensorType: self.! !

!TF_Tensor class methodsFor: 'instance creation' stamp: 'gera 12/26/2016 11:14:21'!
sizeForType: anInteger
	anInteger = self typeFloat ifTrue: [^8].
	anInteger = self typeInt64 ifTrue: [^8].
	anInteger = self typeInt32 ifTrue: [^4].
	^ self shouldBeImplemented.! !

!TF_Tensor class methodsFor: 'instance creation' stamp: 'gera 12/27/2016 18:12:04'!
type: anInteger dimensions: anIntegerArray
	" I understand dimenssions are:
		#()		-> Scalar
		#(7)		-> Unidimensional array of 7 elements
		#(7 4)		-> 7x4 elements matrix
		#(2 5 9)	-> 2x5x9 elements cube
	"
		
	| externalized answer byteSize elementSize |
	externalized _ Int64Array externalFromArray: anIntegerArray.
	elementSize _ self sizeForType: anInteger.
	byteSize _ anIntegerArray inject: elementSize into: [:prev :each | prev * each].
	self assert: byteSize ~= 0.
		
	[answer _ TensorFlowCAPI current
		allocateTensorType: anInteger
		dimensions: externalized
		dimensionsCount: anIntegerArray size
		length: byteSize] ensure: [externalized free].
	^ answer
! !

!TF_Tensor class methodsFor: 'types' stamp: 'gera 12/26/2016 09:41:58'!
typeFloat
	^1! !

!TF_Tensor class methodsFor: 'types' stamp: 'gera 12/26/2016 11:14:05'!
typeInt32
	^3! !

!TF_Tensor class methodsFor: 'types' stamp: 'gera 12/26/2016 09:42:07'!
typeInt64
	^9! !
TF_Structure initialize!
