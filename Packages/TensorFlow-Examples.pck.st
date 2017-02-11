'From Cuis 5.0 of 7 November 2016 [latest update: #3040] on 11 February 2017 at 2:28:32 pm'!
'Description Please enter a description for this package'!
!provides: 'TensorFlow-Examples' 1 8!
!requires: 'Protocols-HTTP' 1 25 nil!
!classDefinition: #TensorFlowExamplesSlowTest category: #'TensorFlow-Examples'!
TestCase subclass: #TensorFlowExamplesSlowTest
	instanceVariableNames: ''
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Examples'!
!classDefinition: 'TensorFlowExamplesSlowTest class' category: #'TensorFlow-Examples'!
TensorFlowExamplesSlowTest class
	instanceVariableNames: ''!

!classDefinition: #TensorFlowExamplesTest category: #'TensorFlow-Examples'!
TestCase subclass: #TensorFlowExamplesTest
	instanceVariableNames: ''
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Examples'!
!classDefinition: 'TensorFlowExamplesTest class' category: #'TensorFlow-Examples'!
TensorFlowExamplesTest class
	instanceVariableNames: ''!

!classDefinition: #ExamplePlan category: #'TensorFlow-Examples'!
Object subclass: #ExamplePlan
	instanceVariableNames: 'graph inputs outputs session'
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Examples'!
!classDefinition: 'ExamplePlan class' category: #'TensorFlow-Examples'!
ExamplePlan class
	instanceVariableNames: ''!

!classDefinition: #BackpropagationBackwardPlan category: #'TensorFlow-Examples'!
ExamplePlan subclass: #BackpropagationBackwardPlan
	instanceVariableNames: ''
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Examples'!
!classDefinition: 'BackpropagationBackwardPlan class' category: #'TensorFlow-Examples'!
BackpropagationBackwardPlan class
	instanceVariableNames: ''!

!classDefinition: #BackpropagationForwardPlan category: #'TensorFlow-Examples'!
ExamplePlan subclass: #BackpropagationForwardPlan
	instanceVariableNames: ''
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Examples'!
!classDefinition: 'BackpropagationForwardPlan class' category: #'TensorFlow-Examples'!
BackpropagationForwardPlan class
	instanceVariableNames: ''!

!classDefinition: #BackpropagationPlan category: #'TensorFlow-Examples'!
ExamplePlan subclass: #BackpropagationPlan
	instanceVariableNames: 'weights activation target learn delta lastDelta'
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Examples'!
!classDefinition: 'BackpropagationPlan class' category: #'TensorFlow-Examples'!
BackpropagationPlan class
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

!classDefinition: #MNIST3LayersNNExamplePlan category: #'TensorFlow-Examples'!
Object subclass: #MNIST3LayersNNExamplePlan
	instanceVariableNames: 'session graph weights1 biases1 weights2 biases2 weights3 biases3 prediction input expectedLabel loss netInput'
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Examples'!
!classDefinition: 'MNIST3LayersNNExamplePlan class' category: #'TensorFlow-Examples'!
MNIST3LayersNNExamplePlan class
	instanceVariableNames: ''!

!classDefinition: #MNISTFile category: #'TensorFlow-Examples'!
Object subclass: #MNISTFile
	instanceVariableNames: 'magic count items'
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Examples'!
!classDefinition: 'MNISTFile class' category: #'TensorFlow-Examples'!
MNISTFile class
	instanceVariableNames: ''!

!classDefinition: #MNISTImageFile category: #'TensorFlow-Examples'!
MNISTFile subclass: #MNISTImageFile
	instanceVariableNames: 'rows columns'
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Examples'!
!classDefinition: 'MNISTImageFile class' category: #'TensorFlow-Examples'!
MNISTImageFile class
	instanceVariableNames: ''!

!classDefinition: #MNISTLabelFile category: #'TensorFlow-Examples'!
MNISTFile subclass: #MNISTLabelFile
	instanceVariableNames: ''
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Examples'!
!classDefinition: 'MNISTLabelFile class' category: #'TensorFlow-Examples'!
MNISTLabelFile class
	instanceVariableNames: ''!

!classDefinition: #MNISTSoftMaxExamplePlan category: #'TensorFlow-Examples'!
Object subclass: #MNISTSoftMaxExamplePlan
	instanceVariableNames: 'session graph weights biases prediction netInput input loss expectedLabel learnWeights learnBiases expectedForComparison'
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Examples'!
!classDefinition: 'MNISTSoftMaxExamplePlan class' category: #'TensorFlow-Examples'!
MNISTSoftMaxExamplePlan class
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

!classDefinition: #OLSExample category: #'TensorFlow-Examples'!
Object subclass: #OLSExample
	instanceVariableNames: 'plan'
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Examples'!
!classDefinition: 'OLSExample class' category: #'TensorFlow-Examples'!
OLSExample class
	instanceVariableNames: ''!

!classDefinition: #SimpleNeuralNetworkExample category: #'TensorFlow-Examples'!
Object subclass: #SimpleNeuralNetworkExample
	instanceVariableNames: 'forward backward weights'
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Examples'!
!classDefinition: 'SimpleNeuralNetworkExample class' category: #'TensorFlow-Examples'!
SimpleNeuralNetworkExample class
	instanceVariableNames: ''!


!TensorFlowCAPITestSlowTests methodsFor: '*TensorFlow-Examples' stamp: 'gera 2/7/2017 09:12:18'!
testImages
	| testing training |
	testing _ MNISTImageFile testSet.
	training _ MNISTImageFile trainingSet .
	
	self assert: testing count equals: 10000.
	self assert: training count equals: 60000.
	self assert: testing rows equals: 28.
	self assert: testing columns equals: 28.
	self assert: training rows equals: 28.
	self assert: training columns equals: 28.
	self assert: testing asTensor dimensions equals: {10000. 28*28}.
	self assert: training asTensor dimensions equals: {60000. 28*28}.! !

!TensorFlowCAPITestSlowTests methodsFor: '*TensorFlow-Examples' stamp: 'gera 2/7/2017 09:39:11'!
testLabels
	| test training |
	test _ MNISTLabelFile testSet.
	training _ MNISTLabelFile trainingSet.
	
	self assert: test count equals: 10000.
	self assert: training count equals: 60000.! !

!TensorFlowCAPITestSlowTests methodsFor: '*TensorFlow-Examples' stamp: 'JB 2/5/2017 16:12:18'!
testMagic
	| images labels |
	images _ MNISTImageFile testSet.
	labels _ MNISTLabelFile testSet.
	
	self assert: images magic = MNISTImageFile magic.
	self assert: labels magic = MNISTLabelFile magic.! !

!TensorFlowExamplesSlowTest methodsFor: 'testing - MNIST - SoftMax' stamp: 'JB 2/11/2017 13:18:23'!
testPrediction
	| imageSet images labelSet labels loss plan result |
	plan _ MNISTSoftMaxExamplePlan new.
	imageSet _ MNISTImageFile testSet.
	labelSet _ MNISTLabelFile testSet.
	images _ {(imageSet bytesAt: 1). (imageSet bytesAt: 2)}.
	labels _ {(labelSet at: 1). (labelSet at: 2)}.
	result _ plan predict: images andCompareTo: labels.
	loss _ (result at:2) allFloats.
	self deny: (loss first closeTo: 0).
	100 timesRepeat: [
		plan predict: images andLearnFrom: labels].
	result _ plan predict: images andCompareTo: labels.
	loss _ (result at:2) allFloats.
	self assert: (loss first closeTo: 0) ! !

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

!TensorFlowExamplesTest methodsFor: 'testing-Backpropagation' stamp: 'gera 1/29/2017 08:32:30'!
testBackpropagationPlanBackguard
	| plan inputs rawResult result const graph weights first second sigmoid |
	plan _ BackpropagationPlan new.
	
	inputs _ TF_Tensor fromFloats: {
		{0}.
		{2 ln}
	}.
	
	graph _ plan graph.
	const _ graph operationNamed: 'weights_initialValue'.
	weights _ (const tensorAt: 'value') allFloats.
	
	sigmoid _ [:x | (x negated exp + 1) reciprocal].
	first _ sigmoid value: 2 ln * weights second.
	second _ sigmoid value: 2 ln * weights fourth.

	rawResult _ plan runOn: {inputs}.
	result _ rawResult allFloats.
	
	
	self
		assert: (result first closeTo: first);
		assert: (result second closeTo: second)! !

!TensorFlowExamplesTest methodsFor: 'testing-Backpropagation' stamp: 'gera 1/29/2017 11:21:39'!
testBackpropagationPlanBackguardChangesWeights
	| plan const graph w0 weights inputs result target |
	plan _ BackpropagationPlan new.
	graph _ plan graph.
	
	inputs _ TF_Tensor fromFloats: {{1}. {2 ln}}.
	target _ TF_Tensor fromFloats: #((0.5) (0.5)).
	
	const _ graph operationNamed: 'weights_initialValue'.
	w0 _ const tensorAt: 'value'.
	weights _ plan weights.
	
	self assert: w0 dimensions equals: #(2 2).
	self assert: weights dimensions equals: #(2 2).
	self assert: w0 allFloats equals: weights allFloats.

	result _ plan learn: inputs with: target.

	self assert: weights allFloats equals: result allFloats.
	self assert: weights dimensions equals: result dimensions.
	self deny: weights allFloats = w0 allFloats
! !

!TensorFlowExamplesTest methodsFor: 'testing-Backpropagation' stamp: 'gera 1/29/2017 14:53:31'!
testBackpropagationPlanBackguardConverges
	| plan const graph w0 weights inputs target result |
	plan _ BackpropagationPlan new.
	graph _ plan graph.
	
	inputs _ TF_Tensor fromFloats: {{1}. {2 ln}}.
	target _ TF_Tensor fromFloats: #((0.5) (0.5)).
	
	const _ graph operationNamed: 'weights_initialValue'.
	w0 _ const tensorAt: 'value'.
	weights _ plan weights.
	
	self assert: w0 dimensions equals: #(2 2).
	self assert: weights dimensions equals: #(2 2).
	self assert: w0 allFloats equals: weights allFloats.

	200 timesRepeat: [plan learn: inputs with: target].

	self assert: (plan delta allFloats squared sum closeTo: 0).

	result _ (plan runOn: {inputs}) allFloats.
	
	self assert: (result first closeTo: 0.5).
	self assert: (result second closeTo: 0.5).
! !

!TensorFlowExamplesTest methodsFor: 'testing-Backpropagation' stamp: 'gera 1/29/2017 08:32:46'!
testBackpropagationPlanForward
	| plan inputs rawResult result const graph weights first second sigmoid |
	plan _ BackpropagationPlan basicNew.
	plan initializeGraph.
	plan initializeSession.
	
	inputs _ TF_Tensor fromFloats: {
		{0}.
		{2 ln}
	}.
	
	graph _ plan graph.
	const _ graph operationNamed: 'weights_initialValue'.
	weights _ (const tensorAt: 'value') allFloats.
	
	sigmoid _ [:x | (x negated exp + 1) reciprocal].
	first _ sigmoid value: 2 ln * weights second.
	second _ sigmoid value: 2 ln * weights fourth.

	rawResult _ plan runOn: {inputs}.
	result _ rawResult allFloats.
	
	
	self
		assert: (result first closeTo: first);
		assert: (result second closeTo: second)! !

!TensorFlowExamplesTest methodsFor: 'testing-Backpropagation' stamp: 'gera 1/29/2017 08:20:58'!
testBackpropagationPlantInitializeBackguard
	| bpp |
	bpp _ BackpropagationPlan basicNew.
	bpp instVarNamed: 'graph' put: TF_Graph create.
	bpp initializeVariables.
	bpp initializeForwardGraph.
	bpp initializeBackwardGraph.
	
	self assert: true description: 'No exception should be raised by this test'! !

!TensorFlowExamplesTest methodsFor: 'testing-Backpropagation' stamp: 'gera 1/29/2017 08:20:33'!
testBackpropagationPlantInitializeForward
	| bpp |
	bpp _ BackpropagationPlan new.
	bpp instVarNamed: 'graph' put: TF_Graph create.
	bpp initializeVariables.
	bpp initializeForwardGraph.
	
	self assert: true description: 'No exception should be raised by this test'! !

!TensorFlowExamplesTest methodsFor: 'testing-Backpropagation' stamp: 'gera 1/29/2017 08:33:02'!
testBackpropagationPlantInitializeVariables
	| bpp graph const weights |
	bpp _ BackpropagationPlan basicNew.
	bpp instVarNamed: 'graph' put: TF_Graph create.
	bpp initializeVariables.
	
	graph _ bpp graph.
	
	self assert: graph allInitializers size = 1.
	self assert: graph allVariables size = 1.
	self assert: graph allVariables first name = 'weights'.

	const _ graph operationNamed: 'weights_initialValue'.
	weights _ const tensorAt: 'value'.
	
	self assert: weights dimensions equals: #(2 2).
	weights allFloats do: [:each |
		self assert: (each between: 0 and: 1)].! !

!TensorFlowExamplesTest methodsFor: 'testing-Backpropagation' stamp: 'gera 1/29/2017 02:01:59'!
testForward
	| inputs plan rawResult result weights |
	plan _ BackpropagationForwardPlan new.
	inputs _ TF_Tensor fromFloats: {
		{0}.
		{2 ln}
	}.
	weights _ TF_Tensor fromFloats: #(
		(1 1)
		(1 0)
	).
	rawResult _ plan runOn: {inputs. weights}.
	result _ rawResult allFloats.
	
	self assert: (result first closeTo: (1 + 0.5) reciprocal);
		 assert: (result second closeTo: 0.5)! !

!TensorFlowExamplesTest methodsFor: 'testing MNIST' stamp: 'JB 2/11/2017 14:23:03'!
testMNIST3LayersNNForwardGraph
	" Not sure how to test this better. weights are random, so output is random.
	I believe that choosing the right input value the output should comply with some statistics,
	however there are only 10 output valus, and I believe that's not enough to have statistical meaning"
	
	| mnist result |
	mnist _ MNIST3LayersNNExamplePlan new.
	
	self 
		shouldnt: [
			result _ mnist predict: {
				ByteArray new: mnist inputSize.
				(ByteArray new: mnist inputSize) + 1.
			}]
		raise: Error.

	self assert: {2} equals: result dimensions.
	self assert: 0 equals: result allInt32s first.
	self assert: 0 equals: result allInt32s second.! !

!TensorFlowExamplesTest methodsFor: 'testing MNIST' stamp: 'gera 2/2/2017 19:24:27'!
testMNIST3LayersNNInitialization
	MNIST3LayersNNExamplePlan new.! !

!TensorFlowExamplesTest methodsFor: 'testing MNIST' stamp: 'JB 2/11/2017 14:24:51'!
testMNIST3LayersNNLossGraph
	" Not sure how to test this better. weights are random, so output is random.
	I believe that choosing the right input value the output should comply with some statistics,
	however there are only 10 output valus, and I believe that's not enough to have statistical meaning"
	
	| mnist results |
	mnist _ MNIST3LayersNNExamplePlan new.
	
	self 
		shouldnt: [
			results _ mnist
				predict: {
					ByteArray new: mnist inputSize.
					(ByteArray new: mnist inputSize) + 1}
				andCompareTo: #(1 2).
				]
		raise: Error.

	self assert: {2} equals: results first dimensions.
	self assert: 0 equals: results first allInt32s first.
	self assert: 0 equals: results first allInt32s second.
	
	self assert: #() equals: results second dimensions.
	self assert: (results second allFloats sum abs > 0).! !

!TensorFlowExamplesTest methodsFor: 'testing MNIST' stamp: 'JB 2/11/2017 14:08:37'!
testMNISTSoftMaxForwardGraph
	" Not sure how to test this better. weights are random, so output is random.
	I believe that choosing the right input value the output should comply with some statistics,
	however there are only 10 output valus, and I believe that's not enough to have statistical meaning"
	
	| mnist result |
	mnist _ MNISTSoftMaxExamplePlan new.
	
	self 
		shouldnt: [
			result _ mnist predict: {
				ByteArray new: mnist inputSize.
				(ByteArray new: mnist inputSize) + 1.
			}]
		raise: Error.

	self assert: {2} equals: result dimensions.
	self assert: (result allInt32s first closeTo: 0).
	self assert: (result allInt32s second closeTo: 0).
! !

!TensorFlowExamplesTest methodsFor: 'testing MNIST' stamp: 'JB 2/11/2017 14:28:10'!
testMNISTSoftMaxLossGraph
	" Not sure how to test this better. weights are random, so output is random.
	I believe that choosing the right input value the output should comply with some statistics,
	however there are only 10 output valus, and I believe that's not enough to have statistical meaning"
	
	| mnist results |
	mnist _ MNISTSoftMaxExamplePlan new.
	
	self 
		shouldnt: [
			results _ mnist
				predict: {
					ByteArray new: mnist inputSize.
					(ByteArray new: mnist inputSize) + 1}
				andCompareTo: #(1 2).
				]
		raise: Error.

	self assert: {2} equals: results first dimensions.
	self assert: (results first allInt32s first closeTo: 0).
	self assert: (results first allInt32s second closeTo: 0).

	
	self assert: #() equals: results second dimensions.
	self assert: (results second allFloats sum abs > 0).! !

!TensorFlowExamplesTest methodsFor: 'testing MNIST' stamp: 'gera 2/2/2017 19:33:39'!
testMNISTSoftMaxnitialization
	MNISTSoftMaxExamplePlan new.! !

!TensorFlowExamplesTest methodsFor: 'testing-NearesNeighbor' stamp: 'gera 1/21/2017 05:10:31'!
testNearesNeighbor
	| plan |
	plan _ ExampleNearestNeighborPlan new.
	self testNearesNeighbor: plan.! !

!TensorFlowExamplesTest methodsFor: 'testing-NearesNeighbor' stamp: 'gera 1/21/2017 05:09:49'!
testNearesNeighbor: plan
	| first second third closest references tensorReference tensors predictor |
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

!TensorFlowExamplesTest methodsFor: 'testing-NearesNeighbor' stamp: 'gera 1/29/2017 05:15:13'!
testNearesNeighborWithOperations
	| plan |
	plan _ ExampleNearestNeighborPlan new initializeGraphWithOperations initializeSession.
	self testNearesNeighbor: plan.! !

!TensorFlowExamplesTest methodsFor: 'testing-ols' stamp: 'gera 1/21/2017 05:18:41'!
testOLS
	self testOLS: OLSExample new.! !

!TensorFlowExamplesTest methodsFor: 'testing-ols' stamp: 'gera 1/21/2017 05:18:17'!
testOLS: ols
	| x y betas |
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

!TensorFlowExamplesTest methodsFor: 'testing-ols' stamp: 'gera 1/29/2017 05:16:11'!
testOLSWithOperations
	self testOLS: OLSExample new initializeWithOperations! !

!TensorFlowExamplesTest methodsFor: 'testing-NeuralNetwork' stamp: 'gera 1/29/2017 02:04:33'!
testSimpleNeuralNetwork
	| inputs weights nn prediction target targetValues |
	inputs _ TF_Tensor fromFloats: {
		{1}.
		{2 ln}
		}.
	targetValues _ #(0.5 0.5).
	target _ TF_Tensor fromFloats: targetValues.
	weights _ #(
		(1 1)
		(1 0)
	).
	nn _ SimpleNeuralNetworkExample new.
	nn initialWeights: weights;
	 	learnAll: {inputs} to: {target} steps: 100.
	prediction _ nn predict: inputs.
	targetValues with: prediction do:[:real :predicted| 
		self assert: (real - predicted) abs < 0.1]! !

!ExamplePlan methodsFor: 'initialization' stamp: 'gera 1/29/2017 05:12:34'!
initialize
	self initializeGraph.
	self initializeSession.! !

!ExamplePlan methodsFor: 'initialization' stamp: 'gera 1/21/2017 05:44:31'!
initializeGraph
	graph _ TF_Graph create
	! !

!ExamplePlan methodsFor: 'initialization' stamp: 'gera 1/29/2017 05:20:08'!
initializeSession
	session _ TF_Session onGraph: graph.
	graph initializeOn: session.
! !

!ExamplePlan methodsFor: 'running' stamp: 'gera 1/29/2017 05:19:43'!
runOn: inputValues
	| results |	
		
	results _ session
		runInputs: inputs
		values: inputValues
		outputs: outputs.
		
	^ results first
	! !

!BackpropagationBackwardPlan methodsFor: 'initialization' stamp: 'gera 1/30/2017 17:55:11'!
initializeGraph
	| inputVariable weights actual delta learningRate newWeights target one |
	super initializeGraph.
	inputVariable _ graph placeholder: 'Input' type: TF_Tensor typeFloat.
	target _ graph placeholder: 'target' type: TF_Tensor typeFloat.
	actual _ graph placeholder: 'actual' type: TF_Tensor typeFloat.
	weights _ graph placeholder: 'weights' type: TF_Tensor typeFloat.
	learningRate _ graph const: 'learningRate' value: 0.9 asTensor.
	one _ graph const: 'one' value: 1.0 asTensor.
	delta _ (target - actual) negated @* actual @* (one - actual) @* inputVariable.
	newWeights _ weights - (learningRate @* delta).
	outputs _ {newWeights output: 0}.
	inputs _ {inputVariable input: 0. weights input: 0. target input:0. actual input: 0}! !

!BackpropagationForwardPlan methodsFor: 'initialization' stamp: 'JB 1/25/2017 11:40:47'!
initializeGraph
	| activation inputVariable netInput weights |
	super initializeGraph.
	inputVariable _ graph placeholder: 'Input' type: TF_Tensor typeFloat.
	weights _ graph placeholder: 'weights' type: TF_Tensor typeFloat.
	netInput _ weights * inputVariable.
	activation _ netInput sigmoid.
	outputs _ {activation output: 0}.
	inputs _ {inputVariable input: 0. weights input: 0}! !

!BackpropagationPlan methodsFor: 'accessing' stamp: 'gera 1/29/2017 11:17:26'!
delta
	^ lastDelta! !

!BackpropagationPlan methodsFor: 'accessing' stamp: 'gera 1/29/2017 08:32:19'!
graph
	^ graph! !

!BackpropagationPlan methodsFor: 'initialization' stamp: 'gera 1/30/2017 17:38:24'!
initializeBackwardGraph
	| actual learningRate learningRateValue one input |
	actual _ activation.
	one _ graph const: 'one' value: #((1 1) (1 1)) asFloatTensor.
	
	learningRateValue _ 0.9 asTensor.
	learningRate _ graph const: 'learningRate' value: learningRateValue.

	input _ inputs first operation.
	
	target _ graph placeholder: 'target' type: TF_Tensor typeFloat.
	
	delta _ (target - actual) negated @* actual @* (one - actual) @* input.
	
	"learn _ weights assign: weights - learningRate @* delta."
	"learn _ weights -= learningRate @* delta."
	learn _ weights descent: delta rate: learningRate.! !

!BackpropagationPlan methodsFor: 'initialization' stamp: 'gera 1/30/2017 00:37:21'!
initializeForwardGraph
	| input |
	input _ graph placeholder: 'Input' type: TF_Tensor typeFloat.
	
	activation _ (weights * input) sigmoid.
	
	outputs _ {activation output: 0}.
	inputs _ {input input: 0}! !

!BackpropagationPlan methodsFor: 'initialization' stamp: 'gera 1/29/2017 05:28:36'!
initializeGraph
	super initializeGraph.
	self
		initializeVariables;
		initializeForwardGraph;
		initializeBackwardGraph.! !

!BackpropagationPlan methodsFor: 'initialization' stamp: 'gera 1/29/2017 04:05:17'!
initializeVariables
	| initialWeights |
	Random withDefaultDo: [:random |
		initialWeights _ (1 to: 4) collect: [:unused | random next]].
	initialWeights _ TF_Tensor fromFloats: initialWeights dimensions: #(2 2).
	
	weights _ graph variable: 'weights' initialValue: initialWeights.! !

!BackpropagationPlan methodsFor: 'training' stamp: 'gera 1/29/2017 15:09:30'!
learn: sample with: expected
	| results |	
	results _ session
		runInputs: {inputs first. target input: 0}
		values: {sample. expected}
		outputs: {learn output: 0. delta output: 0}.
	
	lastDelta _ results at: 2.
	^ results first
	! !

!BackpropagationPlan methodsFor: 'accessing' stamp: 'gera 1/29/2017 08:40:28'!
weights
	^ session runOutput: (weights output: 0)! !

!ExampleNearestNeighborPlan methodsFor: 'initialization' stamp: 'gera 1/22/2017 17:47:49'!
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
	
	outputs _ {prediction output: 0}.
	inputs _ {training input: 0. testing input: 0}.! !

!ExampleNearestNeighborPlan methodsFor: 'initialization' stamp: 'JB 1/23/2017 10:49:44'!
initializeGraphWithOperations
	| axis axisValue distance testing training prediction |
	graph _ TF_Graph create.
	
	training _ graph placeholder: 'training' type: TF_Tensor typeFloat.
	testing _ graph placeholder: 'testing' type: TF_Tensor typeFloat.
	axisValue _ TF_Tensor fromInt32: 0.
	axis _ graph const: 'axis' value: axisValue.
	
	distance _ (testing - training) abs sumOn: axis.
	prediction _ distance findMinOn: axis.
	
	outputs _ {prediction output: 0}.
	inputs _ {training input: 0. testing input: 0}.
! !

!ExampleNearestNeighborPlan methodsFor: 'evaluating' stamp: 'gera 1/21/2017 04:19:59'!
predict: covariatesTesting from: covariatesTraining
	| result |

 	result _ self runOn: {covariatesTraining. covariatesTesting}.
	^ result allInt64s + 1! !

!ExampleOLSPlan methodsFor: 'initialization' stamp: 'gera 1/29/2017 05:16:45'!
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
	
	outputs _ {result output: 0}.
	inputs _ {x input: 0. y input: 0}.! !

!ExampleOLSPlan methodsFor: 'initialization' stamp: 'gera 1/22/2017 17:48:34'!
initializeGraphWithOperations
	| x y prediction |
	super initializeGraph.
	
	x _ graph placeholder: 'x' type: TF_Tensor typeFloat.
	y _ graph placeholder: 'y' type: TF_Tensor typeFloat.
	
	prediction _ (x \* x) inverse * (x \* y).
	
	outputs _ {prediction output: 0}.
	inputs _ {x input: 0. y input: 0}.! !

!MNIST3LayersNNExamplePlan methodsFor: 'accessing' stamp: 'gera 1/31/2017 01:44:41'!
graph
	^ graph! !

!MNIST3LayersNNExamplePlan methodsFor: 'accessing' stamp: 'gera 1/31/2017 14:41:15'!
hidden1Size
	^ 128! !

!MNIST3LayersNNExamplePlan methodsFor: 'accessing' stamp: 'gera 1/31/2017 14:41:18'!
hidden2Size
	^ 32! !

!MNIST3LayersNNExamplePlan methodsFor: 'initialization' stamp: 'gera 1/31/2017 09:56:01'!
initialize
	self
		initializeGraph;
		initializeParameters;
		initializeInferenceGraph;
		initializeLossGraph;
		initializeLearningGraph;
		initializeSession.! !

!MNIST3LayersNNExamplePlan methodsFor: 'initialization' stamp: 'gera 1/31/2017 01:08:54'!
initializeGraph
	graph _ TF_Graph create! !

!MNIST3LayersNNExamplePlan methodsFor: 'initialization' stamp: 'JB 2/11/2017 14:21:08'!
initializeInferenceGraph
	| hidden1 hidden2 |
	prediction _ graph fromBlock: [:image | | axis |
		input _ image.
		hidden1 _ (image * weights1 + biases1) rectified.
		hidden2 _ (hidden1 * weights2 + biases2) rectified.
		netInput _ hidden2 * weights3 + biases3.
		axis _ graph const: 1 asInt32Tensor.
		netInput softmax findMaxOn: axis].
! !

!MNIST3LayersNNExamplePlan methodsFor: 'initialization' stamp: 'gera 1/31/2017 01:07:57'!
initializeLearningGraph! !

!MNIST3LayersNNExamplePlan methodsFor: 'initialization' stamp: 'JB 2/11/2017 14:21:18'!
initializeLossGraph
	loss _ graph
		fromBlock: [:expected |
			expectedLabel _ expected.
			(netInput sparseSoftmaxCrossEntropyWithLogits: expected) meanOn: #(0) asInt32Tensor]
		inputTypes: {TF_Tensor typeInt32}! !

!MNIST3LayersNNExamplePlan methodsFor: 'initialization' stamp: 'gera 2/1/2017 02:19:34'!
initializeParameters
	| aux |
	aux _ graph truncatedNormalRandomShaped: {self inputSize. self hidden1Size} stddev: 1.0 / self inputSize sqrt.
	weights1 _ graph variable: 'weights1' initialValueFrom: aux.
	aux _ graph zerosShaped: {self hidden1Size}.
	biases1 _ graph variable: 'biases1' initialValueFrom: aux.

	aux _ graph truncatedNormalRandomShaped: {self hidden1Size. self hidden2Size} stddev: 1.0 / self hidden1Size sqrt.
	weights2 _ graph variable: 'weights2' initialValueFrom: aux.
	aux _ graph zerosShaped: {self hidden2Size}.
	biases2 _ graph variable: 'biases2' initialValueFrom: aux.

	aux _ graph truncatedNormalRandomShaped: {self hidden2Size. self outputSize} stddev: 1.0 / self hidden2Size sqrt.
	weights3 _ graph variable: 'weights3' initialValueFrom: aux.
	aux _ graph zerosShaped: {self outputSize}.
	biases3 _ graph variable: 'biases3' initialValueFrom: aux.
! !

!MNIST3LayersNNExamplePlan methodsFor: 'initialization' stamp: 'gera 1/31/2017 01:42:14'!
initializeSession
	session _ TF_Session onGraph: graph.
	graph initializeOn: session.! !

!MNIST3LayersNNExamplePlan methodsFor: 'accessing' stamp: 'gera 1/31/2017 14:34:12'!
inputSize
	^ 28*28! !

!MNIST3LayersNNExamplePlan methodsFor: 'accessing' stamp: 'gera 1/31/2017 01:47:08'!
intput
	^ input! !

!MNIST3LayersNNExamplePlan methodsFor: 'accessing' stamp: 'gera 2/2/2017 02:27:05'!
lossGradient
	^ (loss output: 1)! !

!MNIST3LayersNNExamplePlan methodsFor: 'accessing' stamp: 'gera 1/31/2017 14:41:30'!
outputSize
	^ 10! !

!MNIST3LayersNNExamplePlan methodsFor: 'running' stamp: 'gera 1/31/2017 02:06:26'!
predict: inputs
	| results |
	results _ session
		runInputs: {input input: 0}
		values: {inputs asFloatTensor}
		outputs: {prediction output: 0}.
	^ results first! !

!MNIST3LayersNNExamplePlan methodsFor: 'running' stamp: 'gera 2/2/2017 02:18:26'!
predict: inputs andCompareTo: label
	| results |
	results _ session
		runInputs: {input input: 0. expectedLabel input: 0}
		values: {inputs asFloatTensor. label asInt32Tensor}
		outputs: {prediction output: 0. loss output: 0}.
	^ results! !

!MNISTFile methodsFor: 'converting' stamp: 'JB 2/5/2017 15:04:26'!
asTensor
	^ self subclassResponsibility ! !

!MNISTFile methodsFor: 'accessing' stamp: 'JB 2/5/2017 15:58:05'!
count
	^ count! !

!MNISTFile methodsFor: 'accessing' stamp: 'JB 2/5/2017 15:58:10'!
items
	^ items! !

!MNISTFile methodsFor: 'accessing' stamp: 'JB 2/5/2017 15:58:01'!
magic
	^ magic! !

!MNISTFile methodsFor: 'initialization' stamp: 'JB 2/5/2017 15:37:24'!
parse: aStream
	magic _ aStream nextUnsignedInt32BigEndian: true.
	count _  aStream nextUnsignedInt32BigEndian: true.

	! !

!MNISTFile class methodsFor: 'private' stamp: 'JB 2/5/2017 15:56:42'!
download: name
		| url response content file |
		url _ 'http://yann.lecun.com/exdb/mnist/', name.
		response _ HTTPClient httpGet: url.
		content _ response content asByteArray.
		file _ ('dataset/', name) asFileEntry.
		file < content.
		^ file! !

!MNISTFile class methodsFor: 'instance creation' stamp: 'JB 2/5/2017 15:36:27'!
fromFile: aString
	| file filename stream compressed|
	filename _ aString, '.gz'.
	file _ ('dataset/', filename) asFileEntry.
	file exists ifFalse:[ file _ self download: filename].
	compressed _ file readStream contentsOfEntireFile.
	stream _ (GZipReadStream on: compressed) upToEnd asByteArray readStream.
	^ self fromStream: stream
	! !

!MNISTFile class methodsFor: 'instance creation' stamp: 'JB 2/5/2017 15:26:40'!
fromStream: aStream
	| answer |
	answer _ self new.
	^ answer parse: aStream! !

!MNISTFile class methodsFor: 'accessing' stamp: 'JB 2/5/2017 15:07:25'!
magic
	^ self subclassResponsibility ! !

!MNISTFile class methodsFor: 'accessing' stamp: 'JB 2/5/2017 15:16:07'!
testName
	^ self subclassResponsibility ! !

!MNISTFile class methodsFor: 'instance creation' stamp: 'JB 2/5/2017 15:28:39'!
testSet
	^self fromFile: self testName
	! !

!MNISTFile class methodsFor: 'accessing' stamp: 'JB 2/5/2017 15:16:03'!
trainName
	^ self subclassResponsibility ! !

!MNISTFile class methodsFor: 'instance creation' stamp: 'JB 2/5/2017 15:28:58'!
trainingSet
	^self fromFile: self trainName
	! !

!MNISTImageFile methodsFor: 'converting' stamp: 'gera 2/6/2017 20:17:27'!
asTensor
	^  TF_Tensor fromFloats:  items dimensions: {count. rows * columns}.! !

!MNISTImageFile methodsFor: 'accessing' stamp: 'JB 2/10/2017 00:50:56'!
at: imageIndex
	| bytes |
	bytes _ self bytesAt: imageIndex.
	^ GrayForm extent: rows @ columns bits: bytes! !

!MNISTImageFile methodsFor: 'accessing' stamp: 'JB 2/10/2017 00:50:21'!
bytesAt: imageIndex
	| size base |
	size _ columns * rows.
	base _ imageIndex - 1 * size + 1.
	^ items copyFrom: base to: base + size - 1.! !

!MNISTImageFile methodsFor: 'accessing' stamp: 'JB 2/5/2017 15:58:35'!
columns
	^ columns! !

!MNISTImageFile methodsFor: 'initialization' stamp: 'JB 2/5/2017 15:10:42'!
parse: aStream
	super parse: aStream.
	rows _ aStream nextUnsignedInt32BigEndian: true.
	columns _ aStream nextUnsignedInt32BigEndian: true.
	items _ aStream upToEnd.! !

!MNISTImageFile methodsFor: 'accessing' stamp: 'JB 2/5/2017 15:58:26'!
rows
	^ rows! !

!MNISTImageFile class methodsFor: 'accessing' stamp: 'JB 2/5/2017 15:09:23'!
magic
	^ 2051! !

!MNISTImageFile class methodsFor: 'accessing' stamp: 'JB 2/5/2017 15:16:30'!
testName
	^ 't10k-images-idx3-ubyte'! !

!MNISTImageFile class methodsFor: 'accessing' stamp: 'JB 2/5/2017 15:16:46'!
trainName
	^ 'train-images-idx3-ubyte'! !

!MNISTLabelFile methodsFor: 'converting' stamp: 'JB 2/10/2017 00:44:04'!
asTensor
	^ items asInt32Tensor ! !

!MNISTLabelFile methodsFor: 'initialization' stamp: 'gera 2/7/2017 09:37:58'!
at: index
	^ items at: index! !

!MNISTLabelFile methodsFor: 'initialization' stamp: 'JB 2/5/2017 15:01:28'!
parse: aStream
	super parse: aStream.
	items _ aStream upToEnd.
	
	! !

!MNISTLabelFile class methodsFor: 'accessing' stamp: 'JB 2/5/2017 13:22:30'!
magic
	^ 2049! !

!MNISTLabelFile class methodsFor: 'accessing' stamp: 'JB 2/5/2017 15:17:11'!
testName
	^ 't10k-labels-idx1-ubyte'! !

!MNISTLabelFile class methodsFor: 'accessing' stamp: 'JB 2/5/2017 15:17:29'!
trainName
	^ 'train-labels-idx1-ubyte'! !

!MNISTSoftMaxExamplePlan methodsFor: 'initialization' stamp: 'gera 2/2/2017 19:26:53'!
initialize
	self
		initializeGraph;
		initializeParameters;
		initializeInferenceGraph;
		initializeLossGraph;
		initializeLearningGraph;
		initializeSession.! !

!MNISTSoftMaxExamplePlan methodsFor: 'initialization' stamp: 'gera 2/2/2017 19:27:02'!
initializeGraph
	graph _ TF_Graph create! !

!MNISTSoftMaxExamplePlan methodsFor: 'initialization' stamp: 'JB 2/11/2017 13:29:41'!
initializeInferenceGraph
	prediction _ graph fromBlock: [:image | | axis |
		input _ image.
		netInput _ image * weights + biases.
		axis _ graph const: 1 asInt32Tensor.
		netInput softmax findMaxOn: axis]
! !

!MNISTSoftMaxExamplePlan methodsFor: 'initialization' stamp: 'JB 2/11/2017 12:20:32'!
initializeLearningGraph
	| learningRate learningRateValue activation gradient |
	learningRateValue _ 0.9 asTensor.
	learningRate _ graph const: 'learningRate' value: learningRateValue.
	activation _ graph
		fromBlock: [:expected |
			expectedLabel _ expected.
			netInput sparseSoftmaxCrossEntropyWithLogits: expected]
		inputTypes: {TF_Tensor typeInt32}.	
	activation useOutput: 1.
	gradient _ activation meanOn: #(0) asInt32Tensor.
	"todo: divide by the number of inputs (the batch size)"
	learnWeights _ weights descent: input \* activation rate: learningRate.
	learnBiases _ biases descent: gradient rate: learningRate
	! !

!MNISTSoftMaxExamplePlan methodsFor: 'initialization' stamp: 'JB 2/11/2017 13:03:35'!
initializeLossGraph
	loss _ graph
		fromBlock: [:expected |
			expectedForComparison _ expected.
			(netInput sparseSoftmaxCrossEntropyWithLogits: expected) meanOn: #(0) asInt32Tensor]
		inputTypes: {TF_Tensor typeInt32}! !

!MNISTSoftMaxExamplePlan methodsFor: 'initialization' stamp: 'gera 2/2/2017 19:29:23'!
initializeParameters
	| aux |
	aux _ graph zerosShaped: {self inputSize. self outputSize}.
	weights _ graph variable: 'weights' initialValueFrom: aux.
	aux _ graph zerosShaped: {self outputSize}.
	biases _ graph variable: 'biases' initialValueFrom: aux.! !

!MNISTSoftMaxExamplePlan methodsFor: 'initialization' stamp: 'gera 2/2/2017 19:27:12'!
initializeSession
	session _ TF_Session onGraph: graph.
	graph initializeOn: session.! !

!MNISTSoftMaxExamplePlan methodsFor: 'accessing' stamp: 'gera 2/2/2017 19:30:14'!
inputSize
	^ 28*28! !

!MNISTSoftMaxExamplePlan methodsFor: 'accessing' stamp: 'gera 2/2/2017 19:29:56'!
outputSize
	^ 10! !

!MNISTSoftMaxExamplePlan methodsFor: 'running' stamp: 'gera 2/2/2017 19:37:26'!
predict: inputs
	| results |
	results _ session
		runInputs: {input input: 0}
		values: {inputs asFloatTensor}
		outputs: {prediction output: 0}.
	^ results first! !

!MNISTSoftMaxExamplePlan methodsFor: 'running' stamp: 'JB 2/11/2017 13:00:39'!
predict: inputs andCompareTo: label
	| results |
	results _ session
		runInputs: {input input: 0. expectedForComparison input: 0}
		values: {inputs asFloatTensor. label asInt32Tensor}
		outputs: {prediction output: 0. loss output: 0}.
	^ results! !

!MNISTSoftMaxExamplePlan methodsFor: 'running' stamp: 'JB 2/11/2017 12:57:01'!
predict: inputs andLearnFrom: label
	| results |
	results _ session
		runInputs: {input input: 0. expectedLabel input: 0}
		values: {inputs asFloatTensor. label asInt32Tensor}
		outputs: {prediction output. learnWeights output. learnBiases output}.
	^ results! !

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

!NearestNeighbor methodsFor: 'running' stamp: 'gera 1/30/2017 17:56:34'!
run
	| covariatesTr covariatesTe predictions size dataset testing training |
	dataset _ self loadDataSet.
	training _ dataset first.
	testing _ dataset second.
	covariatesTr _ training allButLast asFloatTensor.
	size _ testing first size.
	predictions _ (1 to: size) collect: [ :i | | covariates index predicted results trueValue |
		covariates _ testing  collect:[:col| {col at: i}].
		covariatesTe _ covariates allButLast asFloatTensor.
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

!OLSExample methodsFor: 'initialization' stamp: 'JB 1/19/2017 10:29:16'!
initialize
	plan _ ExampleOLSPlan new.! !

!OLSExample methodsFor: 'initialization' stamp: 'gera 1/29/2017 05:15:58'!
initializeWithOperations
	plan _ ExampleOLSPlan new initializeGraphWithOperations initializeSession! !

!OLSExample methodsFor: 'running' stamp: 'gera 1/21/2017 04:20:26'!
regress: x on: y
	| result |
	result _ plan runOn: {x. y}.
	^ result allFloats! !

!SimpleNeuralNetworkExample methodsFor: 'accessing' stamp: 'gera 1/30/2017 17:56:57'!
initialWeights: aCollection
	weights _ aCollection asFloatTensor! !

!SimpleNeuralNetworkExample methodsFor: 'initialization' stamp: 'JB 1/25/2017 15:22:27'!
initialize
	forward _ BackpropagationForwardPlan new.
	backward _ BackpropagationBackwardPlan new.! !

!SimpleNeuralNetworkExample methodsFor: 'learning' stamp: 'JB 1/25/2017 15:25:47'!
learn: sample with: expected
		| result |
	result _ forward runOn: {sample. weights}.
	weights _ backward runOn: {sample. weights. expected. result}.! !

!SimpleNeuralNetworkExample methodsFor: 'learning' stamp: 'JB 1/25/2017 15:29:01'!
learnAll: samples to: targets steps: n
	n timesRepeat: [
		samples with: targets do:[:sample :target| self learn: sample with: target]
		]! !

!SimpleNeuralNetworkExample methodsFor: 'service' stamp: 'JB 1/25/2017 15:26:43'!
predict: sample
		| result |
	result _ forward runOn: {sample. weights}.
	^ result allFloats ! !
