'From Cuis 5.0 of 7 November 2016 [latest update: #3043] on 21 February 2017 at 7:46:53 pm'!
'Description Please enter a description for this package'!
!provides: 'TensorFlow-Examples' 1 34!
!requires: 'Protocols-HTTP' 1 25 nil!
!requires: 'TensorFlow-Kernel' 1 348 nil!
!classDefinition: #TensorFlowExamplesSlowTest category: #'TensorFlow-Examples'!
TestCase subclass: #TensorFlowExamplesSlowTest
	instanceVariableNames: 'images labels'
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

!classDefinition: #BatchTrainer category: #'TensorFlow-Examples'!
Object subclass: #BatchTrainer
	instanceVariableNames: 'plan batchSize imageSet labelSet imageTesting labelTesting'
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Examples'!
!classDefinition: 'BatchTrainer class' category: #'TensorFlow-Examples'!
BatchTrainer class
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
	instanceVariableNames: 'session graph weights1 biases1 weights2 biases2 weights3 biases3 prediction input expectedLabel loss netInput activation learnWeights3 hidden2 learnBiases3 learnWeights2 hidden1 learnBiases2 learnWeights1 learnBiases1'
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Examples'!
!classDefinition: 'MNIST3LayersNNExamplePlan class' category: #'TensorFlow-Examples'!
MNIST3LayersNNExamplePlan class
	instanceVariableNames: ''!

!classDefinition: #MNIST3LayersNNSigmoid category: #'TensorFlow-Examples'!
MNIST3LayersNNExamplePlan subclass: #MNIST3LayersNNSigmoid
	instanceVariableNames: ''
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Examples'!
!classDefinition: 'MNIST3LayersNNSigmoid class' category: #'TensorFlow-Examples'!
MNIST3LayersNNSigmoid class
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
	instanceVariableNames: 'session graph weights biases prediction netInput input loss expectedLabel learnWeights learnBiases activation'
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


!TensorFlowExamplesSlowTest methodsFor: 'initialization' stamp: 'JB 2/17/2017 21:01:05'!
setUp
	| imageSet labelSet N|
	imageSet _ MNISTImageFile testSet.
	labelSet _ MNISTLabelFile testSet.
	N _ 2.
	images _ (1 to: N) collect:[:i| imageSet bytesAt: i].
	labels _ (1 to: N) collect:[:i| labelSet at: i].! !

!TensorFlowExamplesSlowTest methodsFor: 'testing - MNIST files' stamp: 'gera 2/12/2017 23:23:51'!
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
	self assert: testing asTensor shape equals: {10000. 28*28}.
	self assert: training asTensor shape equals: {60000. 28*28}.! !

!TensorFlowExamplesSlowTest methodsFor: 'testing - MNIST files' stamp: 'gera 2/12/2017 23:24:00'!
testLabels
	| test training |
	test _ MNISTLabelFile testSet.
	training _ MNISTLabelFile trainingSet.
	
	self assert: test count equals: 10000.
	self assert: training count equals: 60000.! !

!TensorFlowExamplesSlowTest methodsFor: 'testing - MNIST files' stamp: 'gera 2/12/2017 23:23:56'!
testMagic
	| images labels |
	images _ MNISTImageFile testSet.
	labels _ MNISTLabelFile testSet.
	
	self assert: images magic = MNISTImageFile magic.
	self assert: labels magic = MNISTLabelFile magic.! !

!TensorFlowExamplesSlowTest methodsFor: 'testing - MNIST - SoftMax' stamp: 'JB 2/17/2017 19:45:57'!
testPrediction
	| loss plan prediction result |
	plan _ MNISTSoftMaxExamplePlan new.
	result _ plan predict: images andCompareTo: labels.
	loss _ (result at:2) asNumbers .
	self deny: (loss closeTo: 0).
	100 timesRepeat: [
		plan predict: images andLearnFrom: labels].
	result _ plan predict: images andCompareTo: labels.
	loss _ (result at:2) asNumbers.
	self assert: 0 closeTo: loss.
	prediction _ plan predict: images.
	self assert: prediction asNumbers first equals: labels first.
	self assert: prediction asNumbers second equals: labels second! !

!TensorFlowExamplesSlowTest methodsFor: 'testing - MNIST - 3Layers' stamp: 'JB 2/17/2017 22:00:17'!
testPrediction3Layers
	|   loss plan result prediction |
	plan _ MNIST3LayersNNExamplePlan new.
	result _ plan predict: images andCompareTo: labels.
	loss _ (result at:2) allFloats.
	self deny: (loss first closeTo: 0).
	100 timesRepeat: [ |interval index im lb|
		interval _ 1 to: images size.
		index _ interval collect:[:i| interval atRandom].
		im _ index collect:[:i| images at: i].
		lb _ index collect:[:i| labels at: i].
		plan predict: im andLearnFrom: lb].
	result _ plan predict: images andCompareTo: labels.
	loss _ (result at:2) allFloats.
	self assert: (loss first closeTo: 0).
	prediction _ plan predict: images.
	self assert: prediction asNumbers first equals: labels first.
	self assert: prediction asNumbers second equals: labels second! !

!TensorFlowExamplesSlowTest methodsFor: 'testing - MNIST - 3Layers' stamp: 'gera 2/21/2017 19:41:11'!
testPrediction3LayersSigmoid
	|   loss plan result prediction |
	plan _ MNIST3LayersNNSigmoid new.
	result _ plan predict: images andCompareTo: labels.
	loss _ (result at:2) allFloats.
	self deny: (loss first closeTo: 0).
	100 timesRepeat: [ |interval index im lb|
		interval _ 1 to: images size.
		index _ interval collect:[:i| interval atRandom].
		im _ index collect:[:i| images at: i].
		lb _ index collect:[:i| labels at: i].
		plan predict: im andLearnFrom: lb].
	result _ plan predict: images andCompareTo: labels.
	loss _ (result at:2) asNumbers.
	self assert: (loss < 1).
	prediction _ plan predict: images.
	self assert: prediction asNumbers equals: labels.
! !

!TensorFlowExamplesSlowTest methodsFor: 'testing - MNIST - SoftMax' stamp: 'gera 2/18/2017 17:52:54'!
testPredictionMany
	| imageSet images labelSet labels loss plan result prediction n |
	n _ 200.
	
	plan _ MNISTSoftMaxExamplePlan new.
	imageSet _ MNISTImageFile testSet.
	labelSet _ MNISTLabelFile testSet.
	
	images _ (1 to: n) collect: [:i | imageSet bytesAt: i].
	labels _ (1 to: n) collect: [:i | labelSet at: i].
	
	result _ plan predict: images andCompareTo: labels.
	loss _ (result at:2) asNumbers.
	self deny: (loss closeTo: 0).
	100 timesRepeat: [
		plan predict: images andLearnFrom: labels].
	
	result _ plan predict: images andCompareTo: labels.
	loss _ (result at:2) asNumbers.
	self assert: (loss closeTo: 0).
	
	prediction _ plan predict: images.
	self assert: prediction asNumbers equals: labels.
! !

!TensorFlowExamplesTest methodsFor: 'testing-NearesNeighbor' stamp: 'JB 1/18/2017 23:03:42'!
tensorFrom: points
	| shape transposed |
	rank _ points first size.
	transposed _ OrderedCollection new.
	(1 to: rank) do: [ :i |
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
	
	self assert: w0 shape equals: #(2 2).
	self assert: weights shape equals: #(2 2).
	self assert: w0 allFloats equals: weights allFloats.

	result _ plan learn: inputs with: target.

	self assert: weights allFloats equals: result allFloats.
	self assert: weights shape equals: result shape.
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
	
	self assert: w0 shape equals: #(2 2).
	self assert: weights shape equals: #(2 2).
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
	
	self assert: weights shape equals: #(2 2).
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

!TensorFlowExamplesTest methodsFor: 'testing MNIST' stamp: 'JB 2/17/2017 19:26:22'!
testMNIST3LayersNNForwardGraph
	" Not sure how to test this better. weights are random, so output is random.
	I believe that choosing the right input value the output should comply with some statistics,
	however there are only 10 output valus, and I believe that's not enough to have statistical meaning"
	
	| mnist result first second |
	mnist _ MNIST3LayersNNExamplePlan new.
	
	self 
		shouldnt: [
			result _ mnist predict: {
				(ByteArray new: mnist inputSize) + 2.
				(ByteArray new: mnist inputSize) + 1.
			}]
		raise: Error.

	self assert: {2} equals: result shape.
	first _ result asNumbers first.
	second _ result asNumbers second.
	self assert: (0 <= first and: [ first < 10 ]).
	self assert: (0 <= second and: [ second < 10 ]).! !

!TensorFlowExamplesTest methodsFor: 'testing MNIST' stamp: 'gera 2/2/2017 19:24:27'!
testMNIST3LayersNNInitialization
	MNIST3LayersNNExamplePlan new.! !

!TensorFlowExamplesTest methodsFor: 'testing MNIST' stamp: 'JB 2/17/2017 19:32:50'!
testMNIST3LayersNNLossGraph
	" Not sure how to test this better. weights are random, so output is random.
	I believe that choosing the right input value the output should comply with some statistics,
	however there are only 10 output valus, and I believe that's not enough to have statistical meaning"
	
	| mnist results first second |
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

	self assert: {2} equals: results first shape.
	first _ results first asNumbers first.
	second _ results first asNumbers second.
	self assert: (0 <= first and: [ first < 10 ]).
	self assert: (0 <= second and: [ second < 10 ]).
	
	self assert: #() equals: results second shape.
	self assert: (results second asNumbers > 0).! !

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

	self assert: {2} equals: result shape.
	self assert: (result allInt32s first closeTo: 0).
	self assert: (result allInt32s second closeTo: 0).
! !

!TensorFlowExamplesTest methodsFor: 'testing MNIST' stamp: 'JB 2/17/2017 19:29:28'!
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

	self assert: {2} equals: results first shape.
	self assert: results first asNumbers first equals: 0.
	self assert: results first asNumbers second equals: 0.

	
	self assert: #() equals: results second shape.
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

!BatchTrainer methodsFor: 'initialization' stamp: 'JB 2/18/2017 20:36:29'!
initialize
	super initialize.
	batchSize _ 100.
	imageSet _ MNISTImageFile trainingSet.
	labelSet _ MNISTLabelFile trainingSet.
	imageTesting _ MNISTImageFile testSet.
	labelTesting _ MNISTLabelFile testSet.
	self use3Layers.
! !

!BatchTrainer methodsFor: 'private' stamp: 'JB 2/18/2017 20:23:39'!
predict: images andCompareTo: labels
	| equals result |
	result _ (plan predict: images) asNumbers.
	equals _ result with: labels collect:[:x :y| (x = y) ifTrue:[1] ifFalse:[0]].
	^ equals mean asFloat.
! !

!BatchTrainer methodsFor: 'running' stamp: 'JB 2/18/2017 20:27:19'!
predictOnTesting
	| images labels |
	images _ imageTesting bytesAt: 1 take: imageTesting count.
	labels _ labelTesting at: 1 take: labelTesting count.
	^ self predict: images andCompareTo: labels! !

!BatchTrainer methodsFor: 'running' stamp: 'JB 2/18/2017 20:26:40'!
predictOnTraining
	| images labels |
	images _ imageSet bytesAt: 1 take: 10 * batchSize.
	labels _ labelSet at: 1 take: 10 * batchSize.
	^ self predict: images andCompareTo: labels! !

!BatchTrainer methodsFor: 'running' stamp: 'JB 2/18/2017 20:22:07'!
train
	(1 to: labelSet count -1 by: batchSize) do: [:index| self trainAt: index].! !

!BatchTrainer methodsFor: 'private' stamp: 'JB 2/18/2017 20:21:23'!
trainAt: index
	| images labels loss report result |
	images _ imageSet bytesAt: index take: batchSize.
	labels _ labelSet at: index take: batchSize.
	result _ plan predict: images andLearnFrom: labels.
	loss _ result at: 2.
	report _ index asString, ': ', loss asNumbers asString.
	Transcript show: report;
					newLine
	! !

!BatchTrainer methodsFor: 'initialization' stamp: 'JB 2/18/2017 20:28:05'!
use3Layers
	plan _ MNIST3LayersNNExamplePlan new.! !

!BatchTrainer methodsFor: 'initialization' stamp: 'JB 2/19/2017 17:09:45'!
use3LayersSigmoid
	plan _ MNIST3LayersNNSigmoid new.! !

!BatchTrainer methodsFor: 'initialization' stamp: 'JB 2/18/2017 20:28:22'!
useSoftmax
	plan _ MNISTSoftMaxExamplePlan new.! !

!ExamplePlan methodsFor: 'initialization' stamp: 'gera 1/29/2017 05:12:34'!
initialize
	self initializeGraph.
	self initializeSession.! !

!ExamplePlan methodsFor: 'initialization' stamp: 'gera 1/21/2017 05:44:31'!
initializeGraph
	graph _ TF_Graph create
	! !

!ExamplePlan methodsFor: 'initialization' stamp: 'gera 2/19/2017 16:22:35'!
initializeSession
	session _ TF_Session on: graph.
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

!BackpropagationPlan methodsFor: 'initialization' stamp: 'gera 2/17/2017 00:27:26'!
initializeBackwardGraph
	| actual learningRate learningRateValue one input |
	actual _ activation.
	one _ graph const: 'one' value: #((1 1) (1 1)) asFloatTensor.
	
	learningRateValue _ 0.9 asTensor.
	learningRate _ graph const: 'learningRate' value: learningRateValue.

	input _ inputs first operationOn: graph.
	
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
	initialWeights _ TF_Tensor fromFloats: initialWeights shape: #(2 2).
	
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

!ExampleNearestNeighborPlan methodsFor: 'initialization' stamp: 'gera 2/17/2017 00:48:53'!
initializeGraphWithOperations
	| axis distance testing training prediction |
	graph _ TF_Graph create.
	
	training _ graph placeholder: 'training' type: TF_Tensor typeFloat.
	testing _ graph placeholder: 'testing' type: TF_Tensor typeFloat.
	axis _ 0 asInt32Tensor.
	
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

!MNIST3LayersNNExamplePlan methodsFor: 'initialization' stamp: 'gera 2/19/2017 20:13:54'!
initializeInferenceGraph
	hidden1 _ graph
		fromBlock: [:image |
			input _ image.
			(image * weights1 + biases1) rectified]
		named: 'layer1'.
	hidden2 _ graph fromBlock: [(hidden1 * weights2 + biases2) rectified] named: 'layer2'.
	prediction _ graph
		fromBlock: [:image |
			netInput _ hidden2 * weights3 + biases3.
			netInput softmax findMaxOn: 1 asInt32Tensor]
		named: 'layer3'.
! !

!MNIST3LayersNNExamplePlan methodsFor: 'initialization' stamp: 'gera 2/20/2017 17:59:03'!
initializeLearningGraph
	|  axis0 activationGradient gradient1 gradient2 learningRate biasGradient learningRateBias |
	
	learningRate _ graph const: 0.001 asTensor.
	learningRateBias _ graph const: 0.1 asTensor.
	axis0 _ graph const: #(0) asInt32Tensor. 
	graph
		fromBlock: [
			activationGradient _ activation useOutput: 1.
			biasGradient _ activationGradient meanOn: axis0.
			"todo: divide by the number of inputs (the batch size)"
			learnWeights3 _ weights3 descent: hidden2 \* activationGradient rate: learningRate.
			learnBiases3 _ biases3 descent: biasGradient rate: learningRateBias]
		named: 'learning3'.
		
	graph fromBlock: [
		gradient2 _ (activationGradient *\ weights3) timesRectifiedGradOf: hidden2.
		learnWeights2 _ weights2 descent: hidden1 \* gradient2 rate: learningRate.
		learnBiases2 _ biases2 descent: (gradient2 meanOn: axis0) rate: learningRate] 
			named: 'learning2'.
			
	graph fromBlock: [
		gradient1 _ (gradient2 *\ weights2) timesRectifiedGradOf: hidden1.
		learnWeights1 _ weights1 descent: input \* gradient1 rate: learningRate.
		learnBiases1 _ biases1 descent: (gradient1 meanOn: axis0) rate: learningRateBias] 
			named: 'learning1'.	! !

!MNIST3LayersNNExamplePlan methodsFor: 'initialization' stamp: 'JB 2/15/2017 21:25:29'!
initializeLossGraph
	loss _ graph
		fromBlock: [:expected |
			expectedLabel _ expected.
			activation _ netInput sparseSoftmaxCrossEntropyWithLogits: expected.
			activation meanOn: #(0) asInt32Tensor]
		inputTypes: {TF_Tensor typeInt32}
		named: 'loss'.! !

!MNIST3LayersNNExamplePlan methodsFor: 'initialization' stamp: 'gera 2/16/2017 09:35:09'!
initializeParameters
	| aux |
		graph
			fromBlock: [
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
				biases3 _ graph variable: 'biases3' initialValueFrom: aux]
			named: 'parameters'
! !

!MNIST3LayersNNExamplePlan methodsFor: 'initialization' stamp: 'gera 2/19/2017 16:22:41'!
initializeSession
	session _ TF_Session on: graph.
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

!MNIST3LayersNNExamplePlan methodsFor: 'running' stamp: 'JB 2/19/2017 15:09:02'!
predict: inputs andLearnFrom: label
	| results |
	results _ session
		runInputs: {input input: 0. expectedLabel input: 0}
		values: {inputs asFloatTensor. label asInt32Tensor}
		outputs: {prediction output. loss output:0. learnWeights1 output. learnBiases1 output.
					learnWeights2 output. learnBiases2 output.
					 learnWeights3 output. learnBiases3 output.}.
	^ results! !

!MNIST3LayersNNSigmoid methodsFor: 'initialization' stamp: 'gera 2/21/2017 19:40:20'!
initializeInferenceGraph
	hidden1 _ graph
		fromBlock: [:image |
			input _ image.
			(image * weights1 + biases1) sigmoid]
		named: 'layer1'.
	hidden2 _ graph fromBlock: [(hidden1 * weights2 + biases2) sigmoid] named: 'layer2'.
	prediction _ graph
		fromBlock: [
			netInput _ hidden2 * weights3 + biases3.
			netInput softmax findMaxOn: 1 asInt32Tensor]
		named: 'layer3'.
! !

!MNIST3LayersNNSigmoid methodsFor: 'initialization' stamp: 'gera 2/21/2017 19:45:40'!
initializeLearningGraph
	|  axis0 gradient gradient1 gradient2 learningRate  meanGradient one batchSize |
	
	learningRate _ 0.1 asTensor.
	batchSize _ graph fromBlock: [(input sizeOn: 0) castTo: TF_Tensor typeFloat] named: 'batchSize'.
	axis0 _ #(0) asInt32Tensor. 
	graph
		fromBlock: [
			gradient _ activation useOutput: 1.
			meanGradient _ gradient meanOn: axis0.
			learnWeights3 _ weights3 descent: hidden2 \* gradient @/ batchSize rate: learningRate.
			learnBiases3 _ biases3 descent: meanGradient rate: learningRate]
		named: 'learning3'.
		
	one _ 1.0 asTensor asOperationOn: graph.
	graph fromBlock: [
		gradient2 _ (gradient *\ weights3) @* hidden2 @* (one - hidden2).
		learnWeights2 _ weights2 descent: hidden1 \* gradient2 @/ batchSize  rate: learningRate.
		learnBiases2 _ biases2 descent: (gradient2 meanOn: axis0) rate: learningRate] 
			named: 'learning2'.
			
	graph fromBlock: [
		gradient1 _ (gradient2 *\ weights2) @* hidden1 @* ( one - hidden1).
		learnWeights1 _ weights1 descent: input \* gradient1 @/ batchSize  rate: learningRate.
		learnBiases1 _ biases1 descent: (gradient1 meanOn: axis0) rate: learningRate] 
			named: 'learning1'.	! !

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
	^  TF_Tensor fromFloats:  items shape: {count. rows * columns}.! !

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

!MNISTImageFile methodsFor: 'accessing' stamp: 'JB 2/18/2017 18:23:25'!
bytesAt: base take: batchSize
	^ (base to: base + batchSize -1) collect:[:index| self bytesAt: index]! !

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

!MNISTLabelFile methodsFor: 'accessing' stamp: 'gera 2/7/2017 09:37:58'!
at: index
	^ items at: index! !

!MNISTLabelFile methodsFor: 'accessing' stamp: 'JB 2/18/2017 18:23:10'!
at: base take: batchSize
	^ (base to: base + batchSize - 1) collect:[:index| self at: index]! !

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

!MNISTSoftMaxExamplePlan methodsFor: 'initialization' stamp: 'gera 2/12/2017 23:10:20'!
initializeInferenceGraph
	prediction _ graph
		fromBlock: [:image |
			input _ image.
			netInput _ image * weights + biases.
			netInput softmax findMaxOn: 1 asInt32Tensor]
		named: 'inference'! !

!MNISTSoftMaxExamplePlan methodsFor: 'initialization' stamp: 'JB 2/19/2017 16:28:46'!
initializeLearningGraph
	| learningRate activationGradient biasGradient |
	graph
		fromBlock: [ | batchSize |
			learningRate _ 0.9 asTensor.
			activationGradient _ activation useOutput: 1.
			biasGradient _ activationGradient meanOn: #(0) asInt32Tensor.
			batchSize _ (input sizeOn: 0) castTo: TF_Tensor typeFloat.
			learnWeights _ weights descent: input \* activationGradient @/ batchSize rate: learningRate.
			learnBiases _ biases descent: biasGradient rate: learningRate]
		named: 'learning'! !

!MNISTSoftMaxExamplePlan methodsFor: 'initialization' stamp: 'gera 2/12/2017 23:09:05'!
initializeLossGraph
	loss _ graph
		fromBlock: [:expected |
			expectedLabel _ expected.
			activation _ netInput sparseSoftmaxCrossEntropyWithLogits: expected.
			activation meanOn: #(0) asInt32Tensor]
		inputTypes: {TF_Tensor typeInt32}
		named: 'loss'
! !

!MNISTSoftMaxExamplePlan methodsFor: 'initialization' stamp: 'gera 2/2/2017 19:29:23'!
initializeParameters
	| aux |
	aux _ graph zerosShaped: {self inputSize. self outputSize}.
	weights _ graph variable: 'weights' initialValueFrom: aux.
	aux _ graph zerosShaped: {self outputSize}.
	biases _ graph variable: 'biases' initialValueFrom: aux.! !

!MNISTSoftMaxExamplePlan methodsFor: 'initialization' stamp: 'gera 2/19/2017 16:22:44'!
initializeSession
	session _ TF_Session on: graph.
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

!MNISTSoftMaxExamplePlan methodsFor: 'running' stamp: 'gera 2/12/2017 23:07:52'!
predict: inputs andCompareTo: label
	| results |
	results _ session
		runInputs: {input input: 0. expectedLabel input: 0}
		values: {inputs asFloatTensor. label asInt32Tensor}
		outputs: {prediction output. loss output}.
	^ results! !

!MNISTSoftMaxExamplePlan methodsFor: 'running' stamp: 'gera 2/19/2017 10:52:34'!
predict: inputs andLearnFrom: label
	| results |
	results _ session
		runInputs: {input input: 0. expectedLabel input: 0}
		values: {inputs asFloatTensor. label asInt32Tensor}
		outputs: {loss output. learnWeights output. learnBiases output}.
	^ results! !

!NearestNeighbor methodsFor: 'private-csv' stamp: 'gera 2/19/2017 16:22:46'!
decodeCSV: csvLines graphRun: description
	|  graph output records session results values answer |
		
	graph _ self decodeCSVGraphDefaults: description.
	records _ (graph operationNamed: 'records') input: 0.
	output _ graph operationNamed: 'output'.
	values _ TF_Tensor fromStringArray: csvLines.
	
	session _ TF_Session on: graph.
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

!NearestNeighbor methodsFor: 'private-csv' stamp: 'gera 2/15/2017 22:40:27'!
irisDescription
	^{TF_Tensor fromFloats: #(-1.0).
		TF_Tensor fromFloats: #(-1.0).
		TF_Tensor fromFloats: #(-1.0).
		TF_Tensor fromFloats: #(-1.0).
		TF_Tensor fromInt64s: #(-1)}! !

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
