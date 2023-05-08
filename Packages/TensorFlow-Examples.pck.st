'From Cuis 6.0 [latest update: #5795] on 7 May 2023 at 7:08:46 pm'!
'Description Please enter a description for this package'!
!provides: 'TensorFlow-Examples' 1 53!
!requires: 'WebClient' 1 0 nil!
!requires: 'TensorFlow-Kernel' 1 411 nil!
SystemOrganization addCategory: 'TensorFlow-Examples'!


!classDefinition: #TensorFlowExamplesSlowTest category: 'TensorFlow-Examples'!
TensorFlowTestCase subclass: #TensorFlowExamplesSlowTest
	instanceVariableNames: 'images labels'
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Examples'!
!classDefinition: 'TensorFlowExamplesSlowTest class' category: 'TensorFlow-Examples'!
TensorFlowExamplesSlowTest class
	instanceVariableNames: ''!

!classDefinition: #TensorFlowExamplesTest category: 'TensorFlow-Examples'!
TensorFlowTestCase subclass: #TensorFlowExamplesTest
	instanceVariableNames: ''
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Examples'!
!classDefinition: 'TensorFlowExamplesTest class' category: 'TensorFlow-Examples'!
TensorFlowExamplesTest class
	instanceVariableNames: ''!

!classDefinition: #BatchTrainer category: 'TensorFlow-Examples'!
Object subclass: #BatchTrainer
	instanceVariableNames: 'plan batchSize imageSet labelSet imageTesting labelTesting'
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Examples'!
!classDefinition: 'BatchTrainer class' category: 'TensorFlow-Examples'!
BatchTrainer class
	instanceVariableNames: ''!

!classDefinition: #ExamplePlan category: 'TensorFlow-Examples'!
Object subclass: #ExamplePlan
	instanceVariableNames: 'graph inputs outputs session'
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Examples'!
!classDefinition: 'ExamplePlan class' category: 'TensorFlow-Examples'!
ExamplePlan class
	instanceVariableNames: ''!

!classDefinition: #BackpropagationBackwardPlan category: 'TensorFlow-Examples'!
ExamplePlan subclass: #BackpropagationBackwardPlan
	instanceVariableNames: ''
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Examples'!
!classDefinition: 'BackpropagationBackwardPlan class' category: 'TensorFlow-Examples'!
BackpropagationBackwardPlan class
	instanceVariableNames: ''!

!classDefinition: #BackpropagationForwardPlan category: 'TensorFlow-Examples'!
ExamplePlan subclass: #BackpropagationForwardPlan
	instanceVariableNames: ''
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Examples'!
!classDefinition: 'BackpropagationForwardPlan class' category: 'TensorFlow-Examples'!
BackpropagationForwardPlan class
	instanceVariableNames: ''!

!classDefinition: #BackpropagationPlan category: 'TensorFlow-Examples'!
ExamplePlan subclass: #BackpropagationPlan
	instanceVariableNames: 'weights activation target learn delta lastDelta'
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Examples'!
!classDefinition: 'BackpropagationPlan class' category: 'TensorFlow-Examples'!
BackpropagationPlan class
	instanceVariableNames: ''!

!classDefinition: #ExampleNearestNeighborPlan category: 'TensorFlow-Examples'!
ExamplePlan subclass: #ExampleNearestNeighborPlan
	instanceVariableNames: ''
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Examples'!
!classDefinition: 'ExampleNearestNeighborPlan class' category: 'TensorFlow-Examples'!
ExampleNearestNeighborPlan class
	instanceVariableNames: ''!

!classDefinition: #ExampleOLSPlan category: 'TensorFlow-Examples'!
ExamplePlan subclass: #ExampleOLSPlan
	instanceVariableNames: ''
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Examples'!
!classDefinition: 'ExampleOLSPlan class' category: 'TensorFlow-Examples'!
ExampleOLSPlan class
	instanceVariableNames: ''!

!classDefinition: #MNIST3LayersNNExamplePlan category: 'TensorFlow-Examples'!
Object subclass: #MNIST3LayersNNExamplePlan
	instanceVariableNames: 'session graph weights1 biases1 weights2 biases2 weights3 biases3 prediction input expectedLabel loss netInput activation hidden2 hidden1 learn'
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Examples'!
!classDefinition: 'MNIST3LayersNNExamplePlan class' category: 'TensorFlow-Examples'!
MNIST3LayersNNExamplePlan class
	instanceVariableNames: ''!

!classDefinition: #MNIST3LayersNNSigmoid category: 'TensorFlow-Examples'!
MNIST3LayersNNExamplePlan subclass: #MNIST3LayersNNSigmoid
	instanceVariableNames: ''
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Examples'!
!classDefinition: 'MNIST3LayersNNSigmoid class' category: 'TensorFlow-Examples'!
MNIST3LayersNNSigmoid class
	instanceVariableNames: ''!

!classDefinition: #MNISTFile category: 'TensorFlow-Examples'!
Object subclass: #MNISTFile
	instanceVariableNames: 'magic count items'
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Examples'!
!classDefinition: 'MNISTFile class' category: 'TensorFlow-Examples'!
MNISTFile class
	instanceVariableNames: ''!

!classDefinition: #MNISTImageFile category: 'TensorFlow-Examples'!
MNISTFile subclass: #MNISTImageFile
	instanceVariableNames: 'rows columns'
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Examples'!
!classDefinition: 'MNISTImageFile class' category: 'TensorFlow-Examples'!
MNISTImageFile class
	instanceVariableNames: ''!

!classDefinition: #MNISTLabelFile category: 'TensorFlow-Examples'!
MNISTFile subclass: #MNISTLabelFile
	instanceVariableNames: ''
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Examples'!
!classDefinition: 'MNISTLabelFile class' category: 'TensorFlow-Examples'!
MNISTLabelFile class
	instanceVariableNames: ''!

!classDefinition: #MNISTSoftMaxExamplePlan category: 'TensorFlow-Examples'!
Object subclass: #MNISTSoftMaxExamplePlan
	instanceVariableNames: 'session graph weights biases prediction netInput input loss expectedLabel learnWeights learnBiases activation'
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Examples'!
!classDefinition: 'MNISTSoftMaxExamplePlan class' category: 'TensorFlow-Examples'!
MNISTSoftMaxExamplePlan class
	instanceVariableNames: ''!

!classDefinition: #NearestNeighbor category: 'TensorFlow-Examples'!
Object subclass: #NearestNeighbor
	instanceVariableNames: 'plan'
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Examples'!
!classDefinition: 'NearestNeighbor class' category: 'TensorFlow-Examples'!
NearestNeighbor class
	instanceVariableNames: ''!

!classDefinition: #OLSExample category: 'TensorFlow-Examples'!
Object subclass: #OLSExample
	instanceVariableNames: 'plan'
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Examples'!
!classDefinition: 'OLSExample class' category: 'TensorFlow-Examples'!
OLSExample class
	instanceVariableNames: ''!

!classDefinition: #RegressionNNExample category: 'TensorFlow-Examples'!
Object subclass: #RegressionNNExample
	instanceVariableNames: 'graph input weights1 biases1 weights2 biases2 weights3 biases3 prediction netInput weights4 biases4 hidden1 hidden2 expectedLabel loss session hidden3 learn'
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Examples'!
!classDefinition: 'RegressionNNExample class' category: 'TensorFlow-Examples'!
RegressionNNExample class
	instanceVariableNames: ''!

!classDefinition: #SimpleNeuralNetworkExample category: 'TensorFlow-Examples'!
Object subclass: #SimpleNeuralNetworkExample
	instanceVariableNames: 'forward backward weights'
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Examples'!
!classDefinition: 'SimpleNeuralNetworkExample class' category: 'TensorFlow-Examples'!
SimpleNeuralNetworkExample class
	instanceVariableNames: ''!


!MNIST3LayersNNSigmoid commentStamp: '<historical>' prior: 0!
self new graph writeDefToFileNamed: 'graph.pb'!

!RegressionNNExample commentStamp: '<historical>' prior: 0!
I compute a regression on arbitrary functions.
Implementation of http://cs.stanford.edu/people/karpathy/convnetjs/demo/regression.html

RegressionNNExample exampleTrainedAndPlot!

!TensorFlowExamplesSlowTest methodsFor: 'initialization' stamp: 'JB 2/17/2017 21:01:05'!
setUp
	| imageSet labelSet N|
	imageSet := MNISTImageFile testSet.
	labelSet := MNISTLabelFile testSet.
	N := 2.
	images := (1 to: N) collect:[:i| imageSet bytesAt: i].
	labels := (1 to: N) collect:[:i| labelSet at: i].! !

!TensorFlowExamplesSlowTest methodsFor: 'testing - MNIST files' stamp: 'gera 2/12/2017 23:23:51'!
testImages
	| testing training |
	testing := MNISTImageFile testSet.
	training := MNISTImageFile trainingSet .
	
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
	test := MNISTLabelFile testSet.
	training := MNISTLabelFile trainingSet.
	
	self assert: test count equals: 10000.
	self assert: training count equals: 60000.! !

!TensorFlowExamplesSlowTest methodsFor: 'testing - MNIST files' stamp: 'gera 6/27/2019 15:44:37'!
testMagic
	| imageSet labelSet |
	imageSet := MNISTImageFile testSet.
	labelSet := MNISTLabelFile testSet.
	
 	self assert: imageSet magic = MNISTImageFile magic.
	self assert: labelSet magic = MNISTLabelFile magic.! !

!TensorFlowExamplesSlowTest methodsFor: 'testing - MNIST - SoftMax' stamp: 'JB 2/17/2017 19:45:57'!
testPrediction
	| loss plan prediction result |
	plan := MNISTSoftMaxExamplePlan new.
	result := plan predict: images andCompareTo: labels.
	loss := (result at:2) asNumbers .
	self deny: loss closeTo: 0.
	100 timesRepeat: [
		plan predict: images andLearnFrom: labels].
	result := plan predict: images andCompareTo: labels.
	loss := (result at:2) asNumbers.
	self assert: 0 closeTo: loss.
	prediction := plan predict: images.
	self assert: prediction asNumbers first equals: labels first.
	self assert: prediction asNumbers second equals: labels second! !

!TensorFlowExamplesSlowTest methodsFor: 'testing - MNIST - SoftMax' stamp: 'gera 6/27/2019 15:46:21'!
testPredictionMany
	| loss plan result prediction |
	
	plan := MNISTSoftMaxExamplePlan new.

	result := plan predict: images andCompareTo: labels.
	loss := (result at:2) asNumbers.
	self deny: loss closeTo: 0.
	100 timesRepeat: [
		plan predict: images andLearnFrom: labels].
	
	result := plan predict: images andCompareTo: labels.
	loss := (result at:2) asNumbers.
	self assert: loss closeTo: 0.
	
	prediction := plan predict: images.
	self assert: prediction asNumbers equals: labels.
! !

!TensorFlowExamplesSlowTest methodsFor: 'testing - MNIST - 3Layers' stamp: 'JB 2/17/2017 22:00:17'!
testPrediction3Layers
	|   loss plan result prediction |
	plan := MNIST3LayersNNExamplePlan new.
	result := plan predict: images andCompareTo: labels.
	loss := (result at:2) allFloats.
	self deny: loss first closeTo: 0.
	100 timesRepeat: [ |interval index im lb|
		interval := 1 to: images size.
		index := interval collect:[:i| interval atRandom].
		im := index collect:[:i| images at: i].
		lb := index collect:[:i| labels at: i].
		plan predict: im andLearnFrom: lb].
	result := plan predict: images andCompareTo: labels.
	loss := (result at:2) allFloats.
	self assert: loss first closeTo: 0.
	prediction := plan predict: images.
	self assert: prediction asNumbers first equals: labels first.
	self assert: prediction asNumbers second equals: labels second! !

!TensorFlowExamplesSlowTest methodsFor: 'testing - MNIST - 3Layers' stamp: 'gera 2/21/2017 19:41:11'!
testPrediction3LayersSigmoid
	|   loss plan result prediction |
	plan := MNIST3LayersNNSigmoid new.
	result := plan predict: images andCompareTo: labels.
	loss := (result at:2) allFloats.
	self deny: loss first closeTo: 0.
	100 timesRepeat: [ |interval index im lb|
		interval := 1 to: images size.
		index := interval collect:[:i| interval atRandom].
		im := index collect:[:i| images at: i].
		lb := index collect:[:i| labels at: i].
		plan predict: im andLearnFrom: lb].
	result := plan predict: images andCompareTo: labels.
	loss := (result at:2) asNumbers.
	self assert: (loss < 1).
	prediction := plan predict: images.
	self assert: prediction asNumbers equals: labels.
! !

!TensorFlowExamplesSlowTest methodsFor: 'testing - Regression' stamp: 'JB 2/24/2017 23:17:14'!
testSinglePointRegression
	| function loss net results rnd x y |
	net := RegressionNNExample new.
	function := RegressionNNExample exampleFunction.
	rnd := Random new.
	x := rnd next.
	y := function value: x.
	100 timesRepeat:[net predict: {{x}} andLearnFrom:{{y}}].
	results := net predict: {{x}} andCompareTo: {{y}}.
	loss := results numbersAt: 2.
	self assert: loss first closeTo: 0.
	! !

!TensorFlowExamplesSlowTest methodsFor: 'testing - Regression' stamp: 'JB 2/24/2017 23:17:04'!
testTwoPointsRegression
	| function loss net results  interval xs ys |
	net := RegressionNNExample new.
	function := RegressionNNExample exampleFunction.
	xs := {{0.1}. {0.6}}.
	ys := {{function value: xs first first}. 
			{function value: xs second first}}.
	interval := 1 to: xs size.
	1000 timesRepeat: [ |index |
		index := interval atRandom.
		net predict: {xs at: index} andLearnFrom:{ys at: index}].
	results := net predict: xs andCompareTo: ys.
	loss := results numbersAt: 2.
	self assert: loss first closeTo: 0.
	! !

!TensorFlowExamplesTest methodsFor: 'testing-NearesNeighbor' stamp: 'gera 6/30/2019 00:45:27'!
tensorFrom: points
	| rank transposed |
	rank := points first size.
	transposed := OrderedCollection new.
	(1 to: rank) do: [ :i |
		transposed add: OrderedCollection new ].
	points do: [ :point |
		point withIndexDo: [ :value :coordinate |
			(transposed at: coordinate) add: value ]].
	^ TFTensor fromFloats: transposed.! !

!TensorFlowExamplesTest methodsFor: 'testing-NearesNeighbor' stamp: 'gera 1/21/2017 05:10:31'!
testNearesNeighbor
	| plan |
	plan := ExampleNearestNeighborPlan new.
	self testNearesNeighbor: plan.! !

!TensorFlowExamplesTest methodsFor: 'testing-NearesNeighbor' stamp: 'gera 1/21/2017 05:09:49'!
testNearesNeighbor: plan
	| first second third closest references tensorReference tensors predictor |
	first := #(0 0 0 0 0).
	second := #(1 1 3 4 2).
	third := #(8 1 3 4 2).
	references := {
		first.
		second.
		third}.
	tensorReference := self tensorFrom: references.
	tensors := references collect: [ :point |
		self tensorFrom: {point} ].
	
	predictor := [:unknown | | result | 
		result := plan predict: unknown from: tensorReference.
		result first].
	
	closest := predictor value: tensors first.
	self assert: closest equals: 1.
	
	closest := predictor value:  tensors second.
	self assert: closest equals: 2.
	
	closest := predictor value:  tensors third.
	self assert: closest equals: 3.! !

!TensorFlowExamplesTest methodsFor: 'testing-NearesNeighbor' stamp: 'gera 1/29/2017 05:15:13'!
testNearesNeighborWithOperations
	| plan |
	plan := ExampleNearestNeighborPlan new initializeGraphWithOperations initializeSession.
	self testNearesNeighbor: plan.! !

!TensorFlowExamplesTest methodsFor: 'testing-Backpropagation' stamp: 'gera 6/30/2019 00:45:28'!
testBackpropagationPlanBackguard
	| plan inputs rawResult result const graph weights first second sigmoid |
	plan := BackpropagationPlan new.
	
	inputs := TFTensor fromFloats: {
		{0}.
		{2 ln}
	}.
	
	graph := plan graph.
	const := graph operationNamed: 'weights_initialValue'.
	weights := (const tensorAt: 'value') allFloats.
	
	sigmoid := [:x | (x negated exp + 1) reciprocal].
	first := sigmoid value: 2 ln * weights second.
	second := sigmoid value: 2 ln * weights fourth.

	rawResult := plan runOn: {inputs}.
	result := rawResult allFloats.
	
	
	self
		assert: result first closeTo: first;
		assert: result second closeTo: second! !

!TensorFlowExamplesTest methodsFor: 'testing-Backpropagation' stamp: 'gera 6/30/2019 00:45:27'!
testBackpropagationPlanBackguardChangesWeights
	| plan const graph w0 weights inputs result target |
	plan := BackpropagationPlan new.
	graph := plan graph.
	
	inputs := TFTensor fromFloats: {{1}. {2 ln}}.
	target := TFTensor fromFloats: #((0.5) (0.5)).
	
	const := graph operationNamed: 'weights_initialValue'.
	w0 := const tensorAt: 'value'.
	weights := plan weights.
	
	self assert: w0 shape equals: #(2 2).
	self assert: weights shape equals: #(2 2).
	self assert: w0 allFloats equals: weights allFloats.

	result := plan learn: inputs with: target.

	self assert: weights allFloats equals: result allFloats.
	self assert: weights shape equals: result shape.
	self deny: weights allFloats = w0 allFloats
! !

!TensorFlowExamplesTest methodsFor: 'testing-Backpropagation' stamp: 'gera 6/30/2019 00:45:27'!
testBackpropagationPlanBackguardConverges
	| plan const graph w0 weights inputs target result |
	plan := BackpropagationPlan new.
	graph := plan graph.
	
	inputs := TFTensor fromFloats: {{1}. {2 ln}}.
	target := TFTensor fromFloats: #((0.5) (0.5)).
	
	const := graph operationNamed: 'weights_initialValue'.
	w0 := const tensorAt: 'value'.
	weights := plan weights.
	
	self assert: w0 shape equals: #(2 2).
	self assert: weights shape equals: #(2 2).
	self assert: w0 allFloats equals: weights allFloats.

	200 timesRepeat: [plan learn: inputs with: target].

	self assert: plan delta allFloats squared sum closeTo: 0.

	result := (plan runOn: {inputs}) allFloats.
	
	self assert: result first closeTo: 0.5.
	self assert: result second closeTo: 0.5.
! !

!TensorFlowExamplesTest methodsFor: 'testing-Backpropagation' stamp: 'gera 6/30/2019 00:45:28'!
testBackpropagationPlanForward
	| plan inputs rawResult result const graph weights first second sigmoid |
	plan := BackpropagationPlan basicNew.
	plan initializeGraph.
	plan initializeSession.
	
	inputs := TFTensor fromFloats: {
		{0}.
		{2 ln}
	}.
	
	graph := plan graph.
	const := graph operationNamed: 'weights_initialValue'.
	weights := (const tensorAt: 'value') allFloats.
	
	sigmoid := [:x | (x negated exp + 1) reciprocal].
	first := sigmoid value: 2 ln * weights second.
	second := sigmoid value: 2 ln * weights fourth.

	rawResult := plan runOn: {inputs}.
	result := rawResult allFloats.
	
	
	self
		assert: result first closeTo: first;
		assert: result second closeTo: second! !

!TensorFlowExamplesTest methodsFor: 'testing-Backpropagation' stamp: 'gera 7/3/2019 00:34:29'!
testBackpropagationPlantInitializeBackguard
	| bpp |
	bpp := BackpropagationPlan basicNew.
	bpp instVarNamed: 'graph' put: TFGraph create.
	bpp initializeVariables.
	bpp initializeForwardGraph.
	bpp initializeBackwardGraph.
	
	self assert: true description: 'No exception should be raised by this test'! !

!TensorFlowExamplesTest methodsFor: 'testing-Backpropagation' stamp: 'gera 7/3/2019 00:34:29'!
testBackpropagationPlantInitializeForward
	| bpp |
	bpp := BackpropagationPlan new.
	bpp instVarNamed: 'graph' put: TFGraph create.
	bpp initializeVariables.
	bpp initializeForwardGraph.
	
	self assert: true description: 'No exception should be raised by this test'! !

!TensorFlowExamplesTest methodsFor: 'testing-Backpropagation' stamp: 'gera 7/3/2019 00:34:29'!
testBackpropagationPlantInitializeVariables
	| bpp graph const weights |
	bpp := BackpropagationPlan basicNew.
	bpp instVarNamed: 'graph' put: TFGraph create.
	bpp initializeVariables.
	
	graph := bpp graph.
	
	self assert: graph allInitializers size = 1.
	self assert: graph allVariables size = 1.
	self assert: graph allVariables first name = 'weights'.

	const := graph operationNamed: 'weights_initialValue'.
	weights := const tensorAt: 'value'.
	
	self assert: weights shape equals: #(2 2).
	weights allFloats do: [:each |
		self assert: (each between: 0 and: 1)].! !

!TensorFlowExamplesTest methodsFor: 'testing-Backpropagation' stamp: 'gera 6/30/2019 00:45:27'!
testForward
	| inputs plan rawResult result weights |
	plan := BackpropagationForwardPlan new.
	inputs := TFTensor fromFloats: {
		{0}.
		{2 ln}
	}.
	weights := TFTensor fromFloats: #(
		(1 1)
		(1 0)
	).
	rawResult := plan runOn: {inputs. weights}.
	result := rawResult allFloats.
	
	self assert: result first closeTo: (1 + 0.5) reciprocal;
		 assert: result second closeTo: 0.5! !

!TensorFlowExamplesTest methodsFor: 'testing MNIST' stamp: 'JB 2/17/2017 19:26:22'!
testMNIST3LayersNNForwardGraph
	" Not sure how to test this better. weights are random, so output is random.
	I believe that choosing the right input value the output should comply with some statistics,
	however there are only 10 output valus, and I believe that's not enough to have statistical meaning"
	
	| mnist result first second |
	mnist := MNIST3LayersNNExamplePlan new.
	
	self 
		shouldnt: [
			result := mnist predict: {
				(ByteArray new: mnist inputSize) + 2.
				(ByteArray new: mnist inputSize) + 1.
			}]
		raise: Error.

	self assert: {2} equals: result shape.
	first := result asNumbers first.
	second := result asNumbers second.
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
	mnist := MNIST3LayersNNExamplePlan new.
	
	self 
		shouldnt: [
			results := mnist
				predict: {
					ByteArray new: mnist inputSize.
					(ByteArray new: mnist inputSize) + 1}
				andCompareTo: #(1 2).
				]
		raise: Error.

	self assert: {2} equals: results first shape.
	first := results first asNumbers first.
	second := results first asNumbers second.
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
	mnist := MNISTSoftMaxExamplePlan new.
	
	self 
		shouldnt: [
			result := mnist predict: {
				ByteArray new: mnist inputSize.
				(ByteArray new: mnist inputSize) + 1.
			}]
		raise: Error.

	self assert: {2} equals: result shape.
	self assert: result allInt32s first closeTo: 0.
	self assert: result allInt32s second closeTo: 0.
! !

!TensorFlowExamplesTest methodsFor: 'testing MNIST' stamp: 'JB 2/17/2017 19:29:28'!
testMNISTSoftMaxLossGraph
	" Not sure how to test this better. weights are random, so output is random.
	I believe that choosing the right input value the output should comply with some statistics,
	however there are only 10 output valus, and I believe that's not enough to have statistical meaning"
	
	| mnist results |
	mnist := MNISTSoftMaxExamplePlan new.
	
	self 
		shouldnt: [
			results := mnist
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

!TensorFlowExamplesTest methodsFor: 'testing-ols' stamp: 'gera 1/21/2017 05:18:41'!
testOLS
	self testOLS: OLSExample new.! !

!TensorFlowExamplesTest methodsFor: 'testing-ols' stamp: 'gera 6/30/2019 00:45:25'!
testOLS: ols
	| x y betas |
	x := TFTensor fromFloats: #(
		(1  2   3)
		(6  7   8)
		(4  5   6)
		(9  2   3)
		(1 10  2)
	).

	y := TFTensor fromFloats: #(
		(14)
		(44)
		(32)
		(22)
		(27)
	).
	
	betas := ols regress: x on: y.
	self assert: betas first closeTo: 1;
		assert: betas second closeTo: 2;
		assert: betas third closeTo: 3
	
		
! !

!TensorFlowExamplesTest methodsFor: 'testing-ols' stamp: 'gera 1/29/2017 05:16:11'!
testOLSWithOperations
	self testOLS: OLSExample new initializeWithOperations! !

!TensorFlowExamplesTest methodsFor: 'testing-NeuralNetwork' stamp: 'gera 6/30/2019 00:45:27'!
testSimpleNeuralNetwork
	| inputs weights nn prediction target targetValues |
	inputs := TFTensor fromFloats: {
		{1}.
		{2 ln}
		}.
	targetValues := #(0.5 0.5).
	target := TFTensor fromFloats: targetValues.
	weights := #(
		(1 1)
		(1 0)
	).
	nn := SimpleNeuralNetworkExample new.
	nn initialWeights: weights;
	 	learnAll: {inputs} to: {target} steps: 100.
	prediction := nn predict: inputs.
	targetValues with: prediction do:[:real :predicted| 
		self assert: (real - predicted) abs < 0.1]! !

!BatchTrainer methodsFor: 'initialization' stamp: 'JB 2/18/2017 20:36:29'!
initialize
	super initialize.
	batchSize := 100.
	imageSet := MNISTImageFile trainingSet.
	labelSet := MNISTLabelFile trainingSet.
	imageTesting := MNISTImageFile testSet.
	labelTesting := MNISTLabelFile testSet.
	self use3Layers.
! !

!BatchTrainer methodsFor: 'initialization' stamp: 'JB 2/18/2017 20:28:05'!
use3Layers
	plan := MNIST3LayersNNExamplePlan new.! !

!BatchTrainer methodsFor: 'initialization' stamp: 'JB 2/19/2017 17:09:45'!
use3LayersSigmoid
	plan := MNIST3LayersNNSigmoid new.! !

!BatchTrainer methodsFor: 'initialization' stamp: 'JB 2/18/2017 20:28:22'!
useSoftmax
	plan := MNISTSoftMaxExamplePlan new.! !

!BatchTrainer methodsFor: 'private' stamp: 'JB 2/18/2017 20:23:39'!
predict: images andCompareTo: labels
	| equals result |
	result := (plan predict: images) asNumbers.
	equals := result with: labels collect:[:x :y| (x = y) ifTrue:[1] ifFalse:[0]].
	^ equals mean asFloat.
! !

!BatchTrainer methodsFor: 'private' stamp: 'JB 2/18/2017 20:21:23'!
trainAt: index
	| images labels loss report result |
	images := imageSet bytesAt: index take: batchSize.
	labels := labelSet at: index take: batchSize.
	result := plan predict: images andLearnFrom: labels.
	loss := result at: 2.
	report := index asString, ': ', loss asNumbers asString.
	Transcript show: report;
					newLine
	! !

!BatchTrainer methodsFor: 'running' stamp: 'JB 2/18/2017 20:27:19'!
predictOnTesting
	| images labels |
	images := imageTesting bytesAt: 1 take: imageTesting count.
	labels := labelTesting at: 1 take: labelTesting count.
	^ self predict: images andCompareTo: labels! !

!BatchTrainer methodsFor: 'running' stamp: 'JB 2/18/2017 20:26:40'!
predictOnTraining
	| images labels |
	images := imageSet bytesAt: 1 take: 10 * batchSize.
	labels := labelSet at: 1 take: 10 * batchSize.
	^ self predict: images andCompareTo: labels! !

!BatchTrainer methodsFor: 'running' stamp: 'JB 2/18/2017 20:22:07'!
train
	(1 to: labelSet count -1 by: batchSize) do: [:index| self trainAt: index].! !

!ExamplePlan methodsFor: 'initialization' stamp: 'gera 1/29/2017 05:12:34'!
initialize
	self initializeGraph.
	self initializeSession.! !

!ExamplePlan methodsFor: 'initialization' stamp: 'gera 7/3/2019 00:34:29'!
initializeGraph
	graph := TFGraph create
	! !

!ExamplePlan methodsFor: 'initialization' stamp: 'gera 7/3/2019 03:35:50'!
initializeSession
	session := TFSession on: graph.
	graph initializeOn: session.
! !

!ExamplePlan methodsFor: 'running' stamp: 'gera 1/29/2017 05:19:43'!
runOn: inputValues
	| results |	
		
	results := session
		runInputs: inputs
		values: inputValues
		outputs: outputs.
		
	^ results first
	! !

!BackpropagationBackwardPlan methodsFor: 'initialization' stamp: 'gera 6/30/2019 00:45:25'!
initializeGraph
	| inputVariable weights actual delta learningRate newWeights target one |
	super initializeGraph.
	inputVariable := graph placeholder: 'Input' type: TFTensor typeFloat.
	target := graph placeholder: 'target' type: TFTensor typeFloat.
	actual := graph placeholder: 'actual' type: TFTensor typeFloat.
	weights := graph placeholder: 'weights' type: TFTensor typeFloat.
	learningRate := graph const: 'learningRate' value: 0.9 asTensor.
	one := graph const: 'one' value: 1.0 asTensor.
	delta := (target - actual) negated @* actual @* (one - actual) @* inputVariable.
	newWeights := weights - (learningRate @* delta).
	outputs := {newWeights output: 0}.
	inputs := {inputVariable input: 0. weights input: 0. target input:0. actual input: 0}! !

!BackpropagationForwardPlan methodsFor: 'initialization' stamp: 'gera 6/30/2019 00:45:26'!
initializeGraph
	| activation inputVariable netInput weights |
	super initializeGraph.
	inputVariable := graph placeholder: 'Input' type: TFTensor typeFloat.
	weights := graph placeholder: 'weights' type: TFTensor typeFloat.
	netInput := weights * inputVariable.
	activation := netInput sigmoid.
	outputs := {activation output: 0}.
	inputs := {inputVariable input: 0. weights input: 0}! !

!BackpropagationPlan methodsFor: 'accessing' stamp: 'gera 1/29/2017 11:17:26'!
delta
	^ lastDelta! !

!BackpropagationPlan methodsFor: 'accessing' stamp: 'gera 1/29/2017 08:32:19'!
graph
	^ graph! !

!BackpropagationPlan methodsFor: 'accessing' stamp: 'gera 1/29/2017 08:40:28'!
weights
	^ session runOutput: (weights output: 0)! !

!BackpropagationPlan methodsFor: 'initialization' stamp: 'gera 6/30/2019 00:45:26'!
initializeBackwardGraph
	| actual learningRate learningRateValue one input |
	actual := activation.
	one := graph const: 'one' value: #((1 1) (1 1)) asFloatTensor.
	
	learningRateValue := 0.9 asTensor.
	learningRate := graph const: 'learningRate' value: learningRateValue.

	input := inputs first operationOn: graph.
	
	target := graph placeholder: 'target' type: TFTensor typeFloat.
	
	delta := (target - actual) negated @* actual @* (one - actual) @* input.
	
	"learn := weights assign: weights - learningRate @* delta."
	"learn := weights -= learningRate @* delta."
	learn := weights descent: delta rate: learningRate.! !

!BackpropagationPlan methodsFor: 'initialization' stamp: 'gera 6/30/2019 00:45:25'!
initializeForwardGraph
	| input |
	input := graph placeholder: 'Input' type: TFTensor typeFloat.
	
	activation := (weights * input) sigmoid.
	
	outputs := {activation output: 0}.
	inputs := {input input: 0}! !

!BackpropagationPlan methodsFor: 'initialization' stamp: 'gera 1/29/2017 05:28:36'!
initializeGraph
	super initializeGraph.
	self
		initializeVariables;
		initializeForwardGraph;
		initializeBackwardGraph.! !

!BackpropagationPlan methodsFor: 'initialization' stamp: 'gera 6/30/2019 00:45:26'!
initializeVariables
	| initialWeights |
	Random withDefaultDo: [:random |
		initialWeights := (1 to: 4) collect: [:unused | random next]].
	initialWeights := TFTensor fromFloats: initialWeights shape: #(2 2).
	
	weights := graph variable: 'weights' initialValue: initialWeights.! !

!BackpropagationPlan methodsFor: 'training' stamp: 'gera 1/29/2017 15:09:30'!
learn: sample with: expected
	| results |	
	results := session
		runInputs: {inputs first. target input: 0}
		values: {sample. expected}
		outputs: {learn output: 0. delta output: 0}.
	
	lastDelta := results at: 2.
	^ results first
	! !

!ExampleNearestNeighborPlan methodsFor: 'initialization' stamp: 'gera 6/30/2019 00:45:25'!
initializeGraph
	| neg add abs axis axisValue distance testing training prediction |
	super initializeGraph.
	
	training := graph placeholder: 'training' type: TFTensor typeFloat.
	testing := graph placeholder: 'testing' type: TFTensor typeFloat.
	axisValue := TFTensor fromInt32: 0.
	axis := graph const: 'axis' value: axisValue.
	neg := graph newOperation: 'Neg' named: 'neg' described: [:description |
		description addInput: (testing output: 0)].
	add := graph add: 'add' described:[:description| 
		description addInput: (neg output: 0).
		description addInput: (training output: 0).
		].
	abs := graph newOperation: 'Abs' named: 'abs' described: [:description|
		description addInput: (add output: 0)].
	distance := graph newOperation: 'Sum' named: 'distance' described: [:description|
		description addInput: (abs output: 0).
		description addInput: (axis output: 0).].
	
	prediction := graph newOperation: 'ArgMin' named: 'argmin' described:[:description|
		description addInput: (distance output: 0).
		description addInput: (axis output: 0)].
	
	outputs := {prediction output: 0}.
	inputs := {training input: 0. testing input: 0}.! !

!ExampleNearestNeighborPlan methodsFor: 'initialization' stamp: 'gera 7/3/2019 00:34:29'!
initializeGraphWithOperations
	| axis distance testing training prediction |
	graph := TFGraph create.
	
	training := graph placeholder: 'training' type: TFTensor typeFloat.
	testing := graph placeholder: 'testing' type: TFTensor typeFloat.
	axis := 0 asInt32Tensor.
	
	distance := (testing - training) abs sumOn: axis.
	prediction := distance findMinOn: axis.
	
	outputs := {prediction output: 0}.
	inputs := {training input: 0. testing input: 0}.
! !

!ExampleNearestNeighborPlan methodsFor: 'evaluating' stamp: 'gera 1/21/2017 04:19:59'!
predict: covariatesTesting from: covariatesTraining
	| result |

 	result := self runOn: {covariatesTraining. covariatesTesting}.
	^ result allInt64s + 1! !

!ExampleOLSPlan methodsFor: 'initialization' stamp: 'gera 6/30/2019 00:45:27'!
initializeGraph
	| x xtx y xty inverse result |
	super initializeGraph.
	
	x := graph placeholder: 'x' type: TFTensor typeFloat.
	y := graph placeholder: 'y' type: TFTensor typeFloat.
	
	xtx := graph newOperation: 'MatMul' named:'xTx' described:[:description|
		description addInput: (x output: 0).
		description addInput: (x output: 0).
		description at: 'transpose_a' putBoolean: true.
		].
	
	inverse := graph newOperation: 'MatrixInverse' named:'inv' described:[:description|
		description addInput: (xtx output: 0)].
	
	xty := graph newOperation: 'MatMul' named:'xTy' described:[:description|
		description addInput: (x output: 0).
		description addInput: (y output: 0).
		description at: 'transpose_a' putBoolean: true.
		].
	
	result := graph newOperation: 'MatMul' named:'result' described:[:description|
		description addInput: (inverse output: 0).
		description addInput: (xty output: 0).
		].
	
	outputs := {result output: 0}.
	inputs := {x input: 0. y input: 0}.! !

!ExampleOLSPlan methodsFor: 'initialization' stamp: 'gera 6/30/2019 00:45:26'!
initializeGraphWithOperations
	| x y prediction |
	super initializeGraph.
	
	x := graph placeholder: 'x' type: TFTensor typeFloat.
	y := graph placeholder: 'y' type: TFTensor typeFloat.
	
	prediction := (x \* x) inverse * (x \* y).
	
	outputs := {prediction output: 0}.
	inputs := {x input: 0. y input: 0}.! !

!MNIST3LayersNNExamplePlan methodsFor: 'accessing' stamp: 'gera 1/31/2017 01:44:41'!
graph
	^ graph! !

!MNIST3LayersNNExamplePlan methodsFor: 'accessing' stamp: 'gera 1/31/2017 14:41:15'!
hidden1Size
	^ 128! !

!MNIST3LayersNNExamplePlan methodsFor: 'accessing' stamp: 'gera 1/31/2017 14:41:18'!
hidden2Size
	^ 32! !

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

!MNIST3LayersNNExamplePlan methodsFor: 'initialization' stamp: 'gera 1/31/2017 09:56:01'!
initialize
	self
		initializeGraph;
		initializeParameters;
		initializeInferenceGraph;
		initializeLossGraph;
		initializeLearningGraph;
		initializeSession.! !

!MNIST3LayersNNExamplePlan methodsFor: 'initialization' stamp: 'gera 7/3/2019 00:34:29'!
initializeGraph
	graph := TFGraph create! !

!MNIST3LayersNNExamplePlan methodsFor: 'initialization' stamp: 'gera 2/22/2017 00:10:23'!
initializeInferenceGraph
	hidden1 := graph
		fromBlock: [:image |
			input := image.
			(image * weights1 + biases1) rectified]
		named: 'layer1'.
	hidden2 := graph fromBlock: [(hidden1 * weights2 + biases2) rectified] named: 'layer2'.
	prediction := graph
		fromBlock: [
			netInput := hidden2 * weights3 + biases3.
			netInput softmax findMaxOn: 1 asInt32Tensor]
		named: 'layer3'.
! !

!MNIST3LayersNNExamplePlan methodsFor: 'initialization' stamp: 'gera 6/30/2019 00:45:27'!
initializeLearningGraph
	|  axis0 backprop learningRate batchSize learnBiases1 learnBiases2 learnBiases3 learnWeights1 learnWeights2 learnWeights3 |
	
	learningRate := graph const: 0.1 asTensor.
	batchSize := graph fromBlock: [(input sizeOn: 0) castTo: TFTensor typeFloat] named: 'batchSize'.
	axis0 := graph const: #(0) asInt32Tensor. 
	graph
		fromBlock: [
			| biasGradient activationGradient | 
			activationGradient := activation useOutput: 1.
			biasGradient := activationGradient meanOn: axis0.
			learnWeights3 := weights3 descent: hidden2 \* activationGradient @/ batchSize rate: learningRate.
			learnBiases3 := biases3 descent: biasGradient rate: learningRate.
			backprop := activationGradient *\ weights3]
		named: 'learning3'.
		
	graph fromBlock: [
		| gradient |
		gradient := backprop timesRectifiedGradOf: hidden2.
		learnWeights2 := weights2 descent: hidden1 \* gradient @/ batchSize rate: learningRate.
		learnBiases2 := biases2 descent: (gradient meanOn: axis0) rate: learningRate.
		backprop := gradient *\ weights2] 
			named: 'learning2'.
			
	graph fromBlock: [
		| gradient | 
		gradient := backprop timesRectifiedGradOf: hidden1.
		learnWeights1 := weights1 descent: input \* gradient @/ batchSize rate: learningRate.
		learnBiases1 := biases1 descent: (gradient meanOn: axis0) rate: learningRate] 
			named: 'learning1'.
			
	learn := graph newOperation: 'Identity' named: 'learn' described: [:description |
		description
			addInput: loss output;
			addControlInput: learnWeights1 output;
			addControlInput: learnBiases1 output;
			addControlInput: learnWeights2 output;
			addControlInput: learnBiases2 output;
			addControlInput: learnWeights3 output;
			addControlInput: learnBiases3 output].
! !

!MNIST3LayersNNExamplePlan methodsFor: 'initialization' stamp: 'gera 6/30/2019 00:45:28'!
initializeLossGraph
	loss := graph
		fromBlock: [:expected |
			expectedLabel := expected.
			activation := netInput sparseSoftmaxCrossEntropyWithLogits: expected.
			activation meanOn: #(0) asInt32Tensor]
		inputTypes: {TFTensor typeInt32}
		named: 'loss'.! !

!MNIST3LayersNNExamplePlan methodsFor: 'initialization' stamp: 'gera 2/16/2017 09:35:09'!
initializeParameters
	| aux |
		graph
			fromBlock: [
				aux := graph truncatedNormalRandomShaped: {self inputSize. self hidden1Size} stddev: 1.0 / self inputSize sqrt.
				weights1 := graph variable: 'weights1' initialValueFrom: aux.
				aux := graph zerosShaped: {self hidden1Size}.
				biases1 := graph variable: 'biases1' initialValueFrom: aux.

				aux := graph truncatedNormalRandomShaped: {self hidden1Size. self hidden2Size} stddev: 1.0 / self hidden1Size sqrt.
				weights2 := graph variable: 'weights2' initialValueFrom: aux.
				aux := graph zerosShaped: {self hidden2Size}.
				biases2 := graph variable: 'biases2' initialValueFrom: aux.

				aux := graph truncatedNormalRandomShaped: {self hidden2Size. self outputSize} stddev: 1.0 / self hidden2Size sqrt.
				weights3 := graph variable: 'weights3' initialValueFrom: aux.
				aux := graph zerosShaped: {self outputSize}.
				biases3 := graph variable: 'biases3' initialValueFrom: aux]
			named: 'parameters'
! !

!MNIST3LayersNNExamplePlan methodsFor: 'initialization' stamp: 'gera 7/3/2019 03:35:50'!
initializeSession
	session := TFSession on: graph.
	graph initializeOn: session.! !

!MNIST3LayersNNExamplePlan methodsFor: 'running' stamp: 'gera 2/22/2017 00:10:30'!
predict: inputs
	| results |
	results := session
		runInputs: {input input: 0}
		values: {inputs asFloatTensor}
		outputs: {prediction output}.
	^ results first! !

!MNIST3LayersNNExamplePlan methodsFor: 'running' stamp: 'gera 2/22/2017 00:10:58'!
predict: inputs andCompareTo: label
	| results |
	results := session
		runInputs: {input input: 0. expectedLabel input: 0}
		values: {inputs asFloatTensor. label asInt32Tensor}
		outputs: {prediction output. loss output}.
	^ results! !

!MNIST3LayersNNExamplePlan methodsFor: 'running' stamp: 'gera 2/26/2017 15:13:22'!
predict: inputs andLearnFrom: label
	| results |
	results := session
		runInputs: {input input: 0. expectedLabel input: 0}
		values: {inputs asFloatTensor. label asInt32Tensor}
		outputs: {loss output. learn output}.
	^ results! !

!MNIST3LayersNNSigmoid methodsFor: 'initialization' stamp: 'gera 2/25/2017 11:31:02'!
initializeInferenceGraph
	hidden1 := graph
		fromBlock: [:image |
			input := image.
			(image * weights1 + biases1) sigmoid]
		named: 'layer1'.
	hidden2 := graph fromBlock: [(hidden1 * weights2 + biases2) sigmoid] named: 'layer2'.
	prediction := graph
		fromBlock: [
			netInput := hidden2 * weights3 + biases3.
			netInput findMaxOn: 1 asInt32Tensor]
		named: 'layer3'.
! !

!MNIST3LayersNNSigmoid methodsFor: 'initialization' stamp: 'gera 6/30/2019 00:45:28'!
initializeLearningGraph
	| axis0 activationGradient gradient1 gradient2 learningRate biasGradient one batchSize learnBiases1 learnBiases2 learnBiases3 learnWeights1 learnWeights2 learnWeights3 |
	
	learningRate := graph const: 0.1 asTensor.
	batchSize := graph fromBlock: [(input sizeOn: 0) castTo: TFTensor typeFloat] named: 'batchSize'.
	axis0 := #(0) asInt32Tensor. 
	graph
		fromBlock: [
			activationGradient := activation useOutput: 1.
			biasGradient := activationGradient meanOn: axis0.
			learnWeights3 := weights3 descent: hidden2 \* activationGradient @/ batchSize rate: learningRate.
			learnBiases3 := biases3 descent: biasGradient rate: learningRate]
		named: 'learning3'.
		
	one := 1.0 asTensor asOperationOn: graph.
	graph fromBlock: [
		gradient2 := (activationGradient *\ weights3) @* hidden2 @* (one - hidden2).
		learnWeights2 := weights2 descent: hidden1 \* gradient2 @/ batchSize  rate: learningRate.
		learnBiases2 := biases2 descent: (gradient2 meanOn: axis0) rate: learningRate] 
			named: 'learning2'.
			
	graph fromBlock: [
		gradient1 := (gradient2 *\ weights2) @* hidden1 @* ( one - hidden1).
		learnWeights1 := weights1 descent: input \* gradient1 @/ batchSize  rate: learningRate.
		learnBiases1 := biases1 descent: (gradient1 meanOn: axis0) rate: learningRate] 
			named: 'learning1'.
			
	learn := graph newOperation: 'Identity' named: 'learn' described: [:description |
		description
			addInput: loss output;
			addControlInput: learnWeights1 output;
			addControlInput: learnBiases1 output;
			addControlInput: learnWeights2 output;
			addControlInput: learnBiases2 output;
			addControlInput: learnWeights3 output;
			addControlInput: learnBiases3 output].
! !

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

!MNISTFile methodsFor: 'initialization' stamp: 'jmv 5/7/2023 18:57:06'!
parse: aStream
	magic := aStream nextUint32BigEndian: true.
	count :=  aStream nextUint32BigEndian: true.

	! !

!MNISTFile class methodsFor: 'private' stamp: 'JB 2/5/2017 15:56:42'!
download: aName
		| url response content file |
		url := 'http://yann.lecun.com/exdb/mnist/', aName.
		response := WebClient httpGet: url.
		content := response content asByteArray.
		file := ('dataset/', name) asFileEntry.
		file < content.
		^ file! !

!MNISTFile class methodsFor: 'instance creation' stamp: 'JB 2/5/2017 15:36:27'!
fromFile: aString
	| file filename stream compressed|
	filename := aString, '.gz'.
	file := ('dataset/', filename) asFileEntry.
	file exists ifFalse:[ file := self download: filename].
	compressed := file readStream contentsOfEntireFile.
	stream := (GZipReadStream on: compressed) upToEnd asByteArray readStream.
	^ self fromStream: stream
	! !

!MNISTFile class methodsFor: 'instance creation' stamp: 'JB 2/5/2017 15:26:40'!
fromStream: aStream
	| answer |
	answer := self new.
	^ answer parse: aStream! !

!MNISTFile class methodsFor: 'instance creation' stamp: 'JB 2/5/2017 15:28:39'!
testSet
	^self fromFile: self testName
	! !

!MNISTFile class methodsFor: 'instance creation' stamp: 'JB 2/5/2017 15:28:58'!
trainingSet
	^self fromFile: self trainName
	! !

!MNISTFile class methodsFor: 'accessing' stamp: 'JB 2/5/2017 15:07:25'!
magic
	^ self subclassResponsibility ! !

!MNISTFile class methodsFor: 'accessing' stamp: 'JB 2/5/2017 15:16:07'!
testName
	^ self subclassResponsibility ! !

!MNISTFile class methodsFor: 'accessing' stamp: 'JB 2/5/2017 15:16:03'!
trainName
	^ self subclassResponsibility ! !

!MNISTImageFile methodsFor: 'converting' stamp: 'gera 6/30/2019 00:45:26'!
asTensor
	^  TFTensor fromFloats:  items shape: {count. rows * columns}.! !

!MNISTImageFile methodsFor: 'accessing' stamp: 'JB 2/10/2017 00:50:56'!
at: imageIndex
	| bytes |
	bytes := self bytesAt: imageIndex.
	^ GrayForm extent: rows @ columns bits: bytes! !

!MNISTImageFile methodsFor: 'accessing' stamp: 'JB 2/10/2017 00:50:21'!
bytesAt: imageIndex
	| size base |
	size := columns * rows.
	base := imageIndex - 1 * size + 1.
	^ items copyFrom: base to: base + size - 1.! !

!MNISTImageFile methodsFor: 'accessing' stamp: 'JB 2/18/2017 18:23:25'!
bytesAt: base take: batchSize
	^ (base to: base + batchSize -1) collect:[:index| self bytesAt: index]! !

!MNISTImageFile methodsFor: 'accessing' stamp: 'JB 2/5/2017 15:58:35'!
columns
	^ columns! !

!MNISTImageFile methodsFor: 'accessing' stamp: 'JB 2/5/2017 15:58:26'!
rows
	^ rows! !

!MNISTImageFile methodsFor: 'initialization' stamp: 'jmv 5/7/2023 18:57:11'!
parse: aStream
	super parse: aStream.
	rows := aStream nextUint32BigEndian: true.
	columns := aStream nextUint32BigEndian: true.
	items := aStream upToEnd.! !

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
	items := aStream upToEnd.
	
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

!MNISTSoftMaxExamplePlan methodsFor: 'initialization' stamp: 'gera 7/3/2019 00:34:29'!
initializeGraph
	graph := TFGraph create! !

!MNISTSoftMaxExamplePlan methodsFor: 'initialization' stamp: 'gera 2/12/2017 23:10:20'!
initializeInferenceGraph
	prediction := graph
		fromBlock: [:image |
			input := image.
			netInput := image * weights + biases.
			netInput softmax findMaxOn: 1 asInt32Tensor]
		named: 'inference'! !

!MNISTSoftMaxExamplePlan methodsFor: 'initialization' stamp: 'gera 6/30/2019 00:45:26'!
initializeLearningGraph
	| learningRate activationGradient biasGradient |
	graph
		fromBlock: [ | batchSize |
			learningRate := 0.9 asTensor.
			activationGradient := activation useOutput: 1.
			biasGradient := activationGradient meanOn: #(0) asInt32Tensor.
			batchSize := (input sizeOn: 0) castTo: TFTensor typeFloat.
			learnWeights := weights descent: input \* activationGradient @/ batchSize rate: learningRate.
			learnBiases := biases descent: biasGradient rate: learningRate]
		named: 'learning'! !

!MNISTSoftMaxExamplePlan methodsFor: 'initialization' stamp: 'gera 6/30/2019 00:45:27'!
initializeLossGraph
	loss := graph
		fromBlock: [:expected |
			expectedLabel := expected.
			activation := netInput sparseSoftmaxCrossEntropyWithLogits: expected.
			activation meanOn: #(0) asInt32Tensor]
		inputTypes: {TFTensor typeInt32}
		named: 'loss'
! !

!MNISTSoftMaxExamplePlan methodsFor: 'initialization' stamp: 'gera 2/2/2017 19:29:23'!
initializeParameters
	| aux |
	aux := graph zerosShaped: {self inputSize. self outputSize}.
	weights := graph variable: 'weights' initialValueFrom: aux.
	aux := graph zerosShaped: {self outputSize}.
	biases := graph variable: 'biases' initialValueFrom: aux.! !

!MNISTSoftMaxExamplePlan methodsFor: 'initialization' stamp: 'gera 7/3/2019 03:35:50'!
initializeSession
	session := TFSession on: graph.
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
	results := session
		runInputs: {input input: 0}
		values: {inputs asFloatTensor}
		outputs: {prediction output: 0}.
	^ results first! !

!MNISTSoftMaxExamplePlan methodsFor: 'running' stamp: 'gera 2/12/2017 23:07:52'!
predict: inputs andCompareTo: label
	| results |
	results := session
		runInputs: {input input: 0. expectedLabel input: 0}
		values: {inputs asFloatTensor. label asInt32Tensor}
		outputs: {prediction output. loss output}.
	^ results! !

!MNISTSoftMaxExamplePlan methodsFor: 'running' stamp: 'gera 2/19/2017 10:52:34'!
predict: inputs andLearnFrom: label
	| results |
	results := session
		runInputs: {input input: 0. expectedLabel input: 0}
		values: {inputs asFloatTensor. label asInt32Tensor}
		outputs: {loss output. learnWeights output. learnBiases output}.
	^ results! !

!NearestNeighbor methodsFor: 'private-csv' stamp: 'gera 7/3/2019 03:35:50'!
decodeCSV: csvLines graphRun: description
	|  graph output records session results values answer |
		
	graph := self decodeCSVGraphDefaults: description.
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
			(output output: 3).
			(output output: 4)}.
		
	graph delete.
	
	answer := (1 to: 4) asOrderedCollection collect: [:i |
		(results at: i) allFloats].
	^ answer add: (results at: 5) allInt64s; yourself.! !

!NearestNeighbor methodsFor: 'private-csv' stamp: 'gera 7/3/2019 00:34:29'!
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

!NearestNeighbor methodsFor: 'private-csv' stamp: 'gera 6/30/2019 00:45:26'!
irisDescription
	^{TFTensor fromFloats: #(-1.0).
		TFTensor fromFloats: #(-1.0).
		TFTensor fromFloats: #(-1.0).
		TFTensor fromFloats: #(-1.0).
		TFTensor fromInt64s: #(-1)}! !

!NearestNeighbor methodsFor: 'private-csv' stamp: 'gera 1/16/2017 23:00:14'!
irisFile
	^ self class irisFile! !

!NearestNeighbor methodsFor: 'initialization' stamp: 'JB 1/19/2017 10:46:56'!
initialize
	plan := ExampleNearestNeighborPlan new.! !

!NearestNeighbor methodsFor: 'initialization' stamp: 'jb 1/15/2017 23:28:54'!
loadDataSet
	| dataSet description stream |
	stream := self irisFile.
	stream nextLine.
	dataSet := self testingAndTrainingFrom: stream.
	description := self irisDescription.
	^ dataSet collect: [ :csv |
		self
			decodeCSV: csv
			graphRun: description ].! !

!NearestNeighbor methodsFor: 'running' stamp: 'gera 1/30/2017 17:56:34'!
run
	| covariatesTr covariatesTe predictions size dataset testing training |
	dataset := self loadDataSet.
	training := dataset first.
	testing := dataset second.
	covariatesTr := training allButLast asFloatTensor.
	size := testing first size.
	predictions := (1 to: size) collect: [ :i | | covariates index predicted results trueValue |
		covariates := testing  collect:[:col| {col at: i}].
		covariatesTe := covariates allButLast asFloatTensor.
		results := plan predict: covariatesTe from: covariatesTr.
		index := results first.
		trueValue := covariates fifth first.
		predicted := training fifth at: index.
		trueValue = predicted
			ifTrue: [ 1 ]
			ifFalse: [ 0 ]].
	^ predictions.! !

!NearestNeighbor methodsFor: 'sampling' stamp: 'jb 1/15/2017 19:13:00'!
testingAndTrainingFrom: stream
	| p random testing training lines |
	lines := stream upToEnd lines.
	p := 0.7.
	training := OrderedCollection new.
	testing := OrderedCollection new.
	random := Random new.
	lines collect: [ :line | | collection |
		line isEmpty ifFalse: [
			collection := random next < p
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
	plan := ExampleOLSPlan new.! !

!OLSExample methodsFor: 'initialization' stamp: 'gera 1/29/2017 05:15:58'!
initializeWithOperations
	plan := ExampleOLSPlan new initializeGraphWithOperations initializeSession! !

!OLSExample methodsFor: 'running' stamp: 'gera 1/21/2017 04:20:26'!
regress: x on: y
	| result |
	result := plan runOn: {x. y}.
	^ result allFloats! !

!RegressionNNExample methodsFor: 'accessing' stamp: 'JB 2/22/2017 22:33:25'!
graph
	^ graph! !

!RegressionNNExample methodsFor: 'accessing' stamp: 'JB 2/22/2017 22:33:41'!
hidden1Size
	^ 20! !

!RegressionNNExample methodsFor: 'accessing' stamp: 'JB 2/22/2017 22:33:46'!
hidden2Size
	^ 20! !

!RegressionNNExample methodsFor: 'accessing' stamp: 'JB 2/22/2017 22:33:54'!
hidden3Size
	^ 20! !

!RegressionNNExample methodsFor: 'accessing' stamp: 'JB 2/22/2017 22:33:59'!
inputSize
	^ 1! !

!RegressionNNExample methodsFor: 'accessing' stamp: 'JB 2/22/2017 22:34:03'!
intput
	^ input! !

!RegressionNNExample methodsFor: 'accessing' stamp: 'JB 2/22/2017 22:34:23'!
outputSize
	^ 1! !

!RegressionNNExample methodsFor: 'initialization' stamp: 'JB 2/22/2017 22:33:07'!
initialize
	self
		initializeGraph;
		initializeParameters;
		initializeInferenceGraph;
		initializeLossGraph;
		initializeLearningGraph;
		initializeSession.! !

!RegressionNNExample methodsFor: 'initialization' stamp: 'gera 7/3/2019 00:34:29'!
initializeGraph
	graph := TFGraph create! !

!RegressionNNExample methodsFor: 'initialization' stamp: 'gera 2/26/2017 12:52:24'!
initializeInferenceGraph
	hidden1 := graph
		fromBlock: [:image |
			input := image.
			(image * weights1 + biases1) rectified]
		named: 'layer1'.
	hidden2 := graph fromBlock: [(hidden1 * weights2 + biases2) sigmoid] named: 'layer2'.
	hidden3 := graph fromBlock: [(hidden2 * weights3 + biases3) sigmoid] named: 'layer3'.
	prediction := graph fromBlock: [hidden3 * weights4 + biases4] named: 'layer4'.
! !

!RegressionNNExample methodsFor: 'initialization' stamp: 'gera 6/30/2019 00:45:28'!
initializeLearningGraph
	|  axis0 learningRate batchSize biasGradient one backprop learnBiases1 learnBiases2 learnBiases3 learnBiases4 learnWeights1 learnWeights2 learnWeights3 learnWeights4 |
	
	learningRate := 0.1 asTensor.
	batchSize := graph fromBlock: [(input sizeOn: 0) castTo: TFTensor typeFloat] named: 'batchSize'.
	axis0 := graph const: #(0) asInt32Tensor. 
	one := 1.0 asTensor asOperationOn: graph.
	graph
		fromBlock: [ | gradient|
			gradient := (prediction - expectedLabel).
			biasGradient := gradient meanOn: axis0.
			learnWeights4 := weights4 descent: hidden3 \* gradient @/ batchSize rate: learningRate.
			learnBiases4 := biases4 descent: biasGradient rate: learningRate.
			backprop :=  (gradient *\ weights4)]
		named: 'learning4'.	
	
	graph
		fromBlock: [ | gradient | 
			gradient := backprop @* hidden3 @* (one - hidden3).
			biasGradient := gradient meanOn: axis0.
			learnWeights3 := weights3 descent: hidden2 \* gradient @/ batchSize rate: learningRate.
			learnBiases3 := biases3 descent: biasGradient rate: learningRate.
			backprop := (gradient *\ weights3)]
		named: 'learning3'.
		
	graph fromBlock: [ | gradient |
		gradient :=  backprop @* hidden2 @* (one - hidden2).
		learnWeights2 := weights2 descent: hidden1 \* gradient @/ batchSize  rate: learningRate.
		learnBiases2 := biases2 descent: (gradient meanOn: axis0) rate: learningRate.
		backprop :=  (gradient *\ weights2)]
			named: 'learning2'.
			
	graph fromBlock: [ | gradient |
		gradient := backprop timesRectifiedGradOf: hidden1.
		learnWeights1 := weights1 descent: input \* gradient rate: learningRate.
		learnBiases1 := biases1 descent: (gradient meanOn: axis0) rate: learningRate] 
			named: 'learning1'.

	learn := graph newOperation: 'Identity' named: 'learn' described: [:description |
		description
			addInput: loss output;
			addControlInput: learnWeights1 output;
			addControlInput: learnBiases1 output;
			addControlInput: learnWeights2 output;
			addControlInput: learnBiases2 output;
			addControlInput: learnWeights3 output;
			addControlInput: learnBiases3 output;
			addControlInput: learnWeights4 output;
			addControlInput: learnBiases4 output].
! !

!RegressionNNExample methodsFor: 'initialization' stamp: 'gera 6/30/2019 00:45:26'!
initializeLossGraph
	loss := graph
		fromBlock: [ :expected |
			expectedLabel := expected.
			(prediction - expectedLabel) squared meanOn: #(0) asInt32Tensor ]
		inputTypes: {TFTensor typeFloat}
		named: 'loss'.! !

!RegressionNNExample methodsFor: 'initialization' stamp: 'gera 2/26/2017 17:45:31'!
initializeParameters
	| aux |
	graph
		fromBlock: [
			aux := graph truncatedNormalRandomShaped: {self inputSize. self hidden1Size} stddev: 1.0 / self inputSize sqrt.
			weights1 := graph variable: 'weights1' initialValueFrom: aux.
			aux := graph zerosShaped: {self hidden1Size}.
			biases1 := graph variable: 'biases1' initialValueFrom: aux.

			aux := graph truncatedNormalRandomShaped: {self hidden1Size. self hidden2Size} stddev: 1.0 / self hidden1Size sqrt.
			weights2 := graph variable: 'weights2' initialValueFrom: aux.
			aux := graph zerosShaped: {self hidden2Size}.
			biases2 := graph variable: 'biases2' initialValueFrom: aux.

			aux := graph truncatedNormalRandomShaped: {self hidden2Size. self hidden3Size } stddev: 1.0 / self hidden2Size sqrt.
			weights3 := graph variable: 'weights3' initialValueFrom: aux.
			aux := graph zerosShaped: {self hidden3Size }.
			biases3 := graph variable: 'biases3' initialValueFrom: aux.
			
			aux := graph truncatedNormalRandomShaped: {self hidden3Size. self outputSize } stddev: 1.0 / self hidden3Size sqrt.
			weights4 := graph variable: 'weights4' initialValueFrom: aux.
			aux := graph zerosShaped: {self outputSize }.
			biases4 := graph variable: 'biases4' initialValueFrom: aux]
		named: 'parameters'
		
		
! !

!RegressionNNExample methodsFor: 'initialization' stamp: 'gera 7/3/2019 03:35:50'!
initializeSession
	session := TFSession on: graph.
	graph initializeOn: session! !

!RegressionNNExample methodsFor: 'running' stamp: 'JB 2/22/2017 22:33:07'!
predict: inputs
	| results |
	results := session
		runInputs: {input input: 0}
		values: {inputs asFloatTensor}
		outputs: {prediction output: 0}.
	^ results first! !

!RegressionNNExample methodsFor: 'running' stamp: 'JB 2/22/2017 23:03:05'!
predict: inputs andCompareTo: label
	| results |
	results := session
		runInputs: {input input: 0. expectedLabel input: 0}
		values: {inputs asFloatTensor. label asFloatTensor}
		outputs: {prediction output: 0. loss output: 0}.
	^ results! !

!RegressionNNExample methodsFor: 'running' stamp: 'gera 2/26/2017 15:17:10'!
predict: inputs andLearnFrom: label
	| results |
	results := session
		runInputs: {input input: 0. expectedLabel input: 0}
		values: {inputs asFloatTensor. label asFloatTensor}
		outputs: {loss output:0. learn output}.
	^ results! !

!RegressionNNExample class methodsFor: 'examples' stamp: 'JB 2/23/2017 00:12:28'!
exampleFunction
	^ [ :x | | y |
			y := x * 10 - 5.
			y * y sin ].! !

!RegressionNNExample class methodsFor: 'examples' stamp: 'gera 2/25/2017 12:02:17'!
exampleTrainedAndPlot
	" self exampleTrainedAndPlot "
	| function net predictor g |
	function :=  self exampleFunction.
	net := self exampleTrainedOn: function.
	predictor := [:x |
		| result |
		result := net predict: {{x}}.
		result asNumbers first first] .
	
	g := FunctionGraphMorph new.
	g domain: (0 to: 1).
	g addFunction: function color: Color green.
	g addFunction: predictor color: Color red.
	(g embeddedInMorphicWindowLabeled: 'graph') openInWorld.
	^ net! !

!RegressionNNExample class methodsFor: 'examples' stamp: 'gera 2/25/2017 12:03:03'!
exampleTrainedOn: function
	| net rnd interval xs ys |
	net := self new.
	rnd := Random new.
	xs :=  (1 to: 100) collect: [:i | {rnd next}].
	ys := xs collect: [:x | {function value: x first}].
	interval := 1 to: xs size.
	
	10000 timesRepeat: [
		|x indices y |
		indices := (1 to: 60) collect: [:i| interval atRandom].
		x := indices collect: [:index | xs at: index].
		y :=  indices collect: [:index | ys at: index].
		net predict: x andLearnFrom: y].
	^ net! !

!SimpleNeuralNetworkExample methodsFor: 'accessing' stamp: 'gera 1/30/2017 17:56:57'!
initialWeights: aCollection
	weights := aCollection asFloatTensor! !

!SimpleNeuralNetworkExample methodsFor: 'initialization' stamp: 'JB 1/25/2017 15:22:27'!
initialize
	forward := BackpropagationForwardPlan new.
	backward := BackpropagationBackwardPlan new.! !

!SimpleNeuralNetworkExample methodsFor: 'learning' stamp: 'JB 1/25/2017 15:25:47'!
learn: sample with: expected
		| result |
	result := forward runOn: {sample. weights}.
	weights := backward runOn: {sample. weights. expected. result}.! !

!SimpleNeuralNetworkExample methodsFor: 'learning' stamp: 'JB 1/25/2017 15:29:01'!
learnAll: samples to: targets steps: n
	n timesRepeat: [
		samples with: targets do:[:sample :target| self learn: sample with: target]
		]! !

!SimpleNeuralNetworkExample methodsFor: 'service' stamp: 'JB 1/25/2017 15:26:43'!
predict: sample
		| result |
	result := forward runOn: {sample. weights}.
	^ result allFloats ! !
