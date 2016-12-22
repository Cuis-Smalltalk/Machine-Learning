'From Cuis 5.0 of 7 November 2016 [latest update: #3007] on 21 December 2016 at 11:47:56 pm'!
'Description Please enter a description for this package'!
!provides: 'TensorFlow' 1 9!
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

!classDefinition: #TF_Graph category: #'TensorFlow-Kernel'!
TF_Structure subclass: #TF_Graph
	instanceVariableNames: ''
	classVariableNames: ''
	poolDictionaries: ''
	category: 'TensorFlow-Kernel'!
!classDefinition: 'TF_Graph class' category: #'TensorFlow-Kernel'!
TF_Graph class
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

!TensorFlowCAPITest methodsFor: 'testing options' stamp: 'gera 12/21/2016 09:56:55'!
testCopyToExternalMemory
	| original copy |
	original _ 'hola manola'.
	copy _ library externalizeString: original.
	original withIndexDo: [:each :index |
		self assert: each asciiValue = (copy byteAt: index)].
	self assert: (copy byteAt: original size + 1) = 0.
	copy free.! !

!TensorFlowCAPITest methodsFor: 'testing graph' stamp: 'gera 12/21/2016 22:35:14'!
testNewGraph
	| graph |
	graph _ TF_Graph create.
	self deny: graph getHandle isNil.
	self deny: graph getHandle = 0.
	graph delete.
	self assert: graph getHandle isNil.! !

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

!TensorFlowCAPITest methodsFor: 'testing options' stamp: 'gera 12/21/2016 02:49:10'!
testSessionOptionsSetTarget
	| options |
	options _ library newSessionOptions.
	self deny: options getHandle isNil.
	self deny: options getHandle = 0.
	options target: 'local'.
	options delete.
	self assert: options getHandle isNil.! !

!TensorFlowCAPITest methodsFor: 'testing status' stamp: 'gera 12/21/2016 11:09:31'!
testStatusCodes
	| status  |
	status _ TF_Status create.
	
	status code: 0 message: ''.
	self assert: status isOk.
	self assert: status codeText = 'OK'.
	self shouldnt: [status check] raise: Error.
	
	status code: 1 message: 'You cancelled it!!'.
	self assert: status codeText = 'CANCELLED'.
	self should: [status check] raise: Error.

	status delete.
	self assert: status getHandle isNil.! !

!TensorFlowCAPITest methodsFor: 'testing status' stamp: 'gera 12/21/2016 10:18:08'!
testStatusGetMessage
	| status message |
	status _ TF_Status create.
	
	status code: 0 message: 'All is good'.
	message _ status message.
	self assert: message = 'All is good'.

	status code: 7 message: 'Something is very seven'.
	message _ status message.
	self assert: message = 'Something is very seven'.

	status delete.
	self assert: status getHandle isNil.! !

!TensorFlowCAPITest methodsFor: 'testing status' stamp: 'gera 12/21/2016 09:57:36'!
testStatusSetGetCode
	| status code |
	status _ TF_Status create.
	status code: 1 message: ''.
	code _ status code.
	self assert: code = 1.

	status code: 0 message: ''.
	code _ status code.
	self assert: code = 0.

	status delete.
	self assert: status getHandle isNil.! !

!TensorFlowCAPI methodsFor: 'session' stamp: 'gera 12/21/2016 23:26:55'!
closeSession: aTF_Session status: aTF_Status
	<cdecl: void 'TF_CloseSession' (TF_Session* TF_Status*)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'graph' stamp: 'gera 12/21/2016 22:32:51'!
deleteGraph: aTF_Graph
	<cdecl: void 'TF_DeleteGraph' (TF_Graph*)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'session' stamp: 'gera 12/21/2016 23:23:33'!
deleteSession: aTF_Session status: aTF_Status
	<cdecl: void 'TF_DeleteSession' (TF_Session* TF_Status*)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'session options' stamp: 'GerardoRicharte 12/18/2016 21:41'!
deleteSessionOptions: aTF_SessionOptions
	<cdecl: void 'TF_DeleteSessionOptions' (TF_SessionOptions*)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'status' stamp: 'gera 12/21/2016 02:57:15'!
deleteStatus: aTF_Status
	<cdecl: void 'TF_DeleteStatus' (TF_Status*)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'utils' stamp: 'gera 12/21/2016 09:56:40'!
externalizeString: aString
	| answer |
	answer := ExternalAddress allocate: aString size + 1.
	answer byteAt: aString size + 1 put: 0.
	aString withIndexDo: [:char :index |
		answer byteAt:  index put: char asciiValue].
	^ answer! !

!TensorFlowCAPI methodsFor: 'status' stamp: 'gera 12/21/2016 09:59:30'!
getCode: aTF_Status
	<cdecl: ulong 'TF_GetCode' (TF_Status*)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'status' stamp: 'gera 12/21/2016 10:14:23'!
message: aTF_Status
	<cdecl: char* 'TF_Message' (TF_Status*)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'graph' stamp: 'gera 12/21/2016 22:36:38'!
newGraph
	<cdecl: TF_Graph* 'TF_NewGraph' (void)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'session' stamp: 'gera 12/21/2016 23:21:09'!
newSessionGraph: aTF_Graph options:  aTF_SessionOptions status: aTF_Status
	<cdecl: TF_Session* 'TF_NewSession' (TF_Graph* TF_SessionOptions* TF_Status*)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'session options' stamp: 'GerardoRicharte 12/18/2016 21:35'!
newSessionOptions
	<cdecl: TF_SessionOptions* 'TF_NewSessionOptions' (void)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'status' stamp: 'gera 12/21/2016 02:56:25'!
newStatus
	<cdecl: TF_Status* 'TF_NewStatus' (void)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'session options' stamp: 'gera 12/21/2016 23:47:01'!
sessionOptions: aTF_SessionOptions setConfig: aString configSize: anInteger status: aTF_Status
	" not sure how to use this. Best information found in http://devdocs.io/tensorflow~python/contrib.learn#RunConfig"
	<cdecl: void 'TF_SetConfig' (TF_SessionOptions* void* ulong TF_Status*)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'session options' stamp: 'GerardoRicharte 12/19/2016 10:58'!
sessionOptions: aTF_SessionOptions setTarget: aString
	<cdecl: void 'TF_SetTarget' (TF_SessionOptions* char*)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'status' stamp: 'gera 12/21/2016 09:54:03'!
setStatus: aTF_Status code: anInteger message: anExternalString
	<cdecl: void 'TF_SetStatus' (TF_Status* ulong char*)>
	^self externalCallFailed! !

!TensorFlowCAPI class methodsFor: 'as yet unclassified' stamp: 'gera 12/21/2016 01:56:21'!
current
	^ current ifNil: [current _ self new].
! !

!TensorFlowCAPI class methodsFor: 'as yet unclassified' stamp: 'GerardoRicharte 12/18/2016 12:18'!
moduleName
	^ 'libtensorflow'! !

!TF_Structure methodsFor: 'private' stamp: 'gera 12/21/2016 02:52:42'!
library
	^ TensorFlowCAPI current! !

!TF_Structure class methodsFor: 'field definition' stamp: 'gera 12/21/2016 02:53:51'!
fields
	"self initialize"
	"typedef void* TF_SessionOptions"
	^#()! !

!TF_Structure class methodsFor: 'class initialization' stamp: 'gera 12/21/2016 02:54:06'!
initialize
	super initialize.
	self defineFields! !

!TF_Graph methodsFor: 'initialize-release' stamp: 'gera 12/21/2016 22:35:57'!
delete
	self library deleteGraph: self.
	handle := nil! !

!TF_Graph class methodsFor: 'instance creation' stamp: 'gera 12/21/2016 22:31:20'!
create
	^ TensorFlowCAPI current newGraph! !

!TF_Session methodsFor: 'as yet unclassified' stamp: 'gera 12/21/2016 23:28:28'!
close
	| status |
	status _ TF_Status create.
	self library closeSession: self status: status.
	[status check] ensure: [status delete].! !

!TF_Session methodsFor: 'as yet unclassified' stamp: 'gera 12/21/2016 23:23:02'!
delete
	| status |
	status _ TF_Status create.
	self library deleteSession: self status: status.
	[status check] ensure: [status delete].
	handle := nil! !

!TF_Session class methodsFor: 'instance creation' stamp: 'gera 12/21/2016 23:28:37'!
onGraph: aTF_Graph
	| options status answer |
	options _ TF_SessionOptions create.
	status _ TF_Status create.
	answer _ TensorFlowCAPI current newSessionGraph: aTF_Graph options: options status: status.
	options delete.
	[status check] ensure: [status delete].
	^ answer! !

!TF_SessionOptions methodsFor: 'initialize-release' stamp: 'gera 12/21/2016 02:20:57'!
delete
	self library deleteSessionOptions: self.
	handle := nil! !

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

!TF_Status methodsFor: 'accessing' stamp: 'gera 12/21/2016 23:30:59'!
code: aTF_Code message: aString
	| externalized |
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

!TF_Status methodsFor: 'initialize-release' stamp: 'gera 12/21/2016 02:59:15'!
delete
	self library deleteStatus: self.
	handle := nil! !

!TF_Status methodsFor: 'testing' stamp: 'gera 12/21/2016 10:48:36'!
isOk
	^self code == 0! !

!TF_Status methodsFor: 'accessing' stamp: 'gera 12/21/2016 23:31:33'!
message
	self library message: self! !

!TF_Status class methodsFor: 'instance creation' stamp: 'gera 12/21/2016 02:55:36'!
create
	^ TensorFlowCAPI current newStatus! !
TF_Structure initialize!
