'From Cuis 5.0 of 7 November 2016 [latest update: #3007] on 21 December 2016 at 10:07:39 am'!
'Description Please enter a description for this package'!
!provides: 'TensorFlow' 1 1!
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

!TensorFlowCAPITest methodsFor: 'testing' stamp: 'gera 12/21/2016 09:56:55'!
testCopyToExternalMemory
	| original copy |
	original _ 'hola manola'.
	copy _ library externalizeString: original.
	original withIndexDo: [:each :index |
		self assert: each asciiValue = (copy byteAt: index)].
	self assert: (copy byteAt: original size + 1) = 0.
	copy free.! !

!TensorFlowCAPITest methodsFor: 'testing' stamp: 'gera 12/21/2016 02:58:13'!
testNewSessionOptions
	| options |
	options _ TF_SessionOptions create.
	self deny: options getHandle isNil.
	self deny: options getHandle = 0.
	options delete.
	self assert: options getHandle isNil.! !

!TensorFlowCAPITest methodsFor: 'testing' stamp: 'gera 12/21/2016 02:58:04'!
testNewStatus
	| status |
	status _ TF_Status create.
	self deny: status getHandle isNil.
	self deny: status getHandle = 0.
	status delete.
	self assert: status getHandle isNil.! !

!TensorFlowCAPITest methodsFor: 'testing' stamp: 'gera 12/21/2016 02:49:10'!
testSessionOptionsSetTarget
	| options |
	options _ library newSessionOptions.
	self deny: options getHandle isNil.
	self deny: options getHandle = 0.
	options target: 'local'.
	options delete.
	self assert: options getHandle isNil.! !

!TensorFlowCAPITest methodsFor: 'testing' stamp: 'gera 12/21/2016 09:57:36'!
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

!TensorFlowCAPI methodsFor: 'session options' stamp: 'GerardoRicharte 12/18/2016 21:35'!
newSessionOptions
	<cdecl: TF_SessionOptions* 'TF_NewSessionOptions' (void)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'status' stamp: 'gera 12/21/2016 02:56:25'!
newStatus
	<cdecl: TF_Status* 'TF_NewStatus' (void)>
	^self externalCallFailed! !

!TensorFlowCAPI methodsFor: 'session options' stamp: 'GerardoRicharte 12/19/2016 12:51'!
sessionOptions: aTF_SessionOptions setConfig: aString status: aTF_Status
	<cdecl: void 'TF_SetConfig' (TF_SessionOptions* char*)>
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

!TF_SessionOptions methodsFor: 'initialize-release' stamp: 'gera 12/21/2016 09:56:48'!
config: aString
	^ self library sessionOptions: self setConfig: (self library externalizeString: aString).
! !

!TF_SessionOptions methodsFor: 'initialize-release' stamp: 'gera 12/21/2016 02:20:57'!
delete
	self library deleteSessionOptions: self.
	handle := nil! !

!TF_SessionOptions methodsFor: 'initialize-release' stamp: 'gera 12/21/2016 09:56:51'!
target: aString
	^ self library sessionOptions: self setTarget: (self library externalizeString: aString).
! !

!TF_SessionOptions class methodsFor: 'instance creation' stamp: 'gera 12/21/2016 02:50:29'!
create
	^ TensorFlowCAPI current newSessionOptions! !

!TF_SessionOptions class methodsFor: 'instance creation' stamp: 'GerardoRicharte 12/19/2016 11:01'!
target: aString config: anotherString
	^ self create
		target: aString;
		config: aString! !

!TF_Status methodsFor: 'accessing' stamp: 'gera 12/21/2016 09:59:56'!
code
	^self library getCode: self! !

!TF_Status methodsFor: 'accessing' stamp: 'gera 12/21/2016 09:57:53'!
code: aTF_Code message: aString
	| externalized |
	externalized _ self library externalizeString: aString.
	^ self library setStatus: self code: aTF_Code message: externalized
! !

!TF_Status methodsFor: 'initialize-release' stamp: 'gera 12/21/2016 02:59:15'!
delete
	self library deleteStatus: self.
	handle := nil! !

!TF_Status class methodsFor: 'instance creation' stamp: 'gera 12/21/2016 02:55:36'!
create
	^ TensorFlowCAPI current newStatus! !
TF_Structure initialize!
