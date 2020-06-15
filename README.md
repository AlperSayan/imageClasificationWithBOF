# imageClasificationWithBOF 
classify the dataset via bag of features.

# Details 
A bag	of	Features	system	consists	of	the	following	components,	you	can	see	the hyper-parameters	that	will	be	
adjusted during	experiments.	
- Feature	extraction	(**Grid,	keypoints**)  
  - Sample fixed	size	image	patches	from	a	grid or use keypoints.  
- Feature	Descriptor (**SIFT**)  
- Dictionary	Computation	(**mean-shift	and	k-means**)  
  - Cluster	with	mean-shift	default	parameters.  
  - Dictionaries	with	the	k	values	:	50,	250,	500	and	the	number	of	clusters	meanshift	gives.  	
- Feature	quantization	and	histogram	calculation	(**nearest	neighbor**)  
- Classifier	training (**SVM**)  

