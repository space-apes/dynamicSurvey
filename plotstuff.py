import matplotlib.pyplot as plt

def generateImages(fo, configDict):

	kToConvergenceIterations = []
	convergenceIterationsToK = []
	kToAvgHammingDistance = []
	avgHammingDistanceToK = []
	
	gibbsIterationsToGeneratedOverlap = []
	generatedOverlapToGibbsIterations = []

	trainingSetIterationsToGeneratedOverlap = []
	generatedOverlapToTrainingSetIterations = []

	#populate data point lists

	stringBuffer = fo.readline()
	while stringBuffer != "":
		splitList = stringBuffer.split(" ")
		if splitList[0] == "Kmodes.kClustersOfIndexes::":
			kToConvergenceIterations.append(int(splitList[1]))
			convergenceIterationsToK.append(int(splitList[2]))
		elif splitList[0] == "Kmodes.clusterCountIterationExperiment::":
			kToAvgHammingDistance.append(int(splitList[1]))
			avgHammingDistanceToK.append(float(splitList[2]))
		elif splitList[0] == "RBM.train::":
			trainingSetIterationsToGeneratedOverlap.append(int(splitList[3]))
			generatedOverlapToTrainingSetIterations.append(float(splitList[4]))
				
		stringBuffer = fo.readline()

	#kToConvergenceIterations
	plt.plot(kToConvergenceIterations, convergenceIterationsToK)
	plt.title(f"cluster count to # iterations before convergence using dataset: {configDict['DATASET']}")
	plt.xlabel("k")
	plt.ylabel("iterations until convergence")
	plt.savefig('./images/kToConvergenceIterations.png')
	
	#kToAvgHammingDistancesToMode
	plt.plot(kToAvgHammingDistance, avgHammingDistanceToK)
	plt.title(f"avg intra cluster hamming distance to mode using dataset: {configDict['DATASET']}")
	plt.xlabel("k")
	plt.ylabel("average hamming distance")
	plt.savefig('./images/kToAvgHammingDistance.png')

	#iterationsThroughTrainingSetToGeneratedOverlap
	plt.plot(trainingSetIterationsToGeneratedOverlap, generatedOverlapToTrainingSetIterations)
	plt.title(f"accurately generated samples. LR: {configDict['RBM_LEARNING_RATE']}, gibbs: {configDict['RBM_GIBBS_ITERATIONS']}")
	plt.xlabel("training set iterations")
	plt.ylabel("percent accurately generated samples")
	plt.savefig('./images/trainingSetIterationsToGeneratedAccuracy.png')
	
	#generate 2d lists 
	#generate kmodes k to convergence iterations
	#generate kmodes k to avgHammingDistanceToClusterMode

	#generate rbm k to visible->hidden->generated % overlap
	#
