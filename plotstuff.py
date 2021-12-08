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
		if splitList[0] == "Kmodes.clusterCountIterationExperiment::":
			kToAvgHammingDistance.append(int(splitList[1]))
			avgHammingDistanceToK.append(float(splitList[2]))
		elif splitList[0] == "RBM.train::":
			trainingSetIterationsToGeneratedOverlap.append(int(splitList[3]))
			generatedOverlapToTrainingSetIterations.append(float(splitList[4]))
				
		stringBuffer = fo.readline()

	
	#kToAvgHammingDistancesToMode
	plt.plot(kToAvgHammingDistance, avgHammingDistanceToK)
	plt.title(f"avg intra-cluster hamming distance to mode for {configDict['DATASET']}")
	plt.xlabel("k")
	plt.ylabel("average hamming distance")
	plt.savefig(f'./images/kToAvgHammingDistance.png')

	#iterationsThroughTrainingSetToGeneratedOverlap
	plt.plot(trainingSetIterationsToGeneratedOverlap, generatedOverlapToTrainingSetIterations)
	plt.title(f"data: {configDict['DATASET']}, Hidden: {configDict['RBM_HIDDEN_UNIT_COUNT']}, LR: {configDict['RBM_LEARNING_RATE']}, gibbs: {configDict['RBM_GIBBS_ITERATIONS']}")
	plt.xlabel("training set iterations")
	plt.ylabel("percent accurately generated samples")
	plt.savefig(f'./images/trainingSetIterationsToGeneratedAccuracy-H{configDict["RBM_HIDDEN_UNIT_COUNT"]}-G{configDict["RBM_GIBBS_ITERATIONS"]}-LR{str(configDict["RBM_LEARNING_RATE"])[2:]}.png')
	
	#generate 2d lists 
	#generate kmodes k to convergence iterations
	#generate kmodes k to avgHammingDistanceToClusterMode

	#generate rbm k to visible->hidden->generated % overlap
	#
