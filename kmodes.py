"""
NAME: kmodes.py
AUTHOR: Brian Reza Smith
INSTITUTION: California State University Dominguez Hills
YEAR: 2021
DESCRIPTION:
	definition of Kmodes class. Take in list of binary column vectors 
	and cluster them based on a similarity measure (euclidean or hamming distance). 
	
	notable methods:
		constructor(distanceFunction, dataSet, logFile) 
		kClustersOfIndexes(k) -> 2d list, each list contains indexes clustered together	
		modeOfBinVectorList(listOfVectors) -> representative vector for list 
		avgHammingDistanceToTargetVector(listOfVectors, targetVector)
		avgHammingDistanceForKClusters(k) 
"""

import numpy as np
from random import randint
from mathtools import binVectorHammingDistance, isBinaryColVector, isBinaryColVectorList

class Kmodes:
	distanceFunction = None
	trainingSet = None
	indexClusters = []
	logFile = None

	def __init__(self, _distanceFunction, _totalVectors, _logFile):
		self.distanceFunction = _distanceFunction
		self.totalVectors = _totalVectors
		self.indexClusters = []
		self.logFile = _logFile

	def modeOfBinVectorList(self, listOfVectors):
		firstShape = np.shape(listOfVectors[0])
		numberOfVectors = len(listOfVectors)

		if (not isBinaryColVectorList(listOfVectors)):
			print("Kmodes.modeOfBinVector:: non binary vector")
			return

		vectorCount = np.zeros((np.shape(listOfVectors[0])))
		for vector in listOfVectors:
			if (np.shape(vector) != firstShape):
				print("Kmodes.modeOfBinVector:: shape of vector does not match others")
				return
			vectorCount = np.add(vectorCount, vector)
		return np.reshape(list( map(lambda x: round( x[0]/numberOfVectors ), vectorCount)), (np.shape(listOfVectors[0])))
	
	def avgHammingDistanceToTargetVector(self, listOfVectors, targetVector):
		totalHammingDistances = 0
		for currentVector in listOfVectors:
			totalHammingDistances += binVectorHammingDistance(currentVector, targetVector)
		return totalHammingDistances / len(listOfVectors)
	

	def avgHammingDistancesForKClusters(self, k):
		self.kClustersOfIndexes(k)
		totalHammingDistances = 0
		vectorClusters = []
		#construct vectorClusters
		for clusterIndex in range(len(self.indexClusters)):
			vectorClusters.append([])
			withinClusterHammingDistances = 0
			for vectorIndex in self.indexClusters[clusterIndex]:
				vectorClusters[clusterIndex].append(self.totalVectors[vectorIndex])
			mode = self.modeOfBinVectorList(vectorClusters[clusterIndex])
			for vector in vectorClusters[clusterIndex]:
				withinClusterHammingDistances += binVectorHammingDistance(vector, mode)
			#print(f"for cluster {clusterIndex}, avg hamming distance is: {withinClusterHammingDistances / len(vectorClusters[clusterIndex])}")
			totalHammingDistances += withinClusterHammingDistances
		
		return totalHammingDistances / len(self.totalVectors)


	def clusterCountIterationExperiment(self, maxK):
		for k in range(1, maxK+1):
			self.logFile.write(f"clusterCountIterationExperiment:: {k} {self.avgHammingDistancesForKClusters(k)}\n")




	def intListsHaveSameElements(self,l1, l2):
		#now we know they have same lengths
		if len(l1) != len(l2):
			return False
		for x in l1:
		 	if x not in l2:
		 		return False
		return True
	
	
	def indexOfClosestCentroid(self, vector, centroidList):
		closestCentroidIndex = 0
		for i in range(len(centroidList)):
				if binVectorHammingDistance(vector, centroidList[i]) < binVectorHammingDistance(vector, centroidList[closestCentroidIndex]):
					closestCentroidIndex = i
		return closestCentroidIndex

	def kClustersOfIndexes(self,k):
		indexClusters = []
		centroids = []
		totalVectorsSet = np.copy(self.totalVectors)
		for cluster in range(k):
			indexClusters.append([])
			randIndex = randint(0, len(totalVectorsSet)-1)
			centroids.append(totalVectorsSet[randIndex])
			totalVectorsSet = np.delete(totalVectorsSet, randIndex, 0)
		
		#first assignments to random centroids
		for vectorIndex in range(len(self.totalVectors)):
			closestCentroidIndex = self.indexOfClosestCentroid(self.totalVectors[vectorIndex], centroids)
			indexClusters[closestCentroidIndex].append(vectorIndex)

		
		#now, find modes of clusters, those are new centroids.
		#group vectors into new clusters using new centroids
		#continue doing this until assignments to clusters are same as last iteration

		iterationsBeforeConvergence = 0
		converged = False
		while not converged:
			converged = True		
			oldIndexClusters = np.copy(indexClusters)
			#create vector clusters from indexClusters
			vectorClusters = []
			for i in range(k):
				vectorClusters.append([])
				for innerIndex in range(len(indexClusters[i])):
					vectorClusters[i].append(self.totalVectors[indexClusters[i][innerIndex]])
			#find mode for each cluster and set that as new centroid
			for i in range(k):
				if vectorClusters[i]:
					centroids[i] = self.modeOfBinVectorList(vectorClusters[i])
				else:
					centroids[i] = np.zeros(np.shape(self.totalVectors[0]))

			#assign to new clusters based on new centroids
			indexClusters = []
			for i in range(k):
				indexClusters.append([])
			for vectorIndex in range(len(self.totalVectors)):
				closestCentroidIndex = self.indexOfClosestCentroid(self.totalVectors[vectorIndex], centroids)
				indexClusters[closestCentroidIndex].append(vectorIndex)
			iterationsBeforeConvergence +=1
			
			for i in range(k):
				if not (self.intListsHaveSameElements(indexClusters[i], oldIndexClusters[i])):
					converged = False
					
		
		self.logFile.write(f"Kmodes.kClustersOfIndexes:: {k} {iterationsBeforeConvergence}\n")
		self.indexClusters = indexClusters

		return indexClusters
