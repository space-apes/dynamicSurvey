"""
NAME: mathtools.py
AUTHOR: Brian Reza Smith
INSTITUTION: California State University Dominguez Hills
YEAR: 2021
DESCRIPTION:
	definition of assorted math functions used in implementation
	of RBM machine learning model

"""
from math import exp, pi, sqrt
import numpy as np

def logistic(lnOdds):
        """
        converts logOdds to probability
                log odds of independent events is product of them

        @param float lnOdds
                natural log of ratio of probability event happens / probability it does not happen

        @return float probability
                probability of outcome such that 0 <= probability <= 1
        """
        return 1.0 / (1.0 + (exp(lnOdds)**-1) )


def univariateGaussianCurve(m, sd, x):
	""" calculates value on gaussian curve given mean, standard deviation, x-value 

	@param float m mean value of distribution
	@param float sd standard deviation of distribution
	@param float x value for sample 

	@return float y value on curve, given parameters
	"""
	return (1/(sqrt(2*pi*sd**2))) * exp(-(x-m)**2/(2*sd**2))

def binVectorEuclideanDistance(v1, v2):
	""" 
	euclidean distance given 2 binary vectors

	@param int[] v1 binary vector with values in {0,1}
	@param int[] v2 binary vector with values in {0,1}

	@return float square root of summed squared element-wise subtractions

	"""
	if np.shape(v1) != np.shape(v2):
		print("mathtools.binVectorEuclideanDistance:: vectors unequal shape")
		return 

	if (not isBinaryColVector(v1)) or (not isBinaryColVector(v2)):
		print("mathtools.hammingDistanceBinVectors:: invalid input vector")
		return
	
	total = 0
	for i in len(v1):
		total += (v1[i][0]-v2[i][0])**2
	return sqrt(total)

def binVectorHammingDistance(v1, v2):
	""" 
	hamming distance given 2 binary vectors

	@param int[] v1 binary vector with values in {0,1}
	@param int[] v2 binary vector with values in {0,1}

	@return int count of element-wise matches across length of vectors

	"""
	if np.shape(v1) != np.shape(v2):
		print("mathtools.binVectorHammingDistance:: vectors unequal shape")
		return 

	if (not isBinaryColVector(v1)) or (not isBinaryColVector(v2)):
		print("mathtools.hammingDistanceBinVectors:: invalid input vector")
		return 
	
	total = 0
	for i in range(len(v1)):
		if v1[i][0] != v2[i][0]:	
			total+=1
	return total

def isBinaryColVector(v1):
	shape = np.shape(v1)
	if len(shape) != 2 or shape[1] != 1:
		print("mathtools.isBinaryColVector:: shape is not (x,1)")
		return False
	for x in v1:
		if x[0] not in [0,1]:
			print("mathtools.isBinaryColVector:: value of element is not in {0,1}")
			return False
	return True

def isBinaryColVectorList(vectorList):
	for vector in vectorList:
		if (not isBinaryColVector(vector)):
			return False
	return True

"""
#notes for binary column vectors when i am too sleep deprived to deal with thinking:

testVector1 = np.reshape(np.array([1,1,0]), (3,1))
testVector2 = np.reshape(np.array([1,0,1]), (3,1))
testVector3 = np.reshape(np.array([1,0,1]), (3,1))


#print individual values by iterating through first index and keeping second index at 0
#print(testVector1[0][0])
#print(testVector1[1][0])

#find number of elements by doing len of whole vector
#print(len(testVector1))

"""
