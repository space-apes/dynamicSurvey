import numpy as np
from rbm import RBM
from kmodes import Kmodes
from apiclient.discovery import build
from oauth2client.service_account import ServiceAccountCredentials
from mathtools import *
from random import gauss


#TOY MOVIE DATASET: 25% romance, 75% other
#SHEET_ID = '1QtGC9wmIANYb7DFBBVMrbRqh4QZsx0LCzn8WWH1D474'
#SHEET_RANGE = 'B2:M17'

#TOY MOVIE DATASET: 25% romance, 25%scifi, 25%action, 25% other
#SHEET_ID = '1d2SJXqawVG5g7SuI6-XU6cST392m7h3j6Y-Ngl5gwpo'
#SHEET_RANGE = 'B2:M17'

#59 respondents culture dataset
SHEET_ID='17y07Dg89WMv-bOZkh_GxhuoZJPjKGw9Xg3R5am_nMOs'
SHEET_RANGE = 'F2:BA60'


scopes = ['https://www.googleapis.com/auth/spreadsheets.readonly']

HIDDEN_UNITS = 4
LEARNING_RATE = .001
GIBBS_ITERATIONS = 3
ITERATIONS_THROUGH_TRAININGSET = 10

def get_service(api_name, api_version, key_file_location):
        creds = ServiceAccountCredentials.from_json_keyfile_name(key_file_location, scopes)
        service = build(api_name, api_version, credentials=creds)
        return service

def main():
	#google sheets stuff
	service = get_service('sheets', 'v4', './serviceCred.json')
	sheet = service.spreadsheets()
	result = sheet.values().get(spreadsheetId = SHEET_ID, range=SHEET_RANGE).execute()
	values = result.get('values', [])
	
	#get file object for log file
	fo = open("log.txt", "a")
	
	numpyTrainingSet = []
	
	if not values:
		print('no data found')
	else:
		numpyTrainingSet = np.asarray(values, dtype=np.int8)


	numpyTrainingSet = np.reshape(numpyTrainingSet, (len(numpyTrainingSet),len(numpyTrainingSet[0]),-1))
	
	
	km1 = Kmodes(binVectorHammingDistance, numpyTrainingSet)
	
	
	km1.avgHammingDistancesForKClustersExperiment(6)

	"""
	
	tv1 = np.reshape(np.array([1,1,1]), (3,1))
	tv2 = np.reshape(np.array([1,1,1]), (3,1))
	tv3 = np.reshape(np.array([0,0,1]), (3,1))
	tv4 = np.reshape(np.array([1,0,1]), (3,1))
	tv5 = np.reshape(np.array([0,0,0]), (3,1))

	vectorList = [tv1,tv2,tv3,tv4,tv5]
	km1 = Kmodes(binVectorHammingDistance, vectorList)

	print(f"total mode is: {km1.modeOfBinVectorList(vectorList)}")

	print(f"closest centroid to [1,1,1] is: {km1.indexOfClosestCentroid(tv1, [tv3, tv4, tv5])}")
	"""
	#kClusters = km1.kClustersOfIndexes(3)
	
	"""
	rbm1 = RBM(HIDDEN_UNITS, fo, numpyTrainingSet)	

	rbm1.train(LEARNING_RATE, GIBBS_ITERATIONS, ITERATIONS_THROUGH_TRAININGSET)

	#romance vector	
	testVector = np.reshape([1,1,1,0,0,0,0,0,0,0,0,0], (12,1))
	for i in range(HIDDEN_UNITS):
	       print(f"for hidden unit {i}, probability is: {rbm1.probHGivenXVector(i, testVector)}")a
	"""
	fo.close()
if __name__ =="__main__":
	main()
