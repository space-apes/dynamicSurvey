import numpy as np
from rbm import RBM
from kmodes import Kmodes
from apiclient.discovery import build
from oauth2client.service_account import ServiceAccountCredentials
from mathtools import *
from random import gauss



#expect about 25% to be associated with 'romance' hidden unit rest random
SHEET_ID = '1QtGC9wmIANYb7DFBBVMrbRqh4QZsx0LCzn8WWH1D474'

#expect about 25% to be associated with each of the 3 categories, then 25% random
#SHEET_ID = '1d2SJXqawVG5g7SuI6-XU6cST392m7h3j6Y-Ngl5gwpo'


scopes = ['https://www.googleapis.com/auth/spreadsheets.readonly']
SHEET_RANGE = 'B2:M17'

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
	totalMode = km1.modeOfBinVectorList(numpyTrainingSet)

	print(f"mode for entire training set is: {totalMode}")
	print(f"avg hamming distance from training set to training mode is: {km1.avgHammingDistanceToTargetVector(numpyTrainingSet, totalMode)}")

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
