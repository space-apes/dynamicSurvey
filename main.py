import numpy as np
from rbm import RBM
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
	
	#print(f"shape of numpyTrainingSet: {np.shape(numpyTrainingSet)}")
	#print(f"shape of first element: {np.shape(numpyTrainingSet[0])}")
	#print(f" first element: {numpyTrainingSet[0]}")
	#print(f" first value: {numpyTrainingSet[0][0][0]}")
	
	#testVector1 = np.array([1,0,1])
	#testVector2 = np.array([0,2,0])

	#print(f"shape of testVector1: ", np.shape(testVector1))
	#print(np.matmul(np.reshape(testVector1, (3,1)), np.reshape(testVector2, (1,3))))

	#print(f"numpyTrainingSet shape: {np.shape(numpyTrainingSet)} before:\n{numpyTrainingSet}")
	#print(f"numpyTrainingSet shape: {np.shape(numpyTrainingSet)} after:\n{numpyTrainingSet}")
	
	rbm1 = RBM(HIDDEN_UNITS, fo, numpyTrainingSet)	

	#print(f"probability of h1 given first training vector before training: {rbm1.probHgivenXvector(numpyTrainingSet[0])}")

	#print(f"expected HVector given first training: {rbm1.expectedHVectorGivenXVector(numpyTrainingSet[0])}" )
	#print(f"prob x given hVector: {rbm1.probXGivenHVector(0, np.reshape([1,0,1], (3,1)))}")
	#print(f"xVector given hVector: {rbm1.expectedXVectorGivenHVector(np.reshape([1,0,1], (3,1)))}")
	#print(f"shape of generated Xvector: {np.shape(rbm1.expectedXVectorGivenHVector(np.reshape([1,0,1], (3,1))))}")
	#print(f"shape of generated HVector: {np.shape(rbm1.expectedHVectorGivenXVector(np.reshape([1,0,1,0,1,1,0,1,1,0,0,0], (12,1))))}")

	rbm1.train(.001, 10, 100)
	#print(f"expected XVector given first training: {rbm1.expectedXVectorGivenHVector([0,0,0])}" )

	#romance vector	
	testVector = np.reshape([1,1,1,0,0,0,0,0,0,0,0,0], (12,1))
	for i in range(HIDDEN_UNITS):
	       print(f"for hidden unit {i}, probability is: {rbm1.probHGivenXVector(i, testVector)}")
	fo.close()
if __name__ =="__main__":
	main()
