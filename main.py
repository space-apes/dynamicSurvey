"""
NAME: main.py
AUTHOR: Brian Reza Smith
INSTITUTION: California State University Dominguez Hills
YEAR: 2021
DESCRIPTION:
	
	driver for RBM experiment to use RBMs in cultural classification.
	Given a training set of binary input vectors representing cultural features (binary survey questions)
		1. use K-Modes method to search for a good estimate for number of hidden features
		2. encode probability distribution for training set
		2. generate examples of new input vectors representative of the training distribution
		3. identify hidden features that may represent underlying cultural identities
		4. produce a probability for a sample that it is associated with a hidden feature
       	
	Uses data from google sheets.

	Global parameter variables: 

	SHEET_ID: which google sheets to use
	SHEET_RANGE: which columns/rows do you want to use 
	LEARNING_RATE: modifies how drastically RBM weights and biases are updated during learning
	GIBBS_ITERATIONS: sets how many times RBM samples from hidden to visible layers
	ITERATIONS_THROUGH_TRAINING_SET: number of times RBM cycles during entire training set durin
		g learning phase
"""

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
	
	
	km1 = Kmodes(binVectorHammingDistance, numpyTrainingSet, fo)
	
	km1.clusterCountIterationExperiment(50)	

	#kClusters = km1.kClustersOfIndexes(3)
	
	fo.close()
if __name__ =="__main__":
	main()
