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
       	
"""

import re
import numpy as np
from rbm import RBM
from kmodes import Kmodes
from apiclient.discovery import build
from oauth2client.service_account import ServiceAccountCredentials
from mathtools import *
from random import gauss
from plotstuff import generateImages




scopes = ['https://www.googleapis.com/auth/spreadsheets.readonly']


def get_service(api_name, api_version, key_file_location):
        creds = ServiceAccountCredentials.from_json_keyfile_name(key_file_location, scopes)
        service = build(api_name, api_version, credentials=creds)
        return service

def getConfigDict(configFileName, maxLines):
	configDict = {}
	configFile = open(configFileName, "r")
	lineCount =  0
	while lineCount < maxLines:
		stringBuffer = configFile.readline()
		key = re.search(r"(.+)\{", stringBuffer).group(1)
		rawValue = stringBuffer.split(" ")[1]
		
		#convert to desired datatype
		if re.search(r"\.", rawValue):
			configDict[key] = float(rawValue)
		elif re.search(f"\d", rawValue):
			configDict[key] = int(rawValue)
		else:
			configDict[key] = rawValue.rstrip()
		lineCount +=1

	configFile.close()
	return configDict


def main():
	#create dictionary of top level parameters from config.txt
	config = getConfigDict("config.txt", 9)
	
	#get file object for log file
	fo = open("log.txt", "r+")

	#clear any previous data in log
	fo.truncate(0)
	
	#pick dataset
	if config['DATASET'] == 'toy':	
		#TOY MOVIE DATASET: 25% romance, 75% other
		SHEET_ID = '1QtGC9wmIANYb7DFBBVMrbRqh4QZsx0LCzn8WWH1D474'
		SHEET_RANGE = 'B2:M17'
	if config['DATASET'] == 'toyEven':	
		#TOY MOVIE DATASET: 25% romance, 25% scifi, 25% action, 25% random
		SHEET_ID = '1d2SJXqawVG5g7SuI6-XU6cST392m7h3j6Y-Ngl5gwpo'
		SHEET_RANGE = 'B2:M17'
	elif config['DATASET'] == 'culture':
		SHEET_ID='17y07Dg89WMv-bOZkh_GxhuoZJPjKGw9Xg3R5am_nMOs'
		SHEET_RANGE = 'F2:BA60'
	elif config['DATASET'] == 'generatedCulture':
		#300 samples, 100 features, 30 samples per pattern
		#see generateData.py for details
		SHEET_ID='1oGcEJ_xQFHz61HgFv2gdxM65C6NG3AQEz6o0HD30Rz0'
		SHEET_RANGE = 'Sheet1!A1:CV300'

	
	
	#get 2d list from google sheets
	service = get_service('sheets', 'v4', './serviceCred.json')
	sheet = service.spreadsheets()
	result = sheet.values().get(spreadsheetId = SHEET_ID, range=SHEET_RANGE).execute()
	values = result.get('values', [])

		
	#convert list to numpy array of column vectors
	numpyTrainingSet = []
	if not values:
		print('no data found')
	else:
		numpyTrainingSet = np.asarray(values, dtype=np.int8)

	numpyTrainingSet = np.reshape(numpyTrainingSet, (len(numpyTrainingSet),len(numpyTrainingSet[0]),-1))


	#run experiment	
	if config['EXPERIMENT'] == 'kmodes':
		km1 = Kmodes(binVectorHammingDistance, numpyTrainingSet, fo, config['KMODES_VERBOSITY'])
		km1.clusterCountIterationExperiment(config['KMODES_MAXK'])	
	
	elif config['EXPERIMENT'] == 'rbm':
		rbm1 = RBM(config['RBM_HIDDEN_UNIT_COUNT'], fo, numpyTrainingSet, config['RBM_VERBOSITY'])
		rbm1.train(config['RBM_LEARNING_RATE'], config['RBM_GIBBS_ITERATIONS'], config['RBM_TRAINING_SET_PASSES'])

	#generate results images
	fo.seek(0)
	generateImages(fo, config)

	fo.close()


if __name__ =="__main__":
	main()
