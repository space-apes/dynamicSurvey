import numpy as numpy
from rbm import RBM
from apiclient.discovery import build
from oauth2client.service_account import ServiceAccountCredentials
from mathtools import *
from random import gauss



#expect about 25% to be associated with 'romance' hidden unit rest random
#SHEET_ID = '1QtGC9wmIANYb7DFBBVMrbRqh4QZsx0LCzn8WWH1D474'


#expect about 25% to be associated with each of the 3 categories, then 25% random
SHEET_ID = '1d2SJXqawVG5g7SuI6-XU6cST392m7h3j6Y-Ngl5gwpo'


scopes = ['https://www.googleapis.com/auth/spreadsheets.readonly']
SHEET_RANGE = 'B2:M17'

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
	
	numpyTestSet = []
	
	if not values:
		print('no data found')
	else:
		numpyTestSet = numpy.asarray(values)

	rbm1 = RBM(.001, 3, fo, numpyTestSet)	
	
	fo.close()
if __name__ =="__main__":
	main()
