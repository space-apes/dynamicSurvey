
import numpy as np
from apiclient.discovery import build
from oauth2client.service_account import ServiceAccountCredentials
import random


scopes = ['https://www.googleapis.com/auth/spreadsheets']

def binaryOutput(x, y):
	if x < y:
		return 1
	else:
		return 0

def generateBinaryVectorFromMap(highLowMap, moreLikelyPercent, lessLikelyPercent):
	tempList = [binaryOutput(random.random(), .5) for x in range(100)]

	#increased probability of being 1 for first 8 indexes of highlowmap
	for x in range(8):
		tempList[highLowMap[x]] = binaryOutput(random.random(), moreLikelyPercent/100)

	#decreased probability of being 1 last 2 indexes of highlowmap
	for y in range(8,10):
		tempList[highLowMap[y]] = binaryOutput(random.random(), lessLikelyPercent/100)
	
	return tempList





def get_service(api_name, api_version, key_file_location):
        creds = ServiceAccountCredentials.from_json_keyfile_name(key_file_location, scopes)
        service = build(api_name, api_version, credentials=creds)
        return service



def main():

		
	service = get_service('sheets', 'v4', './serviceCred.json')
	SHEET_ID = '1oGcEJ_xQFHz61HgFv2gdxM65C6NG3AQEz6o0HD30Rz0'

	sheet = service.spreadsheets()
	
	#random feature indexes 
	#all are set to 1,0 with 50% chance
	#first 8 indexes of each highLowMap element are set to higher chance of 1
	#last 2 indexes are set to lower chance of 1

	highLowMap = [
	  [95, 73, 85, 8, 3, 66, 28, 20, 88, 4],
	  [84, 11, 10, 15, 62, 27, 60, 92, 69, 82],
	  [59, 91, 32, 20, 57, 35, 45, 36, 95, 40],
	  [51, 85, 18, 83, 74, 0, 29, 45, 80, 14],
	  [32, 91, 4, 21, 41, 20, 29, 74, 44, 63],
	  [8, 72, 7, 1, 50, 46, 84, 68, 41, 37],
	  [7, 68, 29, 19, 40, 30, 88, 3, 1, 57],
	  [62, 4, 59, 60, 68, 45, 70, 3, 26, 67],
	  [39, 58, 33, 85, 29, 18, 3, 27, 73, 89],
	  [44, 0, 17, 41, 4, 59, 55, 12, 50, 86],
	]

	values = []
	
	#for each highLowMap pattern of some indexes beiing higher prob some not
	#generate 30 binary lists and append them to values list
	#for a total of 300 samples, 30 samples per pattern

	for highLowMapIndex in range(10):
  		for x in range(30):
    			values.append(generateBinaryVectorFromMap(highLowMap[highLowMapIndex], 80, 20))

	"""
	#to test that each 30 representative samplse as similar distribution
	totalCount = [0]*100
	for vector in values[60:99]:
		totalCount = np.add(totalCount, vector)
	"""
	body  = {'values': values}	

	result = service.spreadsheets().values().update(spreadsheetId=SHEET_ID, range="Sheet1!A1", valueInputOption="USER_ENTERED", body=body).execute()
	#print('{0} cells updated.'.format(result.get('updatedCells')))	


if __name__ =="__main__":
	main()
