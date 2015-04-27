import requests
import sys

"""
Wrapper for Google+ API

@author: Luke Lindsey
@date: 13 March 2015
"""


class GoogleAPIWrapper(object):

	@staticmethod
	def get_api_key():
		# this one is old anyway. New key won't be posted publicly.
		api_key="AIzaSyC-980Av1mXcx3ugAqeB_G5lUTyGG-IN5o"

		# try:
		# 	fp = open("google_api.txt")
		# 	api_key = fp.readline()
		# 	if api_key == "":
		# 		print("Please place your API key in the google_api.txt file")
		# 		print("If you do not have an API Key from GOOGLE, please register for one at: http://developers.google.com")
		# 		sys.exit(0)
		#
		# 	fp.close()
		#
		# except IOError:
		# 	print('API Key not found! Please create and fill up google_api.txt file')
		# 	print('If you do not have an API Key from GOOGLE, please register for one at: http://developers.google.com')
		# 	sys.exit(0)
		# except Exception as e:
		# 	print(e)
		# 	sys.exit(0)

		return api_key

	@staticmethod
	def default_search(params, base_url="https://www.googleapis.com/plus/v1/activities"):

		try:
			# GET request sent to google
			r = requests.get(base_url, params=params)

			# response in json format converted into a dictionary
			response = eval(r.text)
			return response

		except Exception as e:
			# this error might be a rate limit, may need to add a sleep once we get the type
			exc_type, exc_obj, exc_tb = sys.exc_info()
			print 'type:' + str(exc_type)
			print("Error for URL: ", r.url)
			print(e)
			print(r.status_code)
			return {'status': 'ERROR', 'statusInfo': r.status_code}
