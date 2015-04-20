from GoogleAPIWrapper import GoogleAPIWrapper as api
from GooglePost import GooglePost
import cassandra.protocol
from haversine import haversine
from dbFacade import dbFacade
from Scorer import Scorer


"""
This class defines google+ api searching functionality.
It should be used in place of interacting with the api
directly.

@author: Luke Lindsey
@date: 14 March 2015
"""


class GooglePlusSearch(object):

	def __init__(self, api_key=None, db=None, scorer=None, query=None, params={}, location_center=None,
				location_radius=None, from_date=None):
		if not isinstance(api_key, str):
			raise TypeError('api_key must be a string')
		elif not isinstance(db, dbFacade):
			raise TypeError('dbFacade instance required')
		elif not isinstance(scorer, Scorer):
			raise TypeError('Scorer instance required')
		elif not isinstance(query, str):
			raise TypeError('Query must be a string')
		elif not isinstance(params, dict):
			raise TypeError('Params must be a dictionary')

		self.api_key = api_key
		self.db = db
		self.scorer = scorer
		self.query = query
		self.params = params
		self.postCount = 0

		# set location fields
		if location_center is None:
			self.location_search = False  # bool containing whether to include location in search
		else:
			self.location_search = True
		self.location_center = location_center
		self.location_radius = location_radius

		# set date fields
		if from_date is None:
			self.date_search = False
		else:
			self.date_search = True
		self.from_date = from_date

	def get_20_search_results(self, next_page_token=None):

		# dictionary for containing the GET query parameters.
		payload = {}
		# setting the query parameter
		payload["query"] = self.query
		# setting the google plus api key
		payload["key"] = self.api_key
		# setting the parameter for the number of returned results to 20
		payload["maxResults"] = 20
		# setting the order of the returned results to recent.
		payload["orderBy"] = "recent"
		# setting the language parameter to English, so that the returned activity feeds are only in english
		payload["language"] = "en"

		if next_page_token is not None:
			payload["pageToken"] = next_page_token

		for param in self.params:
			payload[param] = self.params[param]

		results = api.default_search(payload)

		# if self.location_search:
		# 	# do location stuff to results
		# 	results = GooglePlusSearch.filter_by_location(results, self.location_center, self.location_radius)
		# if self.date_search:
		# 	# do date search to results
		# 	results = GooglePlusSearch.filter_by_date(results, self.from_date)

		return results

	def store_posts_in_database(self, plus_posts=None):

		if plus_posts is None:
			raise TypeError('Posts argument required')
		elif not isinstance(plus_posts, list):
			raise TypeError('Posts argument must be a list of Google Posts')

		for plus_post in plus_posts:
			try:
				author = plus_post.author.encode('utf-8')
			except UnicodeDecodeError:
				author = plus_post.author.decode('utf-8')
			try:
				post = plus_post.post.encode('utf-8').replace("'", "''")
			except UnicodeDecodeError:
				post = plus_post.post.decode('utf-8').replace("'", "''")
				post = post.encode('utf-8')
			try:
				score = self.scorer.score(post)
			except UnicodeDecodeError:
				post = post.decode('utf-8')
				score = self.scorer.score(post)

			try:
				self.db.add_post(author, 'Google+', post, self.query, score)
				self.db.add_user(author, 0, 'Google+')
			except cassandra.protocol.SyntaxException as e:
				# print "failing at line 81 in GooglePlusSearch"
				# exc_type, exc_obj, exc_tb = sys.exc_info()
				# print 'type: ' + str(exc_type)
				# print(e)
				# sys.exit(0)
				post += "'''"
			self.postCount += 1

		print str(self.postCount) + " posts gathered.."

	def create_google_objects(self, search_results=None):
		if search_results is None:
			raise TypeError('Search results argument required')
		elif not isinstance(search_results, list):
			raise TypeError('Search results argument must be a list of Google+ posts')

		google_posts = []
		for search_result in search_results:
			post = self.create_google_object(search_result)
			google_posts.append(post)
		return google_posts

	@staticmethod
	def create_google_object(search_result):
		post = GooglePost(search_result)
		return post

	'''Given a list of 'item's and returns a list of the filtered items
	Expects the date as a string in format "2011-05-10" Where the month and
	date are always 2 digits long (needs a preceding 0 for one digit months/days)'''
	@staticmethod
	def filter_by_date(results, from_date):
		for result in results:
			this_date = result["updated"]  # in format like '2011-10-01T20:47:45.165Z'
			if GooglePlusSearch.num(from_date[0:4]) > GooglePlusSearch.num(this_date[0:4]):
				results.remove(result)
			elif GooglePlusSearch.num(from_date[0:4]) == GooglePlusSearch.num(this_date[0:4]):
				if GooglePlusSearch.num(from_date[5:7]) > GooglePlusSearch.num(this_date[5:7]):
					results.remove(result)
				elif GooglePlusSearch.num(from_date[5:7]) == GooglePlusSearch.num(this_date[5:7]):
					if GooglePlusSearch.num(from_date[8:10]) > GooglePlusSearch.num(this_date[8:10]):
						results.remove(result)

		return results

	'''Given a list of 'item's and returns a list of filtered items.
	Location center is in (lat, long) format. Radius given in km right now.'''
	@staticmethod
	def filter_by_location(results, location_center, location_radius):
		for result in results:
			try:
				this_location = result["location"]
				this_location = this_location["position"]
				this_location = (this_location["latitude"], this_location["longitude"])
				if haversine(this_location, location_center) > location_radius:
					results.remove(result)
			except KeyError:  # location not provided, so remove this result
				results.remove(result)
		return results

	@staticmethod
	def num(string_in):
		try:
			return int(string_in)
		except ValueError:
			raise ValueError("Not passing an int to num. Failed on :" + string_in)
