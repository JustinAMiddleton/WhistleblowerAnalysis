from dbFacade import dbFacade
from Scorer import Scorer
from SocialNetworkSearch.GooglePlus.GooglePlusThread import GooglePlusThread
from SocialNetworkSearch.TwitterThread import TwitterThread
import threading
import time

class SearchInterface:

	threads = []

	def __init__(self):
		self.db = dbFacade()
		self.db.connect()
		self.db.create_keyspace_and_schema()

	def initialize_scorer(self, search_packet):
		self.scorer = Scorer(search_packet)

	'''
	Starts search crawling threads with inputed query string.
	'''
	def search(self, query, args):
		self.threads = []
		if args['GooglePlus']:
			self.threads.append(GooglePlusThread(self.db, self.scorer, query, args))
		if args['Twitter']:
			self.threads.append(TwitterThread(self.db, self.scorer, query, args))

		for thread in self.threads:
			thread.start()
		for thread in self.threads:
			thread.join()

	'''
	Ends search crawling threads; 
	waits for them to terminate before continuing.
	'''
	def stop_search(self):
		print "Closing threads.."
		for thread in self.threads:
			try:		
				thread.raiseExc(KeyboardInterrupt)
			except threading.ThreadError:
				pass

		for thread in self.threads:
			while thread.isAlive():
				time.sleep(1)
			thread.join()


	'''
	Retrieves users, calculates user scores, 
	updates score in database, and prints top 10 results.
	'''	
	def score(self):
		print "Scoring..\n"
		self.db.calculate_user_scores(self.scorer)