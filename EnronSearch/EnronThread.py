import threading
import re
from CrawlThread import CrawlThread
from EnronSearch import EnronSearch
from dbFacade import dbFacade
from Scorer import Scorer

'''
@author: Brenden Romanowski
@date: 18 March 2015
'''

class EnronThread(CrawlThread):

	def __init__(self, db=None, scorer=None, query=None, args=None):
		if not isinstance(db, dbFacade):
			raise TypeError('dbFacade instance required')
		elif not isinstance(scorer, Scorer):
			raise TypeError('Scorer instance required')
		elif not isinstance(query, str):
			raise TypeError('Query must be a string')
		elif not isinstance(args, dict):
			raise TypeError('Args must be a dictionary')
		elif not 'folder_location' in args.keys():
			raise KeyError('Enron folder directory not defined in args')

		threading.Thread.__init__(self)
		self.db = db
		self.scorer = scorer
		self.query = query
		self.args = args
		newQuery = re.sub('"', '', query)
		self.word_deck = newQuery.split(" OR ")

	def run(self):
		self.search()

	def search(self):
		print self.args['folder_location']
		enron = EnronSearch(self.word_deck, self.db, self.scorer, self.args['folder_location'])

		try:
			enron.search_enron()
			print "Done"
		except KeyboardInterrupt:
			print('\n Enron: Terminated by user (search)\n')
