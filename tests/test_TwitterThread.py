import unittest
import twitter
from SocialNetworkSearch.TwitterThread import TwitterThread
from SocialNetworkSearch.TwitterAPIWrapper import TwitterAPIWrapper
from SocialNetworkSearch.Tweet import Tweet
from dbFacade import dbFacade
from Scorer import Scorer
from Attribute import Attribute
from SearchPacket import SearchPacket

'''
Unit tests for the TwitterThread class

@author: Brenden Romanowski
@date: 8 March 2015
'''

class test_TwitterThread(unittest.TestCase):
	@classmethod
	def setUpClass(self):
		words = ['pizza', 'tacos', 'burgers', 'fries']
		weights = [1,3,2,2]
		targetSentiment = [1,1,1,1]    
		self.args = { 'location' : None }
		self.query = "pizza OR tacos OR burgers OR fries"
	
		self.db = dbFacade()
		self.db.connect()
		self.db.create_keyspace_and_schema()
		attr = Attribute("attr", 1, words, weights, targetSentiment)
		self.scorer = Scorer(SearchPacket([attr]))

	@classmethod
	def tearDownClass(self):
		self.db.delete_keyspace()
	
	def test_create_instance_with_valid_arguments(self):
		thread = TwitterThread(self.db, self.scorer, self.query, self.args)
		self.assertIsInstance(thread, TwitterThread)

	def test_create_instance_without_required_arguments(self):	
		try:
			thread = TwitterThread()
			self.fail()
		except TypeError:
			pass

	def test_create_instance_with_invalid_db_argument(self):
		try:
			thread = TwitterThread(None, self.scorer, self.query, self.args)
			self.fail()
		except TypeError:
			pass
			
	def test_create_instance_with_invalid_scorer_argument(self):
		try:
			thread = TwitterThread(self.db, None, self.query, self.args)
			self.fail()
		except TypeError:
			pass

	def test_create_instance_with_invalid_query_argument(self):
		try:
			thread = TwitterThread(self.db, self.scorer, None, self.args)
			self.fail()
		except TypeError:
			pass

	def test_create_instance_with_invalid_args_argument(self):
		try:
			thread = TwitterThread(self.db, self.scorer, self.query, None)
			self.fail()
		except TypeError:
			pass

	def test_create_instance_with_invalid_args_argument(self):
		try:
			thread = TwitterThread(self.db, self.scorer, self.query, None)
			self.fail()
		except TypeError:
			pass

	def test_create_instance_with_no_location_argument(self):
		invalid_args = {}
		try:
			thread = TwitterThread(self.db, self.scorer, self.query, invalid_args)
			self.fail()
		except KeyError:
			pass

if __name__ == '__main__':
    unittest.main()
