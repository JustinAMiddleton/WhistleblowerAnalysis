import twitter
from Scorer import Scorer

'''
This class holds data representing one tweet received by
searching the Twitter API.

@author: Brenden Romanowski
@date: 24 Feb 2015
'''
class Tweet(object):

	def __init__(self, api_data=None, scorer=None):
		if api_data is None:
			raise TypeError('api_data argument required')
		elif scorer is None:
			raise TypeError('scorer argument required')
		elif not isinstance(scorer, Scorer):
			raise TypeError('scorer argument must be Scorer instance')
		elif not isinstance(api_data, dict):
			raise TypeError('api_data argument must be a dictionary')
		elif not 'user' in api_data:
			raise TypeError('api_data dict must contain a "user" key')
		elif not 'text' in api_data:
			raise TypeError('api_data dict must contain a "text" key')
		elif not 'screen_name' in api_data['user']:
			raise TypeError('api_data dict must contain "screen_name" key')

		self.username = api_data['user']['screen_name'].encode('utf-8')
		self.content = api_data['text'].encode('utf-8').replace("'", "''")
		self.score = self.score_post(scorer)

	def score_post(self, scorer):
		try:
			return scorer.score(self.content)
		except UnicodeDecodeError:
			self.content = self.content.decode('utf-8')
			return scorer.score(self.content)
