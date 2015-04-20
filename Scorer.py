#from nltk.corpus import movie_reviews
from textblob import TextBlob, Word
from SearchPacket import SearchPacket
from TextPreprocessor import TextPreprocessor
from sklearn.linear_model import LogisticRegression
import numpy as np

'''
This class will be fed words and their significance.
It will score a sentence based on whether those words appear,
and if they do, how significant they are to our attribute.

@author:  Justin A. Middleton
@date:    24 Feb 2015
'''
class Scorer():
	'''
	What I expect: a list of tuples
		A SearchPacket with attributes.
	'''
	def __init__(self, searchPacket):      
		self.packet = searchPacket
		self.points = []
		self.logit = LogisticRegression(C=1.0)
					
	'''
	Scores an input sentence, currently using the pattern analyzer
	as part of text blob. 
	Ignore subjectivity. Use absolute value of polarity.
	'''
	def score(self, text):
		processed = TextPreprocessor(text)
		bagOfWords = processed.get_words() #LINE WILL CHANGE
		polarity = TextBlob(processed.get_raw()).sentiment.polarity
		scores = []
		
		for attr in self.packet.getAttributes():
			attrScore = 0
			for i in range(0, attr.get_size()):
				expectedSent = attr.get_sentiment_num(i)
				if polarity * expectedSent >= 0:
					word = attr.get_word(i)
					significance = attr.get_weight_num(i)
					attrScore += bagOfWords.count(word) * significance
			
			attrWeight = attr.get_attr_weight_num()
			scores.append(attrScore * attrWeight)
			
		#Fill it up to 5
		for i in range(len(scores),5):
			scores.append(0)
			
		return scores	
		
	'''
	Takes in the list of posts from the database and adds them all together to get the total
	score array for this user.
	This sum will be used to generate the logistic regression and then retrieve a probability from it.
	
	post:	I assume it to be a dictionary from the database, where values "score1" through "score5" are the
				scores from the above function.
	'''
	def sum_posts(self, posts):
		sumVector = [0,0,0,0,0]
		for post in posts:
			postArray = [post["score%d" % i] for i in range (1, 6)]		
			sumVector = [x+y for x, y in zip(sumVector, postArray)]
		self.points.append(sumVector)
		return sumVector
		
	'''
	Fits a logistic regression to the datapoints that we have stored in self.points.
	'''
	def make_graph(self):
		sortedPoints = sorted(self.points, key=lambda point: sum(point))
		sortedPoints = [p for p in sortedPoints if sum(p) > 0]
		testResults = self.random_results(sortedPoints)
		datapoints = np.array(sortedPoints)
		resultpoints = np.array(testResults)
		self.logit.fit(datapoints, resultpoints)
	
	'''
	Currently assigns the 0s (not a whistleblower/whatever we're looking for) and the 1s (a whistleblower/whatever
	we're looking for) for the simulation to get a sample of results.
	The name is such because it was originally going to be kinda of random (the sum of scores was going to be considered
	so that the 1s would skew toward higher results), but we also need the results to be reliable for every time we run
	a search on a dataset. So I have this in-between thing now.
	'''
	def random_results(self, points):
		results = []
		numOfOnes = int(len(points) / 10)
		for i in range(0, len(points)-numOfOnes):
			results.append(0)
		for i in range(len(points)-numOfOnes, len(points)):
			results.append(1)
		return results
		
	'''
	Once the regression has been calculated, this will retrieve the probability
	of matching the "1" class by using the "predict_proba" function from the
	sklearn logistic regression class.
	
	point:	list of five numbers, representing the user's total score, with each
					number being the score for each attribute
	'''
	def get_prob(self, point):
		pointArr = np.array([point])
		probs = self.logit.predict_proba(pointArr)
		return probs[0][1]