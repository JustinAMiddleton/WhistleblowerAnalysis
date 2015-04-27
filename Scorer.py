#from nltk.corpus import movie_reviews
from textblob import TextBlob, Word
from SearchPacket import SearchPacket
from TextPreprocessor import TextPreprocessor
from sklearn.linear_model import LogisticRegression
from math import ceil
import numpy as np

'''
This class will be fed words and their significance.
It will score a sentence based on whether those words appear,
and if they do, how significant they are to our attribute.

@author:  Justin A. Middleton
@date:    26 Apr 2015
'''
class Scorer():
	'''
	What I expect: a list of tuples
		A SearchPacket with attributes.
	'''
	def __init__(self, searchPacket):      
		self.packet = searchPacket
		self.points = []
		self.graphs = []
					
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
			
			scores.append(attrScore)
			
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
		return sumVector
		
	'''Add a datapoint to the list being maintained in this class.'''
	def add_point(self, point):
		self.points.append(point)
		
	'''
	Finds the graph for each of the individual dimensions.

	Special cases:
		If no datapoints with an actual score
		If no datapoints for an individual dimension
			...then add None instead of a logistic regression
			and this will be taken care of later.

		If two datapoints
			...then the yAxis will be [0,1]

	Otherwise:
		A logistic regression needs a classification, and here it's either 0 (not what we're looking
		for) or 1 (what we're looking for). We don't know this beforehand, but we do know the weighted
		word frequencies of each user.
			Here, this function makes a logistic regression for every individual regression for all the
		users added to the scorer.
			It then scatters a number of 1s throughout the set (based on different incrementations upon
		the standard deviation) and fits the graph to be used later.
	'''
	def fit_graphs(self):
		for i in range(0, 5):
			dimension = [x[i] for x in self.points if sum(x) > 0]
			dimension.sort()
			
			#The moment we hit an array with zero as the highest, then there's nothing to be done.
			if len(dimension) == 0 or dimension[-1] == 0:
				self.graphs.append(None)
				continue
			
			#Quick stats information.
			avg = sum(dimension, 0.0) / len(dimension)
			std = self.getSTD(dimension)

			#Sets highest as a hit.
			yAxis = [0 for i in range(0, len(dimension))]
			yAxis[-1] = 1

			#Sets a certain number of users as hits.
			numToHit = ceil(len(dimension) / 10.0)
			numToHit = int(numToHit)

			#Scatter the number of hits about the upper half of the dataset. The increments are even,
			#but the use of the standard deviation with the average will put most of the points 
			#toward the higher end.
			if numToHit > 1:
				increment = 2.0 / (numToHit - 1)
				for j in range(0, numToHit):
					self.setClosestTo(dimension, yAxis, avg + (j*increment)*std)
			else:
				if len(yAxis) >= 3:
					yAxis[-2] = 1

			#If the length is but 2, then the lower number should be 0.
			if len(yAxis) == 2:
				yAxis[0] = 0
			
			#Create the logistic regression.
			l = LogisticRegression(C=1.0)
			data = np.array([[d] for d in dimension])
			results = np.array(yAxis)
			l.fit(data, results)
			self.graphs.append(l)
			
	'''Sends the points through the numpy standard deviation finder and returns the results.'''
	def getSTD(self, dimension):
		dimensionNP = np.array(dimension)
		return np.std(dimensionNP)	

	'''
	Finds the value in dimension which is equal to or just above limit.
	If limit is already considered a hit, set the one below it as a hit.
	If that one is a hit too, oh well, consider it a loss. We don't want to go back
	all the way.
	'''
	def setClosestTo(self, dimension, yAxis, limit):
		for i in range(1, len(dimension)):
			if dimension[i] >= limit:
				if yAxis[i] == 0:
					yAxis[i] = 1
				else:
					yAxis[i-1] = 1
				break		
		
	'''
	Once the regression has been calculated, this will retrieve the probability
	of matching the "1" class by using the "predict_proba" function from the
	sklearn logistic regression class.

	What this really does is get a weighted average of every attribute and its
	own logistic regression.
	
	point:	list of five numbers, representing the user's total score, with each
					number being the score for each attribute
	'''
	def get_prob(self, point):
		prob = 0
		ctr = 0
		
		for i, attr in zip(range(0, 5), self.packet.getAttributes()):
			attrWeight = attr.get_attr_weight_num()
			logit = self.graphs[i]
			#If logit is None, then we had determined there was nothing to be gained
			#from this regression. Perhaps it had no valuable datapoints, or not enough.
			if logit is None:
				continue
				
			num = point[i]
			if num > 0:
				predictedProb = logit.predict_proba([num])[0][1]
				prob += predictedProb*attrWeight
			ctr += attrWeight

		if ctr > 0:
			prob /= ctr

		return prob