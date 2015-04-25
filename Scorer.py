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
		#self.logit = LogisticRegression(C=1.0)
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
		
	'''Finds the graph for each of the individual dimensions.'''
	def fit_graphs(self):
		for i in range(0, 5):
			dimension = [x[i] for x in self.points if sum(x) > 0]
			dimension.sort()
			
			#The moment we hit an array with zero as the highest, then there's nothing to be done.
			if len(dimension) == 0:
				pass
			if dimension[-1] == 0:
				self.graphs.append(None)
				continue
			
			avg = sum(dimension, 0.0) / len(dimension)
			maxVal = dimension[-1]
			std = self.getSTD(dimension)

			yAxis = [0 for i in range(0, len(dimension))]
			yAxis[-1] = 1

			dimension.append(avg + std)
			yAxis.append(1)
			dimension.append(maxVal - std)
			yAxis.append(1)
			
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
	Fits a logistic regression to the datapoints that we have stored in self.points.
	
	def make_graph(self):
		testPoints, testResults = self.random_results()		
		datapoints = np.array(testPoints)
		resultpoints = np.array(testResults)
		self.logit.fit(datapoints, resultpoints)
	
	
	Currently assigns the 0s (not a whistleblower/whatever we're looking for) and the 1s (a whistleblower/whatever
	we're looking for) for the simulation to get a sample of results.
	The name is such because it was originally going to be kinda of random (the sum of scores was going to be considered
	so that the 1s would skew toward higher results), but we also need the results to be reliable for every time we run
	a search on a dataset. So I have this in-between thing now.
	
	def random_results(self):
		sortedPoints = sorted(self.points, key=lambda point: sum(point))
		sortedPoints = [p for p in sortedPoints if sum(p) > 0]
		maxes = [0,0,0,0,0]
		for point in self.points:
			print point
			for i in range(0, len(point)):
				if point[i] > maxes[i]:
					maxes[i] = point[i]
					
		points = []
		results = []
		for point in self.points:
			if sum(point) == 0:
				continue
				
			set = False
			points.append(point)
			for i in range(0, len(points)):
				if point[i] == maxes[i]:
					set = True
					results.append(1)
					break
			if not set:
				results.append(0)
				
		points.append(maxes)
		results.append(1)
		print maxes
				
		return points, results
	'''
		
	'''
	Once the regression has been calculated, this will retrieve the probability
	of matching the "1" class by using the "predict_proba" function from the
	sklearn logistic regression class.
	
	point:	list of five numbers, representing the user's total score, with each
					number being the score for each attribute
	'''
	def get_prob(self, point):
		prob = 0
		ctr = 0
		
		for i, attr in zip(range(0, 5), self.packet.getAttributes()):
			attrWeight = attr.get_attr_weight_num()
			logit = self.graphs[i]
			if logit is None:
				continue
				
			num = point[i]
			predictedProb = logit.predict_proba([num])[0][1]
			prob += predictedProb*attrWeight
			ctr += attrWeight
				
		return prob / ctr

	
		'''
		pointArr = np.array([point])
		probs = self.logit.predict_proba(pointArr)
		return probs[0][1]
		'''