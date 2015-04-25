'''

Created on March 11, 2015
Last edit on April 14, 2015

@author: Brenden, Justin

'''

class Attribute():
	def __init__(self, name = "Attribute", attrWeight = 1,
			words = None, weights = None, sentiments = None):
		self.name = name
		self.attrWeight = attrWeight
		self.words = words
		self.weights = weights
		self.sentiments = sentiments
		
	def __eq__(self, other):
		if self.name != other.name:
			return False
		if self.attrWeight != other.attrWeight:
			return False
		if self.words != other.words:
			return False
		if self.weights != other.weights:
			return False
		if self.sentiments != other.sentiments:
			return False
		
		return True

	def set_name(self, name):
		#We don't really want to change the attribute name, here.
		if name != "Attribute":
			self.name = name
			
		#This is to give each attribute a name of its own that isn't the default "Attribute"
		if name == "Attribute" and self.words is not None and self.weights is not None:
			newName = self.generateName(self.words, self.weights)
			if newName != "":
				self.name = newName
		
	def set_attr_weight(self, weight):
		if weight == "High":
			self.attrWeight = 3
		elif weight == "Medium":
			self.attrWeight = 2
		else:
			self.attrWeight = 1
	
	def set_words(self, words):
		self.words = words

	'''Expects a list of strings.'''
	def set_weights(self, weights):
		newWeights = []
		for weight in weights:
			if weight == "High":
				newWeights.append(3)
			elif weight == "Medium":
				newWeights.append(2)
			else:
				newWeights.append(1)
		self.weights = newWeights
		
	'''Expects a list of numbers.'''
	def set_weights_nums(self, weights):
		self.weights = weights

	'''Expects a list of strings.'''
	def set_sentiments(self, sentiments):
		newSentiments = []
		for sentiment in sentiments:
			if sentiment == "Neutral" or sentiment == "Sentiment": #The latter happens when it goes unset.
				newSentiments.append(0)
			elif sentiment == "Positive":
				newSentiments.append(1)
			else:
				newSentiments.append(-1)
		self.sentiments = newSentiments
		
	'''Expects a list of numbers.'''
	def set_sentiments_nums(self, sentiments):
		self.sentiments = sentiments

	def get_word(self, index):
		if index >= len(self.words):
			raise ValueError("get_word: Index out of bounds.")
		
		return self.words[index]

	def get_weight(self, index):
		if index >= len(self.words):
			raise ValueError("get_word: Index out of bounds.")
	
		if self.weights[index] == 3:
			return "High"
		elif self.weights[index] == 2:
			return "Medium"
		else:
			return "Low"
			
	def get_weight_num(self, index):
		if index >= len(self.words):
			raise ValueError("get_word: Index out of bounds.")
			
		return self.weights[index]

	def get_sentiment(self, index):
		if index >= len(self.words):
			raise ValueError("get_word: Index out of bounds.")
			
		if self.sentiments[index] == 1:
			return "Positive"
		elif self.sentiments[index] == 0:
			return "Neutral"
		else:
			return "Negative"
			
	def get_sentiment_num(self, index):
		if index >= len(self.words):
			raise ValueError("get_word: Index out of bounds.")
			
		return self.sentiments[index]
			
	def get_size(self):
		if self.words is None:
			return 0
		else:
			return len(self.words)
			
	'''
	The following functions are for bulk processing in the SearchPacket class.
	'''
	
	'''Gets the attribute name.'''
	def get_name(self):
		return self.name
		
	'''Gets the attribute weight.'''
	def get_attr_weight_num(self):
		return self.attrWeight
		
	def get_attr_weight(self):
		if self.attrWeight == 3:
			return "High"
		elif self.attrWeight == 2:
			return "Medium"
		else:
			return "Low"
			
	'''Gets the entire list of attribute words.'''
	def get_words(self):
		return self.words
		
	'''Gets the entire list of attribute weights.'''
	def get_weights(self):
		return self.weights
		
	'''Gets the entire list of attribute expected sentiments.'''
	def get_sentiments(self):
		return self.sentiments
		
	def get_max_score(self):
		if self.words is None or len(self.words) == 0:
			raise ValueError("get_word: No words to calculate score.")
			
		return sum(self.weights)
		
	'''Generates a name based on the first highest weighted word.'''
	def generateName(self, words, weights):
		if words is None or weights is None:
			raise ValueError("generateName: words and weights must not be none.")
		if len(words) == 0 or len(words) != len(weights):
			raise ValueError("generateName: words and weights must be the same size.")
			
		max = 0
		name = ""
		for word, weight in zip(words, weights):
			if weight > max and word != "":
				max = weight
				name = word

		return name
