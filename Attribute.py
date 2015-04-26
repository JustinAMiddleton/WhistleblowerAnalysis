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
		
	'''Just checks to see that all values are equal.'''
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
	
	'''
	name:	A string by which this attribute will be referred.
			Special case:	The name "Attribute" will indicate that the name was not set in the GUI.
										In this case, set the name to the first highest-value word.
	'''
	def set_name(self, name):
		#We don't really want to change the attribute name, here.
		if name != "Attribute":
			self.name = name
			
		#This is to give each attribute a name of its own that isn't the default "Attribute"
		if (name == "Attribute" or name == "") and self.words is not None and self.weights is not None:
			newName = self.generate_name(self.words, self.weights)
			#If no words are set, then we don't have anything to change it to.
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
		if len(weights) > 0 and not isinstance(weights[0], basestring):
			raise ValueError("set_weights: This function expects strings.")
	
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
		if len(weights) > 0 and not isinstance(weights[0], int):
			raise ValueError("set_weights_nums: This function expects numbers.")
			
		self.weights = weights

	'''Expects a list of strings.'''
	def set_sentiments(self, sentiments):
		if len(sentiments) > 0 and not isinstance(sentiments[0], basestring):
			raise ValueError("set_sentiments: This function expects strings.")	

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
		if len(sentiments) > 0 and not isinstance(sentiments[0], int):
			raise ValueError("set_weights_nums: This function expects numbers.")	

		self.sentiments = sentiments

	'''Returns the word in an attribute from a given index (int)'''
	def get_word(self, index):
		if not isinstance(index, int) or index >= len(self.words):
			raise ValueError("get_word: Index out of bounds.")
		
		return self.words[index]

	'''
	index:	int, index of word in internal array
	returns:	word "High, "Medium", or "Low"
	'''
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

	'''
	index:	int, index of word in internal array
	returns:	word "Positive", "Neutral", or "Negative"
	'''
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
			
	'''Returns # of words in attribute'''
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
	def generate_name(self, words, weights):
		if words is None or weights is None:
			raise ValueError("generate_name: words and weights must not be none.")
		if len(words) != len(weights):
			raise ValueError("generate_name: words and weights must be the same size.")
			
		max = 0
		name = ""
		for word, weight in zip(words, weights):
			if weight > max and word != "":
				max = weight
				name = word

		return name
