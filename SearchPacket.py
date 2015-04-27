'''
SearchPacket! Made of individual attributes. This will be passed into the scorer for some 
sweet scoring action. Radical, dude!

@author:  Justin A. Middleton
@date:    12 March 2015
'''
from nltk import word_tokenize
from Attribute import Attribute
from Lemmatizer import Lemmatizer
from nltk.corpus import stopwords

class SearchPacket:
	'''
	Creates the search packet by passing in a list of attributes from
	the GUI interaction!
	
	Once the attributes are in, the packet sanitizes them to check for
	any that have errors (e.g. duplicate names, empty strings for words)
	and either ignore the words or ignore the attribute altogether.
	
	What I expect: a LIST of attributes.
	'''
	def __init__(self, attributes):
		self.attributes = []
		self.lemma = Lemmatizer()
	
		for attr in attributes:
			#If the attribute cannot be sanitized to an acceptable degree, skip it.
			try:
				sanitized = self.sanitizeAttribute(attr)
			except ValueError, e:
				continue
				
			self.attributes.append(sanitized)
				
		if len(attributes) < 1:
			raise ValueError("__init__: No valid attributes to search.")
					
	'''
	Turns a rough attribute from the GUI into one that has exactly as many words
	as it needs.
		attr: dirty attribute from the GUI
		returns: clean attribute, without any invalid words
	'''
	def sanitizeAttribute(self, attr):
		if attr.get_name() is None or attr.get_name() == "":
			raise ValueError("sanitizeAttribute: Invalid name for attribute.")
		if attr.get_name() in [a.get_name() for a in self.attributes]:
			raise ValueError("sanitizeAttribute: Duplicate name for an attribute.")
		if attr.get_attr_weight_num() < 1 or attr.get_attr_weight_num() > 3:
			raise ValueError("sanitizeAttribute: Bad attribute weight.")
			
		dirtyWords = attr.get_words()
		dirtyWeights = attr.get_weights()
		dirtySents = attr.get_sentiments()
		
		if dirtyWords is None or dirtyWeights is None or dirtySents is None:
			raise ValueError("sanitizeAttribute: Unassigned values in attribute.")
		if len(dirtyWords) != len(dirtyWeights) or len(dirtyWords) != len(dirtySents):
			raise ValueError("sanitizeAttribute: list length mismatch.")
			
		cleanWords,	cleanWeights, cleanSents = self.cleanInfoLists(dirtyWords, 
			dirtyWeights, dirtySents)
			
		if len(cleanWords) < 1:
			raise ValueError("sanitizeAttribute: no valid words in attribute.")
			
		return Attribute(attr.get_name(), attr.get_attr_weight_num(),
			cleanWords, cleanWeights, cleanSents)
	
	'''
	Removes any invalid combinations from the lists. This includes those that
	have bad numbers for weights and sents or words with invalid names.
		dirtyWords:	list of str
		dirtyWeights:	list of int (between 1 and 3 inclusive)
		dirtySents:	list of int (between -1 and 1 inclusive)
	Returns three lists with all bad combinations removed.
	
	All three must be processed at the same time so corresponding information
	can be discarded if any part of it is bad.
	'''
	def cleanInfoLists(self, dirtyWords, dirtyWeights, dirtySents):
		cleanWords = []
		cleanWeights = []
		cleanSents = []

		'''
		Stop words: a list of the most commonly used words in the english language
	  	I remove them because keeping them will bloat the running time and the words
		probably won't give us any of the information we want.
		'''
		stop = stopwords.words("english")
		for word, weight, sent in zip(dirtyWords, dirtyWeights, dirtySents):
			word = word.lower()
			word = self.lemma.lemmatizeTokens([word])[0] #self.lemma.stem([word])[0]#
			if word in cleanWords or word in stop or word == "":
				continue
			if weight < 1 or weight > 3:
				continue
			if sent < -1 or sent > 1:
				continue
				
			cleanWords.append(word)
			cleanWeights.append(weight)
			cleanSents.append(sent)	

		return cleanWords, cleanWeights, cleanSents
	
	def getAttributes(self):
		return self.attributes
			
	'''
	Get the query from all search terms inside. Just "OR"s them all together.
	Example: ["one", "two", "three"] and ["four, "five", "six"] will yield
						"one OR two OR three OR four OR five OR six"
	'''
	def getQuery(self):
		attributeQueries = []
		for attr in self.attributes:
			query = " OR ".join(attr.get_words())
			attributeQueries.append(query)
		finalQuery = " OR ".join(attributeQueries)
		return finalQuery
		