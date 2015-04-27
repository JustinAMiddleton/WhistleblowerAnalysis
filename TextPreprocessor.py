from Lemmatizer import Lemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize
import time

'''
This class takes in the text from any source and prepares it for actual scoring.
Techniques include:
	Tokenizing
	Spelling correction
	Lemmatizing / Stemming (but lemmatizing for now)

Author: Justin A. Middleton
'''
class TextPreprocessor:
	def __init__(self, text):
		self.lemmatizer = Lemmatizer()
		
		self.raw = text
		self.tokens = []
		self.spellchecked = []
		self.lemmatized = []
		
		self.preprocess(text)
		
	'''
	Sort of the main driver for this class. Will send the text through
	each of the individual steps enumerated at the top-level comment.
	'''
	def preprocess(self, text):
		preprocessed = {}
		text = text.lower()
		preprocessed["original"] = text.lower()
		
		t0 = time.time()
		self.tokens = word_tokenize(text)
		
		stop = stopwords.words("english")
		self.tokens = [word for word in self.tokens if word not in stop] #clear it of pesky stop words
		#self.spellchecked = self.spellcheck(self.tokens)
		self.lemmatized = self.lemmatizer.lemmatizeTokens(self.tokens)#self.lemmatizer.stem(self.tokens) #
		
	'''
	Uses the textblob library to attempt to check the speller.
	The textblob checker is based on Peter Norvig's implementation: http://norvig.com/spell-correct.html
	It has about a 70% accuracy.
	@PROBLEM: This takes way too long. Commented out.
	
	def spellcheck(self, tokens):
		newTokens = []
		for word in tokens:
			w = Word(word)
			#The first result at index 0 will always be the one with the highest probability.
			spelling = w.spellcheck()[0] 
			
			#Make the assumption that if the checker is not 80% confident, 
			#then it's just slang or something we don't know.
			if spelling[1] > .80:
				newTokens.append(spelling[0])
			else:
				newTokens.append(word)
			
		return newTokens
	'''
		
	'''getters'''
	def get_raw(self):
		return self.raw
		
	def get_tokens(self):
		return self.tokens
		
	def get_spellchecked(self):
		return self.spellchecked
		
	def get_lemmatized(self):
		return self.lemmatized
		
	'''A sort of default getter so I don't have to keep changing things elsewhere.'''
	def get_words(self):
		return self.lemmatized