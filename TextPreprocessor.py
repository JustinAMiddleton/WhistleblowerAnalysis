from Lemmatizer import Lemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from textblob import TextBlob, Word
import time

'''
This class takes in the text from any source and prepares it for actual scoring.
Techniques include:
	Tokenizing
	Spelling correction
	Lemmatizing
	
What may come:
	Stemming
'''
class TextPreprocessor:
	def __init__(self, text):
		#self.lemmatizer = Lemmatizer()
		
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
		
		#t0 = time.time()
		self.tokens = word_tokenize(text)
		#self.spellchecked = self.spellcheck(self.tokens)
		#self.lemmatized = self.lemmatizer.lemmatizeTokens(self.spellchecked)
		#print "Time:", time.time()-t0
		
	'''
	Uses the textblob library to attempt to check the speller.
	The textblob checker is based on Peter Norvig's implementation: http://norvig.com/spell-correct.html
	It has about a 70% accuracy.
	'''
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
		
	'''getters'''
	def get_raw(self):
		return raw
		
	def get_tokens(self):
		return self.tokens
		
	def get_spellchecked(self):
		return self.spellchecked
		
	def get_lemmatized(self):
		return self.lemmatized
		