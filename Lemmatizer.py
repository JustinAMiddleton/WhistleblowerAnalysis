'''
This class takes in a text and transforms it into a lemmatized/stemmed form.
Author:	Justin A. Middleton
'''
from nltk import word_tokenize, pos_tag, map_tag, WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

class Lemmatizer():
	def __init__(self):
		self.lemmatizer = WordNetLemmatizer()
		self.stemmer = SnowballStemmer("english", ignore_stopwords=True)

	'''
	Lemmatizes every word in a sentence and then tokenizes it.	
		sentence: str
	'''
	def lemmatize(self, sentence):
		tokens = word_tokenize(sentence)
		lemmas = self.lemmatizeTokens(tokens)
		return " ".join(lemmas)
		
	'''
	Turns phrase tokens into lemmatized tokens, which means into some standard format
	as determined by the nltk lemmatizer. "Dogs" to "dog", "went" to "go", etc.	 
		tokens: list of str
	'''
	def lemmatizeTokens(self, tokens):
		tokens_tagged = pos_tag(tokens)
		#Get simple POS tags.
		tokens_simpleTags = [(word, map_tag('en-ptb', 'universal', tag)) 
			for word, tag in tokens_tagged]
		
		#Actually lemmatize.
		lemmas = []
		for token, tag in tokens_simpleTags:
			lemmatized = ""
			if tag == "VERB":
				lemmatized = self.lemmatizer.lemmatize(token, pos='v')
			elif tag == "ADJ":
				lemmatized = self.lemmatizer.lemmatize(token, pos='a')
			elif tag == "ADV":
				lemmatized = self.lemmatizer.lemmatize(token, pos='r')
			else:
				lemmatized = self.lemmatizer.lemmatize(token) #pos = 'n'
			lemmas.append(lemmatized.encode("utf-8"))
		return lemmas

	'''
	Reduce this word down to its most basic form by removing suffixes or common ending
	and finding the "root" or "stem" of the word.

	Example: "response," "responsive," and "responsivity" all stem from "respons," or 
	something similar.
	'''
	def stem(self, tokens):
		stemmed = []
		for token in tokens:
			stem = self.stemmer.stem(token)
			stemmed.append(stem.encode("utf-8"))
		return stemmed