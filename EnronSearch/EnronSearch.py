import os
import emailformat
import re
from datetime import datetime
from nltk.tokenize import sent_tokenize
from dbFacade import dbFacade
from Scorer import Scorer


class EnronSearch:

	def __init__(self, word_deck=None, db=None, scorer=None, email_main_directory=None):
		if not isinstance(word_deck, list):
			raise TypeError('word_deck must be a list')
		elif not isinstance(word_deck[0], str):
			raise TypeError('word deck must contain strings')
		elif not isinstance(db, dbFacade):
			raise TypeError('dbFacade instance required')
		elif not isinstance(scorer, Scorer):
			raise TypeError('Scorer instance required')
		elif not isinstance(email_main_directory, str):
			raise TypeError('directory must be a string')

		self.word_deck = word_deck
		print self.word_deck
		self.email_main_directory = email_main_directory + '/'
		self.db = db
		self.scorer = scorer
		self.total_users = 0
		self.total_emails = 0
		self.total_sentences_matched = 0
		self.start_time = datetime.now()
		self.end_time = datetime.now()

	def search_enron(self):

		my_sent_folders = [r'/sent/', r'/sent_items/']  # there are more sent folders than this, let's start small though

		user_directory_list = os.listdir(self.email_main_directory)

		for user_dir in user_directory_list:
			for my_sent_folder in my_sent_folders:
				for sent_folder, no_directories, email_files in os.walk(self.email_main_directory + user_dir + my_sent_folder):
					for email_file_name in email_files:
						#The extra forward slash is in case there is another folder inside the sent folder.
						email_file = open((sent_folder + "/" + email_file_name), 'r')
						email = email_file.read()
						email_file.close()
						self.process_email(email, user_dir)
						self.total_emails += 1
			print user_dir

			self.db.add_user(user_dir, 0, 'Enron')
			self.total_users += 1

		self.end_time = datetime.now()

	def process_email(self, email, user):

		email = emailformat.format_email(email)  # remove the junk

		index = 0

		for word in self.word_deck:
			if word.lower() in email.lower():
				sentences = self.extract_sentences(index, email)

				for sentence in sentences:
					sentence = EnronSearch.clean_sentence(sentence)
					scores = self.scorer.score(sentence)
					if sum(scores) > 0:
						self.db.add_post(user, 'Enron', sentence.replace("'", "''"), word, scores)
						self.total_sentences_matched += 1
				break
			index += 1

	def extract_sentences(self, word_index, email):
		"""This method returns a generator of sentences that contain the
		word passed in as a parameter"""
		words_to_search = self.word_deck[word_index:]
		sentence_list = sent_tokenize(email)

		for sentence in sentence_list:
			for word_to_search in words_to_search:
				if word_to_search.lower() in sentence.lower():
					yield sentence
					break

	@staticmethod
	def clean_sentence(sentence):
		space_pattern = re.compile('\s{2,}')
		sentence = sentence.replace('\n', ' ')
		sentence = re.sub(space_pattern, ' ', sentence)
		return sentence