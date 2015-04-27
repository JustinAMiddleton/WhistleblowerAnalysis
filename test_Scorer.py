import unittest
from Attribute import Attribute
from SearchPacket import SearchPacket
from Scorer import Scorer

class test_Scorer(unittest.TestCase):
	def setUp(self):
		self.words = ['pizza', 'tacos', 'burgers', 'fries']
		weights = [1, 3, 2, 2]
		sentiments = [1, 1, 1, 1]
		attribute = Attribute("Attribute1", 1, self.words, weights, sentiments)
		self.packet = SearchPacket([attribute])
		self.scorer = Scorer(self.packet)
		
#init
	def test000_000_init(self):
		score = Scorer(self.packet)
		self.assertIsInstance(score, Scorer)

	def test000_900_init_noparam(self):
		correctError = "__init__: "
		try:
			score = Scorer()
			self.fail("Error: no error!")
		except ValueError, e:
			self.assertEqual(correctError, str(e)[:len(correctError)])
		except Exception, e:
			self.fail(str(e))
			
	def test000_900_init_invalidparam(self):
		correctError = "__init__: "
		try:
			score = Scorer("Invalid")
			self.fail("Error: no error!")
		except ValueError, e:
			self.assertEqual(correctError, str(e)[:len(correctError)])
		except Exception, e:
			self.fail(str(e))
			
#score
	def test100_000_score_pos(self):
		text = "I love tacos and fries."
		#tacos has weight 3, fries 2
		self.assertEquals(self.scorer.score(text), [5, 0, 0, 0, 0])
		
	def test100_001_score_neg(self):
		text = "I hate tacos and fries."
		self.assertEquals(self.scorer.score(text), [0, 0, 0, 0, 0])
		
	def test100_002_score_mixed(self):
		weights = [1, 3, 2, 2]
		sentiments = [0, 0, 0, 0]
		attribute = Attribute("Attribute1", 1, self.words, weights, sentiments)
		packet = SearchPacket([attribute])
		score = Scorer(packet)
		text = "I hate tacos and love fries."
		#tacos has weight 3, fries 2
		self.assertEquals(score.score(text), [5, 0, 0, 0, 0])
		
	def test100_003_score_multiattr(self):
		attr1 = Attribute("1", 1, ["burgers"], [1], [0])
		attr2 = Attribute("2", 1, ["fries"], [2], [0])
		attr3 = Attribute("3", 1, ["tacos"], [3], [0])
		packet = SearchPacket([attr1, attr2, attr3])
		score = Scorer(packet)		
		text = "I hate tacos and love fries and burgers are okay."
		self.assertEquals(score.score(text), [1, 2, 3, 0, 0])
		
	def test100_900_score_nottext(self):
		correctError = "score: "
		try:
			self.scorer.score(1)
			self.fail("Error: no error!")
		except ValueError, e:
			self.assertEqual(correctError, str(e)[:len(correctError)])
		except Exception, e:
			self.fail(str(e))
			
#sum_posts
	def test200_000_sum(self):
		post1 = {"score1": 1, "score2": 0, "score3": 0, "score4": 0, "score5": 0}
		post2 = {"score1": 1, "score2": 1, "score3": 0, "score4": 0, "score5": 0}
		post3 = {"score1": 1, "score2": 1, "score3": 1, "score4": 0, "score5": 0}
		post4 = {"score1": 1, "score2": 1, "score3": 1, "score4": 1, "score5": 0}
		post5 = {"score1": 1, "score2": 1, "score3": 1, "score4": 1, "score5": 1}
		vectors = [post1, post2, post3, post4, post5]
		self.assertEquals(self.scorer.sum_posts(vectors), [5,4,3,2,1])
		
	def test200_000_sum_baddim(self):
		post1 = {"score2": 0, "score3": 0, "score4": 0, "score5": 0}
		post2 = {"score1": 1, "score2": 1, "score3": 0, "score4": 0, "score5": 0}
		post3 = {"score1": 1, "score2": 1, "score3": 1, "score4": 0, "score5": 0}
		post4 = {"score1": 1, "score2": 1, "score3": 1, "score4": 1, "score5": 0}
		post5 = {"score1": 1, "score2": 1, "score3": 1, "score4": 1,}
		vectors = [post1, post2, post3, post4, post5]
		self.assertEquals(self.scorer.sum_posts(vectors), [3,3,2,1,0])
		

			
if __name__ == '__main__':
	unittest.main()  
	
