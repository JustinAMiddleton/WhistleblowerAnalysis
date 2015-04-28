import unittest
from Attribute import Attribute
from SearchPacket import SearchPacket
from Scorer import Scorer
from sklearn.linear_model import LogisticRegression

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
		
	def test200_900_sum_notlist(self):
		correctError = "sum_posts: "
		try:
			self.scorer.sum_posts("Invalid.")
			self.fail("Error: no error!")
		except ValueError, e:
			self.assertEqual(correctError, str(e)[:len(correctError)])
		except Exception, e:
			self.fail(str(e))
		
	def test200_900_sum_emptylist(self):
		correctError = "sum_posts: "
		try:
			self.scorer.sum_posts("Invalid.")
			self.fail("Error: no error!")
		except ValueError, e:
			self.assertEqual(correctError, str(e)[:len(correctError)])
		except Exception, e:
			self.fail(str(e))

	def test200_900_sum_notdictinlist(self):
		correctError = "sum_posts: "
		try:
			self.scorer.sum_posts(["Invalid."])
			self.fail("Error: no error!")
		except ValueError, e:
			self.assertEqual(correctError, str(e)[:len(correctError)])
		except Exception, e:
			self.fail(str(e))

#add_point
	def test300_000_addpoint(self):
		self.scorer.add_point([1,1,1,1,1])
		self.assertEquals(len(self.scorer.points), 1)
		self.scorer.add_point([2,2,2,2,2])
		self.assertEquals(len(self.scorer.points), 2)

	def test300_900_addpoint_notlist(self):
		correctError = "add_point"
		try:
			self.scorer.add_point("Invalid")
			self.fail("Error: no error!")
		except ValueError, e:
			self.assertEqual(correctError, str(e)[:len(correctError)])
		except Exception, e:
			self.fail(str(e))


#fit_graphs
	def test400_000_fit(self):
		dim1 = [i*2 for i in range(0, 30)]
		dim2 = [100-i*3 for i in range(0, 30)]
		dim3 = [50+25*(-1)**i for i in range(0, 30)]
		dim4 = [37*i % 100 for i in range(0, 30)]
		dim5 = [0 for _ in range(0, 30)]
		for point in zip(dim1, dim2, dim3, dim4, dim5):
			self.scorer.add_point(list(point))
		self.scorer.fit_graphs()
		for i in range(0, 4):
			self.assertIsInstance(self.scorer.graphs[i], LogisticRegression)
		self.assertIsNone(self.scorer.graphs[4])
	
#def getSTD
	def test500_000_std(self):
		std = self.scorer.getSTD([2,4,6,8,10])
		self.assertEquals("%.3f" % std, "2.828")

	def test500_001_std2(self):
		std = self.scorer.getSTD([1,9,4,27,1,33,100,95,3,1,1,0,0,0,0])
		self.assertEquals("%.3f" % std, "32.569")

	def test500_900_std_notlist(self):
		correctError = "getSTD: "
		try:
			self.scorer.getSTD("Invalid.")
			self.fail("Error: no error!")
		except ValueError, e:
			self.assertEqual(correctError, str(e)[:len(correctError)])
		except Exception, e:
			self.fail(str(e))
			
	def test500_901_std_notints(self):
		correctError = "getSTD: "
		try:
			self.scorer.getSTD(["Invalid"])
			self.fail("Error: no error!")
		except ValueError, e:
			self.assertEqual(correctError, str(e)[:len(correctError)])
		except Exception, e:
			self.fail(str(e))

	def test500_902_std_empty(self):
		correctError = "getSTD: "
		try:
			print self.scorer.getSTD([])
			self.fail("Error: no error!")
		except ValueError, e:
			self.assertEqual(correctError, str(e)[:len(correctError)])
		except Exception, e:
			self.fail(str(e))

#setClosestTo
	def test600_000_setClosestTo_notincluded(self):
		dimension = [2,4,6,8,10]
		yAxis = [0,0,0,0,0]
		self.scorer.setClosestTo(dimension, yAxis, 8)
		self.assertEquals(yAxis, [0,0,0,1,0])

	def test600_001_setClosestTo_notincluded(self):
		dimension = [2,4,6,8,10]
		yAxis = [0,0,0,0,0]
		self.scorer.setClosestTo(dimension, yAxis, 7)
		self.assertEquals(yAxis, [0,0,0,1,0])	

	def test600_002_setClosestTo_alreadyset(self):
		dimension = [2,4,6,8,10]
		yAxis = [0,0,0,1,0]
		self.scorer.setClosestTo(dimension, yAxis, 7)
		self.assertEquals(yAxis, [0,0,1,1,0])	

	def test600_003_setClosestTo_lowest(self):
		dimension = [2,4,6,8,10]
		yAxis = [0,0,0,0,0]
		self.scorer.setClosestTo(dimension, yAxis, 1)
		self.assertEquals(yAxis, [0,1,0,0,0])

	def test600_004_setClosestTo_outofrange(self):
		dimension = [2,4,6,8,10]
		yAxis = [0,0,0,0,0]
		self.scorer.setClosestTo(dimension, yAxis, 12)
		self.assertEquals(yAxis, [0,0,0,0,1])

	def test600_900_notequallens(self):
		correctError = "setClosestTo: "
		try:
			print self.scorer.setClosestTo([1,2,3], [0, 0], 3)
			self.fail("Error: no error!")
		except ValueError, e:
			self.assertEqual(correctError, str(e)[:len(correctError)])
		except Exception, e:
			self.fail(str(e))

	def test600_901_notenough(self):
		correctError = "setClosestTo: "
		try:
			print self.scorer.setClosestTo([1], [0], 3)
			self.fail("Error: no error!")
		except ValueError, e:
			self.assertEqual(correctError, str(e)[:len(correctError)])
		except Exception, e:
			self.fail(str(e))

#get_prob
	def test700_000_getprob(self):
		dim1 = [i*2 for i in range(0, 30)]
		dim2 = [100-i*3 for i in range(0, 30)]
		dim3 = [50+25*(-1)**i for i in range(0, 30)]
		dim4 = [37*i % 100 for i in range(0, 30)]
		dim5 = [0 for _ in range(0, 30)]
		for point in zip(dim1, dim2, dim3, dim4, dim5):
			self.scorer.add_point(list(point))
		self.scorer.fit_graphs()
		self.assertEquals("%.3f" % self.scorer.get_prob([10, 10, 10, 10, 10]), "0.156")

	def test700_001_getprob_zero(self):
		dim1 = [i*2 for i in range(0, 30)]
		dim2 = [100-i*3 for i in range(0, 30)]
		dim3 = [50+25*(-1)**i for i in range(0, 30)]
		dim4 = [37*i % 100 for i in range(0, 30)]
		dim5 = [0 for _ in range(0, 30)]
		for point in zip(dim1, dim2, dim3, dim4, dim5):
			self.scorer.add_point(list(point))
		self.scorer.fit_graphs()
		self.assertEquals(self.scorer.get_prob([0,0,0,0,0]), 0)

	def test700_900_getprob_pointnotlist(self):
		correctError = "get_prob: "
		try:
			print self.scorer.get_prob("Invalid")
			self.fail("Error: no error!")
		except ValueError, e:
			self.assertEqual(correctError, str(e)[:len(correctError)])
		except Exception, e:
			self.fail(str(e))

	def test700_901_getprob_pointempty(self):
		correctError = "get_prob: "
		try:
			print self.scorer.get_prob([])
			self.fail("Error: no error!")
		except ValueError, e:
			self.assertEqual(correctError, str(e)[:len(correctError)])
		except Exception, e:
			self.fail(str(e))

	def test700_902_getprob_pointnotofints(self):
		correctError = "get_prob: "
		try:
			print self.scorer.get_prob(["Invalid"])
			self.fail("Error: no error!")
		except ValueError, e:
			self.assertEqual(correctError, str(e)[:len(correctError)])
		except Exception, e:
			self.fail(str(e))

	def test700_903_getprob_pointwithbaddim(self):
		correctError = "get_prob: "
		try:
			print self.scorer.get_prob([0,0,0,1])
			self.fail("Error: no error!")
		except ValueError, e:
			self.assertEqual(correctError, str(e)[:len(correctError)])
		except Exception, e:
			self.fail(str(e))

	def test700_904_getprob_graphsnotyetfit(self):
		correctError = "get_prob: "
		try:
			print self.scorer.get_prob([0,0,0,0,1])
			self.fail("Error: no error!")
		except ValueError, e:
			self.assertEqual(correctError, str(e)[:len(correctError)])
		except Exception, e:
			self.fail(str(e))

if __name__ == '__main__':
	unittest.main()  
	
