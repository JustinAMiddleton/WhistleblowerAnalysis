'''
SearchPacket! Made of individual attributes. This will be passed into the scorer for some 
sweet scoring action. Radical, dude!

@author:  Justin A. Middleton
@date:    8 March 2015
'''
import unittest
from SearchPacket import SearchPacket
from Attribute import Attribute

class test_SearchPacket(unittest.TestCase):
	def setUp(self):    
		self.attr1 = Attribute()
		self.attr1.name = "one"
		self.attr1.set_words(["one", "two", "three"])
		self.attr1.set_weights_nums([1,2,3])
		self.attr1.set_sentiments_nums([1,1,1])
		
		self.attr2 = Attribute()
		self.attr2.name = "two"
		self.attr2.set_words(["four", "five", "six"])
		self.attr2.set_weights_nums([3,2,1])
		self.attr2.set_sentiments_nums([-1,1,-1])
		
		self.attrs = [self.attr1, self.attr2]
		self.packet = SearchPacket(self.attrs)
		
#sanitize
	#pass
	def test000_000_sanitizeAttribute(self):
		test = Attribute("test", 1, ["one", "two"], [1,1], [1,1])
		sanitize = self.packet.sanitizeAttribute(test)
		self.assertEquals(sanitize, test)
		
	def test000_001_sanitize_oneBadWord(self):
		test = Attribute("test", 1, ["one", ""], [1,1], [1,1])
		correct = Attribute("test", 1, ["one"], [1], [1])
		self.assertEquals(self.packet.sanitizeAttribute(test), correct)
		
	def test000_002_sanitize_oneBadWeight(self):
		test = Attribute("test", 1, ["one", "two"], [1,0], [1,1])
		correct = Attribute("test", 1, ["one"], [1], [1])
		self.assertEquals(self.packet.sanitizeAttribute(test), correct)
		
	def test000_003_sanitize_oneBadSent(self):
		test = Attribute("test", 1, ["one", "two", "three"], [1,1, 1], [1, 0, 2])
		correct = Attribute("test", 1, ["one", "two"], [1,1], [1,0])
		self.assertEquals(self.packet.sanitizeAttribute(test), correct)
		
	def test000_004_sanitize_oneDupWord(self):
		test = Attribute("test", 1, ["one", "one"], [1,1], [1,1])
		correct = Attribute("test", 1, ["one"], [1], [1])
		self.assertEquals(self.packet.sanitizeAttribute(test), correct)
		
	#fail
	def test000_900_sanitize_badName(self):
		correctError = "sanitizeAttribute: "
		try:
			test = Attribute("", 1, ["one", "two"], [1,1], [1,1])
			self.packet.sanitizeAttribute(test)
			self.fail("Error: no error!")
		except ValueError, e:
			self.assertEqual(correctError, str(e)[:len(correctError)])
		except Exception, e:
			self.fail(str(e))
			
	def test000_901_sanitize_dupName(self):
		correctError = "sanitizeAttribute: "
		try:
			test = Attribute("one", 1, ["one", "two"], [1,1], [1,1])
			self.packet.sanitizeAttribute(test)
			self.fail("Error: no error!")
		except ValueError, e:
			self.assertEqual(correctError, str(e)[:len(correctError)])
		except Exception, e:
			self.fail(str(e))
			
	def test000_902_sanitize_badWeight(self):
		correctError = "sanitizeAttribute: "
		try:
			test = Attribute("test", 0, ["one", "two"], [1,1], [1,1])
			self.packet.sanitizeAttribute(test)
			self.fail("Error: no error!")
		except ValueError, e:
			self.assertEqual(correctError, str(e)[:len(correctError)])
		except Exception, e:
			self.fail(str(e))
			
	def test000_903_sanitize_noWords(self):
		correctError = "sanitizeAttribute: "
		try:
			test = Attribute("one", 1, None, [1,1], [1,1])
			self.packet.sanitizeAttribute(test)
			self.fail("Error: no error!")
		except ValueError, e:
			self.assertEqual(correctError, str(e)[:len(correctError)])
		except Exception, e:
			self.fail(str(e))
			
	def test000_904_sanitize_noWeights(self):
		correctError = "sanitizeAttribute: "
		try:
			test = Attribute("one", 1, ["one", "two"], None, [1,1])
			self.packet.sanitizeAttribute(test)
			self.fail("Error: no error!")
		except ValueError, e:
			self.assertEqual(correctError, str(e)[:len(correctError)])
		except Exception, e:
			self.fail(str(e))
			
	def test000_905_sanitize_NoSents(self):
		correctError = "sanitizeAttribute: "
		try:
			test = Attribute("one", 1, ["one", "two"], [1,1], None)
			self.packet.sanitizeAttribute(test)
			self.fail("Error: no error!")
		except ValueError, e:
			self.assertEqual(correctError, str(e)[:len(correctError)])
		except Exception, e:
			self.fail(str(e))
			
	def test000_906_sanitize_NoGoodWords(self):
		correctError = "sanitizeAttribute: "
		try:
			test = Attribute("one", 1, ["", ""], [1,1], [1,1])
			self.packet.sanitizeAttribute(test)
			self.fail("Error: no error!")
		except ValueError, e:
			self.assertEqual(correctError, str(e)[:len(correctError)])
		except Exception, e:
			self.fail(str(e))
			
#getAttributes
	#pass
	def test100_000_getAttributes(self):
		index = 0
		for attr in self.packet.getAttributes():
			self.assertEquals(attr, self.attrs[index])
			index += 1

#getQuery
	def test200_000_getQuery(self):
		self.assertEquals(self.packet.getQuery(), "one OR two OR three OR four OR five OR six")
			
if __name__ == '__main__':
	unittest.main()  
		