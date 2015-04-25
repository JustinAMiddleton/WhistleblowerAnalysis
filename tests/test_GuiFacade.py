import unittest
from SearchPacket import SearchPacket
from SearchInterface import SearchInterface
from Attribute import Attribute
from GuiThread import GuiThread
import Queue

class test_Tweet(unittest.TestCase):

	@classmethod
	def setUpClass(self):
		self.interface = SearchInterface()
		attribute = Attribute("Test Attribute", attrWeight = 1,
			words = ['pizza'], weights = [3], sentiments = [1])
		self.attributes = [attribute]
		self.que = Queue.Queue()

	@classmethod
	def tearDownClass(self):
		self.interface.db.delete_keyspace()
		self.interface.db.close()
		

	def test_create_instance_with_valid_parameters(self):
		thread = GuiThread(self.attributes, {}, self.que)
		self.assertIsInstance(thread, GuiThread)
	
	def test_create_instance_with_invalid_attributes_parameter(self):
		try:
			thread = GuiThread("Invalid", {})
			self.fail()
		except TypeError:
			pass

	def test_create_instance_with_invalid_args_parameter(self):
		try:
			thread = GuiThread(self.attributes, "Invalid", self.que)
			self.fail()
		except TypeError:
			pass
	
	def test_create_instance_with_missing_attributes_parameter(self):
		try:
			thread = GuiThread(args={}, progress_que=self.que)
			self.fail()
		except TypeError:
			pass

	def test_create_instance_with_missing_args_parameter(self):
		try:
			thread = GuiThread(attributes=self.attributes, progress_que=self.que)
			self.fail()
		except TypeError:
			pass

	def test_create_instance_with_empty_attributes_list(self):
		try:
			thread = GuiThread(attributes=[], args={}, progress_que=self.que)
			self.fail()
		except TypeError:
			pass

	def test_create_instance_with_attributes_list_containing_invalid_objects(self):
		try:
			thread = GuiThread(attributes=["Invalid"], args={}, progress_que=self.que)
			self.fail()
		except TypeError:
			pass

if __name__ == '__main__':
    unittest.main()

