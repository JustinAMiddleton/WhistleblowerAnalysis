import unittest
from dbFacade import dbFacade

class test_dbFacade(unittest.TestCase):

	@classmethod
	def setUpClass(self):
		self.db = dbFacade()
		self.db.connect()
		self.db.create_keyspace_and_schema()
		
	def test_add_user_and_get_users(self):
		self.db.add_user("Charlie", 100, "Twitter")
		users = self.db.get_users()
		self.assertEquals(len(users), 1)
		self.assertEquals(users[0].username, "Charlie")
		self.assertEquals(users[0].score, 100.0)
		self.assertEquals(users[0].website, "Twitter")
		
	def test_add_user_raise_exception_when_no_keyspace(self):
		db = dbFacade()
		db.connect()
		try:
			db.add_user("Charlie", 100, "Twitter")
			self.fail()
		except Exception as e:
			"Invalid query" in str(e)
		
	def test_add_post_and_get_posts(self):
		self.db.add_post("Charlie", "Twitter", "Hi there", "Hi", [100, 90, 80, 70, 60])
		posts = self.db.get_posts("Charlie")
		self.assertEquals(len(posts), 1)
		self.assertEquals(posts[0].username, "Charlie")
		self.assertEquals(posts[0].website, "Twitter")
		self.assertEquals(posts[0].content, "Hi there")
		self.assertEquals(posts[0].query, "Hi")
		self.assertEquals(posts[0].score1, 100.0)
		self.assertEquals(posts[0].score2, 90.0)
		self.assertEquals(posts[0].score3, 80.0)
		self.assertEquals(posts[0].score4, 70.0)
		self.assertEquals(posts[0].score5, 60.0)
		
	def test_add_post_raise_exception_when_no_keyspace(self):
		db = dbFacade()
		db.connect()
		try:
			db.add_post("Charlie", "Twitter", "Hi there", "Hi", 100)
			self.fail()
		except Exception as e:
			"Invalid query" in str(e)
	
	def test_get_scored_users_should_return_in_descending_score_order(self):
		self.db.insert_user_score("Charlie", 100, "Twitter")
		self.db.insert_user_score("Tom", 200, "Twitter")
		self.db.insert_user_score("Jaime", 150, "Twitter")
		users = self.db.get_scored_users()
		self.assertEquals(len(users), 3)
		self.assertEquals(users[0].score, 200.0)
		self.assertEquals(users[1].score, 150.0)
		self.assertEquals(users[2].score, 100.0)		
		
	@classmethod
	def tearDownClass(self):
		self.db.delete_keyspace()
		self.db.close()
	

if __name__ == '__main__':
    unittest.main()
