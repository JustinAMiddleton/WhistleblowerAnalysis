from cassandra.cluster import Cluster
import cassandra.cluster
import cassandra.query
import time
import uuid

'''
This class handles database communication and should
be used instead of interacting with the database directly.

@author: Brenden Romanowski
@date: 24 Feb 2015
'''

class dbFacade(object):
	session = None
	keyspace = None
	
	def connect(self):
		while(True):
			try:
				cluster = Cluster(['131.204.27.98'])
				self.session = cluster.connect()
				break;
			except cassandra.cluster.NoHostAvailable:
				print "Connection failed, retrying.."
				time.sleep(1)
			except Exception as e:
				print str(e)
				time.sleep(1)
		print "Connected to cassandra database."
		
	def close(self):
		time.sleep(1)
		self.session.cluster.shutdown()
		self.session.shutdown()
		
	def add_user(self, username, score, website):
		self.session.execute("""
			INSERT INTO %s.users (username, score, website)
			VALUES ('%s', %s, '%s'); 
			""" % (self.keyspace, username, score, website))
			
	def add_post(self, username, website, content, query, scores):	
		self.session.execute("""
			INSERT INTO %s.posts (id, username, website, content, query, score1, score2, score3, score4, score5)
			VALUES (%s, '%s', '%s', '%s', '%s', %s, %s, %s, %s, %s);
			""" % (self.keyspace, uuid.uuid1(), username, website, content, query, \
				scores[0], scores[1], scores[2], scores[3], scores[4]))
	
	def get_users(self):	
		query = "SELECT * FROM %s.users;" % self.keyspace
		statement = cassandra.query.SimpleStatement(query)
		
		users = []		
		for user in self.session.execute(statement):
			users.append(user)

		return users
	
	def get_users_dict(self):
		self.session.row_factory = cassandra.query.dict_factory
		users = self.get_users()

		return users

	def get_scored_users(self):
		results = self.session.execute("""
				SELECT * FROM %s.scored_users LIMIT 10;
				""" % self.keyspace)
		return results

	'''def get_top_ten_users(self, website):
		results = self.session.execute("""
				SELECT * FROM %s.scored_users WHERE website='%s' LIMIT 10;
				""" % (self.keyspace, website)
		return results'''

	def get_posts(self, username):
		results = self.session.execute("""
				SELECT * FROM %s.posts WHERE username='%s';
				""" % (self.keyspace, username))
		return results

	def insert_user_score(self, username, score, website):
		self.session.execute("""
				INSERT INTO %s.scored_users (username, score, website) 
				VALUES ('%s', %s, '%s');
				""" % (self.keyspace, username, float(score), website))
		
	def create_keyspace_and_schema(self):
		timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")
		self.keyspace = "search_%s" % timestamp

		self.session.execute("""
			CREATE KEYSPACE %s WITH
			replication = {'class':'SimpleStrategy','replication_factor':3};
			""" % self.keyspace)
		
		self.session.execute("""
			CREATE TABLE %s.scored_users (
				username text,
				score float,
				website text,
				PRIMARY KEY (website, score, username)
			) WITH CLUSTERING ORDER BY (score DESC);
			""" % self.keyspace)

		self.session.execute("""
			CREATE TABLE %s.users (
				username text,
				website text,
				score float,
				PRIMARY KEY (username, website)
			);
			""" % self.keyspace)
		
		self.session.execute("""
			CREATE TABLE %s.posts (
				id uuid,
				username text,
				website text,
				content text,
				query text,
				score1 float,
				score2 float,
				score3 float,
				score4 float,
				score5 float,
				PRIMARY KEY (username, id)	
			);
			""" % self.keyspace)

	def delete_keyspace(self):
		self.session.execute("""
			DROP KEYSPACE %s; """ % self.keyspace)

	def clear_database(self):
		keyspaces = self.get_keyspace_names()
		
		for keyspace in keyspaces:
			if "search" in keyspace:
				self.session.execute("""DROP KEYSPACE %s;""" % keyspace)
					
	def get_keyspace_names(self):
		keys = self.session.execute("""
			SELECT * FROM system.schema_keyspaces;"""); 
			
		keyspaces = []
		for key in keys:
			keyspaces.append(key.keyspace_name)
			
		return keyspaces
