import threading
import time
from SearchPacket import SearchPacket
from SearchInterface import SearchInterface
from Attribute import Attribute

'''
This class controls the backend of the Whistleblower tools.
The GUI creates a multithreading environment to allow operations
via the backend to be controlled by the GUI.

Takes in a list of Attribute objects and a dictionary of
search arguments.

@author: Brenden Romanowski
@date: 17 March 2015
'''

class GuiThread(threading.Thread):
	def __init__(self, interface, attributes, args, progress_que):
		if interface is None:
			raise TypeError('interface argument required')
		elif attributes is None:
			raise TypeError('attributes argument required')
		elif args is None:
			raise TypeError('args argument required')
		elif not isinstance(interface, SearchInterface):
			raise TypeError('interface must be a subclass of SearchInterface')
		elif not isinstance(attributes, list):
			raise TypeError('attributes argument must be a list')
		elif not isinstance(args, dict):
			raise TypeError('args argument must be a dictionary')
		elif not len(attributes) > 0:
			raise TypeError('at least one Attribute required')
		elif not isinstance(attributes[0], Attribute):
			raise TypeError('Attributes list must contain Attribute instances')

		threading.Thread.__init__(self)
		#self.interface = SearchInterface()
		self.interface = interface
		self.search_packet = SearchPacket(attributes)
		self.args = args
		self.que = progress_que

	def run(self):
		self.interface.initialize_scorer(self.search_packet)
		self.query = self.search_packet.getQuery()
		self.interface.search(self.query, self.args)
		time.sleep(1)

	def stop(self):
		self.interface.stop_search()


