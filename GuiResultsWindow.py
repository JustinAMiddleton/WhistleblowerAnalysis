from Tkinter import *

'''
This class contains code for the Search Results GUI window.

@author: Brenden Romanowski
@date: 17 March 2015
'''

class ResultsWindow:

	def __init__(self, db):
		self.db = db

	def create_results_window(self):
		toplevel= Toplevel()
		toplevel.title('Results')
		toplevel.focus_set()
		toplevel.geometry('320x320-160+200')
		results_frame = Frame(toplevel)
		results_frame.pack()
		top_users = self.db.get_scored_users()
		postButtons = []
		for i in range(0,len(top_users)):
			Label(results_frame, text=i+1).grid(row=i, column=0)
	
			user = Label(results_frame, text="[%s] %s" % 
				(str(round(top_users[i]['score'],5)), top_users[i]['username']))
			user.grid(row=i, column=1, sticky=W)	
			
			postButton = Button(results_frame, text="Show Posts", 
					command=lambda i=i: self.show_posts(top_users[i]['username']))
			postButton.grid(row=i, column=2)

	def show_posts(self, username):
		root = Tk()
		root.geometry('-160+200')
		root.title("%s's Posts" % username)
		scrollbar_x = Scrollbar(root, orient=HORIZONTAL)
		scrollbar_x.pack(side=BOTTOM, fill=X)

		listbox = Listbox(root, width=50)
		listbox.pack()

		posts = self.db.get_posts(username)

		for i in range(len(posts)):
		    listbox.insert(END, posts[i]['content'].encode('utf-8'))

		listbox.config(xscrollcommand=scrollbar_x.set)
		scrollbar_x.config(command=listbox.xview)
