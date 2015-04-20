'''
Created on Feb 16, 2015

@author: Randy
'''

from Tkinter import *
import ttk
import tkMessageBox
import threading
from tkFileDialog import *
from Attribute import Attribute
from SearchPacket import SearchPacket
from GuiThread import GuiThread
from GuiAttributeWindow import AttributeWindow
from GuiResultsWindow import ResultsWindow
from EnronSearch.EnronInterface import EnronInterface
import Queue as queue

class App:
	
	attributes = []
	args = {}

	def __init__(self, master): 
		self.frame = Frame(master)
		self.frame.pack()
		self.frame.grid(pady=15, padx=15)
		master.title("Whistleblower Analysis")
		master.geometry('320x430-625+200')
		self.root = master

		self.initialize_attributes()
		self.create_main_window_controls()

	def search(self):
		self.args['folder_location'] = self.folder_input_box.get()
		if self.args['folder_location'] == "":
			tkMessageBox.showinfo("Oops", "Must set a folder directory")	
			return 0 

		self.start_button.config(state = DISABLED)
		self.thread = GuiThread(EnronInterface(), 
					self.attributes, 
					self.args, 
					self.que)
		self.thread.start()
		
		while self.thread.interface == None:
			time.sleep(1)

		self.interface = self.thread.interface
		self.stop_button.config(state = NORMAL)

	def stop(self):
		self.stop_button.config(state = DISABLED)

		if self.thread.isAlive():
			self.thread.stop()
			self.thread.join()
	
		self.secondary_thread = threading.Thread(target=self.score_results)
		self.secondary_thread.start()
		self.root.after(50, self.check_que)

	def score_results(self):
		self.interface.score(self.que)
		self.show_results_window()
		self.start_button.config(state = NORMAL)

	def check_que(self):
		while True:
		    try: x = self.que.get_nowait()
		    except queue.Empty:
			self.root.after(25, self.check_que)
			break
		    else:
			current_value = self.progress_var.get()
			self.progress_var.set(current_value + x)
			if x == 100:
			    break

	def initialize_attributes(self):
		pass		
		for i in range(0, 5): 			
			self.attributes.append(Attribute())

	def set_enron_directory(self):
		self.folder_input_box.delete(0, END)
		self.folder_input_box.insert(0, askdirectory(parent=self.frame))

	def show_attribute_window(self, attribute, index):
		attr_window = AttributeWindow()
		attr_window.create_attribute_window(self.attributes, self.attribute_labels, index)

	def show_results_window(self):
		results_window = ResultsWindow(self.interface.db)
		results_window.create_results_window()	

	def clear_attribute(self, index):
		self.attributes[index] = Attribute()
		self.attribute_labels[index]['text'] = "Attribute " + str(index+1)

	def create_main_window_controls(self):
		self.create_main_window_file_input_controls()
		self.create_main_window_attribute_controls()
		self.create_main_window_command_controls()

		self.progress_var = IntVar()
		self.progress_var.set(0)
		self.progress = ttk.Progressbar(self.frame, 
						variable=self.progress_var, 
						mode='determinate')
		self.progress.grid(row=11, column=0, columnspan=4,sticky=W+E)
		self.que = queue.Queue()

	def create_main_window_file_input_controls(self):
		file_input_frame = Frame(self.frame)
		file_input_frame.grid(row=0, column=0, rowspan=3, columnspan=3, padx=10, sticky=W, pady=4)
		Label(file_input_frame, text="Folder Location", font = "Verdana 10 bold").grid(row=0, column=0, pady=4, sticky=W)
		self.folder_input_box = Entry(file_input_frame, width=24)
		self.folder_input_box.grid(row=1, column=0)
		self.dir_opt = options = {}
		browse = Button(file_input_frame, text="Browse")
		browse.grid(row=1, column=1)
		browse.config(command=lambda: self.set_enron_directory())

	def create_main_window_attribute_controls(self):
		self.defAtt = "Define Attribute"
		attributes = Frame(self.frame)
		attributes.grid(row=3, column=0, rowspan=3, columnspan=2, padx=10, sticky=N)

		Label(attributes, text="Attributes", font = "Verdana 10 bold").grid(row=0, column=0, pady=4, sticky=W)

		self.create_attribute_label_controls(attributes)
		self.create_attribute_button_controls(attributes)

	def create_attribute_label_controls(self, frame):
		self.attribute_labels = []
		for i in range(1, len(self.attributes)+1):
			attr_label = Label(frame, text="Attribute "+str(i), width=10, anchor=W)
			attr_label.grid(row=i, column=0, sticky=W, padx=10)	
			self.attribute_labels.append(attr_label)
	
	def create_attribute_button_controls(self, frame):
		for i in range(0, 5):
			Button(frame, text=self.defAtt, command=lambda i=i: self.show_attribute_window(self.attributes[i], i)).grid(
				    row=i+1, column=1, pady=4)
			Button(frame, text="X", command=lambda i=i: self.clear_attribute(i)).grid(row=i+1, column=2, padx=5)

	def create_main_window_command_controls(self):
		buttons = Frame(self.frame)
		buttons.grid(row=8, column=0, rowspan=2, columnspan=4, pady=20)

		self.start_button = Button(buttons, text="Search", 
				command=self.search, font = "Verdana 10")
		self.start_button.grid(row=8)
		
		self.stop_button = Button(buttons, text="Stop", 
				command=self.stop, font = "Verdana 10")
		self.stop_button.grid(row=9, pady=5)
		self.stop_button.config(state = DISABLED)

root=Tk()
app = App(root)
root.mainloop()
