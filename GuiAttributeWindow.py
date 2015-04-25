from Tkinter import *

'''
This class contains code for the Define Attribute GUI window.

@author: Brenden Romanowski
@date: 17 March 2015
'''

class AttributeWindow:
	
	def create_attribute_window(self, attributes, attribute_labels, index):
		self.attribute_labels = attribute_labels
		self.define_attribute(attributes[index], index)

	def define_attribute(self, attribute, index):
		self.toplevel= Toplevel()
		self.toplevel.title('Define Attribute')
		self.toplevel.focus_set()
		self.toplevel.geometry('450x270-160+200')
		self.attribute_frame = Frame(self.toplevel)
		self.attribute_frame.pack()

		values = self.create_attribute_controls(attribute)

		set_attribute = lambda: self.set_attribute_values(attribute, values, index)

		Button(self.attribute_frame, text="Save", command=set_attribute).grid(row=7, column=0,pady=10, padx=5)			

	def clear_attribute(self, index):
		self.attributes[index] = Attribute()
		self.attribute_labels[index]['text'] = "Attribute " + str(index+1)

	def create_attribute_controls(self, attribute):	
		wordBoxes = []
		weightBoxes = []
		sentimentBoxes = []

		new_attribute = True
		if attribute.words is not None:
			new_attribute = False

		Label(self.attribute_frame, text="Name").grid(row=0, column=0, pady=5)
		nameBox = Entry(self.attribute_frame)
		nameBox.grid(row=0, column=1, pady=5)
		nameBox.insert(0, attribute.name)

		Label(self.attribute_frame, text="Weight").grid(row=1, column=0, pady=5)
		attrWeightStr = StringVar(self.toplevel)
		if new_attribute:
			attrWeightStr.set("Medium")
		else:
			attrWeightStr.set(attribute.get_attr_weight())
		attrWeight = OptionMenu(self.attribute_frame, attrWeightStr, "High", "Medium", "Low")
		attrWeight.grid(row=1, column=1, sticky=W)

		for i in range(1,6):
			Label(self.attribute_frame, text="Word "+str(i)).grid(row=i+1, column=0)
			wordBox = Entry(self.attribute_frame)
			wordBox.grid(row=i+1, column=1)

			weightStr = StringVar(self.toplevel)
			sentimentStr = StringVar(self.toplevel)
	
			weightBox = OptionMenu(self.attribute_frame, weightStr, "High", "Medium", "Low")
			sentimentBox = OptionMenu(self.attribute_frame, sentimentStr, "Positive", "Neutral", "Negative")

			weightBox.config(width=7)
			sentimentBox.config(width=7)
			weightBox.grid(row=i+1, column=2)
			sentimentBox.grid(row=i+1, column=3)

			if new_attribute:
				weightStr.set("Weight")
				sentimentStr.set("Sentiment")
			else:
				wordBox.insert(0, attribute.get_word(i-1))
				weightStr.set(attribute.get_weight(i-1))
				sentimentStr.set(attribute.get_sentiment(i-1))

			wordBoxes.append(wordBox)
			weightBoxes.append(weightStr)
			sentimentBoxes.append(sentimentStr)

		return [wordBoxes, weightBoxes, sentimentBoxes, nameBox, attrWeightStr]
	
	def set_attribute_values(self, attribute, values, index):
		words = self.get_control_values(values[0])
		weights = self.get_control_values(values[1])
		sentiments = self.get_control_values(values[2])
	
		attribute.set_attr_weight(values[4].get())
		attribute.set_words(words)
		attribute.set_weights(weights)
		attribute.set_sentiments(sentiments)
		
		#This is below the rest because set_name will choose a name from the words
		#if no name is actually provided.
		attribute.set_name(values[3].get())

		self.attribute_labels[index]['text'] = attribute.name

	def get_control_values(self, controls):
		values = []
		for control in controls:
			value = control.get()
			values.append(value)
		return values
