
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
import re 
import string
from nltk.stem import WordNetLemmatizer

# Plot functions
from Plot_funct import tweet_occurence_graph
from Plot_funct import wordcount
from Plot_funct import character_length
from Plot_funct import plot_wordcloud
# Vectorizer
news_vectorizer = open("resources/vector.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data

raw = pd.read_csv("resources/train.csv")

#some functions for text preprocessing for wordclouds
def pred(prediction):
	if prediction == -1:
		st.success("Your text is categorized as being against the belief of man-made climate change :thumbsdown:")
	elif prediction == 0:
		st.success("Your text is categorized as Neutral, i.e. you are neither for or against the belief of man-made climate change")
	elif prediction == 1:
		st.success("Your text is categorized as supporting the belief of man-made climate change :thumbsup:")	
	else: st.success("Your text is categorized as factual news") 


# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title, made with markdown -
	st.markdown("<h1 style='text-align: center; color: #3498DB;'>Climate Change Tweet Classifier</h1>", unsafe_allow_html=True)
	
	# Creating sidebar with radio -
	# you can create multiple pages this way
	options = ["Classify A Tweet",'Exploratory Data analysis',"About classification models","Background information", "What is climate change?","About this App"]
	st.sidebar.image('resources/imgs/markus.jpg',use_column_width= True)
	st.sidebar.title(":cloud: Tweet Classification :cloud:")
	selection = st.sidebar.radio("What would you like to see?", options)
	st.sidebar.info("Hi! This is an App developed by JHB Classification Team SS3. For more information check out the 'About this App' page")
	


	# Building EDA page
	if selection == 'Exploratory Data analysis':
		st.image('resources/imgs/Datavisual.jpeg',use_column_width= True)
		st.markdown("<h2 style='text-align: center; color: #3498DB;'>Exploratory Data Analysis</h2>", unsafe_allow_html=True)
		#shows a pie chart with the distribution of the data
		sent_dict = {"Anti":-1,"Neutral":0,"Pro":1,'Factual news':2}
		if st.checkbox("Show distribution of the data"):	
			raw['sentiment'].value_counts().plot(kind='pie', autopct='%.1f', labels=['Pro','News','Neutral','Anti'])
			st.pyplot()
			st.info("The categories in the above data is clearly unbalanced. We can see that 52.3% of the tweets supports the belief of man-made climate change (Pro), 21.1% are based on factual news about climate change (News), 17.6% of the tweets are rather neutral on the subject (Neutral), and 9.1% do not believe in man-made climate change (Anti). ")

		if st.checkbox("Distribution of word count"):
			st.info("The graph below shows the distribution of word counts")
			wordcount(raw)
			st.pyplot()

		if st.checkbox("Distribution of character length"):
			st.info("The following graph shows the distribution of the character lengths associated to the various sentiment groups")
			character_length(raw)
			st.pyplot()

		if st.checkbox("Word Cloud"):
			plot_wordcloud(raw['message'])
			st.pyplot()

		#Build the most mentiond twitter handle
		if st.checkbox("Show most mentioned Twitter account"):

			st.info("The graph below shows the most occurring twitter handle amongst the different sentiment groups")
			opt2 = st.selectbox("Select sentiment group",['Anti','Neutral','Pro','Factual news'],key='Pro')
			tweet_occurence_graph(raw, sentiment=sent_dict[opt2], top_n=10, color='cadetblue')
			st.pyplot()
		
		if st.checkbox("Show most occurring hashtags"):
			st.info("The graph below shows the most occurring hashtags amongst the different sentiment groups")
			opt = st.selectbox("Select sentiment group",['Anti','Neutral','Pro','Factual news'])
			tweet_occurence_graph(raw, sentiment= sent_dict[opt],pattern="hashtags", top_n=10, color='cadetblue')
			st.pyplot()

		

	# Building out the "Background information" page
	if selection == "Background information":
		st.subheader("Background information")
		# You can read a markdown file from supporting resources folder
		st.markdown(open('resources/info.md').read())
		#shows a sample of raw data 
		
		if st.checkbox("Show a sample of the raw data"): # data is hidden if box is unchecked
			#st.write(raw['message'].head().values,) # will write the df to the page
			st.table(raw[['message', 'sentiment']].head())

	#Building the Educational page		
	if selection == "What is climate change?":
		st.image('resources/imgs/climate-cold.jpg',use_column_width= True)
		st.markdown("<h2 style='text-align: center; color: #3498DB;'>What Is Climate Change?</h2>", unsafe_allow_html=True)
		st.markdown(open("resources/What_is_climate_change.md").read())
		video_file = open('resources/imgs/climate_change.mp4', 'rb')
		video_bytes = video_file.read()
		st.video(video_bytes)
		
	#Building out the classification models page
	if selection == "Classify A Tweet":
		st.image('resources/imgs/Speech.jpg',use_column_width= True)
		st.subheader("Let's classify your tweets!")
		tweet_text = st.text_area("Enter your text","Type Here")
		st.info("Which classification model would you like to use?")

		#selection of linear regression model
		if st.button(("Linear SVC Regression"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			predictor = joblib.load(open(os.path.join("resources/linear_svc.pkl"),"rb"))
			prediction = predictor.predict(vect_text)
			pred(prediction)
			

		# selection of random forest model 
		if st.button("Logistic regression"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			prediction = predictor.predict(vect_text)
			pred(prediction)
			

		if st.button("Voting Classifier"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			predictor = joblib.load(open(os.path.join("resources/voting_classifier.pkl"),"rb"))
			prediction = predictor.predict(vect_text)
			pred(prediction)
			
		if st.button("Compliment Naive Bayes"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			predictor = joblib.load(open(os.path.join("resources/complement_naive_bayes.pkl"),"rb"))
			prediction = predictor.predict(vect_text)
			pred(prediction)
			

		

	# Building About model page
	if selection == "About classification models":
		st.image("resources/imgs/ai-header.png")
		st.markdown("<h2 style='text-align: center; color: #3498DB;'>What Is Classification?</h2>", unsafe_allow_html=True)
		st.markdown(open("resources/About_class_models.md").read())

	# Building out the "About this App" page
	if selection == "About this App":
		st.image('resources/imgs/blackboard.jpg',use_column_width=True)
		st.subheader("About this App")
		# You can read a markdown file from supporting resources folder
		st.markdown(open('resources/About_file.md').read())
		st.image('resources/imgs/EDSA_logo.png')
		st.image('resources/imgs/kaggle-logo.png', use_column_width= True)		


# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()

