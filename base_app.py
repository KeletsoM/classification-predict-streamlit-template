
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
from wordcloud import WordCloud
import re 
import string
from nltk.stem import WordNetLemmatizer

# Plot functions
from Plot_funct import tweet_occurence_graph
# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data

raw = pd.read_csv("resources/train.csv")

#some functions for text preprocessing for wordclouds


# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title, made with markdown -
	st.markdown("<h1 style='text-align: center; color: #3498DB;'>Climate Change Tweet Classifier</h1>", unsafe_allow_html=True)
	st.image('resources/imgs/Speech.jpg',use_column_width= True)
	# Creating sidebar with radio -
	# you can create multiple pages this way
	options = ["Classify A Tweet","Background information", "About this App",'Exploratory Data analysis', "What is climate change"]
	st.sidebar.image('resources/imgs/markus.jpg',use_column_width= True)
	st.sidebar.title(":cloud: Tweet Classification :cloud:")
	selection = st.sidebar.radio("What would you like to see?", options)
	st.sidebar.info("Hi! This is an App developed by JHB Classification Team SS3. For more information check out the 'About this App' page")
	
	# Building EDA page
	if selection == 'Exploratory Data analysis':
		#shows a pie chart with the distribution of the data
		if st.checkbox("Show distribution of the data"):	
			raw['sentiment'].value_counts().plot(kind='pie', autopct='%.1f', labels=['Pro','News','Neutral','Anti'])
			st.pyplot()
			st.info("The categories in the above data is clearly unbalanced. We can see that 53.9% of the tweets supports the belief of man-made climate change (Pro), 23.0% are based on factual news about climate change (News), 14,9% of the tweets are rather neutral on the subject (Neutral), and 8.2% do not believe in man-made climate change (Anti). ")

		#
		if st.checkbox("Show Most mentioned acount"):

			st.info("The graph below shows the most mentioned account amongst people with the view that climate change is not a man made phenomenon")
			sent = st.slider('sentiment',0,3,0)
			tweet_occurence_graph(raw, sentiment=sent-1, top_n=10, color='cadetblue')
			st.pyplot()
		
		if st.checkbox("Show most ocurring hashtags"):
			st.info("The graph below shows the most mentioned account amongst people with the view that climate change is not a man made phenomenon")
			sent1 = st.slider('sentiment',0,3,1)
			tweet_occurence_graph(raw, sentiment=sent1-1,pattern="hashtags", top_n=10, color='cadetblue')
			st.pyplot()


	# Building out the "Background information" page
	if selection == "Background information":
		st.subheader("Background information")
		# You can read a markdown file from supporting resources folder
		st.markdown(open('resources/info.md').read())

		#shows a sample of raw data 
		st.subheader("Exploratory data analysis of the data used to built the models")
		if st.checkbox("Show a sample of the raw data"): # data is hidden if box is unchecked
			#st.write(raw['message'].head().values,) # will write the df to the page
			st.table(raw[['message', 'sentiment']].head())

		

			
	if selection == "What is climate change":
		st.info("An Educational video on Climate change and it's effects")
		video_file = open('resources/imgs/climate_change.mp4', 'rb')
		video_bytes = video_file.read()
		st.video(video_bytes)
		
	# Building out the classification models page

	if selection == "Classify A Tweet":
		st.subheader("Let's classify your tweets!")
		tweet_text = st.text_area("Enter your text","Type Here")
		st.info("Which classification model would you like to use?")

		#selection of linear regression model
		if st.button("Linear Regression"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			if prediction == -1:
				st.success("Your text is categorized as being against the belief of man-made climate change :thumbsdown:")
			elif prediction == 0:
				st.success("Your text is categorized as Neutral, i.e. you are neither for or against the belief of man-made climate change")
			elif prediction == 1:
				st.success("Your text is categorized as supporting the belief of man-made climate change :thumbsup:")	
			else: st.success("Your text is categorized as factual news") 

		#selection of random forest model 
		if st.button("Random Forest Classifier"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			if prediction == -1:
				st.success("Your text is categorized as being against the belief of man-made climate change :thumbsdown:")
			elif prediction == 0:
				st.success("Your text is categorized as Neutral, i.e. you are neither for or against the belief of man-made climate change")
			elif prediction == 1:
				st.success("Your text is categorized as supporting the belief of man-made climate change :thumbsup:")	
			else: st.success("Your text is categorized as factual news") 

	# Building out the "About this App" page
	if selection == "About this App":
		st.subheader("About this App")
		# You can read a markdown file from supporting resources folder
		st.markdown(open('resources/About_file.md').read())
		st.image('resources/imgs/EDSA_logo.png')		




# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()

