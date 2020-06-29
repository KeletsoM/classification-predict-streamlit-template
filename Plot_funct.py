import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud


def tweet_occurence_graph(dataframe, sentiment, pattern='twitter_handle', top_n=5, color='darkblue'):
    """
    Returns a horizontal bar graph of the top most active topic of interest 
    in a specified sentiment group.
    
    Parameters
    -----------
    dataframe: DataFrame
        Dataframe consisting of tweet messages and their respective sentiment
    sentiment: int, str
        A sentiment value based on category
    pattern: {'twitter_handle', 'hashtags'} default 'twitter_handle'
        A regex pattern to identify a twitter handle, hashtags or any topic of interest
        ```twitter_handle``` is to identify twitter handles.
        ```hashtags``` is for identifying hashtags in a tweet message.
    top_n: int, default=5
        The top n occurence in accordance with the specified regex pattern.
        
    Returns
    --------
    Bar graph
       A horizontal bar graph   
    """
    sns.set_style('darkgrid')
    # I.D twitter handle and hashtags patterns
    twitter_handle_pattern = r'@\w*\d*'
    hashtag_pattern = r'#\w*\d*'
    
    df_filtered = dataframe[dataframe['sentiment'] == sentiment]
    
    # Create a dictionary of twitter handles and number of occurances per sentiment group
    frequency = {}
    
    for tweet in df_filtered['message']:

        if pattern == 'twitter_handle':
            identified_patterns = re.findall(twitter_handle_pattern, tweet)
        elif pattern == 'hashtags':
            identified_patterns = re.findall(hashtag_pattern, tweet)
        else:
            identified_patterns = re.findall(pattern, tweet)
        
        for item in identified_patterns:
            if item in frequency.keys():
                frequency[item] += 1
            else:
                frequency[item] = 1
    
    temp_df = pd.DataFrame(data=frequency.values(), index=frequency.keys(), columns=['Occurences'])
    temp_df.sort_values(by='Occurences', ascending=False, inplace=True)
    temp_df = temp_df[:top_n]
    
    target = {2:'News', 1:'Pro', 0:'Neutral', -1:'Anti'}
    
    plt.figure(figsize=(11, 8))
    sns.barplot(x='Occurences', y=temp_df.index, color=color, data=temp_df)
    plt.title(f'Number of Occurences for {target[sentiment]} category')


def character_length(data):
    """
    Returns a graph showing distribution of character length on each sentiment

     Parameters
    -----------
    data: DataFrame.
    
    """
    sns.set_style('darkgrid')
    fig, ax = plt.subplots(figsize=(8, 7))
    charlen=data['message'].str.len()
    charlen_sentiment=pd.concat([pd.DataFrame(charlen),data['sentiment']],axis=1)
    sns.kdeplot(charlen_sentiment['message'][charlen_sentiment['sentiment'] == -1],label='anti')
    sns.kdeplot(charlen_sentiment['message'][charlen_sentiment['sentiment'] == 0],label='neutral')
    sns.kdeplot(charlen_sentiment['message'][charlen_sentiment['sentiment'] == 1],label='pro')
    sns.kdeplot(charlen_sentiment['message'][charlen_sentiment['sentiment'] == 2],label='news')

def wordcount(data):
        """
        Returns a graph showing distribution of word count for each sentiment

        Parameters
        -----------
        data: DataFrame.
        
        """
        sns.set_style('darkgrid')
        word_count=data['message'].apply(lambda x: len(x.split()))
        wordcount_sentiment=pd.concat([pd.DataFrame(word_count),data['sentiment']],axis=1)
        ax = sns.catplot(x="sentiment", y="message",kind='boxen', data=wordcount_sentiment)
        plt.ylabel('Word count', fontsize=12)

def plot_wordcloud(data):
        """
        Returns Wordcloud plot of top 100 words in our tweets.
        
        Parameters
        -----------
        data: DataFrame column that contains tweets.
        
        """
        plt.figure(figsize = (15,8))
        plt.imshow(WordCloud(max_words = 100 ,background_color ='white', width = 1000 , height = 600).generate(" ".join(data)) , interpolation = 'bilinear')