import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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


