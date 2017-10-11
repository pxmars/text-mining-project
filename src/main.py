import pickle

import matplotlib.pyplot as plt

from Opinion import Opinion
from aspects_extraction import aspectsFromTaggedTweets, tweetAspectMapping
from entities_identification import tweetsEntitiesMapping
from opinion_summarization import summarize
from preprocessing import loadData, process
from sentiment_analysis import lexiconBasedSentimentPrediction, getTweetsLabels, sentimentAnalysis


def getData():
    """
    load Data and process it
    :return: list of Opinions, list of Aspects, list of processed tweets to work on later
    """
    pathToData = '../../data.json'
    data = loadData(pathToData)  # load Data
    tweets = [data[i]['text'] for i in range(len(data))]  # tweets before preprocessing
    processed_tweets = [process(item, False) for item in tweets]  # tweets after preprocessing
    processed_tweets_with_stopwords = [process(item, False, True) for item in
                                       tweets]  # tweets after preprocessing without removing stopwords
    filtered_tweets_result = tweetsEntitiesMapping(processed_tweets)
    filtered_tweets = filtered_tweets_result[1]  # tweets after filtering by target entity
    filtered_tweets_with_stopwords = tweetsEntitiesMapping(processed_tweets_with_stopwords)[
        1]  # tweets without removing stopwords after filtering by target entity
    tweets_entities_mapping = filtered_tweets_result[2]  # map each tweet to its target entity
    ids_list = filtered_tweets_result[3]  # keep track of the ids of filtered tweets from the complete list of tweets

    '''
        This part of code is for POS tagging the filtered tweets.
        Since the process of POS tagging is time consuming here, we used pickling to save the result once and use it
        multiple times later.
        
        # tagged_tweets = [posTag(sentence) for sentence in tweets]
        # with open("../resources/taggedTweets.txt", "wb") as fp:  # Pickling
        #    pickle.dump(tagged_tweets, fp)
    '''

    with open("../resources/taggedTweets.txt", "rb") as fp:  # Unpickling
        tagged_tweets = pickle.load(fp)  # load tagged tweets

    aspects = aspectsFromTaggedTweets(tagged_tweets)  # extract the list of Aspects using a frequency-based method
    aspects += ['job', 'wirtschaft', 'politik']
    print(aspects)
    tweets_aspects_mapping = tweetAspectMapping(tagged_tweets, aspects)  # map each tweet to its target aspect
    opinions = [
        Opinion(data[ids_list[i]]['user_screeen_name'], data[ids_list[i]]['created_at'], data[ids_list[i]]['text']) for
        i in range(len(
            filtered_tweets))]  # create a list of Opinion object that has opinionHolder, postDate and text attributes
    for key in tweets_entities_mapping:
        opinions[key].setTargetEntity(tweets_entities_mapping[key])  # set the target entity for each tweet
        opinions[key].setTargetAspect(tweets_aspects_mapping[key])  # set the target aspect for each tweet

    return opinions, aspects, filtered_tweets, tagged_tweets, filtered_tweets_with_stopwords


def visualizeAspectBasedResults(opinions, aspect=""):
    count = [0] * 7
    entity = ['cdu', 'spd', 'linke', 'grüne', 'csu', 'fdp', 'afd']
    for opinion in opinions:
        # if opinion.targetAspect != aspect:
        #     continue
        for pos in range(7):
            if opinion.targetEntity == entity[pos] and opinion.SO == 1:
                count[pos] = count[pos] + 1
    labels = 'CDU', 'SPD', 'Die Linke', 'Grüne', 'CSU', 'FDP', 'AFD'
    sizes = count
    colors = ['blue', 'red', 'brown', 'green', 'grey', 'yellow', 'orange']
    headerText = "Just a demo :)"

    # use matplotlib to plot the chart
    plt.pie(sizes, labels=labels, colors=colors, shadow=True, startangle=90)
    plt.title("Who is the winner? " + headerText)
    plt.show()


def performLexiconBasedSentimentAnalysis(data):
    """
    perform a lexicon-based sentiment analysis, summarize and visualize results
    :return: list of Opinions
    """
    opinions = data[0]
    taggedTweets = data[3]
    sentiments_mapping = lexiconBasedSentimentPrediction(
        taggedTweets)  # identify the sentiment orientation of each tweet
    for key in sentiments_mapping:
        opinions[key].setSO(sentiments_mapping[key])  # set the sentiment orientation for each tweet
    return opinions


def summarizeOpinions(opinions, aspects):
    with open('../resources/summary.txt', 'w', encoding='utf-8') as file:
        summarize(opinions, aspects, file)  # write the summary to a file


def performSupervisedSentimentAnalysis(data):
    """
    perform supervised approaches for sentiment analysis
    :return: void
    """
    processed_tweets = data[4]  # get the complete dataset to work on
    labels = getTweetsLabels()[1]  # get the labels of the dataset
    tweets = []
    for pos in range(len(processed_tweets)):
        tweets.append(
            (processed_tweets[pos], labels[pos]))  # store each tweet and its corresponding sentiment label in one list
    split_perc = 0.1  # specify the percentage of the dataset splitting into train and test sets (10% for training set)
    split_size = int(len(tweets) * split_perc)  # specify the size of the split
    train_tweets, test_tweets = tweets[split_size:], tweets[:split_size]  # split the dataset into train and test sets
    sentimentAnalysis(train_tweets, test_tweets)  # perform the sentiment analysis based on supervised approaches


if __name__ == "__main__":
    Data = getData()
    Aspects = Data[1]
    Opinions = performLexiconBasedSentimentAnalysis(Data)
    summarizeOpinions(Opinions, Aspects)
    visualizeAspectBasedResults(Opinions)  # visualize results based on the specified aspect
    # performSupervisedSentimentAnalysis(Data)

    print("Done")
