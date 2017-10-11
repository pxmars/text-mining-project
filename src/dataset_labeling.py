import json
import time

from aylienapiclient import textapi

from entities_identification import tweetsEntitiesMapping
from preprocessing import loadData, process


def prepare():
    """
    Get the list of filtered tweets by target entity where each item contains the tweet
    with its original attributes when downloaded from Twitter
    :return:
    """
    path = '../../Data.json'
    List = loadData(path)  # load data
    tweets = [List[i]['text'] for i in range(len(List))]  # store the text of each tweet in a list
    tweets = [process(item, False) for item in tweets]  # get the list of processed tweets
    filtered_tweets = tweetsEntitiesMapping(tweets)  # filter tweets by target entity
    ids_list = filtered_tweets[3]  # get the list of ids of the filtered tweets in the original list
    count = 0
    list_tweets = []  # store the filtered tweet objects
    for item in List:
        if count in ids_list:
            list_tweets.append(item)
        count = count + 1
    return list_tweets


def labeling():
    """
    Perform labeling using the Aylien Sentiment Analysis Rest API
    There is a limit of 1000 requests per day when choosing the free plan.
    :return:
    """
    appID = ["SdiriAppID", "PolatAppID", "MarsAppID"]  # list of app ids for each team member
    key = ["SdiriKey", "PolatKey", "MarsKey"]  # list of keys for each team member
    list_tweets = prepare()  # get the list of tweets; labeling requests
    size = 50  # the window size; number of requests to send to Aylien server at a time
    lists_tweets = [list_tweets[i:i + size] for i in range(0, len(list_tweets), size)]  # split the list of requests
    # into different lists with the same window size
    with open('../resources/labeled_tweets.json', 'w', encoding='utf8') as jsonfile:  # write the Aylien server
        # response to a file
        user_credentials_num = 0
        for pos in range(len(lists_tweets)):  # iterate over all request windows
            if pos != 0 and (pos % 20) == 0:
                user_credentials_num = user_credentials_num + 1  # change the user credentials for every
                # (20 * window size = 20 * 50 = 1000) requests
            client = textapi.Client(appID[user_credentials_num], key[user_credentials_num])  # set the client
            # credentials
            list_tweets = lists_tweets[pos]
            for tweet in list_tweets:  # iterate over all tweets in the same request window
                tweet = tweet['text']  # get the text of the tweet to label
                tweet = tweet.strip()
                if len(tweet) == 0:  # verify if the tweet text is not empty
                    print("empty tweet")
                    continue
                sentiment = client.Sentiment({'text': tweet, 'mode': 'tweet', 'language': 'de'})  # perform the request
                json_object = {
                    'Tweet': sentiment['text'],
                    'Sentiment': sentiment['polarity']
                }  # store the result in json object
                json.dump(json_object, jsonfile)  # write the json object to the file
                jsonfile.write('\n')
            time.sleep(61)  # idle time between the current request window and the next one


if __name__ == "__main__":
    labeling()
    print("done")
