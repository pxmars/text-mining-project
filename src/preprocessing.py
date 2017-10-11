import json
import re

from nltk.corpus import stopwords
from nltk.stem.snowball import GermanStemmer
from nltk.tokenize import TweetTokenizer


def loadData(filePath):
    """
    Load data as a list of tuples and conserve the wanted attributes
    :param filePath:
    :return:
    """
    file = open(filePath, 'r', encoding='utf-8')  # load dataset file
    line = file.readline()
    data = []  # store a list of tweet objects
    l_unwanted = ['user_id', 'user_lang', 'id', 'lang']  # list of attributes to remove from the json object
    unwanted = set(l_unwanted)
    while line:
        tweet = json.loads(line)  # load it as Python dict
        for unwanted_key in unwanted:
            del tweet[unwanted_key]  # remove unwanted attributes
        data.append(tweet)
        line = file.readline()
    file.close()
    return data


def getStopWordSet(stopWordSetFileName=None):
    """
    Get a list of german stop words
    :param stopWordSetFileName:
    :return:
    """
    _stopWords = set(stopwords.words('german'))  # load the list of german stopwords built-in the nltk corpus
    _stopWords.add('at_user')  # add 'at_user' to the list of stopwords
    _stopWords.add('url')  # add 'url' to the list of stopwords

    if stopWordSetFileName is not None:  # in case we want to add few words from a file to the list of stopwords
        fp = open(stopWordSetFileName, 'r')  # load the file that contains extra stopwords
        line = fp.readline()
        while line:
            word = line.strip()
            _stopWords.add(word)  # add the current entry of the file to the list of german stopwords
            line = fp.readline()
        fp.close()
    return _stopWords


def removeStopWords(tweet, preserve_case=True):
    """
    Remove stop words from the tweet
    :param tweet:
    :param preserve_case:
    :return:
    """
    pathStopwords = '../resources/stopwords-de.txt'
    stopWords = getStopWordSet(pathStopwords)  # get the list of german stopwords
    if preserve_case is False:  # in case we want to write tweets in lowercase format
        tweet = tweet.lower()
    tweet = ' '.join([word for word in tweet.split() if word not in stopWords])  # just keep words
    # that are not in the german stopwords
    return tweet


def basicProcess(tweet):
    """
    Perform basic tweet preprocess
    The code was taken from https://www.ravikiranj.net/posts/2012/code/how-build-twitter-sentiment-analyzer/
    :param tweet:
    :return:
    """
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet)  # Convert www.* or https?://* to URL
    tweet = re.sub('@[^\s]+', 'AT_USER', tweet)  # Convert @username to AT_USER
    tweet = re.sub('[\s]+', ' ', tweet)  # Remove additional white spaces
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)  # Replace #word with word
    tweet = re.sub(r'([0-9]*)', "", tweet)  # remove number
    tweet = tweet.strip('\'"')  # trim
    return tweet


def stemming(tweet):
    """
    Perform word stemming using GermanStemmer
    :param tweet:
    :return:
    """
    words = tweet.split()  # split the tweet into a list of words
    stemmer = GermanStemmer()  # get the german stemmer
    stemTweet = []  # store the list of stemmed words
    for word in words:
        word = stemmer.stem(word)  # stem the word
        stemTweet.append(word)  # append it to the list of stemmed words
    return " ".join(stemTweet)  # join the list of stemmed words together


def process(tweet, preserve_case=True, preserve_stopwords=False):
    """
    Perform tweet preprocessing
    :param tweet:
    :param preserve_case:
    :param preserve_stopwords:
    :return:
    """
    tweet = basicProcess(tweet)  # perform basic preprocessing
    if preserve_stopwords is False:  # in case we want to remove stopwords from the tweet
        tweet = removeStopWords(tweet, preserve_case)
    tweet = stemming(tweet)  # perform tweet stemming
    tokens = TweetTokenizer(preserve_case).tokenize(tweet)  # tokenize the tweet to get a list of tokens
    return tokens


if __name__ == "__main__":
    print("Done")
