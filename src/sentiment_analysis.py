import json
import random

import nltk.classify
from nltk import *
from sklearn.svm import SVC

"""
This part of code holds the unsupervised learning approaches for sentiment analysis 
"""


def readWordsLexicon(path):
    """
    Load lexicon of german words
    :param path:
    :return:
    """
    lexicon = dict()  # holding the sentiment score of each word
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            lexicon[line.split(' ')[0]] = float(line.split(' ')[1])  # store the sentiment score
            # of the current word (between 0 and 1)
    return lexicon


def readEmoticonsLexicon(path):
    """
    Load lexicon of emoticons
    :param path:
    :return:
    """
    lexicon = set()  # set for quick look-up
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            lexicon.add(line.strip())  # add the current emoticon to the set
    return lexicon


def getTweetsLabels():
    """
    Get the list of labels of tweets (positive, neutral, negative)
    :return:
    """
    path = "../resources/labeled_tweets.json"
    tweets = []  # store the list of tweets
    labels = []  # store the list of corresponding labels
    with open(path, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            tweet = json.loads(line)
            tweets.append(tweet['Tweet'])  # add the text of the tweet to the list of tweets
            labels.append(tweet['Sentiment'])  # add the sentiment value (class\label) to the list of labels
            line = f.readline()
    return tweets, labels


def lexiconBasedSentimentPrediction(taggedTweets):
    """
    Perform lexicon based sentiment prediction
    :param taggedTweets:
    :return:
    """
    pos_words_path = "../resources/positive-words.txt"
    neg_words_path = "../resources/negative-words.txt"
    pos_emoticons_path = "../resources/emoticons_lexicon_pos.txt"
    neg_emoticons_path = "../resources/emoticons_lexicon_neg.txt"

    pos_words_lex = readWordsLexicon(pos_words_path)  # load the lexicon of positive german words with scores
    neg_words_lex = readWordsLexicon(neg_words_path)  # load the lexicon of negative german words with scores
    pos_emoticons_lex = readEmoticonsLexicon(pos_emoticons_path)  # load the lexicon of positive emoticons
    neg_emoticons_lex = readEmoticonsLexicon(neg_emoticons_path)  # load the lexicon of negative emoticons

    result = dict()  # Map the id of each tweet to its corresponding sentiment value
    cnt = 0  # counter over the list of tagged tweets
    for taggedTweet in taggedTweets:
        tweet_len = len(taggedTweet)
        sum_words_pos = sum([pos_words_lex[tok] if tok in pos_words_lex else 0 for tok, pos in taggedTweet])
        # sum of scores of positive words in the current tweet
        sum_words_neg = sum([neg_words_lex[tok] if tok in neg_words_lex else 0 for tok, pos in taggedTweet])
        # sum of scores of negative words in the current tweet
        num_emoticons_pos = sum([1 if tok in pos_emoticons_lex else 0 for tok, pos in taggedTweet])
        # sum of scores of positive emoticons in the current tweet
        num_emoticons_neg = sum([1 if tok in neg_emoticons_lex else 0 for tok, pos in taggedTweet])
        # sum of scores of negative emoticons in the current tweet
        score = ((float(1 / 3) * sum_words_pos + float(2 / 3) * num_emoticons_pos) - (
            float(1 / 3) * sum_words_neg + float(2 / 3) * num_emoticons_neg)) / float(tweet_len)
        # compute the weighted sentiment score for the current tweet
        result[cnt] = 0  # set the sentiment value initially to 0 //neutral
        if score > 0.0:
            result[cnt] = 1  # set the sentiment value to 1 //positive
        elif score < 0.0:
            result[cnt] = -1  # set the sentiment value to -1 //negative
        cnt = cnt + 1
    return result


"""
This part of code holds the (semi)supervised learning approaches for sentiment analysis 
This part was based on https://www.ravikiranj.net/posts/2012/code/how-build-twitter-sentiment-analyzer/
Code snippets were taken from the link above.
"""

feature_list = []  # store the list of features


def getAllWords(train):
    """
    Get all the words in the training Data
    :param train:
    :return:
    """
    all_words = []  # store the list of all words in the training data
    for (words, sentiment) in train:
        all_words += words  # add the list of words (tokens) in the current tweet from the training dataset
    return all_words


def getFeatureList(word_list):
    """
    Get the feature list using a frequency based approach
    :param word_list:
    :return:
    """
    word_list = FreqDist(word_list)  # calculate the frequency distribution over the list of words
    return [w for (w, c) in word_list.most_common(1000) if len(w) > 2]  # remove words with length less than 2


def extract_features(tweet):
    """
    Extract the features for the tweet
    :param tweet:
    :return:
    """
    tweet_words = set(tweet)  # set of all words in the tweet
    features = {}  # dict to store the features vector
    for word in feature_list:
        features['contains(%s)' % word] = (word in tweet_words)  # boolean feature set to true if the set of all words
        # in the tweet contains the current word from the feature list
    if len(features) == 0:
        features['contains(%s)' % list(tweet_words)[0]] = True
    return features


def buildClassifiers(train, performCrossValidation=False, num_folds=0):
    """
    Train classifiers and perform cross validation on demand by splitting training set into training and test set
    based on the number of folds
    :param train:
    :param performCrossValidation:
    :param num_folds:
    :return:
    """
    training_dataset = [(extract_features(t), c) for (t, c) in train]  # prepare the training set by extracting features

    classifierNaiveBayes = NaiveBayesClassifier.train(training_dataset)  # train the NaiveBayes classifier
    print('----1----')
    classifierSVML = nltk.classify.SklearnClassifier(SVC(kernel='linear')).train(training_dataset)  # train the SVM
    # classifier with a linear kernel
    print('----2----')
    classifierSVMP = nltk.classify.SklearnClassifier(SVC(kernel='poly')).train(training_dataset)  # train the SVM
    # classifier with a polynomial kernel
    print('----3----')
    classifierMaxEntropy = nltk.classify.MaxentClassifier.train(training_dataset, max_iter=10)  # train the maximum
    # entropy classifier with 10 iterations
    print('----4----')

    if performCrossValidation is True:
        subset_size = int(len(train) / num_folds)  # define the size of each subset
        # initialize the average accuracy for each classifier
        accuracy1 = 0
        accuracy2 = 0
        accuracy3 = 0
        accuracy4 = 0
        for i in range(num_folds):
            print('\n')
            print("Cross Validation number " + str(i+1))
            training_set = [(extract_features(d), c) for (d, c) in
                            train[:i * subset_size] + train[i * subset_size + subset_size:]]  # get the training set
            testing_set = [(extract_features(d), c) for (d, c) in
                           train[i * subset_size:i * subset_size + subset_size]]  # get the testing set

            classifierNaiveBayes = NaiveBayesClassifier.train(training_set)  # Naive Bayes Classifier
            print("Naive Bayes Accuracy: " + str(classify.accuracy(classifierNaiveBayes, testing_set)))
            accuracy1 += classify.accuracy(classifierNaiveBayes, testing_set)

            classifierSVML.train(training_set)  # Support Vector Machine with linear kernel
            print("SVM with linear kernel Accuracy: " + str(classify.accuracy(classifierSVML, testing_set)))
            accuracy2 += classify.accuracy(classifierSVML, testing_set)

            classifierSVMP.train(training_set)  # Support Vector Machine with polynomial kernel
            print("SVM with polynomial kernel Accuracy: " + str(classify.accuracy(classifierSVMP, testing_set)))
            accuracy3 += classify.accuracy(classifierSVMP, testing_set)

            classifierMaxEntropy.train(training_set, max_iter=10, trace=0)  # Maximum entropy classifier
            print("MaxEntropy Accuracy: " + str(classify.accuracy(classifierMaxEntropy, testing_set)))
            accuracy4 += classify.accuracy(classifierMaxEntropy, testing_set)

        print("Accuracy of Naive Bayes: " + str(accuracy1 / num_folds))
        print("Accuracy of SVC Linear: " + str(accuracy2 / num_folds))
        print("Accuracy of SVM Poly: " + str(accuracy3 / num_folds))
        print("Accuracy of Maximum Entropy: " + str(accuracy4 / num_folds))

    return classifierNaiveBayes, classifierSVML, classifierSVMP, classifierMaxEntropy


def performTesting(test, classifiers):
    """
    Predict the class label for each classifier
    :param test:
    :param classifiers:
    :return:
    """
    _accuracy_ = [0] * len(classifiers)  # holding the accuracy of each classifier
    pos = 0
    for tweet, sentiment in test:
        cnt = 0
        for classifier in classifiers:  # iterate over all trained classifiers
            predictedSentiment = classifier.classify(extract_features(tweet))  # predict the sentiment value using
            # the corresponding classifier
            if sentiment == predictedSentiment:  # in case of match, increment the counter used to calculate accuracy
                _accuracy_[cnt] += 1
            cnt += 1
        pos += 1
    _accuracy_ = [item / float(len(test)) for item in _accuracy_]  # counter scaling by the size of the test set
    # to get the accuracy of each classifier
    print(_accuracy_)


def sentimentAnalysis(train, test):
    """
    Perform (semi)supervised sentiment analysis
    :param train:
    :param test:
    :return:
    """
    global feature_list

    word_list = getAllWords(train)  # get the list of all words in the training dataset
    feature_list = getFeatureList(word_list)  # get the feature list from the word_list
    num_folds = 10  # number of folds for K-folds cross validation
    train = random.sample(train, len(train))  # randomize the list of training dataset
    classifiers = buildClassifiers(train, True, num_folds)  # train classifiers
    performTesting(test, classifiers)  # perform testing and compute accuracy of each classifier


if __name__ == "__main__":
    print("done")
