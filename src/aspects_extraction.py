import pickle
from collections import Counter

import gensim


def posTag(toked_tweet):
    """
    Perform tweet POS tagging
    :param toked_tweet:
    :return:
    """
    pathTagger = "../resources/nltk_german_classifier_data.pickle"
    with open(pathTagger, 'rb') as f:
        tagger = pickle.load(f)  # unpickle the german POS tagger
    return tagger.tag(toked_tweet)  # tag the toked sentence


def aspectsFromTaggedTweets(taggedTweets):
    """
    Retrieve aspects from tagged tweets using a frequency-based approach
    :param taggedTweets:
    :return:
    """
    word_counter = Counter()  # catch the most common nouns in the tweets
    for tweet in taggedTweets:
        for word, pos in tweet:
            if (pos == 'NE' or pos == 'NN') and len(word) > 3:  # if the word is a proper noun or a noun
                # and has length grater than 3,  increment its counter
                word_counter[word] += 1
    stop_list = []  # store the list of unwanted words that figured to be aspects using the frequency-based approach
    path = '../resources/stop_aspects.txt'
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            stop_list.append(line.strip())  # append the word entry to the list of stop aspects
    return [noun for noun, count in word_counter.most_common(25) if noun not in stop_list]  # get the most common nouns
    # that are not in the list of stop aspects


def getTweetAspectScore(tweet, aspect, model):
    """
    Calculate the score of the aspect in the tweet through computing the similarity between the aspect word
    and all words in the tweet, here we use german words embedding from the model
    :param tweet:
    :param aspect:
    :param model:
    :return:
    """
    score = 0
    if aspect not in model.vocab:  # return zero score if the aspect is not the vocabulary of the model,
        #  i.e it has no representation in the model so far
        return score
    count = 0  # holding the total number of tweet words represented in the model
    for word, pos in tweet:
        if word in model.vocab:
            count = count + 1
            score = score + model.similarity(aspect, word)  # add the similarity value
            # between the current tweet word and the aspect word to the current score
    score = score / max(count, 1)  # score scaling by the total number of tweet words represented in the model
    return score


def tweetAspectMapping(tweets, aspects):
    """
    Map each tweet to its target aspect
    :param tweets:
    :param aspects:
    :return:
    """
    pathModel = '../../german.model'
    model = gensim.models.KeyedVectors.load_word2vec_format(pathModel, binary=True)  # load the word2vec model
    # using gensim
    result = dict()  # holding the target aspect for each tweet
    # (mapping by the id of the tweet in the list of tweets)
    cnt = 0  # counter for the list of tweets
    for tweet in tweets:
        best_score = 0.0  # best score over all aspects
        target_aspect = ""  # the aspect corresponding to the best score
        for aspect in aspects:
            cur_score = getTweetAspectScore(tweet, aspect, model)  # get the aspect score in the tweet
            if cur_score >= best_score:  # update the best score and the target aspect
                best_score = cur_score
                target_aspect = aspect
        result[cnt] = target_aspect
        cnt = cnt + 1
    return result


if __name__ == "__main__":
    # just a demo to test the model and different functions
    Model = gensim.models.KeyedVectors.load_word2vec_format('../../german.model', binary=True)
    Tweet = [('Politik', 'NN'), ('ist', 'V'), ('nicht', 'A'), ('interessant', 'AJ')]
    print(getTweetAspectScore(Tweet, 'Politik', Model))
    print("done")
