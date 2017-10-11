"""
This code was taken from https://datascience.blog.wzb.eu/2016/07/13/accurate-part-of-speech-tagging-of-german-texts-with-nltk/
"""
import pickle
import random

import nltk

from lib.ClassifierBasedGermanTagger import ClassifierBasedGermanTagger

corp = nltk.corpus.ConllCorpusReader('../resources', 'tiger_release_aug07.corrected.16012013.conll09',
                                     ['ignore', 'words', 'ignore', 'ignore', 'pos'],
                                     encoding='utf-8')  # load the corpus
# (to be used for training our german POS Tagger)

tagged_sents = corp.tagged_sents()  # get the tagged sentences in the corpus
tagged_sents = [sentence for sentence in tagged_sents]
random.shuffle(tagged_sents)  # make a random shuffle of the list of tagged sentences

# set a split size: use 90% for training, 10% for testing
split_perc = 0.1
split_size = int(len(tagged_sents) * split_perc)
train_sents, test_sents = tagged_sents[split_size:], tagged_sents[:split_size]  # get the training and testing datasets
tagger = ClassifierBasedGermanTagger(train=train_sents)  # train the german POS tagger
# print accuracy 94 %

with open('../resources/nltk_german_classifier_data.pickle', 'wb') as f:
    pickle.dump(tagger, f, protocol=2)  # pickle the POS tagger to use it later without having to train it each time
