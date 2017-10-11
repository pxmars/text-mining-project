import operator


def summarize(opinions, aspects, file):
    """
    Given a list of opinions and aspects, write the aspect-based opinion summary to a file
    :param opinions:
    :param aspects:
    :param file:
    :return: void
    """
    opinions.sort(key=lambda x: x.targetAspect, reverse=False)  # sort the list of opinions
    # according to the targetAspect attribute
    # here all opinion objects with the same target aspect are occurring one after another in a sequential manner
    # for opinion in Opinions:
    #     print(opinion, file=file)
    list_entities = ['cdu', 'spd', 'linke', 'grüne', 'csu', 'fdp', 'afd']  # store the list of entities
    currentAspect = ""  # holding the aspect for the current iteration
    idx_pos = dict()  # holding the list of tweets text that have positive sentiment value for the current aspect
    idx_neg = dict()  # holding the list of tweets text that have negative sentiment value for the current aspect
    perc_pos = dict()  # holding the number of positive tweets for each entity and the current aspect
    perc_neg = dict()  # holding the number of negative tweets for each entity and the current aspect
    cnt_pos = 0  # holding the total number of positive tweets for the current aspect
    cnt_neg = 0  # holding the total number of negative tweets for the current aspect
    cnt = 0  # counter over the list of opinions
    for aspect in aspects:  # initialize both dictionaries
        idx_pos[aspect] = []
        idx_neg[aspect] = []
    for opinion in opinions:  # iterate over the list of opinions
        if (currentAspect != opinion.getTargetAspect() and cnt != 0) or cnt == len(opinions) - 1:
            # in case we move from the current aspect to the next one or the current aspect is the last one to process
            print("When it comes to " + currentAspect + ":", file=file)
            sorted_perc_pos = sorted(perc_pos.items(), key=operator.itemgetter(1), reverse=True)
            # sort the list of positive tweets for each entity in decreasing order to get the entity that has
            # the largest number of votes first
            for e, n in sorted_perc_pos[:3]:
                perc = int(n / float(cnt_pos) * 100)  # compute the percentage of votes for the entity e with regards
                # to the current aspect
                print(str(perc) + "% of twitters are preferring " + e, file=file)
            if len(idx_pos[currentAspect]) > 0:
                print('\n', file=file)
                print("Here are what twitters liked about political parties vision to " + currentAspect, file=file)
                if len(idx_pos[currentAspect]) > 3:  # keep the 3 first positive tweets having
                    # the current aspect as subject
                    idx_pos[currentAspect] = idx_pos[currentAspect][:3]
                for tweet in idx_pos[currentAspect]:
                    print(tweet, file=file)
            if len(idx_neg[currentAspect]) > 0:
                print('\n', file=file)
                print("Here are what tweeters disliked about " + currentAspect, file=file)
                if len(idx_neg[currentAspect]) > 3:  # keep the 3 first negative tweets having
                    # the current aspect as subject
                    idx_neg[currentAspect] = idx_neg[currentAspect][:3]
                for tweet in idx_neg[currentAspect]:
                    print(tweet, file=file)
            # reinitialize counters
            cnt_pos = 0
            cnt_neg = 0
            print('\n' + '\n' + '\n', file=file)
        if currentAspect != opinion.getTargetAspect():  # reinitialize counters of each entity when moving
            # to another aspect
            for item in list_entities:
                perc_pos[item] = 0
                perc_neg[item] = 0
        cnt += 1  # increment the counter over the list of opinions
        currentAspect = opinion.getTargetAspect()  # update the current aspect
        tweet_text = opinion.getText()  # holding the text for the current opinion
        if opinion.getSO() == 1:  # if the sentiment value for the current opinion is positive
            cnt_pos += 1  # increment the counter for positive tweets
            perc_pos[opinion.getTargetEntity()] += 1  # increment the counter for positive tweets for the current entity
            if len(tweet_text) > 80:
                idx_pos[currentAspect].append(opinion.getText())  # store the text of the tweet
        elif opinion.getSO() == -1:  # if the sentiment value for the current opinion is negative
            cnt_neg += 1  # increment the counter for negative tweets
            perc_neg[opinion.getTargetEntity()] += 1  # increment the counter for negative tweets for the current entity
            if len(tweet_text) > 80:
                idx_neg[currentAspect].append(opinion.getText())  # store the text of the tweet
