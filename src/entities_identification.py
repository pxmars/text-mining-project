def loadEntities():
    """
    Load a list of related terms to each entity
    :return:
    """
    pathEntities = '../resources/entities.txt'
    entities = dict()  # dict for entities quick look-up
    with open(pathEntities, 'r', encoding='utf-8') as f:  # load the entities file
        for line in f:
            line = line.strip()
            list_words = line.split(' ')
            entities[list_words[0]] = [list_words[pos] for pos in range(1, len(
                list_words))]  # append the related terms to the list of he corresponding entity
            # (list_words[0]; first word in each line)
    return entities


def identifyEntity(tweet, entities):
    """
    Identify the target entity of the tweet from the list of entities
    :param tweet:
    :param entities:
    :return:
    """
    best_score = 0  # best score over all entities
    targetEntity = ""  # the entity corresponding to the best score
    for word in tweet:
        for entity in entities:
            cur_score = 0  # the score for the current entity
            if word == entity:
                cur_score = 1  # set the current score to 1 in case the entity name is mentioned in the tweet
            for entity_related_word in entities[entity]:
                if word == entity_related_word:
                    cur_score = cur_score + 1  # increment the current score by 1 in case a related term to
                    # the current entity is mentioned in the tweet
            if cur_score > best_score:  # update the best score and the target entity
                best_score = cur_score
                targetEntity = entity
    return targetEntity


def tweetsEntitiesMapping(tweets):
    """
    Map each tweet to its target entity
    :param tweets:
    :return:
    """
    entities = loadEntities()  # load entities
    filtered_tweets = []  # holding the list of filtered tweets
    result_mapping = dict()  # holding the target entity for each tweet
    # (mapping by the id of the tweet in the list of filtered tweets)
    ids_list = []  # holding the list of ids of filtered tweets in the original list
    length = 0  # counter for the list of filtered tweets
    count = 0  # counter for the list of original tweets
    for tweet in tweets:
        entity_value = identifyEntity(tweet, entities)  # identify the target entity for the current tweet
        if len(
                entity_value) > 0:  # filter the tweets by omitting tweets
            # where the target entity is not explicitly identified (the entity_value is empty)
            ids_list.append(count)
            filtered_tweets.append(tweet)
            result_mapping[length] = entity_value
            length = length + 1
        count = count + 1
    return length, filtered_tweets, result_mapping, ids_list


if __name__ == "__main__":
    print("done")
