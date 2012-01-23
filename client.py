#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import time
import os
import pickle
import nltk

MILK = False

if MILK:
    import milk

from collections import defaultdict

from twitter import Twitter
from twitter.oauth import OAuth, read_token_file


CONSUMER_KEY='uS6hO2sV6tDKIOeVjhnFnQ'
CONSUMER_SECRET='MEYTOS97VvlHX7K1rwHPEqVpTSqZ71HtvoK4sVuYk'
COUNT = 40


def get_features(tweet):
    text = tweet['text']
    words = text.split()
    features = dict(
        num_words=len(words),
        author=tweet['user']['screen_name'],
        question='?' in text,
        has_hashes=any(word for word in words if word.startswith('#')),
        has_url='http' in text,
        many_retweets=tweet['retweet_count'] > 0,
    )
    features.update(
        (name, tweet.get(name, '') or '')
        for name in ('favorited', 'truncated', 'retweeted',  'in_reply_to_screen_name', 'possibly_sensitive')
    )
    features.update(
        ('has_' + name, bool(tweet.get(name, False)))
        for name in ('coordinates', 'place', 'geo')
    )

    #for value in features.values():
    #    if isinstance(value, dict):
    #        import pdb;pdb.set_trace()
    features.update(
        ('word(%s)' % word, True)
        for word in words
    )
    features.update(
        ('hash(%s)' % word, True)
        for word in words
            if word.startswith('#')
    )
    features.update(
        ('mention(%s)' % word, True)
        for word in words
            if word.startswith('@')
    )

    #features = dict(
    #    (key, str(value))
    #    for key, value in features.iteritems()
    #)
    return features

if MILK:
    milk_keys = []
    milk_values = defaultdict(set)


def get_milk_keys(feature_list):
    global milk_values, milk_keys
    keys = set()
    for item in feature_list:
        for key, value in item.iteritems():
            keys.add(key)
            if isinstance(value, basestring):
                milk_values[key].add(value)

    milk_keys[:] = keys
    milk_values = dict(
        (key, list(value))
        for key, value
            in milk_values.iteritems()
    )



def get_milk_features(tweet):
    features = get_features(tweet)
    def get_value(key):
        value = features.get(key, 0)
        if key in milk_values:
            try:
                value = milk_values[key].index(value)
            except ValueError:
                value = -100
        return value

    result = [
        get_value(key)
        for key in milk_keys
    ]
    return result
    #features = [
    #    value for key, value in features.iteritems()
    #    if not (key.startswith('word(') or key.startswith('hash(') or key.startswith('mention(') or isinstance(value, basestring))
    #]
    #return features


def main():
    oauth_filename = os.environ.get('HOME', '') + os.sep + '.twitter_oauth'
    oauth_filename = os.path.expanduser(oauth_filename)

    oauth_token, oauth_token_secret = read_token_file(oauth_filename)
    auth = OAuth(oauth_token, oauth_token_secret, CONSUMER_KEY, CONSUMER_SECRET)
    twitter = Twitter(
        auth=auth,
        secure=True,
        api_version='1',
        domain='api.twitter.com'
    )

    try:
        tweets = pickle.load(open('tweets.pickle'))
    except:
        tweets = []
    print "Horay! I've got %s tweets from the file!" % len(tweets)

    # используем nltk
    featuresets = [(get_features(tweet), tweet['good']) for tweet in tweets]
    total = len(featuresets)
    train_set, test_set = featuresets[total/2:], featuresets[:total/2]

    classifier = nltk.NaiveBayesClassifier.train(train_set)
    #tree_classifier = nltk.DecisionTreeClassifier.train(train_set)
    print nltk.classify.accuracy(classifier, test_set)
    classifier.show_most_informative_features(10)
    #print nltk.classify.accuracy(tree_classifier, test_set)


    if MILK:
        # используем milk
        learner = milk.defaultclassifier()
        get_milk_keys(get_features(tweet) for tweet in tweets)
        features = [get_milk_features(tweet) for tweet in tweets]
        labels = [tweet['good'] for tweet in tweets]
        model = learner.train(features, labels)


    ids = set(tweet['id'] for tweet in tweets)

    tweet_iter = twitter.statuses.friends_timeline(count=COUNT)
    for tweet in tweet_iter:
        if tweet.get('text') and tweet['id'] not in ids:
            print '%s: %s' % (tweet['user']['name'], tweet['text'])
            print '[nltk] I think, this tweet is interesting with probability', classifier.prob_classify(get_features(tweet)).prob(True)
            if MILK:
                print '[milk] I think, this tweet is interesting with probability', model.apply(get_milk_features(tweet))
            good = raw_input('Interesting or not?\n(y/n): ') in ('y', 'Y', 'G', 'g')
            tweet['good'] = good
            tweets.append(tweet)



    pickle.dump(tweets, open('tweets.pickle', 'w'))


if __name__ == '__main__':
    main()

