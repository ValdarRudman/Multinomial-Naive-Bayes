# Python Project B
#   Multinomial Naive Bayes
# By
#   Valdar Rudman
#   R00081134

import matplotlib.pyplot as plt

plt.rcdefaults
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# Clean up the imported words
def cleanUp(tweets):
    stop_words = set(stopwords.words('english'))

    filtered_tweets = []
    filtered_tweet = ""

    # Go through tweets one tweet at a time and
    # split tweet to go through one word at a time
    for tweet in tweets:
        word_tokens = word_tokenize(tweet)

        for w in word_tokens:
            if w not in stop_words:
                # Append all words on the current tweet to each other with a white
                # space at the start of each word
                filtered_tweet += " "
                filtered_tweet += w

        # Add the new tweet string to a list(List of tweets) and remove
        # white space at the start of tweet
        filtered_tweets.append(filtered_tweet.lstrip())
    return filtered_tweets


# Read a file in and split on the white space
def readFile(source):
    return open(source).read().split()


# Read a file in and split on new line
def readFileTweets(source):
    return open(source).read().lower().split("\n")


# Counts the number of words in a set and increments the value in the dict by 1
def no_of_words(list, dict):
    for word in list:
        dict[word] = dict[word] + 1
    return dict


# Gets the probability of the word. This can be the probability
# of a word being positive or negative
def prob_of_word(prob, fullDict):
    words = {}

    for word in fullDict:
        words[word] = (prob[word] / fullDict[word])

    return words


# Gets the probability of tweets being positive or negative
def prob_of_new_tweets(tweets, wordPos, wordNeg):
    posTweets, negTweets, unknownTweets = 0, 0, 0

    # Going through tweets one tweet at a time
    for tweet in tweets:

        # Split tweet into words and go through words
        words = tweet.split()

        # Count the positive words and negative words of
        # current tweet being processed
        posWords, negWords = 0, 0

        for word in words:

            # Checks if word is in positive and negative dictionaries
            # If word in, gets the probability of the word and adds
            # it to the posWords or negWords
            # e.g word love: probability of positive 0.8
            # probability of negative #0.2
            # posWords += 0.8, negWords += 0.2
            if word in wordPos:
                # Originally I multiplied the probability
                # posWords *= wordPos[word]
                # But I found that sometimes the if check
                # if wordPos[word] > 0:
                # would not stop a word with value of 0
                # so I end up with posWords = 0
                # I just change the * to + as I don't need the overall
                # percentage just the sum of pos and neg
                # if wordPos[word] > 0:
                posWords += wordPos[word]
            if word in wordNeg:
                # if wordNeg[word] > 0:
                negWords += wordNeg[word]

        # Compares posWords to negWords and vice versa
        # if posWords value at the end is higher: positive tweet
        # e.g posWords = 1.5 this not a percentage but a total of
        # the percentage probability of the words
        # negWords = 1.3. negWords less => positive tweet
        if negWords > posWords:
            negTweets += 1
        elif posWords > negWords:
            posTweets += 1
        else:
            unknownTweets += 1

    return (posTweets / (posTweets + negTweets + unknownTweets)) * 100


# Graph the before and after results of pre-processing for both negative and positive
def graph(PositiveBeforePP, positiveAfterPP, negativeBeforePP, negativeAfterPP):
    BarTitles = ('Pos Before Pre-Processing',
                 'Pos After Pre-Processing',
                 'Neg before Pre-Processing',
                 'Neg After Pre-Processing')
    plot = [PositiveBeforePP, positiveAfterPP,
            negativeBeforePP, negativeAfterPP]
    y_pos = np.arange(len(BarTitles))
    plt.bar(y_pos, plot, align='center', alpha=0.1)
    plt.xticks(y_pos, BarTitles)
    plt.ylabel("Percentage")
    plt.xlabel("Data")
    plt.title("Tweets Accuracy")
    plt.show()


def main():
    posList = readFile("trainPos.txt")
    negList = readFile("trainNeg.txt")

    # Getting unique words for positive and negative as well as getting a full set of them
    uniquePosSet = set(posList)
    uniqueNegSet = set(negList)
    fullSet = uniquePosSet | uniqueNegSet

    # Creating dictionaries to use to keep count of how many times a word show up
    posDict = dict.fromkeys(fullSet, 0)
    negDict = dict.fromkeys(fullSet, 0)
    fullDict = dict.fromkeys(fullSet, 0)

    # Count words for full dictionary
    for word in posList:
        fullDict[word] = fullDict[word] + 1

    for word in negList:
        fullDict[word] = fullDict[word] + 1

    # Counting words for positive and negative dictionary
    posDict = no_of_words(posList, posDict)
    # print(posDict)
    negDict = no_of_words(negList, negDict)
    # print(negDict)

    # Getting the probability that a word is positive or negative
    wordsPos = prob_of_word(posDict, fullDict)
    # print(wordsPos)
    wordsNeg = prob_of_word(negDict, fullDict)
    # print(wordsNeg)

    # Reading in the test Positive file and cleaning up the result
    posTweets = readFileTweets("testPos.txt")
    # print(prob_of_new_tweets(posTweets, wordsPos, wordsNeg))
    posTweetsCleanUp = cleanUp(readFileTweets("testPos.txt"))
    # print(prob_of_new_tweets(posTweetsCleanUp, wordsPos, wordsNeg))

    # Reading in the negative file and cleaning up the result
    negTweets = readFileTweets("testNeg.txt")
    # print(100 - prob_of_new_tweets(negTweets, wordsPos, wordsNeg))
    negTweetsCleanUp = cleanUp(readFileTweets("testNeg.txt"))
    # print(100 - prob_of_new_tweets(negTweetsCleanUp, wordsPos, wordsNeg))

    # Graph the results from the prob_of_new_tweets function
    graph(prob_of_new_tweets(posTweets, wordsPos, wordsNeg),
          prob_of_new_tweets(posTweetsCleanUp, wordsPos, wordsNeg),
          (100 - prob_of_new_tweets(negTweets, wordsPos, wordsNeg)),
          (100 - prob_of_new_tweets(negTweetsCleanUp, wordsPos, wordsNeg)))


if __name__ == '__main__':
    main()
