# Python Project B
#   Multinomial Naive Bayes
# By
#   Valdar Rudman
#   R00081134

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import numpy as np

# Read a file in and split on the white space
def readFile(source):
    return open(source).read().split()

# Read a file in and split on new line
def readTweetsFile(source):
    return open(source).read().lower().split("\n")

# Gets the probability of the word. This can be the probability
# of a word being positive or negative
def prob_of_word(percentDict, fullDict):
    words = {}

    for word in percentDict:
        words[word] = (percentDict[word] / fullDict[word])
    return words

# Takes a list of words in and returns list without stopwords
def removeStopWords(sentence):

    stopWords = set(stopwords.words('english'))
    words = word_tokenize(sentence)
    wordsFiltered = []

    for w in words:
        if w not in stopWords:
            wordsFiltered.append(w)

    return wordsFiltered

# Working out if tweets are positive or negative
def posNegTweets(tweets, wordsPos, wordsNeg):
    posTweets, negTweets, uknownTweets = 0, 0, 0
    for tweet in tweets:
        words = tweet.split()
        posWords, negWords, uknownWord, count = 0, 0, 0, 1
        for word in words:
            if word in wordsPos:
                posWords += wordsPos[word]
            if word in wordsNeg:
                negWords += wordsNeg[word]
            count += 1
        posWords = posWords / count
        negWords = negWords / count

        if posWords > negWords:
            posTweets += 1
        elif negWords > posWords:
            negTweets += 1
        else:
            uknownTweets += 1
    # Returns a list [percent of positive tweets in the batch, percent of negative tweets in the batch, percent of unkown tweets in the batch]
    return [((posTweets / len(tweets)) * 100), ((negTweets / len(tweets)) * 100), ((uknownTweets / len(tweets)) * 100)]

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
    print("Reading in Training Files...")
    posList = readFile("train\\trainPos.txt")
    negList = readFile("train\\trainNeg.txt")

    posList = [item.lower() for item in posList]
    negList = [item.lower() for item in negList]

    print("Removing stopwords from training files...")
    # print(negList)
    posList = removeStopWords(' '.join(posList))
    negList = removeStopWords(' '.join(negList))

    # Getting unique words for positive and negative as well as getting a full set of them
    posSet = set(posList)
    negSet = set(negList)
    fullSet = posSet|negSet

    print("Creating dictionaries...")
    # Creating dictionaries to use to keep count of how many times a word show up
    posDict = dict.fromkeys(posSet, 0)
    negDict = dict.fromkeys(negSet, 0)
    fullDict = dict.fromkeys(fullSet, 0)

    for word in posList:
        posDict[word] = posDict[word] + 1
        fullDict[word] = fullDict[word] + 1

    for word in negList:
        negDict[word] = negDict[word] + 1
        fullDict[word] = fullDict[word] + 1

    # print("Negative: ", negDict)
    # print("Full: ", fullDict)

    print("Calculate words pos/neg value...")
    wordsPos = prob_of_word(posDict, fullDict)
    wordsNeg = prob_of_word(negDict, fullDict)

    print("Reading in Pos Tweets and removing stopwords...")
    posTweets = readTweetsFile("test\\testPos.txt")
    posTweetsCleanedUp = []
    for tweet in posTweets:
        tweet.lower()
        posTweetsCleanedUp.append(' '.join(removeStopWords(tweet)))

    print("Reading in Neg Tweets and removing stopwords...")
    negTweets = readTweetsFile("test\\testNeg.txt")
    negTweetsCleanedUp = []
    for tweet in negTweets:
        tweet.lower()
        negTweetsCleanedUp.append(' '.join(removeStopWords(tweet)))

    print("Calculating Pre results...")
    posPreResults = posNegTweets(posTweets, wordsPos, wordsNeg)
    negPreResults = posNegTweets(negTweets, wordsPos, wordsNeg)
    print("Pre Results\nPositive: ", posPreResults, "\nNegative: ", negPreResults)

    print("Calculating Post results...")
    posPostResults = posNegTweets(posTweetsCleanedUp, wordsPos, wordsNeg)
    negPostResults = posNegTweets(negTweetsCleanedUp, wordsPos, wordsNeg)
    print("Post Results\nPositive: ", posPostResults, "\nNegative: ", negPostResults)

    graph(posPreResults[0], posPostResults[0], negPreResults[1], negPostResults[1])

if __name__ == '__main__':
    main()
