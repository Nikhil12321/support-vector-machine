import re
import csv
import string
import math

from nltk.stem import*


stemmer = PorterStemmer()

new_feature_list = []

number_of_documents = 4

def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)


def getStopWords():

    stopwords=[]
    sw = open('stopwords.txt', 'r')
    line = sw.readline()
    while line:
        word = line.strip()                     #to remove newline characters
        stopwords.append(word)
        line = sw.readline()
    sw.close()
    return stopwords


def getFeatureVector(tweet, stopwords):
    featureVector = []
    #split tweet into words
    words = tweet.split()
    for w in words:
        #strip punctuation
        table = string.maketrans("","")                 #import string
        w = w.translate(table, string.punctuation)     
        #replace two or more with two occurrences
        w = replaceTwoOrMore(w)
        
        #check if the word stats with an alphabet
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
        #ignore if it is a stop word
        if(w in stopwords or val is None):
            continue
        else:
            featureVector.append(stemmer.stem(w.lower()))
    return featureVector




def processTweet(tweet):
    # process the tweets

    #Convert to lower case
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','AT_USER',tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #trim
    tweet = tweet.strip('\'"')
    return tweet


def extract_features(tweet):
    processedTweet = processTweet(tweet)
    #print "after process"
    #print processedTweet
    featureVector = getFeatureVector(processedTweet, stopwords)
    #print featureVector
    features = {}
    for words in featureList:
        features['contains[%s]'%words] = words in featureVector
    sampleTweet = fp.readline()

    return features



#calculate frequency of all the words in the featureList and remove duplicates
def get_global_vector(featureList):

    global_vector = {}


    for w in featureList:

        stemmed_word = stemmer.stem(w)
        try:
            global_vector[w] = global_vector[w]+1
        except:
            global_vector[w] = 1
            new_feature_list.append(w)


    #print global_vector
    return global_vector

#returns weight array of individual tweets
def calculate_weight(featureVector, featureList, global_vector):

    weight_vector = {}
    new_weight = []

    for w in featureList:

        weight_vector[w] = 0

    for w in featureVector:

        if(w in featureList):

            weight_vector[w] = weight_vector[w]+1


    size_of_vector = len(featureVector)

    for w in featureList:

        value_of_weight = weight_vector[w]/float(size_of_vector)                            #atleast one has to be float
        value_of_weight = value_of_weight*math.log10(number_of_documents/global_vector[w])
        new_weight.append(value_of_weight)
    print new_weight









fp = open('tweets.txt', 'r')
sampleTweet = fp.readline()

inp = csv.reader(open('training.csv', 'rb'), delimiter=',')
stopwords = getStopWords()
tweets = []
featureVector = []
featureList = []


for row in inp:

    sentiment = row[0]
    tweet = row[1]
    processedTweet = processTweet(tweet)
    featureVector = getFeatureVector(processedTweet, stopwords)
    tweets.append((featureVector, sentiment));
    for w in featureVector:
        featureList.append(w)

global_vector = get_global_vector(featureList)
calculate_weight(featureVector, new_feature_list, global_vector)



#print new_feature_list


#print tweets
#print featureList

#print extract_features("awesome world with rainbows")