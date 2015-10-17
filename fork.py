import re
import csv
import string



def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)


def getStopWords():

	stopwords=[]
	sw = open('stopwords.txt', 'r')
	line = sw.readline()
	while line:
		word = line.strip()						#to remove newline characters
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
    	table = string.maketrans("","")					#import string
    	w = w.translate(table, string.punctuation)     
        #replace two or more with two occurrences
        w = replaceTwoOrMore(w)
        
        #check if the word stats with an alphabet
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
        #ignore if it is a stop word
        if(w in stopwords or val is None):
            continue
        else:
            featureVector.append(w.lower())
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

#print tweets
#print featureList

print extract_features("awesome world with rainbows")