import re
import csv
import string
import math
import numpy as np
from numpy import linalg
# import cvxopt
# import cvxopt.solvers
import datetime

# from nltk.stem import*




def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

class SVM(object):
    #this is the constructor
    def __init__(self, kernel=linear_kernel, C=None):
        self.kernel = kernel
        self.C = C
        #if c not empty make it float
        if self.C is not None: self.C = float(self.C)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Gram matrix
        K = np.zeros((n_samples, n_samples)) #matrix of size n_samples*n_samples with zeros
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j])

        P = cvxopt.matrix(np.outer(y,y) * K)  #outer product of matrices
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1,n_samples)) # matrix of size 1*n_samples 
        b = cvxopt.matrix(0.0) # matrix of exponentials

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2))) #vertical stack
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))#horizontal stack

        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        a = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        
        self.sv_y = y[sv]
        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n],sv])
        self.b /= len(self.a)

        # Weight vector
        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None

    def project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel(X[i], sv)
                y_predict[i] = s
            return y_predict + self.b

    def predict(self, X):
        return np.sign(self.project(X))




if __name__ == "__main__":
    import pylab as pl

def replaceTwoOrMore(s):
    # look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)


def getStopWords():

    stopwords = []
    sw = open('stopwords.txt', 'r')
    line = sw.readline()
    while line:
        word = line.strip()  # to remove newline characters
        stopwords.append(word)
        line = sw.readline()
    sw.close()
    return stopwords


def getFeatureVector(tweet, stopwords):
    featureVector = []
    # split tweet into words
    words = tweet.split()
    for w in words:
        # strip punctuation
        table = string.maketrans("", "")  # import string
        w = w.translate(table, string.punctuation)     
        # replace two or more with two occurrences
        # w = replaceTwoOrMore(w)
        
        # check if the word stats with an alphabet
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
        # ignore if it is a stop word
        if(w in stopwords or val is None):
            continue
        else:
            featureVector.append(stemmer.stem(w.lower()))
    return featureVector




def processTweet(tweet):
    # process the tweets

    # Convert to lower case
    tweet = tweet.lower()
    # Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet)
    # Convert @username to AT_USER
    tweet = re.sub('@[^\s]+', 'AT_USER', tweet)
    # Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    # Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    # removing punctuations
    exclude = set(string.punctuation)
    tweet = ''.join(ch for ch in tweet if ch not in exclude)
    tweet = tweet.strip('\'"')
    return tweet


def extract_features(tweet):
    processedTweet = processTweet(tweet)
    # print "after process"
    # print processedTweet
    featureVector = getFeatureVector(processedTweet, stopwords)
    # print featureVector
    features = {}
    for words in featureList:
        features['contains[%s]' % words] = words in featureVector
    sampleTweet = fp.readline()

    return features



# calculate frequency of all the words in the featureList and remove duplicates
def get_global_vector(featureList):

    global_vector = {}
    new_global_vector = {}


    for w in featureList:

        stemmed_word = stemmer.stem(w)
        try:
            global_vector[w] = global_vector[w] + 1
        except:
            global_vector[w] = 1
            #new_feature_list.append(w)

    for w in global_vector:

        if(global_vector[w]>2):
            new_feature_list.append(w)
            new_global_vector[w] = global_vector[w]

    # print global_vector
    return new_global_vector



# returns weight array of individual tweets using tf-idf
def calculate_weight(featureVector, featureList, global_vector):

    global counter
    weight_vector = {}
    new_weight = []

    for w in featureList:

        weight_vector[w] = 0

    for w in featureVector:

        if(w in featureList):

            weight_vector[w] = weight_vector[w] + 1


    size_of_vector = len(featureVector);
    size_of_vector = 1.0*size_of_vector;

    for w in featureList:

        value_of_weight = weight_vector[w] / float(size_of_vector)  # atleast one has to be float
        value_of_weight = value_of_weight * math.log10(number_of_documents / global_vector[w])

        if(value_of_weight < 0):
            value_of_weight = 0
        
        new_weight.append(value_of_weight)
    # print new_weight
    

    #   FOR PRINTING PERCENTAGE AND SHIZZZ
    ############################################

    if(counter == 1):
        print "processing inputs \n"
        counter = counter+1
    else:
        perc = (counter/number_of_documents)*100
        print "%d%% complete "%perc
        counter = counter+1
        if(perc == 100):
            counter = 1

    #############################################

    return new_weight





def gen_non_lin_separable_data():
    X1 = np.array(weight_feature_vector[:num_positive])
    y1 = np.ones(len(X1))
    X2 = np.array(weight_feature_vector[num_positive:])
    y2 = np.ones(len(X2)) * -1
    return X1, y1, X2, y2

def split_train(X1, y1, X2, y2):

    num = split_ratio*num_positive

    X1_train = X1[:num]
    y1_train = y1[:num]
    X2_train = X2[:num]
    y2_train = y2[:num]
    X_train = np.vstack((X1_train, X2_train))
    y_train = np.hstack((y1_train, y2_train))
    return X_train, y_train

def split_test(X1, y1, X2, y2):

    num = split_ratio*num_positive

    X1_test = X1[num:]
    y1_test = y1[num:]
    X2_test = X2[num:]
    y2_test = y2[num:]
    X_test = np.vstack((X1_test, X2_test))
    y_test = np.hstack((y1_test, y2_test))
    return X_test, y_test


def test_non_linear():
    X1, y1, X2, y2 = gen_non_lin_separable_data()
    X_train, y_train = split_train(X1, y1, X2, y2)
    X_test, y_test = split_test(X1, y1, X2, y2)
    clf = SVM(gaussian_kernel)
    clf.fit(X_train, y_train)
    
    y_predict = clf.predict(X_test)
    correct = np.sum(y_predict == y_test)
    #print "%d out of %d predictions correct" % (correct, len(y_predict))

    ####################################################
    #   PRINTING THE ACCURACY

    correct = correct * 1.0
    total = 1.0 * len(y_predict)
    accuracy = correct/total
    accuracy = accuracy*100
    print "accuracy is %f%%" %accuracy
    
    ####################################################



# main

#stemmer = PorterStemmer()

new_feature_list = []

counter = 1

weight_feature_vector = []

inp = csv.reader(open('training.csv', 'rb'), delimiter=',')
stopwords = getStopWords()
tweets = []
featureVector = []
featureList = []
number_of_documents = len(list(inp))  # number of training sets
num_positive = number_of_documents/2
num_negative = num_positive
split_ratio = 0.9


getting featureList and featureVector
for row in inp:

    sentiment = row[0]
    tweet = row[1]
    processedTweet = processTweet(tweet)
    featureVector = getFeatureVector(processedTweet, stopwords)
    tweets.append(featureVector);
    for w in featureVector:
        featureList.append(w)
global_vector = get_global_vector(featureList)

for w in tweets:
    weight_of_tweet = calculate_weight(w, new_feature_list, global_vector)
    weight_feature_vector.append(weight_of_tweet)

test_non_linear()
