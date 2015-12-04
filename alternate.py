import re
import csv
import string
import math
import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers

from nltk.stem import*


stemmer = PorterStemmer()

new_feature_list = []

number_of_documents = 84.0  # number of training sets

weight_feature_vector = []




def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x - y) ** 2 / (2 * (sigma ** 2)))

class SVM(object):
    # this is the constructor
    def __init__(self, kernel=linear_kernel, C=None):
        self.kernel = kernel
        self.C = C
        # if c not empty make it float
        if self.C is not None: self.C = float(10)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Gram matrix
        K = np.zeros((n_samples, n_samples))  # matrix of size n_samples*n_samples with zeros
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.kernel(X[i], X[j])

        P = cvxopt.matrix(np.outer(y, y) * K)  # outer product of matrices
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples))  # matrix of size 1*n_samples 
        b = cvxopt.matrix(0.0)  # matrix of exponentials
        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))  # vertical stack
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samplese) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))  # horizontal stack

        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        a = np.ravel(solution['x'])
        print a
        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        print "sv"
        print sv
        print "ind"
        print ind
        print "self.a"
        print self.a
        print "self.sv"
        print self.sv
        print "self.sv_y"
        print self.sv_y
        print "%d support vectors out of %d points" % (len(self.a), n_samples)

        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n], sv])
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
    # trim
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

        if(global_vector[w]!=1):
            new_feature_list.append(w)
            new_global_vector[w] = global_vector[w]

    # print global_vector
    return new_global_vector



# returns weight array of individual tweets using tf-idf
def calculate_weight(featureVector, featureList, global_vector):

    weight_vector = {}
    new_weight = []

    for w in featureList:

        weight_vector[w] = 0

    for w in featureVector:

        if(w in featureList):

            weight_vector[w] = weight_vector[w] + 1


    size_of_vector = len(featureVector)

    for w in featureList:

        value_of_weight = weight_vector[w] / float(size_of_vector)  # atleast one has to be float
        value_of_weight = value_of_weight * math.log10(number_of_documents / global_vector[w])

        if(value_of_weight < 0):
            value_of_weight = 0
        
        new_weight.append(value_of_weight)
    # print new_weight

    return new_weight



# main

fp = open('tweets.txt', 'r')
sampleTweet = fp.readline()

inp = csv.reader(open('trainingo.csv', 'rb'), delimiter=',')
stopwords = getStopWords()
tweets = []
featureVector = []
featureList = []


# getting featureList and featureVector
for row in inp:

    sentiment = row[0]
    tweet = row[1]
    processedTweet = processTweet(tweet)
    featureVector = getFeatureVector(processedTweet, stopwords)
    # print featureVector
    # tweets.append((featureVector, sentiment));
    tweets.append(featureVector);
    for w in featureVector:
        featureList.append(w)


global_vector = get_global_vector(featureList)

#print tweets
print global_vector
# print len(global_vector)
# print featureVector
# print featureList




'''
print "enter data\n"

review  = "This gibberish, flop, insufficient fool exaggerate. Bad ridiculous, absolutely raddi"
processedTweet = processTweet(review)
featureVector = getFeatureVector(processedTweet, stopwords)

print calculate_weight(featureVector, new_feature_list, global_vector)
'''

for w in tweets:
    weight_of_tweet = calculate_weight(w, new_feature_list, global_vector)
    weight_feature_vector.append(weight_of_tweet)

    #print weight_of_tweet
#print "weight weight_of_tweet"

print weight_feature_vector[1]
# print new_feature_list
# print tweets
# print featureList
# print extract_features("awesome world with rainbows")





if __name__ == "__main__":
    import pylab as pl

    def gen_lin_separable_data():
        # generate training data in the 2-d case
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[0.8, 0.6], [0.6, 0.8]])
        #X1 = np.random.multivariate_normal(mean1, cov, 100)
        X1 = np.array(weight_feature_vector[:50])
        y1 = np.ones(len(X1))
        X2 = np.array(weight_feature_vector[50:])
        #X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def gen_non_lin_separable_data():
        mean1 = [-1, 2]
        mean2 = [1, -1]
        mean3 = [4, -4]
        mean4 = [-4, 4]
        cov = [[1.0, 0.8], [0.8, 1.0]]
        X1 = np.random.multivariate_normal(mean1, cov, 50)
        X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, 50)))
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 50)
        X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov, 50)))
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def gen_lin_separable_overlap_data():
        # generate training data in the 2-d case
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[1.5, 1.0], [1.0, 1.5]])
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def split_train(X1, y1, X2, y2):
        X1_train = X1[:40]
        y1_train = y1[:40]
        X2_train = X2[:33]
        y2_train = y2[:33]
        X_train = np.vstack((X1_train, X2_train))
        y_train = np.hstack((y1_train, y2_train))
        return X_train, y_train

    def split_test(X1, y1, X2, y2):
        X1_test = X1[40:]
        y1_test = y1[40:]
        X2_test = X2[33:]
        y2_test = y2[33:]
        print len(X1_test)
        print len(X2_test)
        X_test = np.vstack((X1_test, X2_test))
        y_test = np.hstack((y1_test, y2_test))
        return X_test, y_test

    def plot_margin(X1_train, X2_train, clf):
        def f(x, w, b, c=0):
            # given x, return y such that [x,y] in on the line
            # w.x + b = c
            return (w[0] * x - b + c) / w[1]

        pl.plot(X1_train[:], X1_train[:], "ro")
        pl.plot(X2_train[:], X2_train[:], "bo")
        pl.scatter(clf.sv[:, 0], clf.sv[:, 1], s=100, c="g")

        # w.x + b = 0
        a0 = -4; a1 = f(a0, clf.w, clf.b)
        b0 = 4; b1 = f(b0, clf.w, clf.b)
        pl.plot([a0, b0], [a1, b1], "k")

        # w.x + b = 1
        a0 = -4; a1 = f(a0, clf.w, clf.b, 1)
        b0 = 4; b1 = f(b0, clf.w, clf.b, 1)
        pl.plot([a0, b0], [a1, b1], "k--")

        # w.x + b = -1
        a0 = -4; a1 = f(a0, clf.w, clf.b, -1)
        b0 = 4; b1 = f(b0, clf.w, clf.b, -1)
        pl.plot([a0, b0], [a1, b1], "k--")

        pl.axis("tight")
        pl.show()

    def plot_contour(X1_train, X2_train, clf):
        pl.plot(X1_train[:, 0], X1_train[:, 1], "ro")
        pl.plot(X2_train[:, 0], X2_train[:, 1], "bo")
        pl.scatter(clf.sv[:, 0], clf.sv[:, 1], s=100, c="g")

        X1, X2 = np.meshgrid(np.linspace(-6, 6, 50), np.linspace(-6, 6, 50))
        X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
        Z = clf.project(X).reshape(X1.shape)
        pl.contour(X1, X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
        pl.contour(X1, X2, Z + 1, [0.0], colors='grey', linewidths=1, origin='lower')
        pl.contour(X1, X2, Z - 1, [0.0], colors='grey', linewidths=1, origin='lower')

        pl.axis("tight")
        pl.show()

    def test_linear():
        X1, y1, X2, y2 = gen_lin_separable_data()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

        clf = SVM()
        clf.fit(X_train, y_train)

        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print "%d out of %d predictions correct" % (correct, len(y_predict))

        plot_margin(X_train[y_train == 1], X_train[y_train == -1], clf)

    def test_non_linear():
        X1, y1, X2, y2 = gen_non_lin_separable_data()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

        clf = SVM(gaussian_kernel)
        clf.fit(X_train, y_train)

        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print "%d out of %d predictions correct" % (correct, len(y_predict))

        plot_contour(X_train[y_train == 1], X_train[y_train == -1], clf)

    def test_soft():
        X1, y1, X2, y2 = gen_lin_separable_overlap_data()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

        clf = SVM(C=0.1)
        clf.fit(X_train, y_train)

        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print "%d out of %d predictions correct" % (correct, len(y_predict))

        plot_contour(X_train[y_train == 1], X_train[y_train == -1], clf)

    test_linear()
