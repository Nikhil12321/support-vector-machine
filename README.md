# Sentiment Analysis using Support Vector Machine

**This is an end to end implementation of Sentiment Analysis using SVM. This uses a self-made SVM classifier with tf-idf as the feature.**
For the details of SVM classifier and the source referred see:[link](https://github.com/Nikhil12321/support-vector-machine/blob/master/Docs/SVM_implemet.pdf)


## Any Pre-requisites?
YES!
1. **CVXopt**: This is an optimization library used for solving the quadratic programming problem that is used to make the classifier.
2. **Natural Language Toolkit**: This toolkit is used for various operations like stemming of words.
3. numpy

These libraries have to be pre-installed

## How to use?
1. svm2.py is the file you want to run
2. this uses a .csv file as data. See training.csv to know the format of the data to be kept
3. variable 'inp' contains the entire data and the filename/location to be used. Change it with the data you want. 
4. stopwords.txt is the file containing the stopwords or the useless words classifier doesn't want to care about.
5. You can change the split_ratio to determine the number of training/testing examples
6. This uses tf idf as feature vector. You can apply anything.

## How to prepare the data?
1. The csv file should consist of two columns. The first column should be the sentiment and the next column should be the data (strictly text)
2. The sentiments should be organized, that is, all examples of a single sentiment should appear before the other and should not be intermixed.
3. Only two sentiments are supported right now.

## How do I see the results?
1. in the function test_non_linear, you can use the number of correctly/incorrectly classified to calculate precision, recall, etc.

## Where can I find data?
The training.csv and trainingo.csv contain small amounts of data. However, inside Data folder, training.csv contains a 2000 set data you can use.
