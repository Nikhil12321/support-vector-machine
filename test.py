tweet = "just had some bloodwork done. my arm hurts"


tweet_words = tweet.split()

features  ={}

for word in tweet_words:

	features['contains(%s)'%word] = (word in tweet_words)
print features
