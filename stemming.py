#Stem a text file


from nltk.stem import*



fr = open('vector.txt','r') #input file
fw = open('vec.txt','w')	#output file
stemmer = PorterStemmer()

line = fr.readline()

while line:
	w = line.strip()
	w = stemmer.stem(w)
	fw.write(w+"\n")
	line = fr.readline()



