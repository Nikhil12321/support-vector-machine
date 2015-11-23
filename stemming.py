#Stem a text file


from nltk.stem import*



fr = open('vector.txt','r') #input file
fw = open('vec.txt','w')	#output file
stemmer = PorterStemmer()
p = []

line = fr.readline()

while line:
	w = line.strip()
	w = stemmer.stem(w)
	p.append(w)
	line = fr.readline()
p = set(p)
print len(p)

for w in p:
	fw.write(w+"\n")

