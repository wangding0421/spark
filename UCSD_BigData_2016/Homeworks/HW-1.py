# Name: Ding Wang
# Email: diw005@ucsd.edu
# PID: A53089251
from pyspark import SparkContext
sc = SparkContext()
# coding: utf-8

# ### HomeWork 1
# 
# Unigrams, bigrams, and in general n-grams are 1,2 or n words that appear consecutively in a single sentence. Consider the sentence:
# 
#     "to know you is to love you."
# 
# This sentence contains:
# 
#     Unigrams(single words): to(2 times), know(1 time), you(2 times), is(1 time), love(1 time)
#     Bigrams: "to know","know you","you is", "is to","to love", "love you" (all 1 time)
#     Trigrams: "to know you", "know you is", "you is to", "is to love", "to love you" (all 1 time)
# 
# The goal of this HW is to find the most common n-grams in the text of Moby Dick.
# 
# Your task is to:
# 
# * Convert all text to lower case, remove all punctuations. (Finally, the text should contain only letters, numbers and spaces)
# * Count the occurance of each word and of each 2,3,4,5 - gram
# * List the 5 most common elements for each order (word, bigram, trigram...). For each element, list the sequence of words and the number of occurances.
# 
# Basically, you need to change all punctuations to a space and define as a word anything that is between whitespace or at the beginning or the end of a sentence, and does not consist of whitespace (strings consisiting of only white spaces should not be considered as words). The important thing here is to be simple, not to be 100% correct in terms of parsing English. Evaluation will be primarily based on identifying the 5 most frequent n-grams in correct order for all values of n. Some slack will be allowed in the values of frequency of ngrams to allow flexibility in text processing.   
# 
# This text is short enough to process on a single core using standard python. However, you are required to solve it using RDD's for the whole process. At the very end you can use `.take(5)` to bring the results to the central node for printing.

# The code for reading the file and splitting it into sentences is shown below:

# In[1]:

textRDD = sc.newAPIHadoopFile('/data/Moby-Dick.txt',
                              'org.apache.hadoop.mapreduce.lib.input.TextInputFormat',
                              'org.apache.hadoop.io.LongWritable',
                              'org.apache.hadoop.io.Text',
                               conf={'textinputformat.record.delimiter': "\r\n\r\n"}) \
            .map(lambda x: x[1])

sentences=textRDD.flatMap(lambda x: x.split(". "))


# In[2]:

#Data preprocessing
import string
pSentences = sentences.map(lambda x: x.lower().replace('\r\n', ' ')
                               .translate(dict.fromkeys(map(ord, string.punctuation))))


# #### Note: 
# By default, the delimiter string in Spark is "\n". Thus, values in each partition of textRDD describe lines from the file rather than sentences. As a result, sentences may be split over multiple lines. For this input file, a good approach will be to delimit by paragraph so that each value in the RDD is one paragraph (instead of one line). Then we can split the paragraphs into sentences. 
# 
# This is done by setting the `textinputformat.record.delimiter` parameter to `"\r\n\r\n"` in the configuration file. 

# Let `freq_ngramRDD` be the final result RDD containing the n-grams sorted by their frequency in descending order. Use the following function to print your final output:

# In[3]:

def printOutput(n,freq_ngramRDD):
    top=freq_ngramRDD.take(5)
    print '\n============ %d most frequent %d-grams'%(5,n)
    print '\nindex\tcount\tngram'
    for i in range(5):
        print '%d.\t%d: \t"%s"'%(i+1,top[i][0],' '.join(top[i][1]))


# Your output for unigrams should look like:
# ```
# ============ 5 most frequent 1-grams
# 
# index	count	ngram
# 1.       40: 	 "a"
# 2.	   25: 	 "the"
# 3.	   21: 	 "and"
# 4.	   16: 	 "to"
# 5.	   9:  	 "of"
# 
# ```
# Note: This is just a sample output and does not resemble the actual results in any manner.
# 
# Your final program should generate an output using the following code:

# In[4]:

for n in range(1,6):
    freq_ngramRDD = pSentences.map(lambda x:x.split())    .flatMap(lambda x: [(tuple(x[i:(i+n)]),1) for i in range(0,len(x)-n+1)])    .reduceByKey(lambda x,y:x+y).map(lambda x:(x[1],x[0])).sortByKey(False)
    # Put your logic for generating the sorted n-gram RDD here and store it in freq_ngramRDD variable
    printOutput(n,freq_ngramRDD)

