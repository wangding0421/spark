# -*- coding: utf-8 -*-

# Name: Qian Wang
# Email: qiw087@ucsd.edu
# PID: A53088113

from pyspark import SparkContext
sc = SparkContext()

def print_count(rdd):
    print 'Number of elements:', rdd.count()

with open('../Data/hw2-files-1gb.txt') as f:
    files = [l.strip() for l in f.readlines()]
dataRDD = sc.textFile(','.join(files)).map(lambda x: x.encode('utf-8')).cache()
print_count(dataRDD)

import ujson

json_example = '''
{
    "id": 1,
    "name": "A green door",
    "price": 12.50,
    "tags": ["home", "green"]
}
'''

json_obj = ujson.loads(json_example)
json_obj

import ujson

def safe_parse(raw_json):
    try:
        jsonObj = ujson.loads(raw_json)
    except ValueError, e:
        return False
    if 'id_str' in raw_json and 'text' in raw_json:
        return True
    return False

usrTxtRDD = dataRDD.filter(safe_parse).map(lambda x: ujson.loads(x)).map(lambda x: (x['user']['id_str'], x['text'])).cache()

def print_users_count(count):
    print 'The number of unique users is:', count

print_users_count(usrTxtRDD.map(lambda x: x[0]).distinct().count())

import pickle
fd = open("../Data/users-partition.pickle")
partitionDict = pickle.load(fd)

def getGroup(x):
    if x in partitionDict:
        return partitionDict[x]
    return 7

groupCntRDD = usrTxtRDD.map(lambda x: (getGroup(x[0]), 1)).reduceByKey(lambda a,b: a + b).sortByKey().cache()
# groupRDD.take(5)

def print_post_count(counts):
    for group_id, count in counts:
        print 'Group %d posted %d tweets' % (group_id, count)

print_post_count(groupCntRDD.collect())

"""
This code implements a basic, Twitter-aware tokenizer.

A tokenizer is a function that splits a string of text into words. In
Python terms, we map string and unicode objects into lists of unicode
objects.

There is not a single right way to do tokenizing. The best method
depends on the application.  This tokenizer is designed to be flexible
and this easy to adapt to new domains and tasks.  The basic logic is
this:

1. The tuple regex_strings defines a list of regular expression
   strings.

2. The regex_strings strings are put, in order, into a compiled
   regular expression object called word_re.

3. The tokenization is done by word_re.findall(s), where s is the
   user-supplied string, inside the tokenize() method of the class
   Tokenizer.

4. When instantiating Tokenizer objects, there is a single option:
   preserve_case.  By default, it is set to True. If it is set to
   False, then the tokenizer will downcase everything except for
   emoticons.

The __main__ method illustrates by tokenizing a few examples.

I've also included a Tokenizer method tokenize_random_tweet(). If the
twitter library is installed (http://code.google.com/p/python-twitter/)
and Twitter is cooperating, then it should tokenize a random
English-language tweet.


Julaiti Alafate:
  I modified the regex strings to extract URLs in tweets.
"""

__author__ = "Christopher Potts"
__copyright__ = "Copyright 2011, Christopher Potts"
__credits__ = []
__license__ = "Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License: http://creativecommons.org/licenses/by-nc-sa/3.0/"
__version__ = "1.0"
__maintainer__ = "Christopher Potts"
__email__ = "See the author's website"

######################################################################

import re
import htmlentitydefs

######################################################################
# The following strings are components in the regular expression
# that is used for tokenizing. It's important that phone_number
# appears first in the final regex (since it can contain whitespace).
# It also could matter that tags comes after emoticons, due to the
# possibility of having text like
#
#     <:| and some text >:)
#
# Most imporatantly, the final element should always be last, since it
# does a last ditch whitespace-based tokenization of whatever is left.

# This particular element is used in a couple ways, so we define it
# with a name:
emoticon_string = r"""
    (?:
      [<>]?
      [:;=8]                     # eyes
      [\-o\*\']?                 # optional nose
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth      
      |
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
      [\-o\*\']?                 # optional nose
      [:;=8]                     # eyes
      [<>]?
    )"""

# The components of the tokenizer:
regex_strings = (
    # Phone numbers:
    r"""
    (?:
      (?:            # (international)
        \+?[01]
        [\-\s.]*
      )?            
      (?:            # (area code)
        [\(]?
        \d{3}
        [\-\s.\)]*
      )?    
      \d{3}          # exchange
      [\-\s.]*   
      \d{4}          # base
    )"""
    ,
    # URLs:
    r"""http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"""
    ,
    # Emoticons:
    emoticon_string
    ,    
    # HTML tags:
     r"""<[^>]+>"""
    ,
    # Twitter username:
    r"""(?:@[\w_]+)"""
    ,
    # Twitter hashtags:
    r"""(?:\#+[\w_]+[\w\'_\-]*[\w_]+)"""
    ,
    # Remaining word types:
    r"""
    (?:[a-z][a-z'\-_]+[a-z])       # Words with apostrophes or dashes.
    |
    (?:[+\-]?\d+[,/.:-]\d+[+\-]?)  # Numbers, including fractions, decimals.
    |
    (?:[\w_]+)                     # Words without apostrophes or dashes.
    |
    (?:\.(?:\s*\.){1,})            # Ellipsis dots. 
    |
    (?:\S)                         # Everything else that isn't whitespace.
    """
    )

######################################################################
# This is the core tokenizing regex:
    
word_re = re.compile(r"""(%s)""" % "|".join(regex_strings), re.VERBOSE | re.I | re.UNICODE)

# The emoticon string gets its own regex so that we can preserve case for them as needed:
emoticon_re = re.compile(regex_strings[1], re.VERBOSE | re.I | re.UNICODE)

# These are for regularizing HTML entities to Unicode:
html_entity_digit_re = re.compile(r"&#\d+;")
html_entity_alpha_re = re.compile(r"&\w+;")
amp = "&amp;"

######################################################################

class Tokenizer:
    def __init__(self, preserve_case=False):
        self.preserve_case = preserve_case

    def tokenize(self, s):
        """
        Argument: s -- any string or unicode object
        Value: a tokenize list of strings; conatenating this list returns the original string if preserve_case=False
        """        
        # Try to ensure unicode:
        try:
            s = unicode(s)
        except UnicodeDecodeError:
            s = str(s).encode('string_escape')
            s = unicode(s)
        # Fix HTML character entitites:
        s = self.__html2unicode(s)
        # Tokenize:
        words = word_re.findall(s)
        # Possible alter the case, but avoid changing emoticons like :D into :d:
        if not self.preserve_case:            
            words = map((lambda x : x if emoticon_re.search(x) else x.lower()), words)
        return words

    def tokenize_random_tweet(self):
        """
        If the twitter library is installed and a twitter connection
        can be established, then tokenize a random tweet.
        """
        try:
            import twitter
        except ImportError:
            print "Apologies. The random tweet functionality requires the Python twitter library: http://code.google.com/p/python-twitter/"
        from random import shuffle
        api = twitter.Api()
        tweets = api.GetPublicTimeline()
        if tweets:
            for tweet in tweets:
                if tweet.user.lang == 'en':            
                    return self.tokenize(tweet.text)
        else:
            raise Exception("Apologies. I couldn't get Twitter to give me a public English-language tweet. Perhaps try again")

    def __html2unicode(self, s):
        """
        Internal metod that seeks to replace all the HTML entities in
        s with their corresponding unicode characters.
        """
        # First the digits:
        ents = set(html_entity_digit_re.findall(s))
        if len(ents) > 0:
            for ent in ents:
                entnum = ent[2:-1]
                try:
                    entnum = int(entnum)
                    s = s.replace(ent, unichr(entnum))	
                except:
                    pass
        # Now the alpha versions:
        ents = set(html_entity_alpha_re.findall(s))
        ents = filter((lambda x : x != amp), ents)
        for ent in ents:
            entname = ent[1:-1]
            try:            
                s = s.replace(ent, unichr(htmlentitydefs.name2codepoint[entname]))
            except:
                pass                    
            s = s.replace(amp, " and ")
        return s


from math import log

tok = Tokenizer(preserve_case=False)

def get_rel_popularity(c_k, c_all):
    return log(1.0 * c_k / c_all) / log(2)


def print_tokens(tokens, gid = None):
    group_name = "overall"
    if gid is not None:
        group_name = "group %d" % gid
    print '=' * 5 + ' ' + group_name + ' ' + '=' * 5
    for t, n in tokens:
        print "%s\t%.4f" % (t.encode('utf-8'), n)
    print


# your code here
tokenRDD = usrTxtRDD.flatMap(lambda x: tok.tokenize(x[1])).distinct().cache()
print_count(tokenRDD)

usrDiffTknRDD = usrTxtRDD.map(lambda x: (x[0], tok.tokenize(x[1]))).reduceByKey(lambda a,b: a+b).map(lambda x: (x[0], list(set(x[1])))).cache()
freqTokenRDD = usrDiffTknRDD.flatMap(lambda x: x[1]).map(lambda x: (x, 1)).reduceByKey(lambda a,b: a+b).filter(lambda x: x[1] >= 100).cache()
print_count(freqTokenRDD)
print_tokens(freqTokenRDD.sortBy(lambda x: x[1], False).take(20))

# %%time
freqTokenDict = freqTokenRDD.collectAsMap()
groupDiffTknRDD = usrDiffTknRDD.flatMapValues(lambda x: x).filter(lambda x: x[1] in freqTokenDict).map(lambda x: ((getGroup(x[0]), x[1]), 1)).reduceByKey(lambda a,b: a + b).map(lambda x: (x[0][0], (x[0][1], get_rel_popularity(x[1], freqTokenDict[x[0][1]])))).partitionBy(8).glom().collect()
            
for i in range(8):
    ansRDD = sc.parallelize(groupDiffTknRDD[i]).map(lambda x: x[1])
    print_tokens(ansRDD.sortBy(lambda x: (x[1], x[0]), False).take(10), i)

users_support = [
    (-1, "Bernie Sanders"),
    (-1, "Ted Cruz"),
    (-1, "Donald Trump")
]

for gid, candidate in users_support:
    print "Users from group %d are most likely to support %s." % (gid, candidate)

