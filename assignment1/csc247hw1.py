import re
import string
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import pyLDAvis
import pyLDAvis.gensim_models

from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel

df = pd.read_csv('covid_tweet_dataset.csv')
tweets = df["tweet"].to_list()

# Search URLs in a given tweet string using regex
# "search" method and extracting the starting index
# of the instance if found. 
# Then, as all URLs were found to be 23 characters
# in length, replace the 23-character-long string
# starting at "h" with an empty string using regex
# "sub" method.
def remove_urls(tweets):
    index = 0
    while index < len(tweets):
        tweet = tweets[index]
        x = re.search("https://", tweet)
        if x:
            url = tweet[x.start():(x.start() + 23)]
            tweets[index] = re.sub(url, "", tweet)
            tweet = tweets[index]
        y = re.search("https://", tweet)
        if y:
            url = tweet[y.start():(y.start() + 23)]
            tweets[index] = re.sub(url, "", tweet)
        index += 1
    return tweets
            
    #raise NotImplementedError

# Utilizing string methods in Python, we may
# remove any instances of punctuation from each
# tweet.
def remove_punctuation(tweets):
    index = 0
    while index < len(tweets):
        tweet = tweets[index]
        tweets[index] = tweet.translate(str.maketrans('', '', string.punctuation))
        index += 1
    return tweets
        

# Using an online list of NLTK stop words
# as a reference point, each word is searched
# and substituted using the regex "sub" feature.
#
# (This is due to the fact that I was unable to
# resolve an error with NLTK's stopwords.
# I spent the better part of two hours hoping
# to find a solution, but was unable to.
# This is my workaround.)
def remove_stopwords(tweets):
    index = 0
    while index < len(tweets):
        tweet = tweets[index].lower()
        spaced = re.sub('\s', " ", tweet)
        # Word searches include spaces directly
        # before and after the word in question
        # in order to avoid instances where, for
        # example, the word 'in' is being
        # searched, and is found and deleted due
        # to its place in a larger word such as
        # 'Minnesota' or 'shopping.'
        a = re.sub(" for ", " ", spaced)
        aa = re.sub(" those ", " ", a)
        aaa = re.sub(" your ", " ", aa)
        aaaa = re.sub(" can ", " ", aaa)
        b = re.sub("because", "", aaaa)
        bb = re.sub(" were ", " ", b)
        bbb = re.sub(" their ", " ", bb)
        bbbb = re.sub(" also ", " ", bbb)
        c = re.sub(" if ", " ", bbbb)
        cc = re.sub(" has ", " ", c)
        ccc = re.sub(" they ", " ", cc)
        cccc = re.sub(" other ", " ", ccc)
        d = re.sub(" are ", " ", cccc)
        dd = re.sub(" been ", " ", d)
        ddd = re.sub(" our ", " ", dd)
        dddd = re.sub(" without ", " ", ddd)
        e = re.sub(" that ", " ", dddd)
        ee = re.sub(" having ", " ", e)
        eee = re.sub(" here ", " ", ee)
        eeee = re.sub(" only ", " ", eee)
        f = re.sub(" this ", " ", eeee)
        ff = re.sub(" be ", " ", f)
        fff = re.sub(" i'm ", " ", ff)
        ffff = re.sub(" so ", " ", fff)
        g = re.sub(" we ", " ", ffff)
        gg = re.sub(" do ", " ", g)
        ggg = re.sub(" i'd ", " ", gg)
        gggg = re.sub(" like ", " ", ggg)
        h = re.sub(" my ", " ", gggg)
        hh = re.sub(" was ", " ", h)
        hhh = re.sub(" is ", " ", hh)
        hhhh = re.sub(" such ", " ", hhh)
        i = re.sub(" up ", " ", hhhh)
        ii = re.sub(" had ", " ", i)
        iii = re.sub(" you're ", " ", ii)
        iiii = re.sub(" no ", " ", iii)
        j = re.sub(" you ", " ", iiii)
        jj = re.sub(" being ", " ", j)
        jjj = re.sub(" they're ", " ", jj)
        jjjj = re.sub("would", "", jjj)
        k = re.sub(" me ", " ", jjjj)
        kk = re.sub(" am ", " ", k)
        l = re.sub(" i ", " ", kk)
        ll = re.sub(" did ", " ", l)
        lll = re.sub(" didn't ", " ", ll)
        llll = re.sub(" isn't ", " ", lll)
        m = re.sub(" in ", " ", llll)
        mm = re.sub(" but ", " ", m)
        mmm = re.sub(" will ", " ", mm)
        mmmm = re.sub("could", "", mmm)
        n = re.sub(" to ", " ", mmmm)
        nn = re.sub(" while ", " ", n)
        nnn = re.sub(" im ", " ", nn)
        nnnn = re.sub(" any ", " ", nnn)
        o = re.sub(" as ", " ", nnnn)
        oo = re.sub(" with ", " ", o)
        ooo = re.sub(" after ", " ", oo)
        oooo = re.sub(" one ", " ", ooo)
        p = re.sub(" and ", " ", oooo)
        pp = re.sub(" against ", " ", p)
        ppp = re.sub(" it's ", " ", pp)
        pppp = re.sub(" who ", " ", ppp)
        q = re.sub(" at ", " ", pppp)
        qq = re.sub(" between ", " ", q)
        qqq = re.sub(" it ", " ", qq)
        qqqq = re.sub(" his ", " ", qqq)
        r = re.sub(" how ", " ", qqqq)
        rr = re.sub(" during ", " ", r)
        rrr = re.sub(" its ", " ", rr)
        rrrr = re.sub(" her ", " ", rrr)
        s = re.sub(" the ", " ", rrrr)
        ss = re.sub(" before ", " ", s)
        sss = re.sub(" can ", " ", ss)
        ssss = re.sub(" she ", " ", sss)
        t = re.sub(" a ", " ", ssss)
        tt = re.sub(" through ", " ", t)
        ttt = re.sub(" don't ", " ", tt)
        tttt = re.sub(" he ", " ", ttt)
        u = re.sub(" an ", " ", tttt)
        uu = re.sub(" to ", " ", u)
        uuu = re.sub(" come ", " ", uu)
        uuuu = re.sub(" just ", " ", uuu)
        v = re.sub(" of ", " ", uuuu)
        vv = re.sub(" about ", " ", v)
        vvv = re.sub(" going ", " ", vv)
        vvvv = re.sub(" go ", " ", vvv)
        w = re.sub(" on ", " ", vvvv)
        ww = re.sub(" from ", " ", w)
        www = re.sub(" us ", " ", ww)
        wwww = re.sub(" not ", " ", www)
        x = re.sub(" or ", " ", wwww)
        xx = re.sub(" down ", " ", x)
        xxx = re.sub(" doing ", " ", xx)
        xxxx = re.sub(" what ", " ", xxx)
        y = re.sub(" have ", " ", xxxx)
        yy = re.sub(" off ", " ", y)
        yyy = re.sub(" then ", " ", yy)
        yyyy = re.sub(" who ", " ", yyy)
        z = re.sub(" by ", " ", yyyy)
        zz = re.sub(" over ", " ", z)
        zzz = re.sub(" into ", " ", zz)
        zzzz = re.sub(" where ", " ", zzz)
        tweets[index] = zzzz
        index += 1
    return tweets

# Utilizes regex's built-in features for
# searching and substituting numbers from
# strings of text.
def remove_numbers(tweets):
    index = 0
    while index < len(tweets):
        x = re.sub(r'[0-9]', '', tweets[index])
        tweets[index] = x
        index += 1
    return tweets

# Tokenize tweets one by one before appending
# to a previously blank array.
def tokenize(tweets):
    tokens = []
    for tweet in tweets:
        tokens.append(word_tokenize(tweet))
    return tokens

def print_list(items):
    for item in items:
        print(item)

tweets = remove_urls(tweets)

tweets = remove_stopwords(tweets)

tweets = remove_punctuation(tweets)

tweets = remove_numbers(tweets)

tweets = tokenize(tweets)
print(tweets)

# Create dictionary from tokenized array.
text_dict = Dictionary(tweets)
#print(text_dict)
# Generate bag-of-word format data.
tweets_bow = [text_dict.doc2bow(tweet) for tweet in tweets]

# Assign tweets_bow to corpus.
corpus = tweets_bow

# Initialize LDA Model using pyLDAvis
# gensim.
tweets_lda = LdaModel(corpus, num_topics=10, id2word=text_dict, iterations=20)
#tweets_lda = None
#print(tweets_lda.show_topics())
vis = pyLDAvis.gensim_models.prepare(tweets_lda, corpus, tweets_lda.id2word)
pyLDAvis.save_html(vis, '/Users/audreynovoa/Desktop/nt10i20.html')
