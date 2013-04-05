import urllib
import json
import time
import sys
import os
import sqlite3

HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'


from custom_classifier import Classifier
#from nltk.classify.naivebayes import NaiveBayesClassifier




def main():
    os.system("clear")
    print "Sentiment Analysis by Luca Giacomel. Follow the instructions."
    conn = sqlite3.connect('tweets.db')
    c = conn.cursor()
    # Create table, force uniqueness on the tweet
    c.execute('''CREATE TABLE IF NOT EXISTS tweets (tweet TEXT, sentiment INT, UNIQUE(tweet) ON CONFLICT REPLACE)''')

    def update_progress(progress,current_operation_message,p):
        df=2 #dimension factor, len of the graph = 100/df
        sys.stdout.write('\r[{0}{1}] {2}% (Page: {4}) Current operation: {3}\r\r'.format('#'*(progress/df)," "*(100/df-(progress/df)), progress,current_operation_message,p))
        sys.stdout.flush()
    
    load_from_hd="n"
    
    if c.execute("SELECT COUNT(*) FROM tweets").fetchone()[0] > 0:
        proceed=raw_input("There are some tweets already stored, do you want me to use them [y=Yes | n=No | a=Append]? [y/n/a] ").lower()
        while proceed not in ["","y","n","a"]:
            proceed=raw_input("There are some tweets already stored, do you want me to use them [y=Yes | n=No | a=Append]? [y/n/a] ").lower()
        load_from_hd=proceed.lower()
            
    if load_from_hd=="y" or load_from_hd=="":
        test_tweets=set()
        nb=Classifier(db_path="tweets.db",categories=[1,0])
        print "Done. Classifier loaded"
        search_value=raw_input("What keyword do you want to use to perform the analysis? (you can use @ # :) :( as special operators) ")
        print "Downloading 100 tweets for keywords %s.." % search_value
        z=json.loads(urllib.urlopen("http://search.twitter.com/search.json?q=%s&rpp=100&lang=en" % (urllib.quote(search_value))).read())
        print "Done."
        for m in z['results']:
            test_tweets.add(m['text'])
        test_tweets = list(test_tweets)
        
                    
    elif load_from_hd=="n" or load_from_hd=="a":
        pages_to_load=raw_input("How many pages of tweets should I load? [default=20] ")
        while 1:
            try:
                if pages_to_load=="":
                    pages_to_load=20
                    break
                pages_to_load=int(pages_to_load)
                break
            except:
                pages_to_load=raw_input("How many pages of tweets should I load? [default=20] ")
        
        if load_from_hd=="a":
            neg_tweets = list(c.execute("SELECT * FROM tweets WHERE sentiment = 0"))
            pos_tweets = list(c.execute("SELECT * FROM tweets WHERE sentiment = 1"))
        else:
            pos_tweets,neg_tweets=[],[]
        
        for p in range(1,pages_to_load+1):
            perc=int(float(p*100)/pages_to_load)
            isleep=0
            download_positive = len(pos_tweets) < len(neg_tweets)
            while 1:
                try:
                    if not download_positive:
                        raw_neg_tweets=json.loads(urllib.urlopen("http://search.twitter.com/search.json?page=%s&q=%s&rpp=100&lang=en" % (p,urllib.quote(":("))).read()) 
                        if len(neg_tweets) > len(pos_tweets):
                            download_positive = True
                        for i in raw_neg_tweets['results']:
                            if neg_tweets.count((i['text'],0))==0:
                                neg_tweets.append((i['text'],0))
                    else:
                        raw_pos_tweets=json.loads(urllib.urlopen("http://search.twitter.com/search.json?page=%s&q=%s&rpp=100&lang=en" % (p,urllib.quote(":)"))).read())
                        if len(neg_tweets) < len(pos_tweets):
                            download_positive = False
                        for i in raw_pos_tweets['results']:
                            if pos_tweets.count((i['text'],1))==0:
                                pos_tweets.append((i['text'],1))
                    time.sleep(1)
                    update_progress(perc, "Elements: %s positive, %s negative." % (len(pos_tweets),len(neg_tweets)),p)
                    break
                except:
                    update_progress(perc, "Failed to fetch the json, trying again in %s seconds" % 2**isleep ,p)
                    time.sleep(2**isleep)
                    isleep+=1
                    if 2**isleep>64:
                        update_progress(perc, "Load time >64sec. Skipping page.. "+str(p),p)
                        break        
        try:
            update_progress(perc, "\n",p)
        except:
            1
        
        c.executemany('INSERT INTO tweets VALUES (?,?)', pos_tweets+neg_tweets)
        conn.commit()

        training_start=time.time()
        
        index=max(len(pos_tweets),len(neg_tweets))
        test_tweets=set()
        search_value=raw_input("What keyword do you want to use to perform the analysis? (you can use @ # :) :( as special operators) ")
        print "Downloading 100 tweets for keywords %s.." % search_value
        z=json.loads(urllib.urlopen("http://search.twitter.com/search.json?q=%s&rpp=100&lang=en" % (urllib.quote(search_value))).read())
        print "Done."
        for m in z['results']:
            test_tweets.add(m['text'])
        test_tweets = list(test_tweets)
        print "Training the classifier. This might take a while, grab a coffe while I work."

        nb=Classifier(db=[{},{}], categories=['negative','positive'])
        model = nb.train(pos_tweets[:-100]+neg_tweets[:-100])        
        accuracy = nb.score(pos_tweets[-100:]+neg_tweets[-100:])
        print "Accuracy of the model: %s" % accuracy
        
        print "Done. Training based on a set of %s elements took %s seconds." % (index*2,time.time()-training_start)
    
    for tx in test_tweets:
        print "Tweet: "+OKBLUE+tx+ENDC
        r=nb.classify(tx.lower())
        if r.startswith("positive"):
            print "Result: "+OKGREEN+r+ENDC
        elif r.startswith("negative"):
            print "Result: "+FAIL+r+ENDC
        #else:
        #print "Result: "+WARNING+"neutral (was %s with accuracy %s)" % (r[0],r[1]) +ENDC
    nb.show_most_informative()
    print
    inp = raw_input("Classify some custom input: (quit to exit or anything else for classification) ")
    while inp!="quit":
        r=nb.classify(inp)
        if r.startswith("positive"):
            print "Result: "+OKGREEN+r+ENDC
        elif r.startswith("negative"):
            print "Result: "+FAIL+r+ENDC
        inp = raw_input("Custom input: ")
        
    print "bye!"
    nb.save_to_hard_disk()
    
    
    
if __name__=="__main__":
    main()