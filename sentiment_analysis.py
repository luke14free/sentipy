import urllib,json,time
import sys,os,pickle

HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'


try:
    import pyximport; pyximport.install()
    from naive_bayes_cython import MyNaiveBayesClassifier
    print "Loaded cython version"
except:
    from naive_bayes import MyNaiveBayesClassifier
    print "Failed to load cython version.. Downgrading to python version"


def main():
    os.system("clear")
    print "Sentiment Analysis 0.01 alpha. (c) Luca Giacomel. Disclaimer: this very simple algorithm wont probably work, but it might be worth a try."
    
    def update_progress(progress,current_operation_message,p):
        df=2 #dimension factor, len of the graph = 100/df
        sys.stdout.write('\r[{0}{1}] {2}% (Page: {4}) Current operation: {3}\r\r'.format('#'*(progress/df)," "*(100/df-(progress/df)), progress,current_operation_message,p))
        sys.stdout.flush()
    
    if os.path.exists("/tmp/db.bin") and os.path.exists("/tmp/neg.tweets") and os.path.exists("/tmp/pos.tweets"):
        proceed=raw_input("I found some tweets already stored, do you want me to use them [y=Yes | n=No | a=Append]? [y/N/a] ").lower()
        while proceed not in ["","y","n","a"]:
            proceed=raw_input("I found some tweets already stored, do you want me to use them? [y/N] ").lower()
        load_from_hd=proceed.lower()
            
    if load_from_hd=="y" or load_from_hd=="":
        test_tweets=[]
        db=pickle.load(open("/tmp/db.bin","rb"))
        print "Done. Read a db of %s words" % len(db)
        search_value=raw_input("What keyword do you want to use to perform the analysis? (you can use @ # :) :( as special operators) ")
        print "Downloading 30 tweets for keywords %s.." % search_value
        z=json.loads(urllib.urlopen("http://search.twitter.com/search.json?q=%s&rpp=30&lang=en" % (urllib.quote(search_value))).read())
        print "Done."
        for m in z['results']:
            test_tweets.append(m['text'])
        z=MyNaiveBayesClassifier(db=db,categories=['positive','negative'])
                    
    elif load_from_hd=="n" or load_from_hd=="a":
        pages_to_load=raw_input("How many pages should I load? [default=20] ")
        while 1:
            try:
                if pages_to_load=="":
                    pages_to_load=20
                    break
                pages_to_load=int(pages_to_load)
                break
            except:
                pages_to_load=raw_input("How many pages should I load? [default=20] ")
        
        if load_from_hd=="a":
            pos_tweets=json.load(open("/tmp/neg.tweets"))
            neg_tweets=json.load(open("/tmp/pos.tweets"))
        else:
            pos_tweets,neg_tweets=[],[]
        
        for p in range(1,pages_to_load+1):
            perc=int(float(p*100)/pages_to_load)
            isleep=0
            #update_progress(perc, "Loading page..",p)
            cycle=True
            while 1:
                try:
                    if cycle:
                        raw_pos_tweets=json.loads(urllib.urlopen("http://search.twitter.com/search.json?page=%s&q=%s&rpp=100&lang=en" % (p,urllib.quote(":)"))).read())
                        raw_neg_tweets=json.loads(urllib.urlopen("http://search.twitter.com/search.json?page=%s&q=%s&rpp=100&lang=en" % (p,urllib.quote(":("))).read()) 
                        if len(neg_tweets)<len(pos_tweets):
                            cycle=False
                    else:
                        raw_neg_tweets=json.loads(urllib.urlopen("http://search.twitter.com/search.json?page=%s&q=%s&rpp=100&lang=en" % (p,urllib.quote(":("))).read()) 
                        raw_pos_tweets=json.loads(urllib.urlopen("http://search.twitter.com/search.json?page=%s&q=%s&rpp=100&lang=en" % (p,urllib.quote(":)"))).read())
                        if len(neg_tweets)>len(pos_tweets):
                            cycle=True
                    raw_pos_tweets['results'],raw_neg_tweets['results']
                    time.sleep(1)
                    for i in raw_pos_tweets['results']:
                        if pos_tweets.count((i['text'],'positive'))==0:
                            pos_tweets.append((i['text'],'positive'))
                    for i in raw_neg_tweets['results']:
                        if neg_tweets.count((i['text'],'negative'))==0:
                            neg_tweets.append((i['text'],'negative'))
                    update_progress(perc, "Elements: %s positive, %s negative." % (len(pos_tweets),len(neg_tweets)),p)
                    break
                except:
                    update_progress(perc, "Failed to fetch the json, trying again in %s seconds" % 2**isleep ,p)
                    time.sleep(2**isleep)
                    isleep+=1
                    if 2**isleep>64:
                        update_progress(perc, "Load time >64sec. Skipping page.. "+str(p),p)
                        break                    
        update_progress(perc, "\n",p)
        open("/tmp/pos.tweets","w").write(json.dumps(pos_tweets))
        open("/tmp/neg.tweets","w").write(json.dumps(neg_tweets))

        training_start=time.time()
        
        index=min(len(pos_tweets),len(neg_tweets))
        test_tweets=[]
        search_value=raw_input("What keyword do you want to use to perform the analysis? (you can use @ # :) :( as special operators) ")
        print "Downloading 30 tweets for keywords %s.." % search_value
        z=json.loads(urllib.urlopen("http://search.twitter.com/search.json?q=%s&rpp=30&lang=en" % (urllib.quote(search_value))).read())
        print "Done."
        for m in z['results']:
            test_tweets.append(m['text'])
        print "Training the classifier. This might take a while, grab a coffe while I work."
        j=MyNaiveBayesClassifier()
        z=j.train(pos_tweets[:index]+neg_tweets[:index],['negative','positive'])
        pickle.dump(z.db,open("/tmp/db.bin","wb"))
        
        print "Done. Training based on a set of %s elements took %s seconds." % (index*2,time.time()-training_start)
    
    for tx in test_tweets:
        print "Tweet: "+OKBLUE+tx+ENDC
        r=z.classify(tx.lower())
        if r[0]=="positive" and r[1]>.2:
            print "Result: "+OKGREEN+r[0]+" (accuracy: "+str(r[1])+")"+ENDC
        elif r[0]=="negative" and r[1]>.2:
            print "Result: "+FAIL+r[0]+" (accuracy: "+str(r[1])+")"+ENDC
        else:
            print "Result: "+WARNING+"neutral (was %s with accuracy %s)" % (r[0],r[1]) +ENDC
            
    most_informative=sorted(z.db.keys(),key=lambda x: float(z.db[x][True])/(z.db[({'positive':'negative','negative':'positive'}[x[0]],x[1])][True]+1),reverse=True)[:30]
    for k in most_informative:
        print "Contained ratio: %s : 1 | Word: %s - Sentiment: %s" %(z.db[k][True]-z.db[({'positive':'negative','negative':'positive'}[k[0]],k[1])][True],k[1],k[0])
        
if __name__=="__main__":
    main()