#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import pylab as pl
import random
import sys

from backend import sqlite3
from matplotlib.mlab import PCA
from matplotlib_utils import center_spines
from scipy import percentile as scoreatpercentile
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import AffinityPropagation
from stemming.porter2 import stem

if sys.version_info[1] == 7:
    from collections import Counter
else:
    from collections_fallback import Counter


STOPWORDS = [
    'rt',
    're',
    ':(',
    ':)',
    ':))',
    ':((',
    'http',
    ]


def to_alphanum(s):
    import re
    return ' '.join(re.findall("\w+", s))


def ngrams(tokens, MIN_N, MAX_N):
    n_tokens = len(tokens)
    for i in xrange(n_tokens):
        for j in xrange(i + MIN_N, min(n_tokens, i + MAX_N) + 1):
            yield tuple(tokens[i:j])


class Classifier:

    def __init__(self, *args, **kwargs):

        db = kwargs.get('db', [{}, {}])

        model = MultinomialNB()

        categories = kwargs.get('categories', [])
        db_path = kwargs.get('db_path', '/tmp/db.bin')
        persistent = kwargs.get('persistent', True)

        self.conn = sqlite3.connect('tweets.db')
        self.conn.text_factory = str
        c = self.conn.cursor()
        self.tot_count = float(c.execute('SELECT COUNT(*) FROM tweets'
                               ).fetchone()[0])
        self.neg_count = \
            float(c.execute('SELECT COUNT(*) FROM tweets WHERE sentiment = 0'
                  ).fetchone()[0])
        self.pos_count = \
            float(c.execute('SELECT COUNT(*) FROM tweets WHERE sentiment = 1'
                  ).fetchone()[0])

        decile = int(self.tot_count * .01)

        self.cutoff = \
            c.execute('SELECT pos_count+neg_count FROM features ORDER BY pos_count+neg_count DESC LIMIT ?,?'
                      , [decile, decile]).fetchone()[0]
        print self.cutoff

        if persistent and not os.path.exists(db_path):
            f = open(db_path, 'wb')
            pickle.dump(db, f)
            f.close()
            (self.index_translator, self.db) = db
            self.db_path = db_path
        elif persistent and os.path.exists(db_path):

            (self.index_translator, self.db) = \
                pickle.load(open(db_path, 'rb'))
            self.db_path = db_path
        else:

            self.db = db
            self.db_path = None

        self.persistent = persistent
        self.categories = categories
        self.model = model

    @staticmethod
    def clean_up(tweet):

        # Perform porter stemmer and remove any STOPWORDS

        tweet = ' '.join([word for word in tweet.split(' ')
                         if not word.startswith('#')
                         and not word.startswith('@')
                         and not word.startswith('http')
                         and not word.startswith('www')])

        tweet = to_alphanum(tweet).lower()
        tweet = tweet.split(' ')
        sw = set(STOPWORDS)  # Allows for O(1) lookup

        # return [stem(word) for word in tweet if (word not in sw and stem(word) not in sw)]

        return ['+'.join(i) for i in ngrams([stem(word) for word in
                tweet if word not in sw and stem(word) not in sw
                and len(word) > 2], 1, 2)]

    def train(self, samples):
        """
        Samples should have the following structure:
            [
                ('some-sample-text','positive'),
                ('some-sample-text','negative'),
                ('some-sample-text','neutral')
            ]
        """

        print 'Compressing features'

        counter = Counter(item for sample in samples for item in
                          self.clean_up(sample[0]))
        sorted_words = sorted([i[0] for i in
                              counter.most_common(int(len(counter)
                              * 0.009)) if i[0].find('+') == -1
                              and i[0] > 16 or i[0].find('+') != -1
                              and i[1] > 16])
        self.index_translator = dict(zip(*[sorted_words,
                range(len(sorted_words))]))

        print 'Post-processed features #: %s' \
            % len(self.index_translator)
        print 'Allocating memory & word counting..'

        pos_counter = Counter()
        neg_counter = Counter()

        self.table = np.zeros((len(samples),
                              len(self.index_translator)))
        for sample_idx in range(len(samples)):
            c = Counter(self.clean_up(samples[sample_idx][0]))
            for i in c.keys():
                if i in self.index_translator:
                    self.table[sample_idx][self.index_translator[i]] = \
                        c[i]

                    ((neg_counter if samples[sample_idx][1]
                     == 0 else pos_counter))[i] += c[i]

        self.pos_counter = pos_counter
        self.neg_counter = neg_counter

        classification = [sample[1] for sample in samples]
        print 'Memory allocated, now training..'
        self.model.fit(self.table, classification)
        print 'Trained.'
        self.db = (self.index_translator, self.model)
        return self.model

    def score(self, test_samples):
        self.test_table = np.zeros((len(test_samples),
                                   len(self.index_translator)))
        for sample_idx in range(len(test_samples)):
            cleaned_sample_words = \
                self.clean_up(test_samples[sample_idx][0])
            counter = Counter(cleaned_sample_words)
            for i in counter.keys():
                if i in self.index_translator:
                    self.test_table[sample_idx][self.index_translator[i]] = \
                        counter[i]

        classification = [sample[1] for sample in test_samples]
        return self.model.score(self.test_table, classification)

    def my_score(self):
        import csv
        (tot, ok) = (0., 0.)
        with open('testdata.csv', 'rb') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                translate = {'0': 0, '2': 2, '4': 1}
                (tweet, sentiment) = (row[-1], translate[row[0]])
                if sentiment == 2:
                    sentiment = 1
                r = self.my_classify(tweet)

                # r = self.my_classify(tweet)

                if r == sentiment:
                    ok += 1
                tot += 1

        return ok / tot

    def my_classify(self, sample):
        c = self.conn.cursor()
        positive = 0  #  P
        negative = 0  #  not P

        fwords = self.clean_up(sample)
        fwords_no_bigrams = [i for i in fwords if i.find('#') != -1]
        exclude = []
        for feature in fwords:
            if feature.find('+') != -1 \
                and c.execute("""SELECT pos_count, neg_count 
                                 FROM features 
                                 WHERE feature = ? AND pos_count + neg_count > ?""",
                              [feature, 50]).fetchone():
                for i in range(len(fwords_no_bigrams)):
                    if fwords_no_bigrams[i:i + 1] == feature.split('+'):
                        exclude.extend(fwords_no_bigrams[i:i + 1])
        for i in exclude:
            fwords.remove(i)

        for word in self.clean_up(sample):
            r = \
                c.execute('SELECT pos_count, neg_count FROM features WHERE feature = ? AND pos_count + neg_count > ?'
                          , [word, self.cutoff]).fetchone()
            if r and all(r):
                (pos, neg) = map(float, r)

                cond_pos = pos / self.pos_count  # P(feature|positive)
                cond_neg = neg / self.neg_count  # P(feature|negative)

                positive += math.log(cond_pos)
                negative += math.log(cond_neg)

        positive -= math.log(self.pos_count / self.tot_count)  # P(positive)
        negative -= math.log(self.neg_count / self.tot_count)  # P(negative)

        positive = math.e ** positive
        negative = math.e ** negative

        if int(positive >= negative):
            return True
        else:
            if negative / positive > 1.2:
                return False
            else:
                return True

    def classify(self, sample):
        elem = np.zeros(len(self.index_translator))
        c = Counter(self.clean_up(sample[0]))
        for i in c.keys():
            if i in self.index_translator:
                elem[self.index_translator[i]] = c[i]

        try:
            prob = str(max(self.model.predict_proba(elem)[0])
                       * 100)[:5] + '%'
        except:
            prob = 'n.a.'

        return (('positive (probability: %s)'
                 if self.model.predict(elem)[0]
                == 1 else 'negative (probability: %s)')) % prob

    def show_most_informative(self, remove_outliers=False):
        c = self.conn.cursor()
        tm = \
            c.execute("""SELECT SUM(pos_count_male)+SUM(neg_count_male) 
                         FROM features_by_gender"""
                      ).fetchone()[0]
        tf = \
            c.execute("""SELECT SUM(pos_count_female)+SUM(neg_count_female) 
                         FROM features_by_gender"""
                      ).fetchone()[0]

        query_good = \
            list(c.execute("""SELECT feature,pos_count_male,neg_count_male,pos_count_female,neg_count_female 
                              FROM features_by_gender 
                              WHERE (pos_count_male+neg_count_male+pos_count_female+neg_count_female) > 100 
                              ORDER BY ((pos_count_male+pos_count_female)/(neg_count_female+neg_count_male)) 
                              DESC LIMIT 25"""))
        query_bad = \
            list(c.execute("""SELECT feature,pos_count_male,neg_count_male,pos_count_female,neg_count_female 
                              FROM features_by_gender
                              WHERE (pos_count_male+neg_count_male+pos_count_female+neg_count_female) > 100 
                              ORDER BY ((neg_count_female+neg_count_male)/(pos_count_male+pos_count_female)) 
                              DESC LIMIT 25"""))

        male_most = []
        female_most = []

        for i in query_good + query_bad:
            
            """
            i = [i[0], math.log(i[1] + 1) - math.log(tm), math.log(i[2]
                 + 1) - math.log(tm), math.log(i[3] + 1)
                 - math.log(tf), math.log(i[4] + 1) - math.log(tf)]
            """
            if i[1] + i[2] > i[3] + i[4]:
                male_most.append(i)
            else:
                female_most.append(i)

        raw_cross_data = [i[1:] for i in male_most] + [i[1:]
                                  for i in female_most]

        for i in (raw_cross_data):
            print [j for j in i]

        most_common_words = [[i[0], True] for i in male_most] + [[i[0],
                False] for i in female_most]

        pca_results = PCA(raw_cross_data)

        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        results = pca_results.Y

        if male_most:
            male_avg = np.average([[i[:2] for i in
                                  results[:len(male_most)]]], axis=1)[0]
        else:
            male_avg = (0, 0)

        if female_most:
            female_avg = np.average([[i[:2] for i in
                                    results[len(male_most):]]],
                                    axis=1)[0]
        else:
            female_avg = (0, 0)

        pl.plot(
            male_avg[0],
            male_avg[1],
            'o',
            markerfacecolor='#000000',
            markeredgecolor='k',
            markersize=14,
            )
        pl.plot(
            female_avg[0],
            female_avg[1],
            'o',
            markerfacecolor='#000000',
            markeredgecolor='k',
            markersize=14,
            )

        ax.text(s='Male', x=male_avg[0], y=male_avg[1], fontsize=12)
        ax.text(s='Female', x=female_avg[0], y=female_avg[1],
                fontsize=12)

        af = AffinityPropagation().fit(results)

        ax.spines['left'].set_position('center')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('center')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        center_spines(ax)

        if remove_outliers:
            q1 = scoreatpercentile(results, 25)
            q2 = scoreatpercentile(results, 75)
            iqr = q2 - q1
            filtered_results = [list(i) for i in results if q1 - iqr
                                * 1.5 < i[0] < q2 + iqr * 1.5]  # Exclude outliers for the principal components
            x_plot_interval = max(x[0] for x in filtered_results) \
                - min(x[0] for x in filtered_results)
            ax.set_xlim(-x_plot_interval / 2, x_plot_interval / 2)  # 16:9
            ax.set_ylim(-x_plot_interval / (2 * (1980 / 1024)),
                        +x_plot_interval / (2 * (1980 / 1024)))
        else:
            filtered_results = [list(i) for i in results]

        it = 0
        for (word, _) in most_common_words:
            if filtered_results.count(list(results[it])) > 0:
                ax.text(s=word, x=results[it][0], y=results[it][1],
                        fontsize=8, color=('b' if _ == True else 'r'))
            it += 1

        cluster_centers_indices = af.cluster_centers_indices_
        labels = af.labels_
        X = results
        n_clusters_ = len(cluster_centers_indices)

        sep = 1.0 / n_clusters_
        colors = [(i * sep, random.random(), random.random(), 0.4)
                  for i in range(n_clusters_)]

        for (k, col) in zip(range(n_clusters_), colors):
            class_members = labels == k
            cluster_center = X[cluster_centers_indices[k]]
            pl.plot(X[class_members, 0], X[class_members, 1], '.',
                    color=col)
            pl.plot(
                cluster_center[0],
                cluster_center[1],
                'o',
                markerfacecolor=col,
                markeredgecolor='k',
                markersize=14,
                )
            for x in X[class_members]:

                if remove_outliers and -x_plot_interval / 2 < x[0] \
                    < x_plot_interval / 2 and -x_plot_interval / (2
                        * (1980 / 1024)) < x[1] < x_plot_interval / (2
                        * (1980 / 1024)):
                    pl.plot([cluster_center[0], x[0]],
                            [cluster_center[1], x[1]], color=col)

        print 'Variance explained by the PCA components: %s ' \
            % pca_results.fracs

        try:
            matplotlib.rcParams.update({'font.size': 6})
            plt.savefig('/var/partners/static/mca.png', dpi=500)
            print 'Analysis here: http://109.169.46.22/static/mca.png'
        except:
            matplotlib.rcParams.update({'font.size': 8})
            plt.show()

    def save_to_hard_disk(self, db_path=None):

        # Not implemented yet.

        pass
