'''
Testing the following versions of Naive Bayes
1. Naïve Bayes with present features
2. Naïve Bayes with present and absent features
3. Naïve Bayes with present and absent features with Laplace smoothing
4. Expectation Maximization of Naïve Bayes with present and absent features with Laplace smoothing
5. Complement of Naïve Bayes with present and absent features with Laplace smoothing
6. Expectation Maximization of complement of Naïve Bayes with present and absent features with Laplace smoothing
'''

import naivebayes
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re
import codecs
import numpy as np
import itertools
import seminb
import semiCnb
import pandas as pd


class docparser(object):
    """
    parser to get the CSV dataset
    """

    def __init__(self):
        pass

    def get_data(self, file_path, Text=True):
        if Text:
            data = []
            labels = []
            f = codecs.open(file_path, 'r', encoding="utf8", errors='ignore')
            for line in f:
                doc, label = self.parse_line(line)
                data.append(doc)
                labels.append(label)
            return data, np.array(labels)
        else:
            df = pd.read_csv(file_path)
            # df.columns = ['Label', 'Rating', 'Review']
            df.columns = ['Label', 'Review']
            # rating = pd.get_dummies(df.loc[:, 'Rating'])
            text = []
            labels = []
            for ii, ij in zip(df.loc[:, 'Review'].values, df.loc[:, 'Label'].values):
                # print(ii, ij)
                if ii == ' ':
                    pass
                else:
                    text.append(ii)
                    labels.append(ij)
            # return text, np.array(labels), rating
            return text, np.array(labels)

if __name__ == '__main__':
    # ---------Global parameters-----------------
    Text = False
    max_features = None
    n_sets = 10
    set_size = 1.0 / n_sets
    cumulative_percent = 0
    # set project directory
    abs_path = 'C:/nb_process/VodafoneNZ_pre_process2.csv'     #mytracks_NaiveBayes_Filter #flutter-reviews-NB.csv
    if Text:
        pass
    else:
        # data, labels, rating = docparser().get_data(abs_path, Text)
        data, labels = docparser().get_data(abs_path, Text)
        # ----Naive Bayes with present features only
        labeled_reviews = naivebayes.get_labeled_reviews(abs_path)
        all_words = {}
        for (r, label) in labeled_reviews:
            for word in r.split(" "):
                if len(word) > 1:
                    all_words[word] = 0
        # featureset for NB
        # print(len(all_words))
        featuresets = [(naivebayes.review_features(r, {}), label) for (r, label) in labeled_reviews]


        # EM of NB with present features and laplace smoothing
        naivebayes.cross_validation(featuresets, n_sets, emNB=True, alpha=1.0)

      

       


