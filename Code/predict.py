# Create your views here.

from django.shortcuts import render
from rest_framework import views
from rest_framework import status
from rest_framework.response import Response

from .naivebayes import review_features, nb_prediction
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re
import codecs
import numpy as np
import itertools
from .seminb import *
import pandas as pd
from pandas.io.json import json_normalize

"""
This file utilizes the Expectation Maximization of Multinomial Naïve Bayes to predict the tags of the remaining 57-62 % of the reviews. 
This is the file that does the prediction of reviews with the support of naivebayes.py and seminb.py

Reviews will come in 2 data frames:
1. Untagged reviews (60%) - 1 column (reviews)
2. Tagged reviews (40%) - 2 columns (reviews, tag/label)
"""

class Naiveb(views.APIView):
    def post(self, request):
        # ---------Global parameters-----------------
        Text = False
        max_features = None
        n_sets = 10
        set_size = 1.0 / n_sets
        cumulative_percent = 0
        # # set project directory
        # abs_path = 'C:/Users/malsa876/Desktop/Pytest/mytracks_NaiveBayes_Filter.csv'     #mytracks_NaiveBayes_Filter #flutter-reviews-NB.csv

            # Testing purposes only
        # -------
        # initialize list of lists 
        # tag_data = [['really briliant app intuitive informative give information could need seemingly accurate', 'yes'], 
        # ['connect gps app connect gps matter long  gps set high accuracy setting appear set  app useless cant track workout', 'no'], 
        # ['wish would interest google provide weekly monthly summary', 'yes'],
        # ['useless talk gps phone  20 minute run data','no'],
        # ['great app glad used track perfectly','yes'],
        # ['excellent thank','no'],
        # ['update wish app quick sharing could view device without drive ability view google map instead google earthadv version map cuz home stuff tablet exercise phone hassle overall like app need update keep rest google product become competitive product','yes'],
        # ['nice needs work used app time every time take age locate position via gps  change apps rely gps locate straight away  return app still search  last time took five minute locate position took enjoyment use app  continue use see improves  35','yes'],
        # ['dr anand venugopal good','no'],
        # ['interesting app','no'],
        # ]

        # untag_data = [['really briliant app intuitive informative give information could need seemingly accurate'], 
        # ['connect gps app connect gps matter long  gps set high accuracy setting appear set  app useless cant track workout'], 
        # ['wish would interest google provide weekly monthly summary'],
        # ['useless talk gps phone  20 minute run data'],
        # ['great app glad used track perfectly'],
        # ['excellent thank'],
        # ['update wish app quick sharing could view device without drive ability view google map instead google earthadv version map cuz home stuff tablet exercise phone hassle overall like app need update keep rest google product become competitive product'],
        # ['nice needs work used app time every time take age locate position via gps  change apps rely gps locate straight away  return app still search  last time took five minute locate position took enjoyment use app  continue use see improves  35'],
        # ['dr anand venugopal good'],
        # ['interesting app'],
        # ]

        # Create the pandas DataFrame
        # tag_dfc = pd.DataFrame(tag_data, columns = ['review', 'label'])
        # untag_dfc = pd.DataFrame(untag_data, columns = ['review'])

        # tag_json = tag_dfc.to_json()
        # request.session['tag_data'] = tag_json

        # Retreiveing the output of preprocessing
        taggged_data = request.session['taggedReviews']
        print(taggged_data)
        untagged_data = request.session['untaggedReviews']
        print(untagged_data)
        
        # tag_df = pd.DataFrame.from_dict(json_normalize(tag_data2), orient='columns')
        tag_df = pd.read_json(taggged_data, orient='columns')
        untag_dfc = pd.read_json(untagged_data, orient='columns')

        # Chnaging column names for our dataframes
        tag_df.columns = ['review','label']
        untag_dfc.columns = ['review']
        print("Preprocessed tagged reviews - 40%")
        print(tag_df)
        print("Preprocessed untagged reviews - 60%")
        print(untag_dfc)
        # -------

        data = tag_df[['review']] # We need to bring the review column of our data frame
        labels = tag_df[['label']].to_numpy() # We need to bring the label (informative/non-informative) column of our data frame

        # data, labels = docparser().get_data(tag_df, untag_df)
        labeled_reviews = tag_df.values.tolist() # Converting our tag_df to a list
        unlabeled_reviews = untag_dfc.values.tolist()

        all_tagged_words = {} # Dictionary to store words for labelled reviews
        all_untagged_words = {} # Dictionary to store words for unlabelled reviews
        for (r, label) in labeled_reviews:
            for word in r.split(" "):
                if len(word) > 1:
                    all_tagged_words[word] = 0 #dict with word key

        for row in unlabeled_reviews:
            for r in row:
                for word in r.split(" "):
                    if len(word) > 1:
                        all_untagged_words[word] = 0 #dict with word key
        
        # featureset for NB
        featuresets = [(review_features(r, {}), label) for (r, label) in labeled_reviews]
        # Unlabelled featureset - reviews which will have a label predicted for them
        featuresets2 = [(review_features(r, {}), label) for row in unlabeled_reviews for r in row]

        # EM of NB with present features and laplace smoothing
        predicted_labels = nb_prediction(featuresets, featuresets2, n_sets, alpha=1.0)
        
        # Adding the predicted_labels to our unlabelled data
        untag_dfc['label'] = predicted_labels
        # Removing square brackets from labels - caused due to predicted_labels being a numpy array
        untag_dfc['label'] = untag_dfc['label'].str[0]
        # Merging our 40% and 60% df's to create one df with all reviews tagged.
        allreview_df = pd.concat([tag_df, untag_dfc], ignore_index=True)
        informative_reviews = allreview_df[allreview_df['label'] == True]
        print(informative_reviews)
        informative_reviews = informative_reviews.loc[:, ['review']]
        print("All reviews tagged reviews - 100%")
        print(allreview_df)
        print("All informative reviews which will be passed to COALS")
        print(informative_reviews)
        # Converting our new df's into JSON then storing the JSON in a session variable.
        allreview_json = allreview_df.to_json()
        informative_json =informative_reviews.to_json()
        request.session['all_reviews'] = allreview_json
        request.session['informative_reviews'] = informative_json

        return Response(status=status.HTTP_200_OK)


        