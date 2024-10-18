import csv
import random
import nltk
import numpy as np
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from statistics import mean


def Average(lst):
    return mean(lst)


def get_labeled_reviews(path_to_csv):
    labeled_reviews = []
    with open(path_to_csv, newline='', encoding='utf-8') as csvfile:
        review_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(review_reader, None)  # Skip csv headers
        for row in review_reader:
            label = int(row[0])
            review_text = row[1]

            review = (review_text, label)
            labeled_reviews.append(review)

    return labeled_reviews


def review_features(review, all_words):
    # features = {}
    features = all_words.copy()
    # features["review"] = review
    for word in str.split(review, " "):
        if len(word) > 1:
            if word in features:
                features[word] += 1
            else:
                features[word] = 1
    return features


def cross_validation(all_data, n_sets, mNB=False, cNB=False, NB=False, emNB=False, emCNM=False, alpha=0.0):
    start = time.time()

    if mNB == True:
        from sklearn.naive_bayes import MultinomialNB
    if cNB == True:
        from sklearn.naive_bayes import ComplementNB
    if emNB == True:
        from seminb import Semi_EM_MultinomialNB
    if emCNM == True:
        from semiCnb import Semi_cNB

    set_size = 1.0 / n_sets
    shuffled_data = all_data.copy()
    #print(shuffled_data)
    random.shuffle(shuffled_data)
    cumulative_percent = 0
    prec = 0
    rec = 0
    fmeasure = 0

    for i in range(0, n_sets):
        n_training = int(set_size * len(all_data))
        split_start = i * n_training
        split_end = (i + 1) * n_training
        # print("Train split_start: " + str(split_start) + " - split_end: " + str(split_end))
        train_data_before = shuffled_data[:split_start]
        train_data_after = shuffled_data[split_end:]
        train_data = train_data_before + train_data_after
        test_data = shuffled_data[split_start:split_end]
        # print("train size: " + str(len(train_data)) + " - test size: " + str(len(test_data)))

        train_data2x = []
        train_data2y = []
        for i, v in enumerate(train_data):
            train_data2x.append(v[0])
            train_data2y.append(v[1])


        test_data2x = []
        test_data2y = []
        for i, v in enumerate(test_data):
            test_data2x.append(v[0])
            #print(test_data2x)
            test_data2y.append(v[1])




        ###DEbug.data
        # train_data2x=train_data2x[:100]
        # train_data2y=train_data2y[:100]
        # test_data2x=test_data2x[:100]
        # test_data2y=test_data2y[:100]
        ###end.debug.data

        count_vect = DictVectorizer()

        X_train_tf = count_vect.fit_transform(train_data2x)


        X_test_counts = count_vect.transform(test_data2x)


        if mNB == True:
            clf = MultinomialNB(alpha=alpha, class_prior=None, fit_prior=True).fit(X_train_tf, train_data2y)
        if cNB == True:
            clf = ComplementNB(alpha=alpha, class_prior=None, fit_prior=True, norm=False).fit(X_train_tf, train_data2y)
        if NB == True:
            clf = nltk.NaiveBayesClassifier.train(train_data)
            clf.predict = clf.classify
        if emNB == True:
            clf = Semi_EM_MultinomialNB(alpha=alpha)
            # em_nb_clf.fit(np.array(train_data2x), np.array(train_data2y), np.array(test_data2x))
            clf.fit(X_train_tf, np.array(train_data2y), X_test_counts)
        if emCNM == True:
            clf = Semi_cNB(alpha=alpha)
            # em_nb_clf.fit(np.array(train_data2x), np.array(train_data2y), np.array(test_data2x))
            clf.fit(X_train_tf, np.array(train_data2y), X_test_counts)

        if NB == True:
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            #correct = 0
            for i, t in enumerate(test_data2x):
                l = test_data2y[i]
                classified = clf.predict(t)
                #if classified == l:
                #    correct += 1
                if classified == 1:
                    if l == 1:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if l == 0:
                        tn += 1
                    else:
                        fn += 1
            #print("True positives: {}".format(tp))
            #print("True negatives: {}".format(tn))
            #print("False positives: {}".format(fp))
            #print("Flase negatives: {}".format(fn))
            correct_percent = (tp + tn) / len(test_data)
            cumulative_percent += correct_percent
            print('\nAccuracy:' + str((correct_percent) * 100) + "%")
            precision = tp / (tp + fp)
            prec += precision
            print("Precision: {}".format(precision))
            recall = tp / (tp + fn)
            rec += recall
            print("Recall: {}".format(recall))
            f1 = 2 * (precision * recall) / (precision + recall)
            fmeasure += f1
            print("F1-Score: {}".format(f1))
            #print(str(correct) + "/" + str(len(test_data)))
            #correct_percent = correct / len(test_data)
            # correct_percent = (tp + tn) / len(test_data)
            # cumulative_percent += correct_percent
            # print('Accuracy:' + str((correct_percent) * 100) + "%")
            print("Time: " + str(time.time() - start) + " seconds")

        else:
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            for i, t in enumerate(test_data2x):
                l = test_data2y[i]

                # if modified==True:
                #    count_vect = DictVectorizer(sparse=False)
                # else:
                #    count_vect = DictVectorizer()
                b = count_vect.transform(t)
                classified = clf.predict(b)

                # classified = classifier.classify(t)
                # actual = labeled_reviews[split_point:][i][1]
                #if classified[0] == l:
                    #correct += 1
                if classified[0] == 1:
                    if l == 1:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if l == 0:
                        tn += 1
                    else:
                        fn += 1
            #print("True positives: {}".format(tp))
            #print("True negatives: {}".format(tn))
            #print("False positives: {}".format(fp))
            #print("False negatives: {}".format(fn))
            correct_percent = (tp + tn) / len(test_data)
            cumulative_percent += correct_percent
            print('\nAccuracy:' + str((correct_percent) * 100) + "%")
            precision = tp / (tp + fp)
            prec += precision
            print("Precision: {}".format(precision))
            recall = tp / (tp + fn)
            rec += recall
            print("Recall: {}".format(recall))
            f1 = 2 * (precision * recall) / (precision + recall)
            fmeasure += f1
            print("F1-Score: {}".format(f1))
            # print(str(correct) + "/" + str(len(test_data)))
            # correct_percent = correct / len(test_data)

                # print("Time: " + str(time.time() - start) + " seconds")
            #print(str(correct) + "/" + str(len(test_data)))
            #correct_percent = correct / len(test_data)
            #cumulative_percent += correct_percent
            #print(str((correct_percent) * 100) + "%")
            #print("Time ",time.time()-start)
            print("Time: " + str(time.time() - start) + " seconds")
    print("\nAverage accuracy: " + str((cumulative_percent / n_sets) * 100) + "%")
    print("Average precision: " + str((prec / n_sets)) + "%")
    print("Average recall:" + str((rec/ n_sets)) + "%")
    print("Average F-measure: " + str((fmeasure / n_sets)) + "%" )
    #return str((cumulative_percent / n_sets) * 100)