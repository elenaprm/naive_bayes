#!/usr/bin/python
# FileName: Subsampling.py
# Version 1.0 by Tao Ban, 2010.5.26
# This function extract all the contents, ie subject and first part from the .eml file
# and store it in a new file with the same name in the dst dir.

import email.parser
import os, sys, stat

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

import email
import csv

import numpy as np
import pandas as pd

#######################################################################################

class MultinomialNB():

    def fit(self, X_train, y_train):
        num_emails, num_words = X_train.shape
        self._classes = np.unique(y_train)
        num_classes = len(self._classes)

        self._priors = np.zeros(num_classes)
        self._likelihoods = np.zeros((num_classes, num_words))

        for index, c in enumerate(self._classes):
            class_X_train = X_train[c == y_train]
            self._priors[index] = class_X_train.shape[0] / num_emails
            self._likelihoods[index, :] = ((class_X_train.sum(axis=0)) + 1) / (
                np.sum(class_X_train.sum(axis=0) + 1))

    def predict(self, X_test):
        return [self.predict_each(x_test) for x_test in X_test]

    def predict_each(self, x_test):
        posteriors = []
        for index, c in enumerate(self._classes):
            class_likelihood = x_test * np.transpose(np.log(self._likelihoods[index, :]))
            class_posteriors = np.sum(class_likelihood) + np.log(self._priors[index])
            posteriors.append(class_posteriors)

        return self._classes[np.argmax(posteriors)]


#######################################################################################


def read_file(filename):
    '''
    use email to process the email content
    :param filename: email path
    :return: email title and content
    '''
    with open(filename, encoding='latin-1') as fp:
        msg = email.message_from_file(fp)
        payload = msg.get_payload()
        if type(payload) == type(list()):
            payload = payload[0]
        if type(payload) != type(''):
            payload = str(payload)

        sub = msg.get('subject')
        sub = str(sub)
        return sub + payload

def ExtractSubPayload(filename):
    ''' Extract the subject and payload from the .eml file.
    '''
    if not os.path.exists(filename):  # dest path doesnot exist
        print("ERROR: input file does not exist:", filename)
        os.exit(1)
    fp = open(filename, encoding='latin1')
    msg = email.message_from_file(fp)
    payload = msg.get_payload()
    if type(payload) == type(list()):
        payload = payload[0]  # only use the first part of payload
    sub = msg.get('subject')
    sub = str(sub)
    if type(payload) != type(''):
        payload = str(payload)

    return sub + payload


def ExtractBodyFromDir(srcdir, dstdir):
    '''Extract the body information from all .eml files in the srcdir and
    save the file to the dstdir with the same name.'''
    if not os.path.exists(dstdir):  # dest path doesnot exist
        os.makedirs(dstdir)
    files = os.listdir(srcdir)
    for file in files:
        srcpath = os.path.join(srcdir, file)
        dstpath = os.path.join(dstdir, file)
        src_info = os.stat(srcpath)
        if stat.S_ISDIR(src_info.st_mode):  # for subfolders, recurse
            ExtractBodyFromDir(srcpath, dstpath)
        else:  # copy the file
            body = ExtractSubPayload(srcpath)
            dstfile = open(dstpath, 'w')
            dstfile.write(body)
            dstfile.close()


###################################################################
# main function start here
# srcdir is the directory where the .eml are stored
print('Input source directory: ')  # ask for source and dest dirs
srcdir = input()
if not os.path.exists(srcdir):
    print('The source directory %s does not exist, exit...' % (srcdir))
    sys.exit()
# dstdir is the directory where the content .eml are stored
print('Input destination directory: ')  # ask for source and dest dirs
dstdir = input()
if not os.path.exists(dstdir):
    print('The destination directory is newly created.')
    os.makedirs(dstdir)

###################################################################
ExtractBodyFromDir(srcdir, dstdir)

df_tr = pd.read_csv('/home/elipeke/Escritorio/ML/A2/NBClassifier/spam-mail.tr.label')
spams = df_tr['Prediction']

header = ['text','spam']
data = []
for i in range(1,2499):
    data.append([read_file("".join(['/home/elipeke/Escritorio/ML/A2/NBClassifier/dst/TRAIN_',str(i),'.eml'])), spams[i-1]])

with open('training_emails.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(data)

df = pd.read_csv('/home/elipeke/Escritorio/ML/A2/NBClassifier/training_emails.csv')

X_train, X_test, y_train, y_test = train_test_split(df.text,df.spam,test_size=0.25)

v = CountVectorizer()
X_train_count = v.fit_transform(X_train.values)

model = MultinomialNB()
model.fit(X_train_count, y_train)

emails = []
for i in range(1,1828):
    emails.append(read_file("".join(['/home/elipeke/Escritorio/ML/A2/NBClassifier/TT/TEST_',str(i),'.eml'])))

emails_count = v.transform(emails)
test = model.predict(emails_count)

header2 = ['Id','Prediction']
data2 = []
count=1
for i in test:
    data2.append([count,i])
    count=count+1

with open('results.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header2)
    writer.writerows(data2)
