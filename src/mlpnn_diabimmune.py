""" MLPNN for microbial samples from DIABIMMUNE subjects """
## v1.0

from __future__ import division, print_function, absolute_import
from collections import defaultdict, OrderedDict
import random
import os
import datetime
import timeit
import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn.metrics as sk
import sys
from optparse import OptionParser


######## Retrieve all features of one subject
def getbatch_MLPNN(df, timepoints, subjects, meta, batch_size = 1, idx = 0):
    subjs = list()
    subLen = list()
    label = list()
    samples =  []

    if(idx + batch_size >= len(subjects)):
        batch_size = len(subjects) - idx

    for i in range(0, batch_size):
        s = subjects[i + idx]
        s_samples = timepoints[s]
        numSamplesSubject = len(s_samples)
        SubjectSample = s_samples[numSamplesSubject-1]
        r = meta.loc[meta["subjectID"] == s, "allergy"]
        target = np.unique(r.values)

        if (target == True):
            output = [1, 0]
        elif (target == False):
            output = [0, 1]
        else:
            print("\n\nTarget is not unqiue for subject's samples = ", s)
            print("target = ", target)
            exit()

        subjs.append(s)
        label.append(output)
        subLen.append(numSamplesSubject)
        samples.append(SubjectSample)

    idx = idx + batch_size
    data_sample_arr = df.loc[:, samples]
    data_sample_arr = data_sample_arr.values.transpose()
    return data_sample_arr, np.asarray(label), idx, subjs




def trainingMLPNN(df, subjects, timepoints, meta, numFeatures):
    n_classes = 2
    hid1_size = 128 # First layer
    hid2_size = 256  # Second layer
    epochs = 50
    num_fold = 10
    learning_rate = 0.05

    inputs = tf.placeholder(tf.float32, [None, numFeatures], name='inputs')
    label = tf.placeholder(tf.float32, [None, n_classes], name='labels')
    w1 = tf.Variable(tf.random_normal([hid1_size, numFeatures], stddev=0.01), name='w1')
    b1 = tf.Variable(tf.constant(0.1, shape=(hid1_size, 1)), name='b1')
    y1 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(w1, tf.transpose(inputs)), b1)), keep_prob=0.5)
    w2 = tf.Variable(tf.random_normal([hid2_size, hid1_size], stddev=0.01), name='w2')
    b2 = tf.Variable(tf.constant(0.1, shape=(hid2_size, 1)), name='b2')
    y2 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(w2, y1), b2)), keep_prob=0.5)
    wo = tf.Variable(tf.random_normal([2, hid2_size], stddev=0.01), name='wo')
    bo = tf.Variable(tf.random_normal([2, 1]), name='bo')
    yo = tf.transpose(tf.add(tf.matmul(wo, y2), bo))

    # Loss function and optimizer
    lr = tf.placeholder(tf.float32, shape=(), name='learning_rate')
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=yo, labels=label))
    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)

    # Prediction
    pred = tf.nn.softmax(yo)
    y_true = tf.argmax(label, 1)
    y_pred = tf.argmax(pred, 1)
    correct_prediction = tf.equal(y_pred, y_true)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # Create operation which will initialize all variables
    init = tf.global_variables_initializer()

    # Configure GPU not to use all memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Start a new tensorflow session and initialize variables
    sess = tf.InteractiveSession(config=config)
    sess.run(init)


    testAUC = list()
    testMCC = list()
    for loop in range(0,num_fold):

        print("\n Loop =  ", loop)

        random.shuffle(subjects)
        subjAll = np.unique(meta.loc[meta["allergy"] == True, "subjectID"])
        subjNonAll = np.unique(meta.loc[meta["allergy"] == False, "subjectID"])

        testFoldSizeAllergy = round(0.2 * len(subjAll))
        testAllergy = subjAll[0: testFoldSizeAllergy]
        testFoldSizeNonAllergy = round(0.2 * len(subjNonAll))
        testNonAllergy = subjNonAll[0: testFoldSizeNonAllergy]
        testSet = np.concatenate([testAllergy, testNonAllergy])
        random.shuffle(testSet)
        idx_test = 0
        X_test, labels_test, indx, test_subjects = getbatch_MLPNN(df, timepoints, testSet, meta, batch_size=len(testSet),
                                                                 idx=idx_test)

        print("Total # subjects = ", len(subjects))
        print("Total # allergic subjects = ", len(subjAll),
              "   Total # non-allergic subjects = ", len(subjNonAll))


        subjAll = [x for x in subjAll if x not in testAllergy]
        subjNonAll = [x for x in subjNonAll if x not in testNonAllergy]
        print("Training # allergic subjects = ", len(subjAll),
              "   Training # non-allergic subjects = ", len(subjNonAll))
        print("Testing # allergic subjects = ", len(testAllergy),
              "   Testing # non-allergic subjects = ", len(testNonAllergy))


        for i in range(0, num_fold):
            print("fold = ", i)

            foldSize = round((1 / num_fold) * (len(subjAll) + len(subjNonAll)))
            #print("FoldSize = ", foldSize)

            ### Prepare validation fold
            validateFoldSizeAllergy = round((1 / num_fold) * len(subjAll))
            validateAllergy = subjAll[i * validateFoldSizeAllergy: (i + 1) * validateFoldSizeAllergy]
            validateFoldSizeNonAllergy = round((1 / num_fold) * len(subjNonAll))
            validateNonAllergy = subjNonAll[i * validateFoldSizeNonAllergy: (i + 1) * validateFoldSizeNonAllergy]
            validateSet = np.concatenate([validateAllergy, validateNonAllergy])
            random.shuffle(validateSet)

            ### Prepare training fold
            trainAllergy = [x for x in subjAll if x not in validateAllergy]
            trainNonAllergy = [x for x in subjNonAll if x not in validateNonAllergy]
            trainSet = np.concatenate([trainAllergy, trainNonAllergy])
            random.shuffle(trainSet)

            idx_validate = 0
            idx_train = 0
            X_validate, labels_validate, indx, validate_subjects = getbatch_MLPNN(df, timepoints, validateSet, meta,
                                                                     batch_size=len(validateSet),
                                                                      idx=idx_validate)


            for epoch in range(epochs):
                X_train, labels_train, indx, train_subjects = getbatch_MLPNN(df, timepoints, trainSet, meta,
                                                                                 batch_size=len(trainSet),
                                                                                 idx=idx_train)

                avg_cost = 0.0
                for i in range(X_train.shape[0]):
                    _, c = sess.run([optimizer, loss], feed_dict={lr: learning_rate,
                                                                  inputs: X_train,
                                                                  label: labels_train})
                    avg_cost += c
                avg_cost /= X_train.shape[0]

                if epoch % 5 == 0:
                    print("Epoch: {:3d}    Train Cost: {:.4f}".format(epoch, avg_cost))

                idx_train = 0
                acc_train = accuracy.eval(feed_dict={inputs: X_train, label: labels_train})
                y_t, y_p, y_sm = sess.run([y_true, y_pred, pred], feed_dict={inputs: X_train, label: labels_train})
                precision = sk.precision_score(y_t, y_p, average=None)
                recall = sk.recall_score(y_t, y_p, average=None)
                fscore = sk.f1_score(y_t, y_p, average=None)
                mcc = sk.matthews_corrcoef(y_t, y_p, sample_weight=None)
                fpr, tpr, _ = sk.roc_curve(y_t, y_sm[:, 1])
                roc_auc = sk.auc(fpr, tpr)


            acc_test = accuracy.eval(feed_dict={inputs: X_validate, label: labels_validate})

        y_t, y_p, y_sm = sess.run([y_true, y_pred, pred], feed_dict={inputs: X_test, label: labels_test})
        precision = sk.precision_score(y_t, y_p, average=None)
        recall = sk.recall_score(y_t, y_p, average=None)
        fscore = sk.f1_score(y_t, y_p, average=None)
        mcc = sk.matthews_corrcoef(y_t, y_p, sample_weight=None)
        fpr, tpr, _ = sk.roc_curve(y_t, y_sm[:, 1])
        roc_auc = sk.auc(fpr, tpr)
        testAUC.append(roc_auc)
        testMCC.append(mcc)

    print("average AUC = %0.2f" % np.mean(testAUC))
    print("average MCC = %0.2f" % np.mean(testMCC))
    return



def main(_):
    optparser = OptionParser()
    optparser.add_option('-i', '--inputFile',
                         dest='input',
                         help='path to input file',
                         default=None)
    optparser.add_option('-m', '--metadata',
                         dest='meta',
                         help='path to metadata file',
                         default=None)
    optparser.add_option('-o', '--outputDir',
                         dest='out',
                         help='name of the output directory',
                         default="TEST",
                         type='string')


    (options, args) = optparser.parse_args()
    InputFile = None
    if options.input is not None:
        InputFile = options.input
    else:
        print('No input filename specified, system with exit\n')
        sys.exit('System will exit')

    MetadataFile = None
    if options.meta is not None:
        MetadataFile = options.meta
    else:
        print('No metadata file specified, system with exit\n')
        sys.exit('System will exit')

    OutDir = None
    if options.out is not None:
        OutDir = options.out
    else:
        print('No OutDir specified, system with exit\n')
        sys.exit('System will exit')

    print("InputFile = ", InputFile)
    print("MetadataFile = ", MetadataFile)
    print("OutDir = ", OutDir)



    ## Read OTU matrix
    df = pd.read_csv(InputFile, sep=",", index_col = 0)

    features = list(df.index)
    samples = list(df)
    numFeatures = len(features)
    numSamples = len(samples)

    ## Read Metadatafile
    meta = pd.read_csv(MetadataFile, sep=",", index_col = 0)
    unique, counts = np.unique(meta['subjectID'], return_counts=True)
    maxLen = max(counts)

    ## Calculate # samples per subject
    timepoints = defaultdict(list)
    for i in range(0, numSamples):
        timepoints[meta["subjectID"][i]].append(meta["gid_wgs"][i])

    ### Extract subjects
    subjects = list(timepoints.keys())
    numSubject = len(subjects)

    print("# of subjects = ", numSubject)
    print("# of samples = ", numSamples)
    print("max sequence len = ", maxLen)
    print("# of features = ", numFeatures)


    #### Remove subjects with less than 3 datapoints
    finn = 0
    russ = 0
    est = 0

    for row in meta['country']:
        if(row == "FIN"):
            finn = finn + 1
        if (row == "RUS"):
            russ = russ + 1
        if (row == "EST"):
            est = est + 1

    print("Before filtration:   # Finnish samples = ", finn, "   # Russian samples = ", russ, "   # Estonian samples = ", est)

    ctr = 0
    for i in subjects:
        if (len(timepoints[i]) < 3):
            ctr = ctr +1
            df = df.drop(timepoints[i], axis=1)
            meta = meta.drop(meta[meta.subjectID == i].index)
            timepoints.pop(i, None)
            subjects.remove(i)

    print("# of removed subjects = ", ctr)
    print("After filtration:   # subjects = ", len(subjects))

    finn = 0
    russ = 0
    est = 0
    for row in meta['country']:
        if(row == "FIN"):
            finn = finn + 1
        if (row == "RUS"):
            russ = russ + 1
        if (row == "EST"):
            est = est + 1

    print("After filtration:   # Finnish samples = ", finn, "   # Russian samples = ", russ, "   # Estonian samples = ", est)

    russ = 0
    finn = 0
    est = 0
    for i in subjects:
        cntry = np.unique(meta.loc[meta.subjectID == i, "country"])[0]
        if(cntry == "RUS"):
            russ = russ + 1
        if(cntry == "FIN"):
            finn = finn + 1
        if(cntry == "EST"):
            est = est + 1

    print("After filtration:   # Finnish subjects = ", finn, "   # Russian subjects= ", russ,
          "   # Estonian subjects= ", est)


    fAllergy = 0
    nonfallergy = 0
    for i in subjects:
        allergyy = np.unique(meta.loc[meta.subjectID == i, "allergy"])[0]
        if (allergyy):
            fAllergy = fAllergy + 1
        else:
            nonfallergy = nonfallergy + 1


    print("After filtration:   # allergic subjects = ", fAllergy, "   # non-allergic subjects = ", nonfallergy)
    print("df shape = ", df.shape)
    print("meta shape = ", meta.shape)

    #### Training MLPNN
    start = timeit.default_timer()
    trainingMLPNN(df, subjects, timepoints, meta, numFeatures)
    stop = timeit.default_timer()
    print("Elapsed time = %0.2f Seconds" % (stop - start))


if __name__ == '__main__':
	tf.app.run(main = main)