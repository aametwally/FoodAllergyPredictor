""" LSTM for variable length sequence of microbial samples from DIABIMMUNE subjects """

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
import scipy as sp
import sys
from optparse import OptionParser



# Hyperparameters
learning_rate = 0.05
training_steps = 50
batch_size = 5
display_step = 5
numFold = 2
n_hidden = 50
n_classes = 2
lambda_l2 = 0.05



######## Get batch of subjects' samples
def getbatch(df, timepoints, subjects, maxLen, meta, numFeatures, batch_size = 1, idx = 0):
    d = list()
    subLen = list()
    label = list()

    if(idx + batch_size >= len(subjects)):
        batch_size = len(subjects) - idx

    for i in range(0,batch_size):
        s = subjects[i + idx]
        s_samples = timepoints[s]
        numSamplesSubject = len(s_samples)

        ## Convert it to array then list
        data_samples = df.loc[:, s_samples]
        data_sample_arr = data_samples.values.transpose()
        e = data_sample_arr.tolist()

        ## Pad by zeros for seuence with length < maxLen
        e += [[0.]*numFeatures for i in range(maxLen - numSamplesSubject)]

        r = meta.loc[meta["subjectID"] == s, "allergy"]
        target = np.unique(r.values)

        if (target == True):
            output = [1, 0]
        elif (target == False):
            output = [0, 1]
        else:
            print("\n\nTarget is not unique for subject's samples = ", s)
            print("target = ", target)
            exit()

        label.append(output)
        d.append(e)
        subLen.append(numSamplesSubject)

    idx = idx + batch_size
    return d, np.asarray(label), subLen, idx




def dynamicRNN(x, seqlen, weights, biases, seq_max_len, n_hidden):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, seq_max_len, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)

    # Get lstm cell output, providing 'sequence_length' will perform dynamic calculation.
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32,
                                                sequence_length=seqlen)

    # Performing dynamic calculation and retrieve the last of each sequence.
    outputs = tf.stack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])

    # Build the indexing and retrieve the right output.
    batch_size = tf.shape(outputs)[0]
    # Start indices for each sample
    index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
    # Indexing
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)

    # Linear activation, using outputs computed above
    return tf.matmul(outputs, weights['out']) + biases['out']



def trainingLSTM(maxLen, numFeatures, subjects, meta, timepoints, df):
    seq_max_len = maxLen

    # tf Graph input
    x = tf.placeholder("float", [None, seq_max_len, numFeatures], name = "x") #1
    y = tf.placeholder("float", [None, n_classes], name = "y")
    seqlen = tf.placeholder(tf.int32, [None], name="seqLen") # A placeholder for indicating each sequence length to be able to extract output after certain timepoints
    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    ## Define loss and optimizer
    pred = dynamicRNN(x, seqlen, weights, biases, seq_max_len, n_hidden)
    y_sftmx = tf.nn.softmax(logits = pred)
    y_softmxEntropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y)
    tv = tf.trainable_variables()
    regularization_cost = lambda_l2*tf.reduce_sum([ tf.nn.l2_loss(v) for v in tv ])
    cost = tf.reduce_mean(y_softmxEntropy) + regularization_cost
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


    ## Model Evaluation
    y_true = tf.argmax(y, 1, name="y_true")
    y_pred = tf.argmax(pred, 1, name="y_pred")
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar("LSTMLoss", cost)
    tf.summary.scalar("LSTMAccuracy", accuracy)
    merged_summary = tf.summary.merge_all()


    ## Training
    testAccuracy = list()
    testPrecision = list()
    testRecall = list()
    testFscore = list()
    testAUC = list()
    tprs = []
    testMCC = list()
    testingLoss = list()
    mean_fpr = np.linspace(0, 1, 100)
    LogDir_base = os.getcwd() + "/logs/LSTM" + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


    for loop in range(0, numFold):
        random.shuffle(subjects)
        subjAll = np.unique(meta.loc[meta["allergy"] == True, "subjectID"])
        subjNonAll = np.unique(meta.loc[meta["allergy"] == False, "subjectID"])

        ### Prepare test set
        testFoldSizeAllergy = round(0.1 * len(subjAll))
        testAllergy = subjAll[0: testFoldSizeAllergy]
        testFoldSizeNonAllergy = round(0.1 * len(subjNonAll))
        testNonAllergy = subjNonAll[0: testFoldSizeNonAllergy]
        testSet = np.concatenate([testAllergy, testNonAllergy])
        random.shuffle(testSet)
        idx_test = 0
        testData, testLabel, testLen, idx_test = getbatch(df=df, timepoints=timepoints,
                                                                          subjects=testSet,
                                                                          maxLen=maxLen, meta=meta, numFeatures=numFeatures,
                                                                          batch_size=len(testSet), idx=idx_test)

        print("\nDataset: # subjects = ", len(subjects), "   # allergic subjects = ", len(subjAll), "   # non-allergic subjects = ", len(subjNonAll))
        # print("Total # allergic subjects = ", len(subjAll),
        #       "Total # non-allergic subjects = ", len(subjNonAll))
        subjAll = [x for x in subjAll if x not in testAllergy]
        subjNonAll = [x for x in subjNonAll if x not in testNonAllergy]
        print("TrainingSet: # allergic subjects = ", len(subjAll),
              "    # non-allergic subjects = ", len(subjNonAll))
        print("TestSet: # allergic subjects = ", len(testAllergy),
              "    # non-allergic subjects = ", len(testNonAllergy))




        for i in range(0, numFold):
            print('\n### Loop = %i\tFold = %i' % (loop, i))
            saver = tf.train.Saver(max_to_keep=10)
            init = tf.global_variables_initializer()  # Initialize the variables (i.e. assign their default value)
            LogDir = LogDir_base + "_Fold_" + str(i)

            trainingLoss = list()
            validationLoss = list()

            ### Prepare valdiation fold
            validateFoldSizeAllergy = round((1 / numFold) * len(subjAll))
            validateAllergy = subjAll[i * validateFoldSizeAllergy : (i + 1) * validateFoldSizeAllergy]
            validateFoldSizeNonAllergy = round((1 / numFold) * len(subjNonAll))
            validateNonAllergy = subjNonAll[i * validateFoldSizeNonAllergy: (i + 1) * validateFoldSizeNonAllergy]
            validateSet = np.concatenate([validateAllergy, validateNonAllergy])
            random.shuffle(validateSet)

            ### Prepare Training fold
            trainAllergy = [x for x in subjAll if x not in validateAllergy]
            trainNonAllergy = [x for x in subjNonAll if x not in validateNonAllergy]
            trainSet = np.concatenate([trainAllergy, trainNonAllergy])
            random.shuffle(trainSet)

            idx_train = 0
            idx_validate = 0

            ## Get validateing data
            validateData, validateLabel, validateLen, idx_validate = getbatch(df = df, timepoints = timepoints, subjects = validateSet,
                                                              maxLen = maxLen, meta = meta, numFeatures = numFeatures,
                                                              batch_size = len(validateSet), idx = idx_validate)

            with tf.Session() as sess:
                sess.run(init)

                TrainLogDir = str(LogDir + "_Train")
                TestLogDir = str(LogDir + "_Test")
                TrainWriter = tf.summary.FileWriter(TrainLogDir)
                TrainWriter.add_graph(sess.graph)
                TestWriter = tf.summary.FileWriter(TestLogDir)
                TestWriter.add_graph(sess.graph)

                for step in range(1, training_steps + 1):
                    while(idx_train < len(trainSet)):
                        batch_x, batch_y, batch_seqlen, idx_train = getbatch(df = df, timepoints = timepoints, subjects = trainSet,
                                                                             maxLen = maxLen, meta = meta, numFeatures = numFeatures,
                                                                             batch_size = len(trainSet), idx = idx_train)
                        opt, training_cost = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})

                    idx_train = 0
                    if step % display_step == 0 or step == 1:
                        with tf.name_scope("TrainingMeasurements"):
                           summary = sess.run(merged_summary, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})
                           TrainWriter.add_summary(summary, step)


                        saver.save(sess, os.path.join(TrainLogDir, 'model.ckpt'), global_step=step)

                        # Evaluate validatingSet for tensorboard
                        validate_data = validateData
                        validate_label = validateLabel
                        validate_seqlen = validateLen
                        summary = sess.run(merged_summary, feed_dict={x: validate_data, y: validate_label, seqlen: validate_seqlen})
                        TestWriter.add_summary(summary, step)

                        acc, y_t, y_p, y_sm, y_sftent, validate_cost = sess.run(
                            [accuracy, y_true, y_pred, y_sftmx, y_softmxEntropy, cost],
                            feed_dict={x: validate_data, y: validate_label, seqlen: validate_seqlen})

                        #print("Step = ", step, "     Training loss = ", training_cost, "     Validate loss = ", validate_cost)
                        print('Step = %i\tTraining Loss = %f\t Validation Loss = %f' % (step, training_cost, validate_cost))
                        trainingLoss.append(training_cost)
                        validationLoss.append(validate_cost)


                np.savetxt(LogDir + "_trainingLoss.csv", trainingLoss, delimiter=",")
                np.savetxt(LogDir + "_validationLoss.csv", validationLoss, delimiter=",")


                # Evaluate validateingSet
                test_data = testData
                test_label = testLabel
                test_seqlen = testLen
                acc, y_t, y_p, y_sm, y_sftent, test_cost = sess.run([accuracy, y_true, y_pred, y_sftmx, y_softmxEntropy, cost], feed_dict={x: test_data, y: test_label, seqlen: test_seqlen})


                print("** Test Loss = ", test_cost)
                testingLoss.append(test_cost)


                precision = sk.precision_score(y_t, y_p, average=None)
                recall = sk.recall_score(y_t, y_p, average=None)
                fscore = sk.f1_score(y_t, y_p, average=None)
                mcc = sk.matthews_corrcoef(y_t, y_p, sample_weight=None)
                fpr, tpr, _ = sk.roc_curve(y_t,y_sm[:,1])
                roc_auc = sk.auc(fpr, tpr)
                testAccuracy.append(acc)
                testPrecision.append(precision)
                testRecall.append(recall)
                testFscore.append(fscore)
                testAUC.append(roc_auc)
                tprs.append(sp.interp(mean_fpr, fpr, tpr))
                testMCC.append(mcc)
                tprs[-1][0] = 0.


    np.savetxt(LogDir + "_testingLoss.csv", testingLoss, delimiter=",")
    testMCC = np.asarray(testMCC)
    MCC_resolved = testMCC[testMCC != 0]
    print("Average AUC = %0.2f,   Average MCC = %0.2f" % (np.mean(testAUC),np.mean(MCC_resolved)))
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
    print("OutputPrefix = ", OutDir)


    ## Read OTU matrix
    df = pd.read_csv(InputFile, sep=",", index_col = 0)
    features = list(df.index)
    samples = list(df)
    numFeatures = len(features)
    numSamples = len(samples)

    ## Read Metadatafile
    meta = pd.read_csv(MetadataFile, sep=",", index_col = False)
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

    print("Before filtration:  # Finnish samples = ", finn, "   # Russian samples = ", russ, "   # Estonian samples = ", est)
    #print('Before filtration:  # Finnish samples = %i\t# Russian samples = %i\t# Estonian samples = %i' %(finn,russ,est))
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

    print("After filtration:   # Finnish subjects = ", finn, "   # Russian subjects = ", russ,
          "   # Estonian subjects = ", est)

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


    ### Training LSTM
    start = timeit.default_timer()
    trainingLSTM(maxLen, numFeatures, subjects, meta, timepoints, df)
    stop = timeit.default_timer()
    print("Elapsed time = %0.2f Seconds" % (stop - start))


if __name__ == '__main__':
	tf.app.run(main = main)