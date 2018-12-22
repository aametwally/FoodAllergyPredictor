#!/usr/bin/env python -W ignore::DeprecationWarning

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# Third-party libraries
import numpy as np
import os
import sys
import struct
import argparse
from array import array as pyarray
from seqlearn.hmm import MultinomialHMM
from seqlearn.evaluation import whole_sequence_accuracy
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit, StratifiedKFold
from sklearn.metrics import matthews_corrcoef, accuracy_score, roc_auc_score
import pandas as pd
from math import floor
from hmmlearn import hmm
import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")


parser = argparse.ArgumentParser(description='RF on data')

parser.add_argument("--data", help="raw or latent")

args = parser.parse_args()


if __name__ == '__main__':
	if args.data == None:
		print("Please specify raw or latent for data flag")
	else:
		dataset = args.data
		hmm_ws_accuracy = []
		hmm_last_accuracy =[]
		hmm_ws_mcc = []
		hmm_last_mcc = []
		hmm_pred = []
		hmm_prob = []
		hmm_auc = []

		fp = pd.read_csv("diabimmune_metadata_allcountries_allergy_noQuotes.csv", index_col=3)
		allergy = fp["allergy"]
		allergy = pd.factorize(allergy)
		subject = fp["subjectID"]

		labels = allergy[1]
		allergy = allergy[0]

		subject_data = {'ID': subject, 'label': allergy}
		split_df = pd.DataFrame(data=subject_data)
		lengths = split_df.groupby("ID").count()
		ids = np.unique(split_df["ID"])
		split_lab = np.array(split_df.groupby("ID").median()[["label"]]).reshape(-1)

		if dataset == "latent":
			data = pd.read_csv("diabimmune_embeddeddata_50_addedHeader.csv")
			
		elif dataset == "raw":
			data = pd.read_csv("diabimmune_taxa_genus_allcountries.csv", index_col=0)
		elif dataset == "latent_25":
			data = pd.read_csv("diabimmune_taxa_genus_allcountries_selected_" + str(dataset) + ".csv")
			
		else:
			data = pd.read_csv("diabimmune_taxa_genus_allcountries_selected_" + str(dataset) + ".csv", index_col=0)


		data = data.transpose().as_matrix()


		for set in range(0,10):
			skf = StratifiedKFold(n_splits=10, shuffle=True)
			cv = 0
			for id_train_index, id_test_index in skf.split(ids, split_lab):
				train_index = []
				test_index = []
				for i in range(0, len(subject)):
					if subject[i] in ids[id_train_index]:
						train_index.append(i)
					else:
						test_index.append(i)

				train_lengths = []
				test_lengths = []
				
				lengths = pd.DataFrame(lengths)
				for i in ids[id_train_index]:
					train_lengths.append(lengths.loc[i]["label"])
				for i in ids[id_test_index]:
					test_lengths.append(lengths.loc[i]["label"])
				
				num_features = np.array(data).shape[-1]
						
				x = data[train_index]
				y = allergy[train_index]
				tx = data[test_index]
				ty = allergy[test_index]
				
				print("Running fold %d for set %d" % (cv, set))
				clf=hmm.GMMHMM(n_components=2,n_mix=4,n_iter=100)
				clf.fit(x, train_lengths)
				pred = [row for row in clf.predict(tx, test_lengths)]
				pred_last = []
				ty_last = []
				length_count = 0
				for i in range(0, len(test_lengths)):
					length_count += test_lengths[i]
					pred_last.append(pred[length_count - 1])
					ty_last.append(ty[length_count-1])																																																																																																																																	
				hmm_pred.append(pred)
				
				
				acc_ws_0 = whole_sequence_accuracy(ty, pred, test_lengths)
				acc_last_0 = accuracy_score(ty_last, pred_last)
				mcc_ws_0 = matthews_corrcoef(ty, pred)
				mcc_last_0 = matthews_corrcoef(ty_last, pred_last)

				
				acc_ws_1 = whole_sequence_accuracy([(z + 1)%2 for z in ty], pred, test_lengths)
				acc_last_1 = accuracy_score([(z + 1)%2 for z in ty_last], pred_last)
				mcc_ws_1 = matthews_corrcoef([(z + 1)%2 for z in ty], pred)
				mcc_last_1 = matthews_corrcoef([(z + 1)%2 for z in ty_last], pred_last)	
				
				if acc_last_0 > acc_last_1:
					acc_ws = acc_ws_0
					acc_last = acc_last_0
					mcc_ws = mcc_ws_0
					mcc_last = mcc_last_0
					prob = [row[1] for row in clf.predict_proba(tx)]

				
				else:
					acc_ws = acc_ws_1
					acc_last = acc_last_1
					mcc_ws = mcc_ws_1
					mcc_last = mcc_last_1
					prob = [row[0] for row in clf.predict_proba(tx)]
				roc = roc_auc_score(ty, prob)
				hmm_prob.append(prob)
				print(acc_last)
				print(roc)
				print(mcc_last)
				hmm_ws_accuracy.append(acc_ws)
				hmm_last_accuracy.append(acc_last)
				hmm_ws_mcc.append(mcc_ws)
				hmm_last_mcc.append(mcc_last)
				hmm_auc.append(roc)
				
				cv += 1



		print("Whole Sequence Accuracy = " + str(np.mean(hmm_ws_accuracy)) + " (" + str(np.std(hmm_ws_accuracy)) + ")\n")
		print(hmm_ws_accuracy)
		print("\n\nLast Position Accuracy = " + str(np.mean(hmm_last_accuracy)) + " (" + str(np.std(hmm_last_accuracy)) + ")\n")
		print(hmm_last_accuracy)
		print("\n\nROC AUC = " + str(np.mean(hmm_auc)) + " (" + str(np.std(hmm_auc)) + ")\n")
		print(hmm_auc)
		print("\n\nWhole Sequence MCC = " + str(np.mean(hmm_ws_mcc)) + " (" + str(np.std(hmm_ws_mcc)) + ")\n")
		print(hmm_ws_mcc)
		print("\n\nLast Position MCC = " + str(np.mean(hmm_last_mcc)) + " (" + str(np.std(hmm_last_mcc)) + ")\n")
		print(hmm_last_mcc)

		f = open(dataset + "_hmm.txt", 'w')
		f.write("Mean Whole Sequence Accuracy: " + str(np.mean(hmm_ws_accuracy)) + " (" + str(np.std(hmm_ws_accuracy))+ ")\n")
		f.write(str(hmm_ws_accuracy) + "\n")
		f.write("\Last Position Accuracy: " + str(np.mean(hmm_last_accuracy)) + " (" + str(np.std(hmm_last_accuracy))+ ")\n")
		f.write(str(hmm_last_accuracy) + "\n")
		f.write("\nMean ROC: " + str(np.mean(hmm_auc)) + " (" + str(np.std(hmm_auc))+ ")\n")
		f.write(str(hmm_auc) + "\n")
		f.write("\Whole Sequence MCC: " + str(np.mean(hmm_ws_mcc)) + " (" + str(np.std(hmm_ws_mcc))+ ")\n")
		f.write(str(hmm_ws_mcc) + "\n")
		f.write("\nLast Position MCC: " + str(np.mean(hmm_last_mcc)) + " (" + str(np.std(hmm_last_mcc))+ ")\n")
		f.write(str(hmm_last_mcc) + "\n")
	

		for i in range(0,100):
			f.write("\nPredictions for " + str(i) + "\n")
			f.write("\n" + str(hmm_pred[i]) + "\n")
		f.close()
		  
