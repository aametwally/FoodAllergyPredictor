
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import sys
import numpy as np
import os
import struct
import argparse
from array import array as pyarray
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, matthews_corrcoef
import pandas as pd


parser = argparse.ArgumentParser(description='RF on data')

parser.add_argument("--data", help="raw or latent")

args = parser.parse_args()


if __name__ == '__main__':
	if args.data == None:
		print("Please specify raw or latent for data flag")
	else:
		dataset=args.data
		svm_accuracy = []
		svm_roc_auc = []
		svm_precision = []
		svm_recall = []
		svm_f_score = []
		svm_pred = []
		svm_prob = []
		svm_mcc = []
		
		fp = pd.read_csv("diabimmune_metadata_allcountries_allergy_noQuotes.csv", index_col=3)
		allergy = fp["allergy"]
		allergy = pd.factorize(allergy)
		subject = fp["subjectID"]

		labels = allergy[1]
		allergy = allergy[0]

		subject_data = {'ID': subject, 'label': allergy}
		split_df = pd.DataFrame(data=subject_data).groupby("ID").median()

		split_sub = split_df.index.values
		split_lab = np.array(split_df[["label"]].as_matrix()).reshape(-1)
		
		print(len(split_sub))
		print(len(split_lab))
		
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
			skf = StratifiedKFold(split_lab, n_folds=10, shuffle=True)
			cv = 0
			for id_train_index, id_test_index in skf:
			
				train_index = []
				test_index = []
				for i in range(0, len(subject)):
					if subject[i] in split_sub[id_train_index]:
						train_index.append(i)
					else:
						test_index.append(i)

			
				x = data[train_index]
				y = allergy[train_index]
				tx = data[test_index]
				ty = allergy[test_index]
				
				x = (x - x.min())/(x.max() - x.min())
				#x = x.fillna(value=0)
				tx = (tx - tx.min())/(tx.max() - tx.min())
				#x = tx.fillna(value=0)
				print("Running fold %d for set %d" % (cv, set))
				
				cv_grid = [ {'C': [1, 10, 100, 1000], 'kernel': ['linear']}, {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']} ]
				clf = GridSearchCV(SVC(C=1, probability=True), param_grid=cv_grid, cv=StratifiedKFold(y, n_folds=5, shuffle=True), n_jobs=-1, scoring="accuracy")
				clf.fit(x, y)
				prob = [row[1] for row in clf.predict_proba(tx)]
				pred = [row for row in clf.predict(tx)]
				svm_accuracy.append(clf.score(tx,ty))
				
				test_data = {"ID": subject[test_index], "prob":prob, "y": ty, "pred":pred }
				test_df = pd.DataFrame(data=test_data)
				test_df = test_df.groupby("ID").median()
				
				prob = test_df["prob"]
				ty = test_df["y"]
				pred = [round(x) for x in (test_df["pred"]-0.10)]
				
				svm_roc_auc.append(roc_auc_score(ty, prob))
				svm_precision.append(precision_score(ty, pred, average='weighted'))
				svm_recall.append(recall_score(ty, pred, average='weighted'))
				svm_f_score.append(f1_score(ty, pred, average='weighted'))
				svm_pred.append(pred)
				svm_prob.append(prob)
				svm_mcc.append(matthews_corrcoef(ty, pred))
				cv += 1
				
		print("Accuracy = " + str(np.mean(svm_accuracy)) + " (" + str(np.std(svm_accuracy)) + ")\n")
		print(svm_accuracy)
		print("\n\nROC AUC = " + str(np.mean(svm_roc_auc)) + " (" + str(np.std(svm_roc_auc)) + ")\n")
		print(svm_roc_auc)
		print("\n\nMCC = " + str(np.mean(svm_mcc)) + " (" + str(np.std(svm_mcc)) + ")\n")
		print(svm_mcc)
		print("\n\nPrecision = " + str(np.mean(svm_precision)) + " (" + str(np.std(svm_precision)) + ")\n")
		print("Recall = " + str(np.mean(svm_recall)) + " (" + str(np.std(svm_recall)) + ")\n")
		print("F1 = " + str(np.mean(svm_f_score)) + " (" + str(np.std(svm_f_score)) + ")\n")


		f = open(dataset + "_svm.txt", 'w')
		f.write("Mean Accuracy: " + str(np.mean(svm_accuracy)) + " (" + str(np.std(svm_accuracy))+ ")\n")
		f.write(str(svm_accuracy) + "\n")
		f.write("\nMean ROC: " + str(np.mean(svm_roc_auc)) + " (" + str(np.std(svm_roc_auc))+ ")\n")
		f.write(str(svm_roc_auc) + "\n")
		f.write("\nMean MCC: " + str(np.mean(svm_mcc)) + " (" + str(np.std(svm_mcc))+ ")\n")
		f.write(str(svm_mcc) + "\n")
		f.write("\nMean Precision: " + str(np.mean(svm_precision)) + " (" + str(np.std(svm_precision))+ ")\n")
		f.write(str(svm_precision) + "\n")
		f.write("\nMean Recall: " + str(np.mean(svm_recall)) + " (" + str(np.std(svm_recall))+ ")\n")
		f.write(str(svm_recall) + "\n")
		f.write("\nMean F-score: " + str(np.mean(svm_f_score)) + " (" + str(np.std(svm_f_score))+ ")\n")
		f.write(str(svm_f_score) + "\n")

		for i in range(0,100):
			f.write("\nPredictions for " + str(i) + "\n")
			f.write("\n" + str(svm_pred[i]) + "\n")
			f.write("\n" + str(svm_prob[i]) + "\n")
		f.close()   

