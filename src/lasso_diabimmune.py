
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
import pandas as pd
from array import array as pyarray
from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, matthews_corrcoef


parser = argparse.ArgumentParser(description='RF on data')

parser.add_argument("--data", help="raw or latent")

args = parser.parse_args()


if __name__ == '__main__':
	if args.data == None:
		print("Please specify raw or latent for data flag")
	else:
		dataset=args.data
		lasso_accuracy = []
		lasso_roc_auc = []
		lasso_precision = []
		lasso_recall = []
		lasso_f_score = []
		lasso_pred = []
		lasso_prob = []
		lasso_mcc = []
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
			skf = StratifiedKFold(n_splits=10, shuffle=True)
			cv = 0
			for id_train_index, id_test_index in skf.split(split_sub, split_lab):
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
				x = data[train_index]
				y = allergy[train_index]
				tx = data[test_index]
				ty = allergy[test_index]
				print("Running fold %d for set %d" % (cv, set))
				clf = LassoCV(alphas=np.logspace(-4, -0.5, 50), cv=5, n_jobs=-1)
				clf.fit(x, y)
				prob = [row for row in clf.predict(tx)]
				pred = [int(i > 0.5) for i in prob]
				lasso_accuracy.append(clf.score(tx,ty))
				test_data = {"ID": subject[test_index], "prob":prob, "y": ty, "pred":pred }
				test_df = pd.DataFrame(data=test_data)
				test_df = test_df.groupby("ID").median()
				
				prob = test_df["prob"]
				ty = test_df["y"]
				pred = [round(x) for x in (test_df["pred"]-0.10)]
				lasso_roc_auc.append(roc_auc_score(ty, prob))
				lasso_precision.append(precision_score(ty, pred, average='weighted'))
				lasso_recall.append(recall_score(ty, pred, average='weighted'))
				lasso_f_score.append(f1_score(ty, pred, average='weighted'))
				lasso_pred.append(pred)
				lasso_prob.append(prob)
				lasso_mcc.append(matthews_corrcoef(ty, pred))
				cv += 1
				


		print("Accuracy = " + str(np.mean(lasso_accuracy)) + " (" + str(np.std(lasso_accuracy)) + ")\n")
		print(lasso_accuracy)
		print("\n\nROC AUC = " + str(np.mean(lasso_roc_auc)) + " (" + str(np.std(lasso_roc_auc)) + ")\n")
		print(lasso_roc_auc)
		print("\n\nMCC = " + str(np.mean(lasso_mcc)) + " (" + str(np.std(lasso_mcc)) + ")\n")
		print(lasso_mcc)
		print("\n\nPrecision = " + str(np.mean(lasso_precision)) + " (" + str(np.std(lasso_precision)) + ")\n")
		print("Recall = " + str(np.mean(lasso_recall)) + " (" + str(np.std(lasso_recall)) + ")\n")
		print("F1 = " + str(np.mean(lasso_f_score)) + " (" + str(np.std(lasso_f_score)) + ")\n")

		f = open(dataset + "_lasso.txt", 'w')
		f.write("Mean Accuracy: " + str(np.mean(lasso_accuracy)) + " (" + str(np.std(lasso_accuracy))+ ")\n")
		f.write(str(lasso_accuracy) + "\n")
		f.write("\nMean ROC: " + str(np.mean(lasso_roc_auc)) + " (" + str(np.std(lasso_roc_auc))+ ")\n")
		f.write(str(lasso_roc_auc) + "\n")		
		f.write("\nMean MCC: " + str(np.mean(lasso_mcc)) + " (" + str(np.std(lasso_mcc))+ ")\n")
		f.write(str(lasso_mcc) + "\n")
		f.write("\nMean Precision: " + str(np.mean(lasso_precision)) + " (" + str(np.std(lasso_precision))+ ")\n")
		f.write(str(lasso_precision) + "\n")
		f.write("\nMean Recall: " + str(np.mean(lasso_recall)) + " (" + str(np.std(lasso_recall))+ ")\n")
		f.write(str(lasso_recall) + "\n")
		f.write("\nMean F-score: " + str(np.mean(lasso_f_score)) + " (" + str(np.std(lasso_f_score))+ ")\n")
		f.write(str(lasso_f_score) + "\n")

		for i in range(0,100):
			f.write("\nPredictions for " + str(i) + "\n")
			f.write("\n" + str(lasso_pred[i]) + "\n")
			f.write("\n" + str(lasso_prob[i]) + "\n")
		f.close()
			
