"""
	Metrics used to evaluate performance.

	Dimity Miller, 2020
"""
import numpy as np
import sklearn.metrics

# Original accuracy
def accuracy(x, gt):
	predicted = np.argmin(x, axis = 1)
	total = len(gt)
	acc = np.sum(predicted == gt)/total
	return acc

# Original auroc
def auroc(inData, outData, in_low = True):
	inDataMin = np.min(inData, 1)
	outDataMin = np.min(outData, 1)
	
	allData = np.concatenate((inDataMin, outDataMin))
	labels = np.concatenate((np.zeros(len(inDataMin)), np.ones(len(outDataMin))))
	fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, allData, pos_label = in_low)

	return sklearn.metrics.auc(fpr, tpr)

# Added changes
def get_mask(y, num_train_classes):
	mask_known = y < num_train_classes
	mask_unk = y >= num_train_classes
	return mask_known, mask_unk

def accuracy_th(y_pred, y):
	total = len(y)
	acc = np.sum(y_pred == y)/total
	return acc

def auroc_th(inData, outData, in_low = True):

	allData = np.concatenate((inData, outData))
	labels = np.concatenate((np.zeros(len(inData)), np.ones(len(outData))))
	fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, allData, pos_label = in_low)

	return sklearn.metrics.auc(fpr, tpr)


def calc_accuracies(labels, predictions, train_class_num):

    labels = np.array(labels)
    predictions = np.array(predictions)

    known_mask = labels < train_class_num
    known_accuracy = np.mean(predictions[known_mask] == labels[known_mask])

    unknown_mask = labels >= train_class_num
    unknown_accuracy = np.mean(predictions[unknown_mask] == train_class_num)

    print(
        "Known classes={:.4f}, Unknown classes={:.4f}".format(
            known_accuracy,
            unknown_accuracy
        )
    )
