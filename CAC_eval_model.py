import argparse
import json

from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tf
import torch
import torch.nn as nn

from networks import openSetClassifier
import datasets.utils as dataHelper
from utils import find_anchor_means, gather_outputs

import metrics
import scipy.stats as st
import numpy as np
from dataset_utils import create_dataset
from sklearn.metrics import  accuracy_score

def find_threshold(num_train_classes, dists, y, y_pred):
    # Initialize list to store the best threshold for each class
    best_thresholds = [0] * 12

    # Iterate over each class
    for _class in range(12):
        best_accuracy = 0
        
        # Iterate over potential threshold values
        for pot in range(-1, 8):
            for K in range(9, 0, -1):
                threshold = K * 10**(-pot)
                pred_labels = []
                correct_labels = []
                
                # Evaluate accuracy using the current threshold
                for i in range(y.shape[0]):
                    proposed_class = y_pred[i]
                    if proposed_class == _class:
                        correct_labels.append(y[i])
                        if dists[i, proposed_class] > threshold:
                            proposed_class = num_train_classes
                        pred_labels.append(proposed_class)
                
                # Calculate accuracy
                acc = accuracy_score(correct_labels, pred_labels)
                
                # Update best accuracy and threshold if current accuracy is better
                if acc >= best_accuracy:
                    best_accuracy = acc
                    best_threshold = threshold
        
        # Store the best threshold for the current class
        best_thresholds[_class] = best_threshold
        print(f"Class {_class}: Best accuracy = {best_accuracy}, Best threshold = {best_threshold}")

    return best_thresholds


transform = v2.Compose(
    [
        v2.Grayscale(num_output_channels=1),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)



# Define path
path = "../main_dataset"
path = "../main_dataset"


parser = argparse.ArgumentParser(description='Closed Set Classifier Training')
parser.add_argument('--dataset', default = "PLANKTON", type = str, help='Dataset for evaluation', 
									choices = ['PLANKTON'])
parser.add_argument('--num_trials', default = 5, type = int, help='Number of trials to average results over?')
parser.add_argument('--start_trial', default = 0, type = int, help='Trial number to start evaluation for?')
parser.add_argument('--backbone', default = None, type = str, help='Define backbone model', choices = ['resnet', 'densenet', 'vit'])
parser.add_argument('--name', default = '', type = str, help='Name of training script?')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from utils import find_anchor_means, gather_outputs

def find_anchor_means(net, mapping, loader, only_correct = False):
    ''' Tests data and fits a multivariate gaussian to each class' logits. 
        If dataloaderFlip is not None, also test with flipped images. 
        Returns means and covariances for each class. '''
    #find gaussians for each class
    logits, labels = gather_outputs(net, mapping, loader, only_correct = only_correct)

    num_classes = cfg['num_known_classes']
    means = [None for i in range(num_classes)]

    for cl in range(num_classes):
        x = logits[labels == cl]
        x = np.squeeze(x)
        means[cl] = np.mean(x, axis = 0)

    return means

all_accuracy = []
all_auroc = []

accuracy_known = []
accuracy_known_th = []
accuracy_unknown_th = []

for trial_num in range(args.start_trial, args.start_trial + args.num_trials):
	print('==> Preparing data for trial {}..'.format(trial_num))
	with open('datasets/config.json') as config_file:
		cfg = json.load(config_file)[args.dataset]

	test_idx = [3 * (trial_num), 3 * (trial_num) + 1, 3 * (trial_num) + 2]

	#Create dataloaders for evaluation
	train_dataset, test_dataset = create_dataset(path, transform, test_idx=test_idx)
	batch_size = cfg["batch_size"]
	knownloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	unknownloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

	mapping = [i for i in range(cfg["num_classes"])]

	print('==> Building open set network for trial {}..'.format(trial_num))
	net = openSetClassifier.openSetClassifier(cfg['num_known_classes'], cfg['im_channels'], cfg['im_size'], dropout = cfg['dropout'], backbone = args.backbone)
	checkpoint = torch.load('networks/weights/{}/{}_{}_{}CACclassifierAnchorLoss.pth'.format(args.dataset, args.dataset, trial_num, args.name))

	net = net.to(device)
	net_dict = net.state_dict()
	pretrained_dict = {k: v for k, v in checkpoint['net'].items() if k in net_dict}
	if 'anchors' not in pretrained_dict.keys():
		pretrained_dict['anchors'] = checkpoint['net']['means']
	net.load_state_dict(pretrained_dict)
	net.eval()

	#find mean anchors for each class
	anchor_means = find_anchor_means(net, mapping, knownloader,  only_correct = True)
	
	net.set_anchors(torch.Tensor(np.array(anchor_means)))

	
	print('==> Evaluating open set network accuracy for trial {}..'.format(trial_num))
	x, y = gather_outputs(net, mapping, unknownloader, data_idx = 1, calculate_scores = True)

 
 	# Get mask for known and unknown classes
	mask_known, mask_unk = metrics.get_mask(y, cfg["num_known_classes"])

	# Change every unknown class to same class
	y[mask_unk] = 12

	# Get the predicted classes
	y_pred = np.argmin(x, axis=1)
	print("==> Testing known class accuracy before threshold")
	acc_known, _ = metrics.calc_accuracies(y, y_pred, cfg["num_known_classes"])

	threshold = find_threshold(cfg['num_known_classes'], x, y, y_pred)
	pred_labels=[]
	for i in range(y.shape[0]):
		proposed_Class=y_pred[i]
		if(x[i, proposed_Class]>threshold[proposed_Class]):
			proposed_Class=cfg['num_known_classes']
		pred_labels.append(proposed_Class)
  
	acc_known_th, acc_unk_th = metrics.calc_accuracies(y, pred_labels, cfg["num_known_classes"])
	pred_labels = np.array(pred_labels)

	accuracy_known += [round(acc_known, 4)]
	accuracy_known_th += [round(acc_known_th, 4)]
	accuracy_unknown_th += [round(acc_unk_th, 4)]
 
	# Full accuracy score
	accuracy = metrics.accuracy_th(pred_labels, y)
	all_accuracy += [round(accuracy, 4)]
	
	print('==> Evaluating open set network AUROC for trial {}..'.format(trial_num))
	auroc = metrics.auroc_th(pred_labels[mask_known], pred_labels[mask_unk])
	all_auroc += [round(auroc, 4)]

mean_auroc = np.mean(all_auroc)
mean_acc = np.mean(all_accuracy)


print('Known class accuracy: {}'.format(accuracy_known))
print('Known class accuracy after threshold: {}'.format(accuracy_known_th))
print('Unknown class accuracy after threshold: {}'.format(accuracy_unknown_th))
print()
print('Raw Top-1 Accuracy: {}'.format(all_accuracy))
print('Raw AUROC: {}'.format(all_auroc))
print('Average Top-1 Accuracy: {}'.format(mean_acc))
print('Average AUROC: {}'.format(mean_auroc))


