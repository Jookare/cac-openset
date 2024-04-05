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


transform = v2.Compose(
    [
        v2.Grayscale(num_output_channels=1),
        v2.RandomAffine(degrees=(0, 180), translate=(0, 0.1), scale=(0.95, 1.05), fill=255),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0], 
                     std=[1]),
    ]
)
# Define path
path = "./main_dataset"


parser = argparse.ArgumentParser(description='Closed Set Classifier Training')
parser.add_argument('--dataset', default = "MNIST", type = str, help='Dataset for evaluation', 
									choices = ['PLANKTON'])
parser.add_argument('--num_trials', default = 5, type = int, help='Number of trials to average results over?')
parser.add_argument('--start_trial', default = 0, type = int, help='Trial number to start evaluation for?')
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

	mapping = [None for i in range(cfg["num_classes"])]
	
	for i in range(15):
		mapping[i] = i


	print('==> Building open set network for trial {}..'.format(trial_num))
	net = openSetClassifier.openSetClassifier(cfg['num_known_classes'], cfg['im_channels'], cfg['im_size'], dropout = cfg['dropout'])
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
	
	net.set_anchors(torch.Tensor(anchor_means))

	
	print('==> Evaluating open set network accuracy for trial {}..'.format(trial_num))
	x, y = gather_outputs(net, mapping, knownloader, data_idx = 1, calculate_scores = True)
	accuracy = metrics.accuracy(x, y)
	all_accuracy += [accuracy]

	print('==> Evaluating open set network AUROC for trial {}..'.format(trial_num))
	xK, yK = gather_outputs(net, mapping, knownloader, data_idx = 1, calculate_scores = True)
	xU, yU = gather_outputs(net, mapping, unknownloader, data_idx = 1, calculate_scores = True, unknown = True)

	auroc = metrics.auroc(xK, xU)
	all_auroc += [auroc]

mean_auroc = np.mean(all_auroc)
mean_acc = np.mean(all_accuracy)

print('Raw Top-1 Accuracy: {}'.format(all_accuracy))
print('Raw AUROC: {}'.format(all_auroc))
print('Average Top-1 Accuracy: {}'.format(mean_acc))
print('Average AUROC: {}'.format(mean_auroc))