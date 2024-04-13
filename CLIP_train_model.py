import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import random_split
from dataset_utils import create_dataset
import torch.nn as nn
import torch.optim as optim

import json

import argparse

import datasets.utils as dataHelper

from networks import openSetClip

from utils import progress_bar

import os
import numpy as np

parser = argparse.ArgumentParser(description='Open Set Classifier Training')
parser.add_argument('--dataset', required = True, type = str, help='Dataset for training', 
									choices = ['PLANKTON'])
parser.add_argument('--trial', required = True, type = int, help='Trial number, 0-4 provided')
parser.add_argument('--resume', '-r', action='store_true', help='Resume from the checkpoint')
parser.add_argument('--alpha', default = 10, type = int, help='Magnitude of the anchor point')
parser.add_argument('--lbda', default = 0.1, type = float, help='Weighting of Anchor loss component')
parser.add_argument('--tensorboard', '-t', action='store_true', help='Plot on tensorboardX')
parser.add_argument('--name', default = "myTest", type = str, help='Optional name for saving and tensorboard')
parser.add_argument('--clip', default = "", type = str, help='Define clip model', choices = ['clip', 'bioclip'])
args = parser.parse_args()

if args.tensorboard:
	from torch.utils.tensorboard import SummaryWriter

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#parameters useful when resuming and finetuning
best_acc = 0
best_cac = 10000
best_anchor = 10000
start_epoch = 0

#Create dataloaders for training
print('==> Preparing data..')
with open('datasets/config.json') as config_file:
	cfg = json.load(config_file)[args.dataset]

# Define path
path = "../main_dataset"

# Dataset id for open-set test set
test_idx = [3 * (args.trial), 3 * (args.trial) + 1, 3 * (args.trial) + 2]

if args.clip == "clip":
	import clip
	model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
else:
	import open_clip
	model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip', device=device, jit=False)

train_dataset, test_dataset = create_dataset(path, preprocess, test_idx=test_idx)

# Split the dataset to training and validation and further split the validation to also testing
training_set, validation_set = random_split(train_dataset, [0.8, 0.2])

batch_size = cfg["batch_size"]
trainloader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
valloader = DataLoader(validation_set, batch_size=batch_size, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

mapping = [i for i in range(cfg['num_classes'])]

print('==> Building network..')
net = openSetClip.openSetClassifier(cfg['num_known_classes'], cfg['im_channels'], cfg['im_size'], dropout = cfg['dropout'])

# initialising with anchors
anchors = torch.diag(torch.Tensor([args.alpha for i in range(cfg['num_known_classes'])]))	
net.set_anchors(anchors)

net = net.to(device)
training_iter = int(args.resume)

# Train and validation

net.train()
optimizer = optim.SGD(net.parameters(), lr = cfg['openset_training']['learning_rate'][0], 
							momentum = 0.9, weight_decay = cfg['openset_training']['weight_decay'])

if args.tensorboard:
	writer = SummaryWriter('runs/{}_{}_{}ClosedSet'.format(args.dataset, args.trial, args.name))


def CACLoss(distances, gt):
	'''Returns CAC loss, as well as the Anchor and Tuplet loss components separately for visualisation.'''
	true = torch.gather(distances, 1, gt.view(-1, 1)).view(-1)
	non_gt = torch.Tensor([[i for i in range(cfg['num_known_classes']) if gt[x] != i] for x in range(len(distances))]).long().cuda()
	others = torch.gather(distances, 1, non_gt)
	
	anchor = torch.mean(true)

	tuplet = torch.exp(-others+true.unsqueeze(1))
	tuplet = torch.mean(torch.log(1+torch.sum(tuplet, dim = 1)))

	total = args.lbda*anchor + tuplet

	return total, anchor, tuplet


# Training
def train(epoch):
	print('\nEpoch: %d' % epoch)
	net.train()
	train_loss = 0
	correctDist = 0
	total = 0

	for batch_idx, (inputs, targets) in enumerate(trainloader):
		inputs, targets = inputs.to(device), targets.to(device)
		#convert from original dataset label to known class label
		targets = torch.Tensor(targets).long().to(device)

		optimizer.zero_grad()

		features = model.encode_image(inputs).float()
		outputs = net(features)
		cacLoss, anchorLoss, tupletLoss = CACLoss(outputs[1], targets)


		cacLoss.backward()

		optimizer.step()

		train_loss += cacLoss.item()

		_, predicted = outputs[1].min(1)

		total += targets.size(0)
		correctDist += predicted.eq(targets).sum().item()

		progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
			% (train_loss/(batch_idx+1), 100.*correctDist/total, correctDist, total))
  
	if args.tensorboard:
		acc = 100.*correctDist/total
		writer.add_scalar('train/accuracy', acc, epoch)
  
def val(epoch):
	global best_acc
	global best_anchor
	global best_cac
	net.eval()
	anchor_loss = 0
	cac_loss = 0
	correct = 0
	total = 0
	with torch.no_grad():
		for batch_idx, (inputs, targets) in enumerate(valloader):
			inputs = inputs.to(device)
			targets = torch.Tensor(targets).long().to(device)

			features = model.encode_image(inputs).float()
			outputs = net(features)

			cacLoss, anchorLoss, tupletLoss = CACLoss(outputs[1], targets)

			anchor_loss += anchorLoss
			cac_loss += cacLoss

			_, predicted = outputs[1].min(1)
			
			total += targets.size(0)

			correct += predicted.eq(targets).sum().item()

			progress_bar(batch_idx, len(valloader), 'Acc: %.3f%% (%d/%d)'
				% (100.*correct/total, correct, total))
   
	anchor_loss /= len(valloader)
	cac_loss /= len(valloader)
	acc = 100.*correct/total

	# Save checkpoint.
	state = {
		'net': net.state_dict(),
		'acc': acc,
		'epoch': epoch,
	}
	if not os.path.isdir('networks/weights/{}'.format(args.dataset)):
		os.mkdir('networks/weights/{}'.format(args.dataset))
  
	save_name = '{}_{}_{}CACclassifier'.format(args.dataset, args.trial, args.name)
	if anchor_loss <= best_anchor:
		print('Saving..')
		torch.save(state, 'networks/weights/{}/'.format(args.dataset)+save_name+'AnchorLoss.pth')
		best_anchor = anchor_loss


	if cac_loss <= best_cac:
		print('Saving..')
		torch.save(state, 'networks/weights/{}/'.format(args.dataset)+save_name+'CACLoss.pth')
		best_cac = cac_loss

	if acc >= best_acc:
		print('Saving..')
		torch.save(state, 'networks/weights/{}/'.format(args.dataset)+save_name+'Accuracy.pth')
		best_acc = acc
  
	if args.tensorboard:
		writer.add_scalar('val/accuracy', acc, epoch)
  
max_epoch = cfg['openset_training']['max_epoch'][training_iter]+start_epoch
for epoch in range(start_epoch, max_epoch):
	train(epoch)
	val(epoch)
