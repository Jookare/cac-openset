from torch.utils.data import Dataset, Sampler
import torch
import random
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset

class BalancedSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.num_classes = dataset.num_classes
        self.class_indices = [[] for _ in range(self.num_classes)]
        for idx in range(len(dataset)):
            _, _, label = dataset[idx]
            self.class_indices[label].append(idx)

        self.num_samples = min(len(indices) for indices in self.class_indices)

    def __iter__(self):
        indices = []
        for _ in range(self.num_samples):
            for class_indices in self.class_indices:
                indices.append(random.choice(class_indices))
        return iter(indices)

    def __len__(self):
        return self.num_samples * self.num_classes


class CustomDataset(Dataset):
    def __init__(self, image_paths, image_titles, classes, class_to_label, preprocess_train, tokenizer):
        # Image path
        self.image_paths = image_paths
        self.titles = image_titles
        self.classes = classes
        self.num_classes = len(classes)
        

        # dictionary of title to label
        self.title_to_label = class_to_label
        self.preprocess_train = preprocess_train
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, idx):
        image = self.preprocess_train(Image.open(self.image_paths[idx]))
        title = self.tokenizer(self.titles[idx])  # Assuming tokenizer is defined somewhere
        label = self.title_to_label[self.titles[idx]]
        return image, title.squeeze(0), label   # Returning idx for debugging or tracking purposes
    

def getImages(main_folder_path, classes, training_set=True, start=None, end=None):
    if training_set:
        path = os.path.join(main_folder_path, "training")
    else:
        path = os.path.join(main_folder_path, "testing")

    images = []
    labels = []

    # Loop through each subfolder
    for idx, _class in enumerate(classes):
        folder_path = os.path.join(path, _class)
        # Get a list of image files in the subfolder
        image_files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith((".jpg", "png","jpeg"))
        ]
        if (start is not None and end is not None ):
            labels += [_class] * (end - start)
            images += image_files[start:end]
        else:
            labels += [_class] * len(image_files)
            images += image_files

    return images, labels
    


def create_dataset(main_folder_path, preprocess, tokenizer, train_idx=None, test_idx=None):
    # Get a list of classes
    train_path = os.path.join(main_folder_path, "training")
    classes = sorted([f for f in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, f))])

    # If test indices are given we assume open-set recognition
    if test_idx:
        assert test_idx, "Need to have indices of test classes"
        test_classes = [classes[i] for i in test_idx]
        train_classes = [classes[i] for i in train_idx] if train_idx else [label for label in classes if label not in test_classes]

        test_classes_names = []
        for idx, label in enumerate(train_classes):
            if label in test_classes:
                label = f"Unk_{label}"
            test_classes_names.append(label)

        test_classes_names += [f"Unk_{c}" for c in test_classes if c not in train_classes]
        
        # Add also the train classes to test classes
        test_classes = train_classes + [c for c in test_classes if c not in train_classes]

    else:
        train_classes = [classes[i] for i in train_idx] if train_idx else classes
        test_classes = train_classes.copy()
        test_classes_names = train_classes.copy()

    # Get the images and the corresponding numerical labels
    
    train_images, train_labels = getImages(main_folder_path, train_classes, training_set=True)
    test_images, test_labels   = getImages(main_folder_path, test_classes, training_set=False)

    
    class_to_label = {cls: label for label, cls in enumerate(classes)}
    # Create dataset
    train_dataset = CustomDataset(train_images, train_labels, train_classes, class_to_label, preprocess, tokenizer)
    test_dataset  = CustomDataset(test_images,  test_labels,  test_classes,  class_to_label, preprocess, tokenizer)

    return train_dataset, test_dataset


def test_bioclip_model(model, val_dataloader, tokenizer, device):
    
    classes = val_dataloader.dataset.classes
    classes = tokenizer(classes).to(device)

    predicted_labels = []
    correct_labels = []
    with torch.no_grad():
        for batch in val_dataloader:

            images, _, labels = batch 
            correct_labels.extend(list(labels.cpu().numpy()))
            
            images = images.to(device)
            labels = labels.to(device)
            
            with torch.cuda.amp.autocast():
                image_features = model.encode_image(images)
                text_features = model.encode_text(classes)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

            # print(text_probs)
            predictions = torch.argmax(text_probs, dim=-1)        
            predicted_labels.extend(list(predictions.cpu().numpy()))

    predicted_labels = np.array(predicted_labels)
    correct_labels = np.array(correct_labels)

    print("Validation accuracy:")
    accuracy = np.sum(predicted_labels == correct_labels) / (len(predicted_labels))*100
    print(f"\t accuracy percentage: {accuracy:.2f}%")
    return predicted_labels, correct_labels

def test_clip_model(model, val_dataloader, tokenizer, device):
    
    classes = val_dataloader.dataset.classes
    classes = tokenizer(classes).to(device)

    predicted_labels = []
    correct_labels = []
    with torch.no_grad():
        for batch in val_dataloader:

            images, _, labels = batch 
            correct_labels.extend(list(labels.cpu().numpy()))
            
            images = images.to(device)
            labels = labels.to(device)
            
            logits_per_image, logits_per_text = model(images, classes)

            # print(text_probs)
            predictions = torch.argmax(logits_per_image, dim=-1)        
            predicted_labels.extend(list(predictions.cpu().numpy()))

    predicted_labels = np.array(predicted_labels)
    correct_labels = np.array(correct_labels)

    print("Validation accuracy:")
    accuracy = np.sum(predicted_labels == correct_labels) / (len(predicted_labels))*100
    print(f"\t accuracy percentage: {accuracy:.2f}%")
    return predicted_labels, correct_labels

