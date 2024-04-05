
import os
import torch
import random
import numpy as np
from PIL import Image, ImageOps
from torchvision.transforms import v2
from torch.utils.data import Dataset
import cv2

# Taken from object detection utils. Removes the scale and other defects from the image leaving only plankton.
def clean_image(image):
    """Input: PIL Image
    Output: cleaned image
    """
    cv_image = np.array(image)
    threshold_value = 254
    # Apply thresholding
    _, thresholded_image = cv2.threshold(cv_image, threshold_value, 255, cv2.THRESH_BINARY)

    # Binarize image
    inverted_image = cv2.bitwise_not(thresholded_image)

    # Find all contours
    contours, _ = cv2.findContours(inverted_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort from largest to smallest
    c = sorted(contours, key=cv2.contourArea, reverse=True)

    # Fill all but the largest bounding box with white
    cv_image = cv2.drawContours(cv_image, c[1:-1], -1, color=(255), thickness=cv2.FILLED)

    image = Image.fromarray(cv_image)
    return image

def resizeImage(image_path, size):
    """Resizes the given image to a given size while keeping aspect ratio.
    Returns an Image object.
    """
    image = Image.open(image_path).convert("L")
    img = ImageOps.contain(image, size)

    # Create a new white image
    output_img = Image.new("L", size, 255)

    # Paste to the center of the image
    output_img.paste(img, ((size[0] - img.size[0]) // 2, (size[1] - img.size[1]) // 2))
    return output_img

def image_augmentation(image_files, folder_name, _class, data_transforms, image_size, N):

    destination_folder = os.path.join(folder_name, _class)
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # No need to augment we can just downsample
    if len(image_files) >= N:
        selected_images = random.sample(image_files, N)
        for count, path in enumerate(selected_images):
            image = resizeImage(path, size=image_size)
            # image = clean_image(image)
            image.save(f"{destination_folder}/{_class}_{str(count)}.jpg")

    else:
        for count in range(N):
            path = random.choice(image_files)

            image = resizeImage(path, size=image_size)
            # image = clean_image(image)
            image = data_transforms(image)

            image.save(f"{destination_folder}/{_class}_{str(count)}.jpg")
            

def remove_small_classes(main_folder_path, min_amount_images):
    subfolders = [
        f
        for f in os.listdir(main_folder_path)
        if os.path.isdir(os.path.join(main_folder_path, f))
    ]
    subfolders.sort()

    folders = []
    for idx, _class in enumerate(subfolders):
        folder_path = os.path.join(main_folder_path, _class)
        # Get a list of image files in the subfolder
        image_files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith((".jpg", "png","jpeg"))
        ]
        if (len(image_files) > min_amount_images):
            folders.append(_class)
            print(f"{_class}, {len(image_files)}")

    return folders


def create_dataset_folder(
    main_folder_path, image_size, num_images, transform, folder_name, min_amount_images=1, split=0.6, 
):
    """Takes in a path to image folder and divides it to training and testing datasets by applying given transformations.
    Inputs:
    main_folder_path: path to image folder
    num_classes:  (N, N) Number of classes in training and testing sets. Testing can have more classes than training but not less.
    num_images:  (N, N) Number of images per class in training and testing sets
    split: F, How large portion of the original images should go to training set. Rest go to testing.
    """
    # Find all subfolders
    classes = remove_small_classes(main_folder_path, min_amount_images)

    for idx, _class in enumerate(classes):
        folder_path = os.path.join(main_folder_path, _class)
        # Get a list of image files in the subfolder
        image_files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith((".jpg", "png","jpeg"))
        ]
        N = np.ceil(len(image_files) * split).astype("int")

        # Shuffle images
        random.shuffle(image_files)

        # Currently not taking into account that there can be more images than the required amount => no need to augment
        image_files1 = image_files[:N]
        image_files2 = image_files[N:]
        image_augmentation(image_files1, folder_name+"/training", _class, transform, image_size, num_images[0])
        image_augmentation(image_files2, folder_name+"/testing" , _class, transform, image_size, num_images[1])


# Dataset functions
class CustomDataset(Dataset):
    def __init__(self, images, labels, classes, transform):
        self.transform = transform
        self.images = images
        self.labels = labels
        self.classes = classes
        self.num_classes = len(classes)
     
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # *Randomly select two images
        image_path = self.images[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert("L")
        
        # Apply transformation
        if self.transform:
            image = self.transform(image)
        
        return image, label

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
            labels += [idx] * (end - start)
            images += image_files[start:end]
        else:
            labels += [idx] * len(image_files)
            images += image_files

    return images, labels
    


def create_dataset(main_folder_path, transform, train_idx=None, test_idx=None):
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
    # valid_images, valid_labels = getImages(main_folder_path, train_classes, training_set=True, start=100, end=1000)
    test_images, test_labels   = getImages(main_folder_path, test_classes, training_set=False)
    
    # Create dataset
    train_dataset = CustomDataset(train_images, train_labels, train_classes, transform=transform)
    # valid_dataset = CustomDataset(valid_images, valid_labels, train_classes, transform=transform)
    test_dataset = CustomDataset(test_images, test_labels, test_classes_names, transform=transform)

    return train_dataset,  test_dataset
    

