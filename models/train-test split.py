import os
import shutil
from sklearn.model_selection import train_test_split

# define paths
dataset_dir = 'data/preprocessed_moroccan_money_dataset'
train_dir = 'data/train_test_datasets/train'
test_dir = 'data/train_test_datasets/validation'

# Create train and test directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

denominations = os.listdir(dataset_dir)

# divide the images for the test and train
for denomination in denominations:
    denomination_dir = os.path.join(dataset_dir, denomination)
    
    if os.path.isdir(denomination_dir):
        images = os.listdir(denomination_dir)
        
        # train-test split
        train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)
        
        # create subfolders for labeling in the train and test directories
        os.makedirs(os.path.join(train_dir, denomination), exist_ok=True)
        os.makedirs(os.path.join(test_dir, denomination), exist_ok=True)
        
        # Copy the images into the appropriate folders
        for image in train_images:
            shutil.copy(os.path.join(denomination_dir, image), os.path.join(train_dir, denomination, image))
        
        for image in test_images:
            shutil.copy(os.path.join(denomination_dir, image), os.path.join(test_dir, denomination, image))

print("Séparation des données effectuée avec succès !")
