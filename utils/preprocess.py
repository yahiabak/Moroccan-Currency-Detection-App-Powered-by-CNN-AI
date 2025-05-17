import cv2
import os

def preprocess_images(input_dir, output_dir, image_size=(224, 224)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for label in os.listdir(input_dir):
        label_path = os.path.join(input_dir, label)
        output_label_path = os.path.join(output_dir, label)
        if not os.path.exists(output_label_path):
            os.makedirs(output_label_path)
        
        for image_file in os.listdir(label_path):
            img_path = os.path.join(label_path, image_file)
            
            # check if image is valid
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Unable to load image {img_path}")
                continue  
            
            # resizing the image
            img_resized = cv2.resize(img, image_size)
            
            # ensure the output path has a valid file extension
            filename = os.path.splitext(image_file)[0] + '.jpg'  
            output_path = os.path.join(output_label_path, filename)
            
            # write image
            cv2.imwrite(output_path, img_resized)

            print(f"Processed and saved: {output_path}")

preprocess_images("data/moroccan_money_dataset", "data/preprocessed_moroccan_money_dataset")
