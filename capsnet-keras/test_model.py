import os
import cv2
import argparse
import numpy as np
from PIL import Image
import capsulenet as CAPS
import matplotlib.pyplot as plt
from utils import combine_images
from keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

save_dir = './result'
dataset_path = "../../Dataset/Animals/"

# Argument Parser
parser = argparse.ArgumentParser(description="Parameters for testing the model")
parser.add_argument('-c', '--class_to_classify', default='cat', type=str, help="Class of image to test with.")
parser.add_argument('-i', '--image', default=1, type=int, help="Number of the image to test on.")
args = parser.parse_args()

if args.class_to_classify is None:
    print("Please enter a class to proceed.\n(Either: 'cat' (c), 'dog' (d), 'fox' (f), 'hyena' (h) or 'wolf' (w).")
    exit(0)

if args.class_to_classify not in ['cat', 'dog', 'fox', 'hyena', 'wolf', 'c', 'd', 'f', 'h', 'w']:
    print("Class must be either: 'cat' (c), 'dog' (d), 'fox' (f), 'hyena' (h) or 'wolf' (w).")
    exit(0)

if args.class_to_classify == 'cat' or args.class_to_classify == 'c':
    current_class = 'cat'
    current_class_folder = 'cats'
elif args.class_to_classify == 'dog' or args.class_to_classify == 'd':
    current_class = 'dog'
    current_class_folder = 'dogs'
elif args.class_to_classify == 'fox' or args.class_to_classify == 'f':
    current_class = 'fox'
    current_class_folder = 'foxes'
elif args.class_to_classify == 'hyena' or args.class_to_classify == 'h':
    current_class = 'hyena'
    current_class_folder = 'hyenas'
elif args.class_to_classify == 'wolf' or args.class_to_classify == 'w':
    current_class = 'wolf'
    current_class_folder = 'wolves'

if args.image is None:
    print("No image number entered, by default "+args.class_to_classify+"1.jpg is selected.")
print("No image number entered, by default "+args.class_to_classify+"1.jpg is selected.")

default_routing = 3
number_of_classes = 5
inverse_class_dict = {0:'Cat', 1:"Dog", 2:"Fox", 3:"Hyena", 4:"Wolves"}

# Need to change this line to stop loading the unnecessary training dataset while testing
(x_train, y_train, y_train_output), (x_test, y_test, y_test_output) = CAPS.load_custom_dataset(dataset_path)

model, eval_model, manipulate_model, hierarchy_train_model, hierarchy_eval_model = CAPS.CapsNet(input_shape=x_train.shape[1:], n_class=len(np.unique(np.argmax(y_train, 1))), routings=default_routing)

hierarchy_eval_model.load_weights(save_dir + '/Cosine_similarity_trained_model.h5')

image_number = args.image

current_file = dataset_path + current_class_folder + '/' + current_class + str(image_number)+'.jpg'

test_image = []

img = cv2.imread(current_file)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (28, 28))

test_image.append(img)

test_image = np.array(test_image)
test_image = test_image.reshape(-1, 28, 28, 1).astype('float32') / 255.

prediction = eval_model.predict(test_image, batch_size=100)

print("Predicted as: ", inverse_class_dict[np.argmax(prediction[0], 1)[0]])