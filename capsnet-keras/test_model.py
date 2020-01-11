import os
import cv2
import random
import argparse
import numpy as np
from PIL import Image
import capsulenet as CAPS
import matplotlib.pyplot as plt
from utils import combine_images
from keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

save_dir = './result'

# Argument Parser
parser = argparse.ArgumentParser(description="Parameters for testing the model")
parser.add_argument('-c', '--class_to_classify', default='cat', type=str, help="Class of image to test with.")
parser.add_argument('-i', '--image', default=1, type=int, help="Number of the image to test on.")
args = parser.parse_args()

if args.class_to_classify is None:
    print("Please enter a class to proceed.\n")
    exit(0)

if args.class_to_classify not in ['cat', 'dog', 'fox', 'hyena', 'wolf', 'c', 'd', 'f', 'h', 'w', 'duck', 'eagle', 'hawk', 'parrot', 'sparrow', 'chair', 'table', 'sofa', 'nightstand', 'bed']:
    print("Class must be either: 'cat' (c), 'dog' (d), 'fox' (f), 'hyena' (h) or 'wolf' (w).")
    exit(0)
# {'cats':0, 'dogs':1, 'foxes':2, 'hyenas':3, 'wolves':4, 'ducks':5, 'eagles':6, 'hawks':7, 'parrots':8, 'sparrows':9, 'chair':10, 'sofa':11, 'table':12}
if args.class_to_classify == 'cat' or args.class_to_classify == 'c':
    current_class = 'cat'
    current_class_folder = 'cats'
    current_domain = "Animals"
elif args.class_to_classify == 'dog' or args.class_to_classify == 'd':
    current_class = 'dog'
    current_class_folder = 'dogs'
    current_domain = "Animals"
elif args.class_to_classify == 'fox' or args.class_to_classify == 'f':
    current_class = 'fox'
    current_class_folder = 'foxes'
    current_domain = "Animals"
elif args.class_to_classify == 'hyena' or args.class_to_classify == 'h':
    current_class = 'hyena'
    current_class_folder = 'hyenas'
    current_domain = "Animals"
elif args.class_to_classify == 'wolf' or args.class_to_classify == 'w':
    current_class = 'wolf'
    current_class_folder = 'wolves'
    current_domain = "Animals"
elif args.class_to_classify == 'duck':
    current_class = 'duck'
    current_class_folder = 'ducks'
    current_domain = "Birds"
elif args.class_to_classify == 'eagle'  or args.class_to_classify == 'e':
    current_class = 'eagle'
    current_class_folder = 'eagles'
    current_domain = "Birds"
elif args.class_to_classify == 'hawk':
    current_class = 'hawk'
    current_class_folder = 'hawks'
    current_domain = "Birds"
elif args.class_to_classify == 'parrot' or args.class_to_classify == 'p':
    current_class = 'parrot'
    current_class_folder = 'parrots'
    current_domain = "Birds"
elif args.class_to_classify == 'sparrow' or args.class_to_classify == 's':
    current_class = 'sparrow'
    current_class_folder = 'sparrows'
    current_domain = "Birds"
elif args.class_to_classify == 'chair':
    current_class = 'chair'
    current_class_folder = 'chair'
    current_domain = "Furniture"
elif args.class_to_classify == 'table':
    current_class = 'table'
    current_class_folder = 'table'
    current_domain = "Furniture"
elif args.class_to_classify == 'sofa':
    current_class = 'sofa'
    current_class_folder = 'sofa'
    current_domain = "Furniture"
elif args.class_to_classify == 'nightstand':
    current_class = 'nightstand'
    current_class_folder = 'nightstand'
    current_domain = "Furniture"
elif args.class_to_classify == 'bed':
    current_class = 'bed'
    current_class_folder = 'bed'
    current_domain = "Furniture"


dataset_path = "../Dataset/"+current_domain+"/"

if args.image is None:
    print("No image number entered, by default "+args.class_to_classify+"001.jpg is selected.")

default_routing = 3

inverse_class_dict = {0:'Cat', 1:"Dog", 2:"Fox", 3:"Hyena", 4:"Wolves", 5:"Ducks",6:"Eagles", 7:"Hawks", 8:"Parrots", 9:"Sparrows", 10:"Chair", 11:"Table", 12:"Sofa", 13:"Nightstand", 14:"Bed" }
features_vector = ["Face", "Eyes", "Mouth", "Snout", "Ears", "Whiskers", "Nose", "Teeth", "Beak", "Tongue", 
                "Body", "Wings", "Paws", "Tail", "Legs", "Surface","Arm Rest", "Base", "Pillows", "Cushion", 
                "Drawer", "Knob", "Mattress", "Colour", "Brown", "Black", "Grey", "White", "Purple", "Pink", 
                "Yellow", "Turqoise", "Unknown"]

# Need to change this line to stop loading the unnecessary training dataset while testing
# (x_train, y_train, y_train_output), (x_test, y_test, y_test_output) = CAPS.load_custom_dataset(dataset_path)

model, eval_model, manipulate_model, hierarchy_train_model, hierarchy_eval_model = CAPS.CapsNet(input_shape=(112, 112, 1), n_class=15, routings=default_routing)

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

prediction = hierarchy_eval_model.predict(test_image, batch_size=100)

if(random.randint(1, 2) == 1):
    predicted_class = inverse_class_dict[np.argmax(prediction[0], 1)[0]]
else:
    predicted_class = current_domain

# print(prediction)
print("\nClass predicted:",inverse_class_dict[np.argmax(prediction[0], 1)[0]],"\n")
# for attribute in prediction:
feature_attributes = {}
feature_probabilities = []
for i in range(1, len(prediction)):
    feature_attributes[features_vector[i-1]] = float(prediction[i][0][0])
    feature_probabilities.append(float(prediction[i][0][0]))

print("Features:")
for features in feature_attributes:
    print(features, ":", feature_attributes[features])

# Eliminate empty bars in the bar graph
final_feature_probabilities = []
final_features_vector = []
for i in range(len(features_vector)):
    if feature_probabilities[i] > 0.1:
        final_feature_probabilities.append(feature_probabilities[i])
        final_features_vector.append(features_vector[i])

fig, ax = plt.subplots()
# print(min(feature_probabilities), max(feature_probabilities))
plt.bar( final_features_vector, final_feature_probabilities)
# plt.xticks(x, features_vector)
plt.show()
# print("Predicted as: ", inverse_class_dict[np.argmax(prediction[0], 1)[0]])