import os
import cv2
import numpy as np
from PIL import Image
import capsulenet as CAPS
import matplotlib.pyplot as plt
from utils import combine_images
from keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def test(model, data):
    x_test, y_test = data
    y_pred, x_recon = model.predict(x_test, batch_size=100)
    print('-'*30 + 'Begin: test' + '-'*30)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0])

    img = combine_images(np.concatenate([x_test[:50],x_recon[:50]]))
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save(save_dir + "/real_and_recon.png")
    print()
    print('Reconstructed images are saved to %s/real_and_recon.png' % save_dir)
    print('-' * 30 + 'End: test' + '-' * 30)
    plt.imshow(plt.imread(save_dir + "/real_and_recon.png"))
    plt.show()


save_dir = './result'
dataset_path = "../Dataset/Animals/"

default_routing = 3
number_of_classes = 5
inverse_class_dict = {0:'Cat', 1:"Dog", 2:"Fox", 3:"Hyena", 4:"Wolves"}

(x_train, y_train), (x_test, y_test) = CAPS.load_custom_dataset(dataset_path)

model, eval_model, manipulate_model = CAPS.CapsNet(input_shape=x_train.shape[1:], n_class=len(np.unique(np.argmax(y_train, 1))), routings=default_routing)

model.load_weights(save_dir + '/weights-507.h5')

image_number = int(input())

current_file = dataset_path + 'cats/' + 'cat'+str(image_number)+'.jpg'

test_image = []

img = cv2.imread(current_file)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (28, 28))

test_image.append(img)

test_image = np.array(test_image)
test_image = test_image.reshape(-1, 28, 28, 1).astype('float32') / 255.

# print(test_image)
# print(len(test_image), len(test_image[0][0]))

#test(model=eval_model, data=(x_test, y_test))

prediction = eval_model.predict(test_image, batch_size=100)

print("Predicted as: ", inverse_class_dict[np.argmax(prediction[0], 1)[0]])