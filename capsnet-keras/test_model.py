from keras.models import load_model
import capsulenet as caps

save_dir = './result'
dataset_path = "../Dataset/Animals/"

default_routing = 0.1
number_of_classes = 5
inverse_class_dict = {0:'Cat', 1:"Dog", 2:"Fox", 3:"Hyena", 4:"Wolves"}

model, eval_model, manipulate_model = CapsNet(input_shape=x_train.shape[1:],
                                                  n_class=number_of_classes,
                                                  routings=default_routing)
model = load_model(save_dir + '/weights-507.h5')

#prediction = model.predict(dataset_path+'/dogs/dog1.jpg')
print("Predicted as: ", prediction)