from keras.models import load_model

save_dir = './result'
dataset_path = "../Dataset/Animals/"

inverse_class_dict = {0:'Cat', 1:"Dog", 2:"Fox", 3:"Hyena", 4:"Wolves"}

model = load_model(save_dir + '/weights-507.h5')

prediction = model.predict(dataset_path+'/dogs/dog1.jpg')
print("Predicted as: ", prediction)