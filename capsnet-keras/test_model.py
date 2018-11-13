from keras.models import load_model

save_dir = './result'

model = load_model(save_dir + '/trained_model.h5')