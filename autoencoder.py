from keras.layers import Input, Dense
from keras.models import Model
from keras import backend as K

def train(X,Y,ip,ep): 
	I = Input(shape = (ip,))
	layer_1 = Dense(ip,activation = 'relu')(I)
	layer_2 = Dense(int(ip/2),activation = 'relu')(layer_1)
	layer_3 = Dense(int(ip/3),activation = 'relu')(layer_2)
	layer_4 = Dense(int(ip/4),activation = 'relu')(layer_3)
	layer_5 = Dense(int(ip/5),activation = 'relu')(layer_4)
	layer_6 = Dense(int(ip/4),activation = 'relu')(layer_5)
	layer_7 = Dense(int(ip/3),activation = 'relu')(layer_6)
	layer_8 = Dense(int(ip/2),activation = 'relu')(layer_7)
	layer_9 = Dense(ip,activation = 'relu')(layer_8)
	M = Model(I,layer_9)
	M.compile(optimizer = 'adadelta',loss = 'mse')
	M.fit(X,Y,epochs = ep)
	return M 
