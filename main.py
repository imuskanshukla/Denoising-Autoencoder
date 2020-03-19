import image_preprocessing as img 
import numpy as np
import matplotlib.pyplot as plt
import autoencoder as A
X = img.blur_images(100)
Y = img.pre_process(100)
X = X/X.max()
Y = Y/Y.max()
ip = np.prod(np.array(X[0].shape))
print(X.shape)
print(Y.shape)

model = A.train(X,Y,ip,1)

predictions = model.predict(X)
plt.imshow(predictions[0].reshape(100,100),cmap = 'gray')
plt.show()