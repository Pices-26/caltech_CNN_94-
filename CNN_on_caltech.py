import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, metrics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from scipy.io import loadmat


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.optimizers import adam
from keras.layers import BatchNormalization


############# const
batch_size = 128
num_classes = 100
epochs = 150
img_rows, img_cols = 28, 28
#############

#load the dataset
caltech101 = loadmat('caltech101_silhouettes_28.mat')

X = caltech101['X']
Y = caltech101['Y']
class_names = caltech101['classnames']

Y = Y.flatten()
unique, counts = np.unique(Y, return_counts=True)
classes = np.isin(Y,unique[(counts > 80) & (unique != 2)])

# data contains the flattened images in a (samples, feature) matrix
data = X[classes]
#target 
target = Y[classes]
images = np.reshape(data,(len(target),28,28))
class_names_list = [class_names[0][i][0] for i in range(len(class_names[0]))]

#data shapes
print(data.shape)
print(target.shape)
print(images.shape)

#image view
images_and_labels = list(zip(images, target))
plt.figure('Sample images')
for index, (image, label) in enumerate(images_and_labels[::len(target)//25]):
    if index == 25:
        break
    plt.subplot(5, 5, index+1)
    plt.tight_layout(pad=0.5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title(class_names_list[label-1],fontsize=10,color='r')
plt.show()

# we should 'scale' the data by ensuring zero mean and unit variance
data = scale(data.astype(float))

# Split the data into training and test sets 
X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(data, target, images, test_size=0.25, random_state=42)

#preparing input for CNN
#print(len(X_train))

x_train = X_train.reshape(3198,28,28,1)
x_test = X_test.reshape(1067,28,28,1)


#preparing output for CNN
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
#print(y_train)
#print(y_test)

#model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28,28,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(254, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=categorical_crossentropy,
              optimizer=adam(),
              metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
