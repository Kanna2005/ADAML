#Artificial Neural network
#code:
import math
import pandas as pd
import numpy as np
import random
from sklearn import preprocessing
from tensorflow.keras import models, layers, optimizers
import matplotlib.pyplot as plt

file_name = "C:/Users/Admin/Downloads/SAheart.csv"
data = pd.read_csv(file_name)

data['famhist'] = data['famhist'] == 'Present'

n_test = int(math.ceil(len(data) * 0.3))
random.seed(42)
test_ixs = random.sample(list(range(len(data))), n_test)
train_ixs = [ix for ix in range(len(data)) if ix not in test_ixs]

train = data.iloc[train_ixs, :]
test = data.iloc[test_ixs, :]

features = ['sbp', 'tobacco', 'ldl', 'adiposity', 'typea', 'obesity', 'alcohol', 'age']
response = 'famhist'

x_train = train[features]
y_train = train[response]
x_test = test[features]
y_test = test[response]

x_train = preprocessing.normalize(x_train)
x_test = preprocessing.normalize(x_test)

hidden_units = 10
activation = 'relu'
learning_rate = 0.01
epochs = 10
batch_size = 16

model = models.Sequential()
model.add(layers.Dense(input_dim=len(features), units=hidden_units, activation=activation))
model.add(layers.Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(learning_rate=learning_rate),
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

train_acc = model.evaluate(x_train, y_train, batch_size=32)[1]
test_acc = model.evaluate(x_test, y_test, batch_size=32)[1]

print("Training accuracy:", train_acc)
print("Testing accuracy:", test_acc)

losses = history.history['loss']
plt.plot(range(len(losses)), losses, 'r')
plt.show()



# of an another data set:
#code:

 import math
import pandas as pd
from keras import models, layers, optimizers, regularizers
import numpy as np
import random
from sklearn import model_selection, preprocessing
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

file_name = "C:/Users/Admin/Downloads/SAheart.csv"

# FIX: Add file_name
data = pd.read_csv(file_name)

# FIX: Boolean conversion
data['famhist'] = data['famhist'] == 'Present'
print(data.head())

# FIX: test index
n_test = int(math.ceil(len(data) * 0.3))
random.seed(42)
test_ixs = random.sample(list(range(len(data))), n_test)

# FIX variable name (x → ix)
train_ixs = [ix for ix in range(len(data)) if ix not in test_ixs]

train = data.iloc[train_ixs, :]
test = data.iloc[test_ixs, :]
print(len(train))
print(len(test))

features = ['adiposity', 'age']
response = 'famhist'

x_train = train[features]
y_train = train[response]
x_test = test[features]
y_test = test[response]

# FIX: normalize
x_train = preprocessing.normalize(x_train)
x_test = preprocessing.normalize(x_test)

hidden_units = 10
activation = 'relu'
learning_rate = 0.01
epochs = 5
batch_size = 16

model = models.Sequential()
model.add(layers.Dense(input_dim=len(features),
                       units=hidden_units,
                       activation=activation))

model.add(layers.Dense(units=1, activation='sigmoid'))

# FIX: spelling mistakes — compile, binary_crossentropy
model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.Adam(learning_rate=learning_rate),
    metrics=['accuracy']
)

# FIX: batch_size argument
history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

train_acc = model.evaluate(x_train, y_train, batch_size=32)[1]
test_acc = model.evaluate(x_test, y_test, batch_size=32)[1]

print('Training accuracy: %s' % train_acc)
print('Testing accuracy: %s' % test_acc)

losses = history.history['loss']
plt.plot(range(len(losses)), losses, 'r')
plt.show()
