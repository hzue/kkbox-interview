import preprocess
import dataset
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error

# opt
pick_feature = [0, 12, 13]

# RNN structure setting
BATCH_SIZE = 30
TIME_STEPS = 720
INPUT_SIZE = len(pick_feature)
OUTPUT_SIZE = 120
CELL_SIZE = 400
LR = 0.006

# prepare data
attr_name_list, meta, data = preprocess.run()
attr_name_list, data = dataset.feature.add_weekday(attr_name_list, meta, data)
attr_name_list, data = dataset.feature.add_hour(attr_name_list, meta, data)
(train_X, train_Y), (test_X, test_Y) = \
    dataset.gen_model_data(data, TIME_STEPS, OUTPUT_SIZE, pick_feature)

# define keras rnn
model = Sequential()
model.add(LSTM(
  input_shape=(TIME_STEPS, INPUT_SIZE),
  output_dim=CELL_SIZE,
  # stateful=True
))
model.add(Dense(OUTPUT_SIZE))
adam_grad = Adam(LR)
model.compile(optimizer=adam_grad, loss='mse')

# start training
model.fit(train_X, train_Y, nb_epoch=2, batch_size=30)
print(mean_squared_error(test_Y, model.predict(test_X)))

''' Another training method
batch_start = 0
for step in range(500):
  x, y = dataset.get_batch(train_X, train_Y, BATCH_SIZE, batch_start)
  cost = model.train_on_batch(x, y)
  # batch_start += BATCH_SIZE
  if step % 10 == 0:
    print('train cost: ', cost)
'''

# garbage collection - prevent keras(tensorflow) error
import gc; gc.collect()

