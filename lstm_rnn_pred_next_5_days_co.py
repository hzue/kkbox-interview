import preprocess
import dataset
import numpy as np
import gc
import json
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import RMSprop
from sklearn.metrics import mean_squared_error

# opt
# pick_feature = [0, 12, 13]
pick_feature = range(0, 14)

# RNN structure setting
BATCH_SIZE = 30
TIME_STEPS = 720
INPUT_SIZE = len(pick_feature)
OUTPUT_SIZE = 120
CELL_SIZE = 1200
LR = 0.006

# prepare data
attr_name_list, meta, data = preprocess.run()
attr_name_list, data = dataset.feature.add_weekday(attr_name_list, meta, data)
attr_name_list, data = dataset.feature.add_hour(attr_name_list, meta, data)
(train_X, train_Y), (test_X, test_Y) = \
    dataset.gen_co_model_data(data, TIME_STEPS, OUTPUT_SIZE, pick_feature)

# define keras rnn
model = Sequential()
model.add(LSTM(
  input_shape=(TIME_STEPS, INPUT_SIZE),
  output_dim=CELL_SIZE,
  # dropout_W=0.4,
  # dropout_U=0.4
))
model.add(Dense(OUTPUT_SIZE))
model.compile(optimizer=RMSprop(LR), loss='mse')

# start training
''' Another training method
batch_start = 0
for step in range(500):
  x, y = dataset.get_batch(train_X, train_Y, BATCH_SIZE, batch_start)
  cost = model.train_on_batch(x, y)
  # batch_start += BATCH_SIZE
  if step % 10 == 0:
    print('train cost: ', cost)
'''

model.fit(train_X, train_Y, nb_epoch=20, batch_size=30)
model.save('result/pred_next_5_co.model')
pred_Y = model.predict(train_X)
gc.collect()
print("testing data mse: %f" % mean_squared_error(test_Y, pred_Y))
with open('result/pred_next_5_days_co.result', 'w') as f:
  json.dump({'pred': train_Y.tolist(), 'eval': pred_Y.tolist()}, f)

