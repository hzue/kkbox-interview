import preprocess
import dataset
import numpy as np
import gc
import json
import os
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import RMSprop
from sklearn.metrics import mean_squared_error

# option
pick_feature = [0, 12, 13] # small feature set
# pick_feature = range(0, 14)

# RNN structure setting
BATCH_SIZE = 30
TIME_STEPS = 240
INPUT_SIZE = len(pick_feature)
OUTPUT_SIZE = 1
CELL_SIZE = 700
LR = 0.006

# prepare data
attr_name_list, meta, data = preprocess.run()
attr_name_list, data = dataset.feature.add_weekday(attr_name_list, meta, data)
attr_name_list, data = dataset.feature.add_hour(attr_name_list, meta, data)
(train_X, train_Y), (test_X, test_Y) = \
    dataset.gen_co_model_data(data, TIME_STEPS, OUTPUT_SIZE, pick_feature, strip=1)

# load model if exist
pred_days = 5 * 24
if os.path.exists('result/pred_next_1_co.model'):
  min = data[:,0].min()
  max = data[:,0].max()
  model = keras.models.load_model('result/pred_next_1_co.model')
  result = []
  replace_ans = test_X[0][-1][0] * (max - min) + min
  for i in range(0, pred_days):
    test_X[i][-1][0] = (replace_ans - min) / (max - min)
    pred = model.predict(test_X[i].reshape(1, 240, 3))
    result.append(pred.tolist()[0][0])
    replace_ans = pred[0][0]
  print(result)
  print(test_Y[:120,0])
  with open('result/pred_next_1_days_co.result', 'w') as f:
    json.dump({'pred': result, 'eval': test_Y[:120,0].tolist()}, f)
  gc.collect()
  exit()

# define keras rnn
model = Sequential()
model.add(LSTM(
  input_shape=(TIME_STEPS, INPUT_SIZE),
  output_dim=CELL_SIZE,
  dropout_W=0.4,
  dropout_U=0.4,
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

model.fit(train_X, train_Y, nb_epoch=10, batch_size=30)
pred_Y = model.predict(test_X)
model.save('result/pred_next_1_co.model')
gc.collect()

# eval
print("testing data mse: %f" % mean_squared_error(test_Y, pred_Y))
with open('result/pred_next_1_days_co.result', 'w') as f:
  json.dump({'pred': pred_Y.tolist(), 'eval': test_Y.tolist()}, f)

