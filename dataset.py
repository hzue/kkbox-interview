import numpy as np
import datetime

class feature:

  def add_weekday(attr_name_list, meta, data ):
    attr_name_list = np.append(attr_name_list, 'WEEKDAY')
    weekday = np.array([])
    for i, v in enumerate(meta):
      [d, m, y] = v[0].split('/')
      weekday = np.append(weekday, datetime.datetime(int(y), int(m), int(d)).weekday())
    return attr_name_list, np.c_[data, weekday]

  def add_hour(attr_name_list, meta, data ):
    attr_name_list = np.append(attr_name_list, 'HOUR')
    hour = np.array([])
    for i, v in enumerate(meta):
      hour = np.append(hour, float(v[1].split('.')[0]))
    return attr_name_list, np.c_[data, hour]

def scale(data):
  return data

def get_batch(x, y, batch_size, batch_start):
  total = x.shape[0]
  start = batch_start % total
  if start + batch_size < total:
    return x[start:start+batch_size], y[start:start+batch_size]
  else:
    next_end = (batch_start + batch_size) % total
    return np.concatenate([x[start:], x[:next_end]]), np.concatenate([y[start:], y[:next_end]])

def gen_model_data(data, time_steps, output_size, pick_feature, training_set_rate=0.8, strip=8):
  (total, _) = data.shape
  x = []; y = []; example = 0
  for i in range(0, total - time_steps - output_size, strip):
    x.append(data[i:i+time_steps,pick_feature])
    y.append(data[i+time_steps:i+time_steps+output_size, 0])
    example += 1
  x = np.asarray(x)
  y = np.asarray(y)
  train_set_num = int(example * 0.8)

  # return format => (train_X, train_Y), (test_X, test_Y)
  return (x[:train_set_num], y[:train_set_num]) \
        , (x[train_set_num:], y[train_set_num:])

