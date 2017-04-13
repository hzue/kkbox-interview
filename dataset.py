import numpy as np
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

def scale(data, mode='s', model_file='result/scale_model'):
  assert len(data.shape) == 2, 'Dimension Error'
  if mode == 's':
    f = open(model_file, 'w')
    content = ''
    for i in range(data.shape[1]):
      max = data[:,i].max()
      min = data[:,i].min()
      content += "%d\t%f\t%f\n" % (i, min, max)
      if max - min == 0: data[:,i] = 0
      else: data[:,i] = (data[:,i] - min) / (max - min)
    f.write(content)
    f.close()
    return data
  elif mode == 'r':
    ## [incomplete]
    # f = open(model_file, 'r')
    # f.close()
    return data
  else: raise ValueError('parameter error')

def get_batch(x, y, batch_size, batch_start):
  total = x.shape[0]
  start = batch_start % total
  if start + batch_size < total:
    return x[start:start+batch_size], y[start:start+batch_size]
  else:
    next_end = (batch_start + batch_size) % total
    return np.concatenate([x[start:], x[:next_end]]), np.concatenate([y[start:], y[:next_end]])

def gen_co_model_data(data, time_steps, output_size, pick_feature, training_set_rate=0.8, strip=8):
  (total, _) = data.shape
  x = []; y = []; example = 0
  for i in range(0, total - time_steps - output_size, strip):
    x.append(data[i:i+time_steps,pick_feature])
    y.append(data[i+time_steps:i+time_steps+output_size, 0])
    example += 1
  x = np.asarray(x)
  y = np.asarray(y)
  a, b, c = x.shape
  x = scale(x.reshape(a * b, c))
  x = x.reshape(a, b, c)

  train_set_num = int(example * 0.8)
  # return format => (train_X, train_Y), (test_X, test_Y)
  return (x[:train_set_num], y[:train_set_num]) \
        , (x[train_set_num:], y[train_set_num:])

# for plot
def ma(a, n=3) :
  ret = np.cumsum(a, dtype=float)
  ret[n:] = ret[n:] - ret[:-n]
  return ret[n - 1:] / n
