import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import preprocess
import dataset

with open('result/pred_next_5_days_co.result') as f:
  data = json.load(f)
  print(data)
  plt.plot(data['pred'][0][:120], 'r', data['eval'][0][:120], 'b')
  plt.savefig('result/pred_next_5_days_co_result.png')

