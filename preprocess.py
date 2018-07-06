import numpy as np
import datetime
from collections import defaultdict
from scipy.interpolate import interp1d

opt = {
    'print_flag': False
}

# program entry point
def run():
    attr_name_list, meta, data = read_file('./res/AirQualityUCI.csv')
    rm_attr_ind, rm_example_ind = observe_missing_value(attr_name_list, data)
    attr_name_list, data = clean_attr(rm_attr_ind, attr_name_list, data)
    data = fill_missing_value(meta, data)
    meta, data = clean_example(rm_example_ind, meta, data)
    return attr_name_list, meta, data

# development print
def dprint(string):
    if opt['print_flag']: print(string)

def read_file(path):
    with open(path) as f:
        content = f.readlines()
    data = []; meta = []
    attr_name_list = [x for x in content[0].strip().split(';') if x]
    for i in range(1, len(content)):
        cols = content[i].replace(',', '.').strip().split(';')[:-2]
        if all(col == '' for col in cols): continue
        else:
            cols[2:] = [ float(_) for _ in cols[2:] ]
            data.append(cols[2:])
            meta.append(cols[:2])
    data = np.asarray(data, dtype='float')
    meta = np.asarray(meta)
    return [attr_name_list, meta, data]

def observe_missing_value(attr_name_list, data):
    total = data.shape[0]
    rm_attr_ind = []; rm_example_ind = []

    # observe from attr
    dprint("------------------------------------")
    dprint("[attribute missing rate]")
    for i in range(0, data.shape[1]):
        rate = float(sum(np.where(data[:,i] == -200., 1, 0))) / float(total) * 100
        if rate > 50.: rm_attr_ind.append(i)
        dprint("%-13s:%6.2f" % (attr_name_list[i+2], rate) + "%")

    # observe from every example
    stat = defaultdict(int)
    dprint("------------------------------------")
    dprint("[example missing] \n(format => number of missing attr: number of data)")
    for i, example in enumerate(data):
        miss_num = sum([1 if e == -200 else 0 for e in example])
        stat[str(miss_num)] += 1
        if miss_num > 2: rm_example_ind.append(i)
    for key in stat:
        dprint("miss %2s : %5d" % (key, stat[key]))

    return rm_attr_ind, rm_example_ind

def clean_attr(rm_attr_ind, attr_name_list, data):
    data = np.delete(data, rm_attr_ind, 1)
    rm_attr_ind = [x + 2 for x in rm_attr_ind]
    attr_name_list = np.delete(attr_name_list, rm_attr_ind, 0)
    return attr_name_list, data

def clean_example(rm_example_ind, meta, data):
    # data = np.delete(data, rm_example_ind, 0)
    # meta = np.delete(meta, rm_example_ind, 0)
    return meta, data

def fill_missing_value(meta, data):
    median_map = get_median_map(meta, data)
    interp_by_neighbor = 5
    for i in range(0, data.shape[1]):
        assert (data[:,i][0] != -200. and data[:,i][-1] != -200.), 'need extrapolation'
        missing_ind = np.where(data[:,i] == -200.)[0]
        for ind in missing_ind:
            upstream_miss = sum(np.where(data[:,i][ind+1:ind+interp_by_neighbor+1] == -200., 1, 0))
            downstream_miss = sum(np.where(data[:,i][ind-interp_by_neighbor:ind] == -200., 1, 0))
            if upstream_miss + downstream_miss == 0:
                near = np.concatenate([data[:,i][ind+1:ind+interp_by_neighbor+1], data[:,i][ind-interp_by_neighbor:ind]])
                assert len(near) == interp_by_neighbor * 2, "loss neighbor (may occur at position which is near to boundary)"
                x = list(range(1, interp_by_neighbor + 1)) + list(range(interp_by_neighbor + 2, 2 * interp_by_neighbor + 2))
                f = interp1d(x, near, kind='cubic')
                data[ind,i] = f(interp_by_neighbor + 1)
            else:
                [day, month, year] = meta[ind][0].split('/')
                w = datetime.datetime(int(year), int(month), int(day)).weekday()
                hour = meta[ind][1].split('.')[0]
                data[ind,i] = median_map["%d_%s_%d" % (w, hour, i)]
    return data

def get_median_map(meta, data):
    stat = defaultdict(list)
    row, col = data.shape
    for c in range(col):
        for r in range(row):
            if data[r][c] == -200.: continue
            [day, month, year] = meta[r][0].split('/')
            hour = meta[r][1].split('.')[0]
            w = datetime.datetime(int(year), int(month), int(day)).weekday()
            stat["%d_%s_%d" % (w, hour, c)].append(data[r][c])
    for key in stat:
        stat[key] = np.median(np.array(stat[key]))
    return stat

if __name__ == '__main__':
    opt['print_flag'] = True
    run()
