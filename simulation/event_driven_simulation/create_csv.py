import csv
import pickle
import os
import argparse
from collections import defaultdict
from parallel_eq_sim import processors

with open('sim_data/sim_data.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    processors = [processor[0] for processor in processors]
    writer.writerow(['Load Factor'] + processors)
    sim_data_rows = []
    data_dict = defaultdict(list)
    for filename in os.listdir('sim_data'):
        if 'proc' in filename:
            i = int(filename.split('_')[3].split('.')[0])
            j = int(filename.split('_')[1].split('.')[0])
            with open("sim_data/proc_{}_traffic_{}.p".format(j, i), 'rb') as f:
                data_dict[i].append([j, int(pickle.load(f))])
    sim_data_rows = []

    for x in data_dict:
        ys = data_dict[x]
        ys.sort()
        ys = [latency for proc, latency in ys]
        ys.insert(0, x)
        sim_data_rows.append(ys)
    sim_data_rows.sort()
    for row in sim_data_rows:
        writer.writerow(row)
