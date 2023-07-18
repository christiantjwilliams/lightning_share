import csv
import matplotlib.pyplot as plt
import argparse

x = []
ys = []
with open('sim_data/sim_data.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for i, row in enumerate(reader):
        if i == 0:
            processors_row = row[1:]
        else:
            for j, item in enumerate(row):
                if j > 0:
                    if i == 1:
                        ys.append([float(item)])
                    else:
                        ys[j-1].append(float(item))
            x.append(float(row[0]))

for i, y in enumerate(ys):
    plt.plot(x, y, label=processors_row[i], marker='.')

plt.yscale("log")
plt.xlabel('Gbps')
plt.ylabel('Average Request Completion Time')
plt.title('Lightning, Brainwave, and GPU Performance Comparison')
plt.legend()
plt.xlim(0, 500)
plt.savefig("sim_data/latency_vs_load_fig.png")
