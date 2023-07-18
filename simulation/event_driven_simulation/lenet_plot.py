import pickle
import matplotlib.pyplot as plt

for interarrival in range(500,2000,100):
    with open("sim_data/{}_interarrival_avg_job_completion.p".format(interarrival), 'rb') as f:
        plt.plot(interarrival, float(pickle.load(f)), label="s")

plt.xlabel('Interarrival Length')
plt.ylabel('Average Job Completion')
plt.title('LeNet-300-100 on 300 cores (344 datapath latency)')
plt.savefig("sim_data/avg_job_completion_vs_interarrival.png")