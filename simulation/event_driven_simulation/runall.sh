#!/bin/bash

rm -rf nohup.out

echo "time started sim"; date

bash clear_data.sh

mkdir sim_data
mkdir sim_times
mkdir completed_reqs
mkdir generated_reqs

proc_num=0
processor_set=$(seq 0 1 4)
traffic_set=$(seq 50 1 50) 
inter_arrival_set=$(seq 1 1 1)

if [ "$2" == "true" ]; then
    python3 build_models.py
fi

for processor in ${processor_set}; do
    for traffic in ${traffic_set}; do
        for inter in ${inter_arrival_set}; do 
            nohup python3 parallel_eq_sim.py --proc_id ${processor} --inter ${inter} --time $1 --traffic ${traffic} --core_util True &
            pids[proc_num]=$!;
            ((proc_num=proc_num+1));
        done
    done
done

for pid in ${pids[*]}; do
    wait $pid 
done

scp -r sim_data/ abtin.csail.mit.edu:~/lightning-sim/simulation/event_driven_simulation
scp -r sim_times/ abtin.csail.mit.edu:~/lightning-sim/simulation/event_driven_simulation
scp -r completed_reqs/ abtin.csail.mit.edu:~/lightning-sim/simulation/event_driven_simulation
scp -r generated_reqs/ abtin.csail.mit.edu:~/lightning-sim/simulation/event_driven_simulation

bash plot_data.sh

echo "time finished sim"; date
