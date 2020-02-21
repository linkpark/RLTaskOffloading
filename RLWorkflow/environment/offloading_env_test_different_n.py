from RLWorkflow.environment.offloading_env import Resources
from RLWorkflow.environment.offloading_env import OffloadingEnvironment

import numpy as np
import time

def calculate_qoe(latency_batch, energy_batch, env):
    all_local_time, all_local_energy = env.get_all_locally_execute_time_batch()
    all_local_time = np.squeeze(all_local_time)
    all_local_energy = np.squeeze(all_local_energy)
    latency_batch = np.squeeze(latency_batch)
    energy_batch = np.squeeze(energy_batch)
    qoe_batch = []

    for latency, energy, single_all_local_latency, single_all_local_energy in zip(latency_batch, energy_batch, all_local_time, all_local_energy):
        qoe = env.lambda_t * ((latency - single_all_local_latency) / single_all_local_latency) + \
              env.lambda_e * ((energy - single_all_local_energy) / single_all_local_energy)

        qoe = -qoe
        qoe_batch.append(qoe)

    return qoe_batch

lambda_t = 0.5
lambda_e = 0.5

prefix = ".."

graph_file_pahts = [ prefix+"/offloading_data/offload_random10_test/random.10.",
                     prefix+"/offloading_data/offload_random15_test/random.15.",
                     prefix+"/offloading_data/offload_random20_test/random.20.",
                     prefix+"/offloading_data/offload_random25_test/random.25.",
                     prefix+"/offloading_data/offload_random30_test/random.30.",
                     prefix+"/offloading_data/offload_random35_test/random.35.",
                     prefix+"/offloading_data/offload_random40_test/random.40.",
                     prefix+"/offloading_data/offload_random45_test/random.45.",
                     prefix+"/offloading_data/offload_random50_test/random.50."]

print("=============Test heurestic methods for different n. =============")
print("lambda_t: ", lambda_t)
print("lambda_e: ", lambda_e)

for graph_file_path in graph_file_pahts:
    resource_cluster = Resources(mec_process_capable=(8.0 * 1024 * 1024),
                             mobile_process_capable=(1.0 * 1024 * 1024), bandwith_up=8.0, bandwith_dl=8.0)

    env = OffloadingEnvironment(resource_cluster = resource_cluster,
                           batch_size=100,
                           graph_number=100,
                           graph_file_paths=[graph_file_path],
                           time_major=False,
                           lambda_t=lambda_t,
                           lambda_e=lambda_e)
    if env.task_graphs[0][0].task_number < 20:
        env.calculate_optimal_qoe()

    # Calculate the heft algorithms latency, energy and qoe
    plans, finish_time_batchs = env.greedy_solution(heft=True)
    heft_latency_batch, heft_energy_batch = env.get_running_cost_by_plan_batch(plans[0], env.task_graphs[0])
    latency_batch = np.array(heft_latency_batch)
    energy_batch = np.array(heft_energy_batch)
    heft_qoe_batch = calculate_qoe(latency_batch, energy_batch, env)

    # Calculate the greedy algorithms latency, energy and qoe
    plans, finish_time_batchs = env.greedy_solution(heft=False)
    greedy_latency_batch, greedy_energy_batch = env.get_running_cost_by_plan_batch(plans[0], env.task_graphs[0])
    latency_batch = np.array(greedy_latency_batch)
    energy_batch = np.array(greedy_energy_batch)
    greedy_qoe_batch = calculate_qoe(latency_batch, energy_batch, env)

    # Calculate the round robin latency, energy and qoe
    rrb_latency_batch, rrb_energy_batch = env.round_robin_solution()
    rrb_qoe_batch = calculate_qoe(rrb_latency_batch, rrb_energy_batch, env)

    # Calculate the random latency latency, energy and qoe
    random_latency_batch, random_energy_batch = env.random_solution()
    random_qoe_batch = calculate_qoe(random_latency_batch, random_energy_batch, env)

    # Calculate the all local latency, energy and qoe
    all_local_latency_batch, all_local_energy_batch = env.get_all_locally_execute_time_batch()
    all_local_qoe_batch = calculate_qoe(all_local_latency_batch, all_local_energy_batch, env)

    # Calculate the all remote latency, energy and qoe
    all_remote_latency_batch, all_remote_energy_batch = env.get_all_mec_execute_time_batch()
    all_remote_qoe_batch = calculate_qoe(all_remote_latency_batch, all_remote_energy_batch, env)

    print(graph_file_path)
    print("HEFT algorighm result: ")
    print("latency: ", np.mean(heft_latency_batch))
    print("energy: ", np.mean(heft_energy_batch))
    print("qoe: ", np.mean(heft_qoe_batch))
    print()
    print("Greedy algorighm result: ")
    print("latency: ", np.mean(greedy_latency_batch))
    print("energy: ", np.mean(greedy_energy_batch))
    print("qoe: ", np.mean(greedy_qoe_batch))
    print()
    print("round roubin algorighm result: ")
    print("latency: ", np.mean(rrb_latency_batch))
    print("energy: ", np.mean(rrb_energy_batch))
    print("qoe: ", np.mean(rrb_qoe_batch))
    print()
    print("random algorighm result: ")
    print("latency: ", np.mean(random_latency_batch))
    print("energy: ", np.mean(random_energy_batch))
    print("qoe: ", np.mean(random_qoe_batch))
    print()
    print("all local algorighm result: ")
    print("latency: ", np.mean(all_local_latency_batch))
    print("energy: ", np.mean(all_local_energy_batch))
    print("qoe: ", np.mean(all_local_qoe_batch))
    print()
    print("all remote algorigthm result: ")
    print("latency: ", np.mean(all_remote_latency_batch))
    print("energy: ", np.mean(all_remote_energy_batch))
    print("qoe: ", np.mean(all_remote_qoe_batch))

    print("optimal qoe algorithm result: ")
    print("optimal qoe: ", np.mean(env.optimal_qoe))
    print("optimal qoe latency: ", np.mean(env.optimal_qoe_latency))
    print("optimal qoe energy: ", np.mean(env.optimal_qoe_energy))





