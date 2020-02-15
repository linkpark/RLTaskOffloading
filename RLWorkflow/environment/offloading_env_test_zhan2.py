from RLWorkflow.environment.offloading_env import Resources
from RLWorkflow.environment.offloading_env import OffloadingEnvironment

import numpy as np

g_path = "../data/offloading_data/offload_random15_test/random.15."

print("=============Test heurestic methods for different transmission rate. =============")
resource_cluster = Resources(mec_process_capable=(8.0 * 1024 * 1024),
                             mobile_process_capable=(1.0 * 1024 * 1024), bandwith_up=5.0, bandwith_dl=5.0)

env = OffloadingEnvironment(resource_cluster = resource_cluster,
                           batch_size=100,
                           graph_number=100,
                           graph_file_paths=[g_path],
                           time_major=False)

# env.calculate_optimal_solution()

plans, finish_time_batchs = env.greedy_solution()
cost_batch, energy_batch = env.get_running_cost_by_plan_batch(plans[0], env.task_graphs[0])

rrb_cost_batch, rrb_energy_batch = env.round_robin_solution()
random_cost_batch, random_energy_batch = env.random_solution()
ga_finish_time_batchs = env.ga_solution()

print("trans_rate=5.0Mbps")
print("optimal latency: ")
print("latency ", env.optimal_solution)
print("heft latency:")
print("latency ", np.mean(cost_batch))
print("round robin latency:")
print("latency ", np.mean(rrb_cost_batch))
print("random latency: ")
print("latency ", np.mean(random_cost_batch))
print("ga latency: ", np.mean(ga_finish_time_batchs))
print()


resource_cluster = Resources(mec_process_capable=(8.0 * 1024 * 1024),
                             mobile_process_capable=(1.0 * 1024 * 1024), bandwith_up=8.0, bandwith_dl=8.0)

env = OffloadingEnvironment(resource_cluster = resource_cluster,
                           batch_size=100,
                           graph_number=100,
                           graph_file_paths=[g_path],
                           time_major=False)

# env.calculate_optimal_solution()

plans, finish_time_batchs = env.greedy_solution()
cost_batch, energy_batch = env.get_running_cost_by_plan_batch(plans[0], env.task_graphs[0])

rrb_cost_batch, rrb_energy_batch = env.round_robin_solution()
random_cost_batch, random_energy_batch = env.random_solution()
ga_finish_time_batchs = env.ga_solution()

print("trans_rate=8.0Mbps")
print("optimal latency: ")
print("latency ", env.optimal_solution)
print("heft latency:")
print("latency ", np.mean(cost_batch))
print("round robin latency:")
print("latency ", np.mean(rrb_cost_batch))
print("random latency: ")
print("latency ", np.mean(random_cost_batch))
print("ga latency: ", np.mean(ga_finish_time_batchs))
print()


resource_cluster = Resources(mec_process_capable=(8.0 * 1024 * 1024),
                             mobile_process_capable=(1.0 * 1024 * 1024), bandwith_up=11.0, bandwith_dl=11.0)

env = OffloadingEnvironment(resource_cluster = resource_cluster,
                           batch_size=100,
                           graph_number=100,
                           graph_file_paths=[g_path],
                           time_major=False)

# env.calculate_optimal_solution()

plans, finish_time_batchs = env.greedy_solution()
cost_batch, energy_batch = env.get_running_cost_by_plan_batch(plans[0], env.task_graphs[0])

rrb_cost_batch, rrb_energy_batch = env.round_robin_solution()
random_cost_batch, random_energy_batch = env.random_solution()
ga_finish_time_batchs = env.ga_solution()

print("trans_rate=11.0Mbps")
print("optimal latency: ")
print("latency ", env.optimal_solution)
print("heft latency:")
print("latency ", np.mean(cost_batch))
print("round robin latency:")
print("latency ", np.mean(rrb_cost_batch))
print("random latency: ")
print("latency ", np.mean(random_cost_batch))
print("ga latency: ", np.mean(ga_finish_time_batchs))
print()


resource_cluster = Resources(mec_process_capable=(8.0 * 1024 * 1024),
                             mobile_process_capable=(1.0 * 1024 * 1024), bandwith_up=14.0, bandwith_dl=14.0)

env = OffloadingEnvironment(resource_cluster = resource_cluster,
                           batch_size=100,
                           graph_number=100,
                           graph_file_paths=[g_path],
                           time_major=False)

# env.calculate_optimal_solution()

plans, finish_time_batchs = env.greedy_solution()
cost_batch, energy_batch = env.get_running_cost_by_plan_batch(plans[0], env.task_graphs[0])

rrb_cost_batch, rrb_energy_batch = env.round_robin_solution()
random_cost_batch, random_energy_batch = env.random_solution()
ga_finish_time_batchs = env.ga_solution()

print("trans_rate=14.0Mbps")
print("optimal latency: ")
print("latency ", env.optimal_solution)
print("heft latency:")
print("latency ", np.mean(cost_batch))
print("round robin latency:")
print("latency ", np.mean(rrb_cost_batch))
print("random latency: ")
print("latency ", np.mean(random_cost_batch))
print("ga latency: ", np.mean(ga_finish_time_batchs))
print()


resource_cluster = Resources(mec_process_capable=(8.0 * 1024 * 1024),
                             mobile_process_capable=(1.0 * 1024 * 1024), bandwith_up=17.0, bandwith_dl=17.0)

env = OffloadingEnvironment(resource_cluster = resource_cluster,
                           batch_size=100,
                           graph_number=100,
                           graph_file_paths=[g_path],
                           time_major=False)

# env.calculate_optimal_solution()

plans, finish_time_batchs = env.greedy_solution()
cost_batch, energy_batch = env.get_running_cost_by_plan_batch(plans[0], env.task_graphs[0])

rrb_cost_batch, rrb_energy_batch = env.round_robin_solution()
random_cost_batch, random_energy_batch = env.random_solution()
ga_finish_time_batchs = env.ga_solution()

print("trans_rate=17.0Mbps")
print("optimal latency: ")
print("latency ", env.optimal_solution)
print("heft latency:")
print("latency ", np.mean(cost_batch))
print("round robin latency:")
print("latency ", np.mean(rrb_cost_batch))
print("random latency: ")
print("latency ", np.mean(random_cost_batch))
print("ga latency: ", np.mean(ga_finish_time_batchs))
print()