from RLWorkflow.environment.offloading_env import Resources
from RLWorkflow.environment.offloading_env import OffloadingEnvironment

import numpy as np


#g_path = "../data/offloading_data/offload_random15_test/random.15."
g_path = "../data/offload_random15/random.15."

print("=============Test heurestic methods for different cpu frequency of the server. =============")

resource_cluster = Resources(mec_process_capable=(2.0 * 1024 * 1024),
                             mobile_process_capable=(1.0 * 1024 * 1024), bandwith_up=7.0, bandwith_dl=7.0)

env = OffloadingEnvironment(resource_cluster = resource_cluster,
                           batch_size=100,
                           graph_number=100,
                           graph_file_paths=[g_path],
                           time_major=False)


env.calculate_optimal_solution()
plans, finish_time_batchs = env.greedy_solution()
cost_batch, energy_batch = env.get_running_cost_by_plan_batch(plans[0], env.task_graphs[0])

rrb_cost_batch, rrb_energy_batch = env.round_robin_solution()
random_cost_batch, random_energy_batch = env.random_solution()
ga_finish_time_batchs = env.ga_solution()

greedy_plans, greedy_finish_time_batchs = env.greedy_solution(heft=False)
greedy_cost_batch, greedy_energy_batch = env.get_running_cost_by_plan_batch(greedy_plans[0], env.task_graphs[0])


print("cpu_frequency=2 GHz")
print("optimal latency: ")
print("latency ", env.optimal_solution)
print("energy ", env.optimal_makespan_energy)
print("heft latency:")
print("latency ", np.mean(cost_batch))
print("energy ", np.mean(energy_batch))
print("greedy latency: ")
print("latency ", np.mean(greedy_cost_batch))
print("energy ", np.mean(greedy_energy_batch))
print("round robin latency:")
print("latency ", np.mean(rrb_cost_batch))
print("energy ", np.mean(rrb_energy_batch))
print("random latency: ")
print("latency ", np.mean(random_cost_batch))
print("energy ", np.mean(random_energy_batch))
print("ga latency: ", np.mean(ga_finish_time_batchs))
print()

resource_cluster = Resources(mec_process_capable=(4.0 * 1024 * 1024),
                             mobile_process_capable=(1.0 * 1024 * 1024), bandwith_up=7.0, bandwith_dl=7.0)

env = OffloadingEnvironment(resource_cluster = resource_cluster,
                           batch_size=100,
                           graph_number=100,
                           graph_file_paths=[g_path],
                           time_major=False)

env.calculate_optimal_solution()

plans, finish_time_batchs = env.greedy_solution()
cost_batch, energy_batch = env.get_running_cost_by_plan_batch(plans[0], env.task_graphs[0])

rrb_cost_batch, rrb_energy_batch = env.round_robin_solution()
random_cost_batch, random_energy_batch = env.random_solution()
ga_finish_time_batchs = env.ga_solution()

greedy_plans, greedy_finish_time_batchs = env.greedy_solution(heft=False)
greedy_cost_batch, greedy_energy_batch = env.get_running_cost_by_plan_batch(greedy_plans[0], env.task_graphs[0])

print("cpu_frequency=4 GHz")
print("optimal latency: ")
print("latency ", env.optimal_solution)
print("energy ", env.optimal_makespan_energy)
print("heft latency:")
print("latency ", np.mean(cost_batch))
print("energy ", np.mean(energy_batch))
print("greedy latency: ")
print("latency ", np.mean(greedy_cost_batch))
print("energy ", np.mean(greedy_energy_batch))
print("round robin latency:")
print("latency ", np.mean(rrb_cost_batch))
print("energy ", np.mean(rrb_energy_batch))
print("random latency: ")
print("latency ", np.mean(random_cost_batch))
print("energy ", np.mean(random_energy_batch))
print("ga latency: ", np.mean(ga_finish_time_batchs))
print()


resource_cluster = Resources(mec_process_capable=(6.0 * 1024 * 1024),
                             mobile_process_capable=(1.0* 1024 * 1024), bandwith_up=7.0, bandwith_dl=7.0)

env = OffloadingEnvironment(resource_cluster = resource_cluster,
                           batch_size=100,
                           graph_number=100,
                           graph_file_paths=[g_path],
                           time_major=False)

env.calculate_optimal_solution()

plans, finish_time_batchs = env.greedy_solution()
cost_batch, energy_batch = env.get_running_cost_by_plan_batch(plans[0], env.task_graphs[0])

rrb_cost_batch, rrb_energy_batch = env.round_robin_solution()
random_cost_batch, random_energy_batch = env.random_solution()
ga_finish_time_batchs = env.ga_solution()

greedy_plans, greedy_finish_time_batchs = env.greedy_solution(heft=False)
greedy_cost_batch, greedy_energy_batch = env.get_running_cost_by_plan_batch(greedy_plans[0], env.task_graphs[0])

print("cpu_frequency=6 GHz")
print("optimal latency: ")
print("latency ", env.optimal_solution)
print("energy ", env.optimal_makespan_energy)
print("heft latency:")
print("latency ", np.mean(cost_batch))
print("energy ", np.mean(energy_batch))
print("greedy latency: ")
print("latency ", np.mean(greedy_cost_batch))
print("energy ", np.mean(greedy_energy_batch))
print("round robin latency:")
print("latency ", np.mean(rrb_cost_batch))
print("energy ", np.mean(rrb_energy_batch))
print("random latency: ")
print("latency ", np.mean(random_cost_batch))
print("energy ", np.mean(random_energy_batch))
print("ga latency: ", np.mean(ga_finish_time_batchs))
print()


resource_cluster = Resources(mec_process_capable=(8.0 * 1024 * 1024),
                             mobile_process_capable=(1.0 * 1024 * 1024), bandwith_up=7.0, bandwith_dl=7.0)

env = OffloadingEnvironment(resource_cluster = resource_cluster,
                           batch_size=100,
                           graph_number=100,
                           graph_file_paths=[g_path],
                           time_major=False)

env.calculate_optimal_solution()

plans, finish_time_batchs = env.greedy_solution()
cost_batch, energy_batch = env.get_running_cost_by_plan_batch(plans[0], env.task_graphs[0])

rrb_cost_batch, rrb_energy_batch = env.round_robin_solution()
random_cost_batch, random_energy_batch = env.random_solution()
ga_finish_time_batchs = env.ga_solution()

greedy_plans, greedy_finish_time_batchs = env.greedy_solution(heft=False)
greedy_cost_batch, greedy_energy_batch = env.get_running_cost_by_plan_batch(greedy_plans[0], env.task_graphs[0])

print("cpu_frequency=8 GHz")
print("optimal latency: ")
print("latency ", env.optimal_solution)
print("energy ", env.optimal_makespan_energy)
print("heft latency:")
print("latency ", np.mean(cost_batch))
print("energy ", np.mean(energy_batch))
print("greedy latency: ")
print("latency ", np.mean(greedy_cost_batch))
print("energy ", np.mean(greedy_energy_batch))
print("round robin latency:")
print("latency ", np.mean(rrb_cost_batch))
print("energy ", np.mean(rrb_energy_batch))
print("random latency: ")
print("latency ", np.mean(random_cost_batch))
print("energy ", np.mean(random_energy_batch))
print("ga latency: ", np.mean(ga_finish_time_batchs))
print()

resource_cluster = Resources(mec_process_capable=(10.0 * 1024 * 1024),
                             mobile_process_capable=(1.0 * 1024 * 1024), bandwith_up=7.0, bandwith_dl=7.0)

env = OffloadingEnvironment(resource_cluster = resource_cluster,
                           batch_size=100,
                           graph_number=100,
                           graph_file_paths=[g_path],
                           time_major=False)

env.calculate_optimal_solution()

plans, finish_time_batchs = env.greedy_solution()
cost_batch, energy_batch = env.get_running_cost_by_plan_batch(plans[0], env.task_graphs[0])

rrb_cost_batch, rrb_energy_batch = env.round_robin_solution()
random_cost_batch, random_energy_batch = env.random_solution()
ga_finish_time_batchs = env.ga_solution()

greedy_plans, greedy_finish_time_batchs = env.greedy_solution(heft=False)
greedy_cost_batch, greedy_energy_batch = env.get_running_cost_by_plan_batch(greedy_plans[0], env.task_graphs[0])

print("cpu_frequency=10 GHz")
print("optimal latency: ")
print("latency ", env.optimal_solution)
print("energy ", env.optimal_makespan_energy)
print("heft latency:")
print("latency ", np.mean(cost_batch))
print("energy ", np.mean(energy_batch))
print("greedy latency: ")
print("latency ", np.mean(greedy_cost_batch))
print("energy ", np.mean(greedy_energy_batch))
print("round robin latency:")
print("latency ", np.mean(rrb_cost_batch))
print("energy ", np.mean(rrb_energy_batch))
print("random latency: ")
print("latency ", np.mean(random_cost_batch))
print("energy ", np.mean(random_energy_batch))
print("ga latency: ", np.mean(ga_finish_time_batchs))
print()



resource_cluster = Resources(mec_process_capable=(12.0 * 1024 * 1024),
                             mobile_process_capable=(1.0 * 1024 * 1024), bandwith_up=7.0, bandwith_dl=7.0)

env = OffloadingEnvironment(resource_cluster = resource_cluster,
                           batch_size=100,
                           graph_number=100,
                           graph_file_paths=[g_path],
                           time_major=False)

env.calculate_optimal_solution()

plans, finish_time_batchs = env.greedy_solution()
cost_batch, energy_batch = env.get_running_cost_by_plan_batch(plans[0], env.task_graphs[0])

rrb_cost_batch, rrb_energy_batch = env.round_robin_solution()
random_cost_batch, random_energy_batch = env.random_solution()
ga_finish_time_batchs = env.ga_solution()

greedy_plans, greedy_finish_time_batchs = env.greedy_solution(heft=False)
greedy_cost_batch, greedy_energy_batch = env.get_running_cost_by_plan_batch(greedy_plans[0], env.task_graphs[0])

print("cpu_frequency=12 GHz")
print("optimal latency: ")
print("latency ", env.optimal_solution)
print("energy ", env.optimal_makespan_energy)
print("heft latency:")
print("latency ", np.mean(cost_batch))
print("energy ", np.mean(energy_batch))
print("greedy latency: ")
print("latency ", np.mean(greedy_cost_batch))
print("energy ", np.mean(greedy_energy_batch))
print("round robin latency:")
print("latency ", np.mean(rrb_cost_batch))
print("energy ", np.mean(rrb_energy_batch))
print("random latency: ")
print("latency ", np.mean(random_cost_batch))
print("energy ", np.mean(random_energy_batch))
print("ga latency: ", np.mean(ga_finish_time_batchs))
print()



resource_cluster = Resources(mec_process_capable=(14.0 * 1024 * 1024),
                             mobile_process_capable=(1.0 * 1024 * 1024), bandwith_up=7.0, bandwith_dl=7.0)

env = OffloadingEnvironment(resource_cluster = resource_cluster,
                           batch_size=100,
                           graph_number=100,
                           graph_file_paths=[g_path],
                           time_major=False)

env.calculate_optimal_solution()

plans, finish_time_batchs = env.greedy_solution()
cost_batch, energy_batch = env.get_running_cost_by_plan_batch(plans[0], env.task_graphs[0])

rrb_cost_batch, rrb_energy_batch = env.round_robin_solution()
random_cost_batch, random_energy_batch = env.random_solution()
ga_finish_time_batchs = env.ga_solution()

greedy_plans, greedy_finish_time_batchs = env.greedy_solution(heft=False)
greedy_cost_batch, greedy_energy_batch = env.get_running_cost_by_plan_batch(greedy_plans[0], env.task_graphs[0])

print("cpu_frequency=14 GHz")
print("optimal latency: ")
print("latency ", env.optimal_solution)
print("energy ", env.optimal_makespan_energy)
print("heft latency:")
print("latency ", np.mean(cost_batch))
print("energy ", np.mean(energy_batch))
print("greedy latency: ")
print("latency ", np.mean(greedy_cost_batch))
print("energy ", np.mean(greedy_energy_batch))
print("round robin latency:")
print("latency ", np.mean(rrb_cost_batch))
print("energy ", np.mean(rrb_energy_batch))
print("random latency: ")
print("latency ", np.mean(random_cost_batch))
print("energy ", np.mean(random_energy_batch))
print("ga latency: ", np.mean(ga_finish_time_batchs))
print()


resource_cluster = Resources(mec_process_capable=(16.0 * 1024 * 1024),
                             mobile_process_capable=(1.0 * 1024 * 1024), bandwith_up=7.0, bandwith_dl=7.0)

env = OffloadingEnvironment(resource_cluster = resource_cluster,
                           batch_size=100,
                           graph_number=100,
                           graph_file_paths=[g_path],
                           time_major=False)

env.calculate_optimal_solution()

plans, finish_time_batchs = env.greedy_solution()
cost_batch, energy_batch = env.get_running_cost_by_plan_batch(plans[0], env.task_graphs[0])

rrb_cost_batch, rrb_energy_batch = env.round_robin_solution()
random_cost_batch, random_energy_batch = env.random_solution()
ga_finish_time_batchs = env.ga_solution()

greedy_plans, greedy_finish_time_batchs = env.greedy_solution(heft=False)
greedy_cost_batch, greedy_energy_batch = env.get_running_cost_by_plan_batch(greedy_plans[0], env.task_graphs[0])

print("cpu_frequency=16 GHz")
print("optimal latency: ")
print("latency ", env.optimal_solution)
print("energy ", env.optimal_makespan_energy)
print("heft latency:")
print("latency ", np.mean(cost_batch))
print("energy ", np.mean(energy_batch))
print("greedy latency: ")
print("latency ", np.mean(greedy_cost_batch))
print("energy ", np.mean(greedy_energy_batch))
print("round robin latency:")
print("latency ", np.mean(rrb_cost_batch))
print("energy ", np.mean(rrb_energy_batch))
print("random latency: ")
print("latency ", np.mean(random_cost_batch))
print("energy ", np.mean(random_energy_batch))
print("ga latency: ", np.mean(ga_finish_time_batchs))
print()


