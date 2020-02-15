from RLWorkflow.environment.offloading_env import Resources
from RLWorkflow.environment.offloading_env import OffloadingEnvironment

import numpy as np
import time

def calculate_qoe_for_plans(plans, env):
    task_bach = env.task_graphs[0]
    target_batch = []

    for plan, task_graph in zip(plans, task_bach):
        cost, energy, task_finish_time, total_energy = env.get_scheduling_cost_step_by_step(plan, task_graph)
        all_local_time, all_local_energy = env.get_all_local_cost_for_one_graph(task_graph)

        latency_qoe = -env.score_func_qoe(cost, all_local_cost=all_local_time,
                                      number_of_task=task_graph.task_number)

        energy_qoe = -env.score_func_qoe(energy, all_local_cost=all_local_energy,
                                     number_of_task=task_graph.task_number)

        score = env.lambda_t * np.array(latency_qoe) + env.lambda_e * np.array(energy_qoe)

        target_batch.append(score)
    target_batch = np.array(target_batch)

    return target_batch

def calculate_qoe_for_cost(running_cost, energy_cost, env):
    task_batch = env.task_graphs[0]

    target_qoe = []
    for time, energy, task_graph in zip(running_cost, energy_cost, task_batch):
        all_local_time, all_local_energy = env.get_all_local_cost_for_one_graph(task_graph)

        qoe_time = (time- all_local_time) / all_local_time
        qoe_energy = (energy - all_local_energy) / all_local_energy

        qoe = env.lambda_t * qoe_time + env.lambda_e * qoe_energy

        target_qoe.append(qoe)

    return target_qoe

print("=============Test heurestic methods for different n. =============")
print("heft false")
resource_cluster = Resources(mec_process_capable=(8.0 * 1024 * 1024),
                             mobile_process_capable=(1.0 * 1024 * 1024), bandwith_up=8.0, bandwith_dl=8.0)

env = OffloadingEnvironment(resource_cluster = resource_cluster,
                           batch_size=100,
                           graph_number=100,
                           graph_file_paths=["../data/offloading_data/offload_random10_test/random.10."],
                           time_major=False, lambda_t=0.5, lambda_e=0.5)

# env.calculate_optimal_solution()



plans, finish_time_batchs = env.greedy_solution(heft=True)
cost_batch, energy_batch = env.get_running_cost_by_plan_batch(plans[0], env.task_graphs[0])

greedy_qoe= calculate_qoe_for_cost(cost_batch, energy_batch, env)

rrb_cost_batch, rrb_energy_batch = env.round_robin_solution()
rrb_qoe = calculate_qoe_for_cost(rrb_cost_batch[0], rrb_energy_batch[0], env)

random_cost_batch, random_energy_batch = env.random_solution()
random_qoe = calculate_qoe_for_cost(random_cost_batch, random_energy_batch, env)

all_local_latency, all_local_energy = env.all_local_solution()
all_local_qoe = calculate_qoe_for_cost(all_local_latency, all_local_energy, env)

all_offloading_latency, all_offloading_energy = env.all_offloading_solution()
all_offloading_qoe = calculate_qoe_for_cost(all_offloading_latency, all_offloading_energy, env)

print("n=10")
print("all local: ")
print("latency: ", np.mean(all_local_latency))
print("energy: ", np.mean(all_local_energy))
print("qoe: ", all_offloading_qoe)

print("all remote: ")
print("latency: ", np.mean(all_offloading_latency))
print("energy: ", np.mean(all_offloading_energy))
print("qoe: ", all_offloading_qoe)

# print("optimal algorithm: ")
# print("latency: ", env.optimal_solution)
# print("energy: ", env.optimal_makespan_energy)

print("greedy:")
print("latency: ", np.mean(cost_batch))
print("energy: ", np.mean(energy_batch))
print("qoe: ", np.mean(greedy_qoe))

print("round robin:")
print("latency: ", np.mean(rrb_cost_batch))
print("energy: ", np.mean(rrb_energy_batch))
print("qoe: ", np.mean(rrb_qoe))

print("random latency: ")
print("latency: ", np.mean(random_cost_batch))
print("energy: ", np.mean(random_energy_batch))

print()

resource_cluster = Resources(mec_process_capable=(8.0 * 1024 * 1024),
                             mobile_process_capable=(1.0 * 1024 * 1024), bandwith_up=8.0, bandwith_dl=8.0)

env = OffloadingEnvironment(resource_cluster=resource_cluster,
                           batch_size=100,
                           graph_number=100,
                           graph_file_paths=["../data/offloading_data/offload_random15_test/random.15."],
                           time_major=False)

# env.calculate_optimal_solution()

plans, finish_time_batchs = env.greedy_solution()
cost_batch, energy_batch = env.get_running_cost_by_plan_batch(plans[0], env.task_graphs[0])

rrb_cost_batch, rrb_energy_batch = env.round_robin_solution()
random_cost_batch, random_energy_batch = env.random_solution()
ga_finish_time_batchs = env.ga_solution()

print("n=15")
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
                           graph_file_paths=["../data/offloading_data/offload_random20_test/random.20."],
                           time_major=False)

# env.calculate_optimal_solution()

plans, finish_time_batchs = env.greedy_solution()
cost_batch, energy_batch = env.get_running_cost_by_plan_batch(plans[0], env.task_graphs[0])

rrb_cost_batch, rrb_energy_batch = env.round_robin_solution()
random_cost_batch, random_energy_batch = env.random_solution()
ga_finish_time_batchs = env.ga_solution()

print("n=20")
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
                           graph_file_paths=["../data/offloading_data/offload_random25_test/random.25."],
                           time_major=False)

#env.calculate_optimal_solution()

plans, finish_time_batchs = env.greedy_solution()
cost_batch, energy_batch = env.get_running_cost_by_plan_batch(plans[0], env.task_graphs[0])

rrb_cost_batch, rrb_energy_batch = env.round_robin_solution()
random_cost_batch, random_energy_batch = env.random_solution()

start_time = time.time()
ga_finish_time_batchs = env.ga_solution()
end_time = time.time()

print("ga algorithm time consume: ", (end_time - start_time))

print("n=25")
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
                           graph_file_paths=["../data/offloading_data/offload_random30_test/random.30."],
                           time_major=False)

#env.calculate_optimal_solution()

plans, finish_time_batchs = env.greedy_solution()
cost_batch, energy_batch = env.get_running_cost_by_plan_batch(plans[0], env.task_graphs[0])

rrb_cost_batch, rrb_energy_batch = env.round_robin_solution()
random_cost_batch, random_energy_batch = env.random_solution()
ga_finish_time_batchs = env.ga_solution()

print("n=30")
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
                           graph_file_paths=["../data/offloading_data/offload_random35_test/random.35."],
                           time_major=False)

#env.calculate_optimal_solution()

plans, finish_time_batchs = env.greedy_solution()
cost_batch, energy_batch = env.get_running_cost_by_plan_batch(plans[0], env.task_graphs[0])

rrb_cost_batch, rrb_energy_batch = env.round_robin_solution()
random_cost_batch, random_energy_batch = env.random_solution()
ga_finish_time_batchs = env.ga_solution()

print("n=35")
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
                           graph_file_paths=["../data/offloading_data/offload_random40_test/random.40."],
                           time_major=False)

#env.calculate_optimal_solution()

plans, finish_time_batchs = env.greedy_solution()
cost_batch, energy_batch = env.get_running_cost_by_plan_batch(plans[0], env.task_graphs[0])

rrb_cost_batch, rrb_energy_batch = env.round_robin_solution()
random_cost_batch, random_energy_batch = env.random_solution()
ga_finish_time_batchs = env.ga_solution()

print("n=40")

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
                           graph_file_paths=["../data/offloading_data/offload_random45_test/random.45."],
                           time_major=False)

#env.calculate_optimal_solution()

plans, finish_time_batchs = env.greedy_solution()
cost_batch, energy_batch = env.get_running_cost_by_plan_batch(plans[0], env.task_graphs[0])

rrb_cost_batch, rrb_energy_batch = env.round_robin_solution()
random_cost_batch, random_energy_batch = env.random_solution()
ga_finish_time_batchs = env.ga_solution()


print("n=45")
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
                           graph_file_paths=["../data/offloading_data/offload_random50_test/random.50."],
                           time_major=False)

#env.calculate_optimal_solution()

# plans, finish_time_batchs = env.greedy_solution()
# cost_batch, energy_batch = env.get_running_cost_by_plan_batch(plans[0], env.task_graphs[0])

finish_time_batchs = env.ga_solution()

rrb_cost_batch, rrb_energy_batch = env.round_robin_solution()
random_cost_batch, random_energy_batch = env.random_solution()

ga_finish_time_batchs = env.ga_solution()


print("n=50")
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