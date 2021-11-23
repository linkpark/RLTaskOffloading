import numpy as np
import copy
from rltaskoffloading.environment.task_graph import TaskGraph
from rltaskoffloading.environment.resource_cluster import ResourceCluster
from rltaskoffloading.environment.greedy_workflow import greedy_policy

def schedule(avaliable_task_list, task_graph, cluster, plan):
    while(len(avaliable_task_list) > 0 ):
        EET = np.zeros((len(avaliable_task_list), cluster.resource_number))
        FAT = np.zeros((len(avaliable_task_list), cluster.resource_number))
        for i, task_index in enumerate(avaliable_task_list):
            EET[i, :] = task_graph.dependency[task_index][task_index]
        EAT = np.zeros((len(avaliable_task_list), cluster.resource_number)) + \
              np.array(cluster.resources_available_time)

        for task_index, task in enumerate(avaliable_task_list):
            for r in range(cluster.resource_number):
                pre_task_set = task_graph.pre_task_sets[task]
                for pre_task in pre_task_set:
                    if plan[pre_task] != r:
                        communicate_cost = task_graph.dependency[pre_task][task]
                        start_time = communicate_cost + cluster.resources_available_time[r]

                        if start_time > FAT[task_index][r]:
                            FAT[task_index][r] = start_time

        ECT = EET + np.maximum(EAT, FAT)
        RT = np.argmin(ECT, axis=1)

        min_min_vector = []
        for task, rt in enumerate(RT):
            min_min_vector.append(ECT[task, rt])

        task_index = np.argmin( np.array(min_min_vector))
        schedule_task = avaliable_task_list[task_index]
        avaliable_task_list.remove(schedule_task)
        machine = RT[task_index]
        plan[schedule_task] = machine
        cluster.schedule_task(schedule_task, machine, task_graph)

    return plan

def get_ready_task(pre_task_set):
    ret = []
    for i in range(len(pre_task_set)):
        if len(pre_task_set[i]) == 0:
            ret.append(i)

    return ret

def min_min_policy(cluster, task_graph):
    task_number = task_graph.task_number
    pre_task_set = copy.deepcopy(task_graph.pre_task_sets)
    plan = [-1] * task_number

    while task_number > 0:
        ready_task = get_ready_task(pre_task_set)

        for task_index in ready_task:
            pre_task_set[task_index].add(-1)
            for task_set in pre_task_set:
                if task_index in task_set:
                    task_set.remove(task_index)

        task_number -= len(ready_task)
        plan = schedule(ready_task, task_graph, cluster, plan)

    return plan

if __name__ == "__main__":
    resource_cluster = ResourceCluster(5)
    task_graph = TaskGraph("../data/random20/random.20.0.gv", is_xml=False, is_matrix=False)

    plan = min_min_policy(resource_cluster, task_graph)
    print("final plan is {}".format(plan))
    resource_cluster.reset()
    cost = resource_cluster.get_running_time_through_schedule_plan(plan, task_graph)
    print("min min cost is {}".format(cost))

    resource_cluster.reset()
    plan, _ = greedy_policy(resource_cluster, task_graph)
    resource_cluster.reset()
    cost = resource_cluster.get_running_time_through_schedule_plan(plan, task_graph)
    print("greedy cost is {}".format(cost))


