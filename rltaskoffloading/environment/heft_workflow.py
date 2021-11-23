import numpy as np
from rltaskoffloading.environment.task_graph import TaskGraph
from rltaskoffloading.environment.resource_cluster import ResourceCluster
from rltaskoffloading.environment.greedy_workflow import greedy_policy

'''
    In current environment setting, there is only one type of resource,
'''

def get_avg_time(task_graph):
    # Currently average time is the running time
    tasks_avg_time = [0] * task_graph.task_number

    for i in range(0, task_graph.task_number):
        tasks_avg_time[i] = task_graph.task_list[i].running_time

    return tasks_avg_time


def get_avg_comm_time(task_graph):
    avg_communicate_time = task_graph.dependency

    # set diag to be zero
    avg_communicate_time = avg_communicate_time - \
                           avg_communicate_time * np.eye(task_graph.task_number)

    return avg_communicate_time


def calculate_rank(tasks_avg_time, avg_communicate_time):
    task_rank = [-1] * len(tasks_avg_time)

    def CalculateRankForEach(index):
        if (task_rank[index] != -1):
            return task_rank[index]
        else:
            succ_tasks = np.nonzero(avg_communicate_time[index])

            if len(succ_tasks[0]) != 0:
                task_rank[index] = tasks_avg_time[index] + np.max([avg_communicate_time[index][j] +
                                                                   CalculateRankForEach(j) for j in succ_tasks[0]])
            else:
                task_rank[index] = tasks_avg_time[index]

            return task_rank[index]

    # Tasks have been topological sorted

    for i in range(len(tasks_avg_time)):
        task_rank[i] = CalculateRankForEach(i)

    '''
    i = len(tasks_avg_time)
    while i > 0:
        succ_tasks = np.nonzero(avg_communicate_time[i-1])
        if len(succ_tasks[0]) == 0:
            task_rank[i-1] = tasks_avg_time[i-1]
        else:
            task_rank[i-1] = tasks_avg_time[i-1] + np.max( [avg_communicate_time[i-1][j] +
                                                    task_rank[j] for j in succ_tasks[0]])
        i -= 1



    for i in range(0, len(tasks_avg_time)):
        succ_tasks = np.nonzero(avg_communicate_time[i])
        if len(succ_tasks[0]) != 0:
            task_rank[i] = tasks_avg_time[i] + np.max([avg_communicate_time[i][j] +
                                                   CalculateRankForEach(j) for j in succ_tasks[0]])
        else:
            task_rank[i] = tasks_avg_time[i]
    '''
    task_rank = np.array(task_rank)

    return task_rank


def heft_policy(cluster, task_graph):
    tasks_avg_time = get_avg_time(task_graph)
    communicate_avg_time = get_avg_comm_time(task_graph)

    task_rank = calculate_rank(tasks_avg_time, communicate_avg_time)

    schedule_orders = np.argsort(-task_rank)

    schedule_plan = cluster.best_effort_schedule(schedule_orders, task_graph)
    # finish_time = cluster.running_time()

    final_plan = [(task_index, schedule_plan[task_index]) for task_index in schedule_orders]

    cluster.reset()
    return final_plan


if __name__ == "__main__":
    task_graph = TaskGraph('../data/random20/random.20.1.gv', is_xml=False, is_matrix=False)

    resource_cluster = ResourceCluster(5)
    plan = heft_policy(resource_cluster, task_graph)
    resource_cluster.reset()
    cost = resource_cluster.get_running_time_through_schedule_plan(plan, task_graph)
    print(cost)

    resource_cluster.reset()

    scheduling_plan = [0]*len(plan)
    for i, machine in plan:
        scheduling_plan[i] = machine

    unorder_plan = [(i, machine) for i,machine in enumerate(scheduling_plan)]
    cost = resource_cluster.get_running_time_through_schedule_plan(unorder_plan, task_graph)
    print(cost)

    # plan, _ = greedy_policy(resource_cluster, task_graph)
    #
    # greedy_plan = [(task_index, machine) for task_index, machine in enumerate(plan)]
    # print(greedy_plan)
    # resource_cluster.reset()
    # cost = resource_cluster.get_running_time_through_schedule_plan(greedy_plan, task_graph)
    # print(cost)