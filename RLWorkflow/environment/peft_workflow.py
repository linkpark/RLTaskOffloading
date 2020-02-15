
import numpy as np
from RLWorkflow.environment.task import Task
from RLWorkflow.environment.task_graph import TaskGraph
from RLWorkflow.environment.resource_cluster import ResourceCluster

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

    #set diag to be zero
    avg_communicate_time = avg_communicate_time - \
                           avg_communicate_time * np.eye(task_graph.task_number)

    return avg_communicate_time

def calculate_rank(tasks_avg_time, avg_communicate_time, cluster_resources):
    # calculate OCT
    task_oct = [-1] * len(tasks_avg_time)
    task_eft = [-1] * len(tasks_avg_time)
    rank = [-1] * len(tasks_avg_time)

    def OCT(task_index):
        if task_oct[task_index] != -1:
            return task_oct[task_index]
        else:
            succ_tasks = np.nonzero(avg_communicate_time[task_index])
            if len(succ_tasks[0]) == 0:
                task_oct[task_index] = 0
            else:
                task_oct[task_index] = np.max([ OCT(t_j) + tasks_avg_time[t_j] + avg_communicate_time[task_index][t_j] for t_j in succ_tasks[0]])

            return task_oct[task_index]

    def EFT(task_index):
        if (task_eft[task_index] != -1):
            return task_eft[task_index]
        else:
            succ_tasks = np.nonzero(avg_communicate_time[task_index])

            if len(succ_tasks[0]) != 0:
                task_eft[task_index] = tasks_avg_time[task_index] + np.max([avg_communicate_time[task_index][j] +
                                                                   EFT(j) for j in succ_tasks[0]])
            else:
                task_eft[task_index] = tasks_avg_time[task_index]

            return task_eft[task_index]

    for i in range(len(tasks_avg_time)):
        rank[i] = OCT(i) + EFT(i)

    rank = np.array(rank)

    return rank

def peft_policy( cluster, task_graph):
    tasks_avg_time = get_avg_time(task_graph)
    communicate_avg_time = get_avg_comm_time(task_graph)

    task_rank = calculate_rank(tasks_avg_time, communicate_avg_time, cluster)

    schedule_orders = np.argsort(-task_rank)

    schedule_plan = cluster.best_effort_schedule(schedule_orders, task_graph)

    final_plan = [ (task_index, schedule_plan[task_index]) for task_index in schedule_orders]
    #finish_time = cluster.running_time()

    cluster.reset()

    return final_plan

if __name__ == "__main__":
    task_set = []
    task_set.append(Task("1", 2.3, "A"))
    task_set.append(Task("2", 4.5, "A"))
    task_set.append(Task("3", 7.8, "A"))
    task_set.append(Task("4", 3.0, "A"))
    task_set.append(Task("5", 9.0, "A"))
    task_set.append(Task("6", 4.1, "A"))

    task_graph = TaskGraph(6)
    task_graph.add_task_list(task_set)

    task_graph.add_dependency(0, 1,4.0)
    task_graph.add_dependency(0, 2, 5.0)
    task_graph.add_dependency(1, 3, 8.0)
    task_graph.add_dependency(2, 4, 2.0)
    task_graph.add_dependency(3, 5, 3.0)
    task_graph.add_dependency(4, 5, 4.0)
    task_graph.render("test")

    cluster = ResourceCluster(4)

    schedule_plan = peft_policy(cluster, task_graph)
    cluster.reset()

    print(schedule_plan)