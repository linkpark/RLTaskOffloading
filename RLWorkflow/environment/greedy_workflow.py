import numpy as np

def greedy_policy( cluster, task_graph ):
    '''
    Need to implement the greedy method
    :param cluster:
    :param task_graph:
    :return:
    '''
    schedule_plan = [0] *task_graph.task_number;
    for i in range( 0, len(task_graph.task_list) ):
        best_resource = np.argmin(cluster.resources_available_time)
        # record current finish time
        current_finish_time = max(cluster.resources_available_time)
        schedule_plan[i] = best_resource

        current_task_start_time = cluster.resources_available_time[best_resource]
        task_start_time = current_task_start_time

        for pre_task_index in task_graph.pre_task_sets[i]:
            pre_task_running_machine = schedule_plan[pre_task_index]

            if pre_task_running_machine != best_resource:
                task_start_time = max((task_graph.task_finish_time[pre_task_index] +
                                       task_graph.dependency[pre_task_index][i]), task_start_time)

        task_graph.task_finish_time[i] = task_start_time + task_graph.dependency[i][i]

        cluster.resources_available_time[best_resource] = task_graph.task_finish_time[i]

        finish_time = max(cluster.resources_available_time)
        cost = finish_time - current_finish_time

    return schedule_plan, finish_time