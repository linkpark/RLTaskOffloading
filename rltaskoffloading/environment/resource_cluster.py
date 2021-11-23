import numpy as np

from rltaskoffloading.environment.task_graph import TaskGraph
from rltaskoffloading.environment.task_graph import Task


class ResourceCluster:
    """Resource cluster is used to simulate the cluster environment"""
    def __init__(self, resource_number):
        self.resource_number = resource_number
        self.resources_available_time = [0] * resource_number
        self.current_plan = {}

    def reset(self):
        self.resources_available_time = [0] * self.resource_number
        self.current_plan = {}

    def get_running_time_through_schedule_plan(self, schedule_plan, task_graph):
        """Calculate the running time through schedule plan"""
        schedule_plan_index = [-1] * len(schedule_plan)
        for i, machine in schedule_plan:
            schedule_plan_index[i] = machine

        for i, machine in schedule_plan:
            current_task_start_time = self.resources_available_time[machine]
            task_start_time = current_task_start_time

            for pre_task_index in task_graph.pre_task_sets[i]:
                pre_task_running_machine = schedule_plan_index[pre_task_index]

                if pre_task_running_machine != machine:
                    task_start_time = max((task_graph.task_finish_time[pre_task_index] +
                                           task_graph.dependency[pre_task_index][i]), task_start_time)

            task_graph.task_finish_time[i] = task_start_time + \
                                             task_graph.dependency[i][i]

            self.resources_available_time[machine] = task_graph.task_finish_time[i]

        finish_time = max(self.resources_available_time)

        return finish_time

    def schedule_task_get_finish_time(self, task_index, machine, task_graph):
        """schedule one task in the cluster"""
        self.current_plan.append(machine)
        current_finish_time = np.max(self.resources_available_time)
        pre_task_sequence = task_graph.pre_task_sets[task_index]

        start_time = self.resources_available_time[machine]

        # dependency_start_time = [0] * len(pre_task_sequence)

        for pre_task_index in pre_task_sequence:
            pre_task_plan = self.current_plan[pre_task_index]
            pre_task_finish_time = self.resources_available_time[pre_task_plan]

            if pre_task_plan != machine:
                dependency_start_time = pre_task_finish_time + task_graph.dependency[pre_task_index][task_index]

                if dependency_start_time > start_time:
                    start_time = dependency_start_time

        self.resources_available_time[machine] = start_time + task_graph.dependency[task_index][task_index]

        finish_time_after_schedule = np.max(self.resources_available_time)

        return finish_time_after_schedule

    def running_time(self):
        return max(self.resources_available_time)

    def get_resources_minimal_finish_time(self):
        return min(self.resources_available_time)

    '''
        Calculate finish time for each machine
    '''
    def calculate_finish_time(self, task_index, resouce_number, task_graph, schedule_plan):
        current_task_start_time = self.resources_available_time[resouce_number]
        task_start_time = current_task_start_time

        for pre_task_index in task_graph.pre_task_sets[task_index]:
            pre_task_running_machine = schedule_plan[pre_task_index]

            if pre_task_running_machine != resouce_number:
                task_start_time = max((task_graph.task_finish_time[pre_task_index] +
                                       task_graph.dependency[pre_task_index][task_index]), task_start_time)

        task_finish_time = task_start_time + task_graph.dependency[task_index][task_index]

        return task_finish_time

    def best_effort_schedule(self, task_sequence, task_graph ):
        schedule_plan = [-1] * task_graph.task_number
        final_finish_time = 0

        for task_index in task_sequence:
            finish_time = [0] * self.resource_number
            for i in range(0, self.resource_number):
                finish_time[i] = self.calculate_finish_time(task_index, i, task_graph, schedule_plan)

            best_resource = np.argmin(finish_time)
            self.resources_available_time[best_resource] = finish_time[best_resource]
            task_graph.task_finish_time[task_index] = finish_time[best_resource]
            final_finish_time = finish_time[best_resource]

            schedule_plan[task_index] = best_resource

        task_graph.task_finish_time = [0] * task_graph.task_number

        return schedule_plan

    def get_cost_through_step_by_step_schedule(self, schedule_plan, task_graph):
        cost_list = []
        for i, plan in schedule_plan:
            cost_list.append(self.schedule_task(i, plan, task_graph))

        return cost_list

    def get_norm_cost_through_step_by_step_schedule(self, schedule_plan, task_graph):
        cost_list = []
        for i, plan in enumerate(schedule_plan):
            cost_list.append(self.schedule_task_of_norm_dependencies(i, plan, task_graph))

        return cost_list

    def get_final_cost_time_through_step_by_step_schedule(self, schedule_plan, task_graph):
        cost_list = []
        for i, plan in enumerate(schedule_plan):
            cost_list.append(self.schedule_task_get_finish_time(i, plan, task_graph))

        return cost_list

    def schedule_task_of_norm_dependencies(self, task_index, machine, task_graph):
        """schedule one task in the cluster"""
        self.current_plan.append(machine)
        current_finish_time = np.max(self.resources_available_time)
        pre_task_sequence = task_graph.pre_task_sets[task_index]

        start_time = self.resources_available_time[machine]

        # dependency_start_time = [0] * len(pre_task_sequence)

        for pre_task_index in pre_task_sequence:
            pre_task_plan = self.current_plan[pre_task_index]
            pre_task_finish_time = self.resources_available_time[pre_task_plan]

            if pre_task_plan != machine:
                dependency_start_time = pre_task_finish_time + task_graph.norm_dependencies[pre_task_index][task_index]

                if dependency_start_time > start_time:
                    start_time = dependency_start_time

        self.resources_available_time[machine] = start_time + task_graph.norm_dependencies[task_index][task_index]

        finish_time_after_schedule = np.max(self.resources_available_time)

        delta_cost = finish_time_after_schedule - current_finish_time

        return delta_cost

    def schedule_task(self, task_index, machine, task_graph ):
        """schedule one task in the cluster"""
        self.current_plan[task_index] = machine
        current_finish_time = np.max(self.resources_available_time)
        pre_task_sequence = task_graph.pre_task_sets[task_index]

        start_time = self.resources_available_time[machine]

        #dependency_start_time = [0] * len(pre_task_sequence)

        for pre_task_index in pre_task_sequence:
            pre_task_plan = self.current_plan[pre_task_index]
            pre_task_finish_time = self.resources_available_time[pre_task_plan]

            if pre_task_plan != machine:
                dependency_start_time = pre_task_finish_time + task_graph.dependency[pre_task_index][task_index]

                if dependency_start_time > start_time:
                    start_time = dependency_start_time

        self.resources_available_time[machine] = start_time + task_graph.dependency[task_index][task_index]

        finish_time_after_schedule = np.max(self.resources_available_time)

        delta_cost = finish_time_after_schedule - current_finish_time

        return delta_cost