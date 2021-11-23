import numpy as np
from graphviz import Digraph

from rltaskoffloading.environment.xml_parser import XMLParser
from rltaskoffloading.environment.dot_parser import DotParser
from rltaskoffloading.environment.task import Task

import json


class TaskGraph(object):
    def __init__(self, file_name, is_xml=True, is_matrix=False):
        if is_xml == True:
            self._parse_from_xml(file_name, is_matrix)
        else:
            self._parse_from_dot(file_name, is_matrix)

    # add task list to
    def _parse_from_dot(self, file_name, is_matrix):
        parser = DotParser(file_name, is_matrix)
        task_list = parser.generate_task_list()

        self.task_number = len(task_list)
        self.dependency = np.zeros((self.task_number, self.task_number))
        self.task_list = []

        self.pre_task_sets = []
        self.task_finish_time = [0] * self.task_number
        self.edge_set = []

        for _ in range(self.task_number):
            self.pre_task_sets.append(set([]))
        # add task list to
        self.add_task_list(task_list)

        dependencies = parser.generate_dependency()

        for pair in dependencies:
            self.add_dependency(pair[0], pair[1], pair[2])

        self.max_runtime = np.max(self.dependency[self.dependency > 0.01])
        self.min_runtime = np.min(self.dependency[self.dependency > 0.01])

        # calcualte the heft rank
        tasks_avg_time = [0] * self.task_number

        for i in range(0, self.task_number):
            tasks_avg_time[i] = self.task_list[i].running_time

        avg_communicate_time = self.dependency

        # set diag to be zero
        avg_communicate_time = avg_communicate_time - \
                               avg_communicate_time * np.eye(self.task_number)

        def calculate_rank(tasks_avg_time, avg_communicate_time):
            task_rank = [-1] * len(tasks_avg_time)

            def CalculateRankForEach(index):
                if (task_rank[index] != -1):
                    return task_rank[index]
                else:
                    succ_tasks = np.nonzero(avg_communicate_time[index])

                    if len(succ_tasks[0]) != 0:
                        task_rank[index] = tasks_avg_time[index] + np.max([avg_communicate_time[index][j] +
                                                                           CalculateRankForEach(j) for j in
                                                                           succ_tasks[0]])
                    else:
                        task_rank[index] = tasks_avg_time[index]

                    return task_rank[index]
            for i in range(len(tasks_avg_time)):
                task_rank[i] = CalculateRankForEach(i)

            task_rank = np.array(task_rank)
            return task_rank

        task_rank = calculate_rank(tasks_avg_time, avg_communicate_time)
        self.heft_orders = np.argsort(-task_rank)

    def _parse_from_xml(self, file_name, is_matrix):
        parser = XMLParser(file_name)

        task_list = parser.generate_task_list_by_toplogy()

        self.task_number = len(task_list)
        self.dependency = np.zeros((self.task_number, self.task_number))
        self.task_list = []

        self.pre_task_sets = []
        self.task_finish_time = [0] * self.task_number
        self.edge_set = []

        for _ in range(self.task_number):
            self.pre_task_sets.append(set([]))
        # add task list to
        self.add_task_list(task_list)

        dependencies = parser.generate_dependency()

        for pair in dependencies:
            self.add_dependency(pair[0], pair[1], pair[2])

        self.max_runtime = np.max(self.dependency[ self.dependency > 0.01])
        self.min_runtime = np.min(self.dependency[ self.dependency > 0.01])

        self.mean, self.std = self.return_cost_metric()
        self.norm_dependencies = np.copy(self.dependency)
        self.norm_dependencies[self.norm_dependencies < 0.01] = 0.0
        self.norm_dependencies[self.norm_dependencies > 0.0 ] = ( self.norm_dependencies[self.norm_dependencies > 0.0 ] - self.mean ) / (self.std)

    def add_task_list(self, task_list):
        self.task_list = task_list

        for i in range(0, len(self.task_list)):
            self.dependency[i][i] = task_list[i].running_time

    def add_dependency(self, pre_task_index, succ_task_index, transmission_cost):
        self.dependency[pre_task_index][succ_task_index] = transmission_cost
        self.pre_task_sets[succ_task_index].add(pre_task_index)

        # for each edge, we use a five dimension vector to represent this
        edge = [pre_task_index,
                self.task_list[pre_task_index].depth,
                self.task_list[pre_task_index].running_time,
                transmission_cost,
                succ_task_index,
                self.task_list[succ_task_index].depth,
                self.task_list[succ_task_index].running_time]

        self.edge_set.append(edge)

    def feature_scaling(self, cost):
        return (cost - self.min_runtime) / (self.max_runtime - self.min_runtime)

    def encode_point_sequence(self):
        point_sequence = []
        for i in range(self.task_number):
            cost_time = [ self.feature_scaling(self.dependency[i][i]) ]

            #print("befor norm {}, after norm {}".format(self.dependency[i][i], cost_time))
            #heft_score = [self.task_list[i].heft_score]
            pre_task_cost = []
            pre_task_index_set = []

            succs_task_cost = []
            succs_task_index_set = []

            for pre_task_index in range(0, i):
                # if there is no edge between tasks, the dependency[i][j] will be 0
                if self.dependency[pre_task_index][i] > 0.1:
                    communication_cost = self.feature_scaling(self.dependency[pre_task_index][i])
                    pre_task_cost.append(communication_cost)
                    pre_task_index_set.append(pre_task_index)

            while (len(pre_task_cost) < 6):
                pre_task_cost.append(-1.0)
                pre_task_index_set.append(-1.0)

            for succs_task_index in range(i + 1, self.task_number):
                # if there is no edge between tasks, the dependency[i][j] will be 0
                if self.dependency[i][succs_task_index] > 0.1:
                    communication_cost = self.feature_scaling(self.dependency[i][succs_task_index])
                    succs_task_cost.append(communication_cost)
                    succs_task_index_set.append(succs_task_index)

            while (len(succs_task_cost) < 6):
                succs_task_cost.append(-1.0)
                succs_task_index_set.append(-1.0)

            succs_task_cost = succs_task_cost[0:6]
            succs_task_index_set = succs_task_index_set[0:6]
            pre_task_index_set = pre_task_index_set[0:6]
            pre_task_cost = pre_task_cost[0:6]

            point_vector = cost_time + pre_task_cost + succs_task_cost + pre_task_index_set + succs_task_index_set
            point_sequence.append(point_vector)

        return point_sequence

    def encode_point_sequence_with_heft_sequence(self):
        original_point_sequence = self.encode_point_sequence()
        result_point_sequence = []
        for i in self.heft_orders:
            result_point_sequence.append(original_point_sequence[i])

        return np.array(result_point_sequence)

    def encode_edge_sequence(self):
        edge_array = []
        for i in range(0, len(self.edge_set)):
            if i < len(self.edge_set):
                edge_array.append(self.edge_set[i])
            else:
                edge_array.append([0, 0, 0, 0, 0, 0, 0])

        # input edge sequence refers to start node index
        edge_array = sorted(edge_array)

        return edge_array

    def return_cost_metric(self):
        adj_matrix = np.array(self.dependency)
        cost_set = adj_matrix[np.nonzero( adj_matrix )]
        cost_set = cost_set[cost_set > 0.01]

        mean = np.mean(cost_set)
        std = np.std(cost_set)

        return mean, std

    def print_graphic(self):
        print(self.dependency)
        print("This is pre_task_sets:")
        print(self.pre_task_sets)
        print("This is edge set:")
        print(self.edge_set)

    def render(self, path):
        dot = Digraph(comment='DAG')

        # str(self.task_list[i].running_time)
        for i in range(0, self.task_number):
            dot.node(str(i), str(i) + ":" +str(self.task_list[i].running_time))

        for e in self.edge_set:
            dot.edge(str(e[0]), str(e[4]), constraint='true', label="%.6f" % e[3])

        dot.render(path, view=False)

    def serilaizeToJson(self, path):
        dict = {"graph": [{"nodes": []}, {"edges": []}]}

        for node in self.task_list:
            dict["graph"][0]["nodes"].append([node.depth, node.running_time])

        dict["graph"][1]["edges"] = self.edge_set

        with open(path, 'w') as outfile:
            json.dump(dict, outfile)

    def deserilaizeFromJson(self, path):
        dict = {}
        task_list = []

        with open(path, 'r') as infile:
            dict = json.load(infile)

        for node in dict["graph"][0]["nodes"]:
            task = Task(node[0], node[1])
            task_list.append(task)

        self.add_task_list(task_list)

        for edge in dict["graph"][1]["edges"]:
            self.add_dependency(edge[0], edge[4], edge[3])


if __name__ == "__main__":
    task_graph = TaskGraph('../data/random20/random.20.0.gv', is_xml=False)
    task_graph.render('test')

    print(task_graph.heft_orders)

    encoder_point = task_graph.encode_point_sequence()

    np.set_printoptions(suppress=True)

    print(np.array(encoder_point).shape)
    print(np.array(encoder_point))

    #task_graph = TaskGraph("../data/CyberShake_30/CyberShake.n.30.0.xml")
    #task_graph.render("test")
    #print(np.array(task_graph.encode_point_sequence()))
    #print(np.array(task_graph.encode_point_sequence()).shape)
    #print()
    '''
    
    cost_set_index = np.nonzero(task_graph.dependency)
    none_zero_value = task_graph.dependency[cost_set_index]
    print("len of non zero value is {}".format(len(none_zero_value)))

    dependency = np.copy(task_graph.dependency)
    dependency[dependency < 0.01] = 0.0

    cost_set_index = np.nonzero(task_graph.dependency)
    none_zero_value = task_graph.dependency[cost_set_index]
    print("len of non zero value is {}".format(len(none_zero_value)))

    np.set_printoptions(precision=5, suppress=True)

    cost_set_index = np.nonzero(dependency)
    none_zero_value = dependency[cost_set_index]
    print("len of non zero value is {}".format(len(none_zero_value)))

    print()
    print("Norm dependency is: ")
    print(task_graph.norm_dependencies)
    print(len(task_graph.norm_dependencies[np.nonzero(task_graph.norm_dependencies)]))

    from rltaskoffloading.environment.resource_cluster import ResourceCluster
    resource_cluster = ResourceCluster(5)
    plan = np.zeros(20, dtype=np.int32)

    cost = resource_cluster.get_norm_cost_through_step_by_step_schedule(plan, task_graph)

    print(cost)

    # Test max running time and min running time
    print(task_graph.max_runtime)
    print(task_graph.min_runtime)
    '''
