import pydotplus
from rltaskoffloading.environment.task import Task

class DotParser(object):
    def __init__(self, file_name, is_matrix):
        self.succ_task_for_ids = {}
        self.pre_task_for_ids = {}

        self.dot_ob = pydotplus.graphviz.graph_from_dot_file(file_name)
        self._parse_task()
        self._parse_dependecies()
        self._calculate_depth()

    def _parse_task(self):
        jobs = self.dot_ob.get_node_list()
        self.task_list = [0] * len(jobs)

        for job in jobs:
            job_id = job.get_name()
            data_size = int(eval(job.obj_dict['attributes']['size']))
            #running_cost = float(data_size) / (40.0 * 1024.0 * 1024.0 )
            #running_cost = float(data_size) / ( 30*100 * 1024 * 1024 / 8.0 )
            running_cost = float(data_size) / (100.0 * 1024.0 * 1024.0 / 8.0)
            #running_cost = float(data_size) / (1024 * 1024 * 1024 * 8.0)
            task = Task(job_id, running_cost, "compute")
            id = int(job_id) - 1
            self.task_list[id] = task

    def _parse_dependecies(self):
        edge_list = self.dot_ob.get_edge_list()
        dependencies = []

        for i in range(len(self.task_list)):
            self.pre_task_for_ids[i] = []
            self.succ_task_for_ids[i] = []

        for edge in edge_list:
            source_id = int(edge.get_source()) - 1
            destination_id = int(edge.get_destination()) - 1
            data_size = int(eval(edge.obj_dict['attributes']['size']))

            self.pre_task_for_ids[destination_id].append(source_id)
            self.succ_task_for_ids[source_id].append(destination_id)

            communication_cost = float(data_size) / (100.0 * 1024.0 * 1024.0 / 8.0)
            dependency = [source_id, destination_id, communication_cost]

            dependencies.append(dependency)

        self.dependencies = dependencies

    def _calculate_depth(self):
        ids_to_depth = dict()

        def caluclate_depth_value(id):
            if id in ids_to_depth.keys():
                return ids_to_depth[id]
            else:
                if len(self.pre_task_for_ids[id]) != 0:
                    depth = 1 + max([caluclate_depth_value(pre_task_id) for
                                     pre_task_id in self.pre_task_for_ids[id]])
                else:
                    depth = 0

                ids_to_depth[id] = depth

            return ids_to_depth[id]

        for id in range(len(self.task_list)):
            ids_to_depth[id] = caluclate_depth_value(id)

        for id, depth in ids_to_depth.items():
            self.task_list[id].depth = depth

    def generate_task_list(self):
        return self.task_list

    def generate_dependency(self):
        return self.dependencies






