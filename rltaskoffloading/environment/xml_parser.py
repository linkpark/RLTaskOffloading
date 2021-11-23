import xml.dom.minidom
from rltaskoffloading.environment.task import Task

class XMLParser:
    def __init__(self, filename):
        self.DOMTree = xml.dom.minidom.parse(filename)
        self.collection = self.DOMTree.documentElement
        self.names_to_ids = {}
        self.ids_to_task = {}
        self.ids_to_index = {}
        self.ids_to_output_size = {}

        #store the original dependencies for Parents to Child
        self.succ_task_for_ids = {}
        self.pre_task_for_ids = {}

        self._parse_task()
        self.dependencies = self._parse_dependency_by_ids()
        self.ids_to_depth = self._calculate_depth()

        self.sorted_task = self.sort_task_by_EFT()

    def _parse_task(self):
        jobs = self.collection.getElementsByTagName("job")

        for job in jobs:
            name = job.getAttribute("name")
            job_id = job.getAttribute("id")
            running_time = float(job.getAttribute("runtime"))

            self.succ_task_for_ids[job_id] = []
            self.pre_task_for_ids[job_id] = []

            task = Task(job_id, running_time, name)

            self.ids_to_task[job_id] = task

            if name in self.names_to_ids.keys():
                self.names_to_ids[name].append(job_id)
            else:
                self.names_to_ids[name] = []
                self.names_to_ids[name].append(job_id)

            uses = job.getElementsByTagName("uses")
            output_size = 0.0

            for use in uses:
                if use.getAttribute("link") == "output":
                    output_size += float(use.getAttribute("size"))

            self.ids_to_output_size[job_id] = output_size

    def generate_task_list(self):
        task_list = []
        sequential_time = 0.0

        for i, task in enumerate(self.sorted_task):
            #task[0] means the id
            self.ids_to_task[task[0]].depth = self.ids_to_depth[task[0]]
            task_list.append(self.ids_to_task[task[0]])
            self.ids_to_index[task[0]] = i
            sequential_time += self.ids_to_task[task[0]].running_time

        return task_list, sequential_time

    def generate_dependency(self):
        children = self.collection.getElementsByTagName("child")
        dependencies = []

        for child in children:
            child_id = child.getAttribute("ref")
            parents = child.getElementsByTagName("parent")

            for parent in parents:
                parent_id = parent.getAttribute("ref")
                # set the environment as 100Mb networks
                comunicate_delay = self.ids_to_output_size[parent_id] / (1000.0 *1024.0 * 1024.0 / 8.0)

                dependency = [self.ids_to_index[parent_id],
                              self.ids_to_index[child_id],
                              comunicate_delay]

                dependencies.append(dependency)

        return dependencies

    def generate_task_list_by_toplogy(self):
        task_list = []
        sorted_task = []
        for task_id in self.names_to_ids["ExtractSGT"]:
            task_list.append(self.ids_to_task[task_id])
            sorted_task.append(task_id)

        for task_id in self.names_to_ids["SeismogramSynthesis"]:
            task_list.append(self.ids_to_task[task_id])
            sorted_task.append(task_id)

        for task_id in self.names_to_ids["ZipSeis"]:
            task_list.append(self.ids_to_task[task_id])
            sorted_task.append(task_id)

        for task_id in self.names_to_ids["PeakValCalcOkaya"]:
            task_list.append(self.ids_to_task[task_id])
            sorted_task.append(task_id)

        for task_id in self.names_to_ids["ZipPSA"]:
            task_list.append(self.ids_to_task[task_id])
            sorted_task.append(task_id)

        for index, task_id in enumerate(sorted_task):
            self.ids_to_index[task_id] = index

        return task_list

    def sort_task_by_EFT(self):
        #sort all task by EFT
        heft_rank = {}

        def CalculateRankForEach(task_id):
            if task_id in heft_rank.keys():
                return heft_rank[task_id]
            else:
                if len(self.succ_task_for_ids[task_id]) != 0:
                    score = self.ids_to_task[task_id].running_time + max( [self.dependencies[(task_id, succ_task_id)] +
                                                  CalculateRankForEach(succ_task_id) for succ_task_id in self.succ_task_for_ids[task_id]])
                else:
                    score = self.ids_to_task[task_id].running_time

                heft_rank[task_id] = score
                self.ids_to_task[task_id].heft_score = score

                return score

        for id in self.succ_task_for_ids:
            CalculateRankForEach(id)

        result = sorted(heft_rank.items(), key=lambda item:item[1], reverse=True)
        self.sorted_task = result

        return result

    def _parse_dependency_by_ids(self):
        children = self.collection.getElementsByTagName("child")
        dependencies = {}

        for child in children:
            child_id = child.getAttribute("ref")
            parents = child.getElementsByTagName("parent")


            for parent in parents:
                parent_id = parent.getAttribute("ref")

                self.pre_task_for_ids[child_id].append(parent_id)
                self.succ_task_for_ids[parent_id] += [child_id]

                comunicate_delay = self.ids_to_output_size[parent_id] / (100.0 * 1024.0 * 1024.0 / 8.0)

                dependencies[(parent_id, child_id)] = comunicate_delay

        return dependencies


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

        for id in self.pre_task_for_ids:
            ids_to_depth[id] = caluclate_depth_value(id)

        return ids_to_depth


    def _sort_task_by_topology(self):
        #sort all task by topology
        pass

# Test basic parse method
if __name__ == "__main__":
    task_parser = XMLParser('../data/CyberShake_20/CyberShake.n.20.0.xml')
    #task_parser.parse_task()
    #print(len(task_parser.succ_task_for_ids))
    #dependencies = task_parser.parse_dependency_by_ids()

    #print(task_parser.succ_task_for_ids)

    print(task_parser.sorted_task)
    print(task_parser.dependencies)

    task_list, _ = task_parser.generate_task_list()
    task_depth = task_parser._calculate_depth()

    print(task_depth)

    for i, task in enumerate(task_list):
        print("{}: {}".format(i, task.id_name))
        print("heft score {}".format(task.heft_score))

    print(task_parser.pre_task_for_ids)
    print(task_parser.dependencies[('ID00002', 'ID00003')])

