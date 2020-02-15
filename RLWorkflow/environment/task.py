class Task(object):
    def __init__(self, id_name, running_time, type_name, depth=0, heft_score=0 ):
        self.id_name = id_name
        self.running_time = running_time
        self.type_name = type_name
        self.depth = depth
        self.heft_score = heft_score

    def print_task(self):
        print("task id name: {}, task type name: {} task run time: {}".format(
                                self.id_name, self.type_name, self.running_time))
