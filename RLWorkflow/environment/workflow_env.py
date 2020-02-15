import numpy as np
import os

from RLWorkflow.environment.resource_cluster import ResourceCluster
from RLWorkflow.environment.task_graph import TaskGraph
from RLWorkflow.environment.heft_workflow import heft_policy
from RLWorkflow.environment.greedy_workflow import greedy_policy
from RLWorkflow.environment.peft_workflow import peft_policy
from RLWorkflow.environment.min_min_workflow import min_min_policy
from RLWorkflow.environment.max_min_workflow import max_min_policy


def generate_point_batch_for_multi_graphs(batch_size, graph_class_number, graph_file_path,
                                          start_symbol=0, time_major=True, is_xml=True, resource_number=5):
    """
        using vertice as basic sequence unit
    """
    encoder_batch = []
    decoder_batch = []
    target_batch = []
    task_graph_list = []
    return_task_graph_list = []
    resource_cluster = ResourceCluster(resource_number)

    schedule_plan_list = []

    for i in range(graph_class_number):
        if is_xml:
            task_graph = TaskGraph(graph_file_path + str(i) + '.xml', is_matrix=False, is_xml=True)
        else:
            task_graph = TaskGraph(graph_file_path + str(i) + '.gv', is_matrix=False, is_xml=False)
        task_graph_list.append(task_graph)

    for i in range(graph_class_number):
        task_graph = task_graph_list[i]

        plan = heft_policy(resource_cluster, task_graph)
        schedule_plan = []

        for i, machine in plan:
            schedule_plan.append(machine)

        schedule_plan_list.append(schedule_plan)

        for j in range(int(batch_size/graph_class_number) ):
            point_matrix = np.array(task_graph.encode_point_sequence_with_heft_sequence())

            encoder_batch.append(point_matrix)
            decoder_batch.append([start_symbol] + schedule_plan[0:-1])
            target_batch.append(schedule_plan)
            return_task_graph_list.append(task_graph)

    # time major for encoder, decoder and target
    if time_major == True:
        encoder_batch = np.array(encoder_batch).swapaxes(0, 1)
        decoder_batch = np.array(decoder_batch).swapaxes(0, 1)
        target_batch = np.array(target_batch).swapaxes(0, 1)
        decoder_full_length = np.asarray([decoder_batch.shape[0]] * decoder_batch.shape[1])
    else:
        encoder_batch = np.array(encoder_batch)
        decoder_batch = np.array(decoder_batch)
        target_batch = np.array(target_batch)
        decoder_full_length = np.asarray([decoder_batch.shape[1]] * decoder_batch.shape[0])

    return encoder_batch, decoder_batch, target_batch, decoder_full_length,return_task_graph_list


def generate_point_batch_for_random_graphs(batch_size, graph_number, graph_file_path, time_major=True):
    encoder_list = []
    task_graph_list = []

    encoder_batchs = []
    encoder_lengths = []
    task_graph_batchs = []
    decoder_full_lengths = []
    max_running_time_vector = []
    min_running_time_vector = []

    max_running_time_batchs = []
    min_running_time_batchs = []

    for i in range(graph_number):
        task_graph = TaskGraph(graph_file_path + str(i) + '.gv', is_matrix=False, is_xml=False)
        task_graph_list.append(task_graph)
        max_running_time_vector.append(task_graph.max_runtime)
        min_running_time_vector.append(task_graph.min_runtime)

        task_encode = np.array(task_graph.encode_point_sequence_with_heft_sequence())
        encoder_list.append(task_encode)

    for i in range(int(graph_number / batch_size)):
        start_batch_index = i*batch_size
        end_batch_index = (i+1)*batch_size

        task_encode_batch = encoder_list[start_batch_index:end_batch_index ]
        if time_major:
            task_encode_batch = np.array(task_encode_batch).swapaxes(0,1)
            decoder_full_length = np.asarray([task_encode_batch.shape[0]] * task_encode_batch.shape[1])
        else:
            task_encode_batch = np.array(task_encode_batch)
            decoder_full_length = np.asarray([task_encode_batch.shape[1]] * task_encode_batch.shape[0])

        decoder_full_lengths.append(decoder_full_length)

        encoder_batchs.append(task_encode_batch)
        task_graph_batch = task_graph_list[start_batch_index:end_batch_index]
        task_graph_batchs.append(task_graph_batch)
        max_running_time_batchs.append(max_running_time_vector[start_batch_index:end_batch_index])
        min_running_time_batchs.append(min_running_time_vector[start_batch_index:end_batch_index])

    return encoder_batchs, task_graph_batchs, decoder_full_lengths, max_running_time_batchs, min_running_time_batchs

class WorkflowEnvironment(object):
    def __init__(self, batch_size, graph_number, graph_file_path, is_xml, time_major=True, start_symbol=0, resource_number = 5, input_dim=25):
        self.resource_cluster = ResourceCluster(resource_number)
        self.encoder_batchs = []
        self.decoder_full_lengths = []
        self.task_graphs = []
        self.target_batchs = []
        self.decoder_batchs = []
        self.time_major = time_major
        self.graph_file_path = graph_file_path
        self.is_xml = is_xml
        self.input_dim = input_dim

        # get graphs from environment
        encoder_batchs, task_graph_list, decoder_full_lengths, max_running_time_batchs, min_running_time_batchs =\
            generate_point_batch_for_random_graphs(batch_size, graph_number, graph_file_path, time_major)

        '''
        encoder_batch, decoder_batch, target_batch, decoder_full_length, task_graph_list = \
                    generate_point_batch_for_multi_graphs(batch_size=batch_size,
                                                          graph_class_number=graph_number,
                                                          graph_file_path=graph_file_path,
                                                          start_symbol=start_symbol,
                                                          time_major=self.time_major,
                                                          is_xml=is_xml
                                                         )
        '''
        self.max_running_time_batchs = max_running_time_batchs
        self.min_running_time_batchs = min_running_time_batchs

        self.max_running_time = task_graph_list[0][0].max_runtime
        self.min_running_time = task_graph_list[0][0].min_runtime

        self.encoder_batchs = encoder_batchs
        self.encoder_lengths = decoder_full_lengths
        self.decoder_full_lengths = decoder_full_lengths
        self.task_graphs = task_graph_list

        cost_batchs = []
        for task_graph_batch in self.task_graphs:
            cost_batch = self.get_heft_cost_by_graph_batch(task_graph_batch)
            cost_batchs.append(cost_batch)

        self.heft_cost = np.mean(cost_batchs)

        self.start_symbol = start_symbol
        self.batch_size = batch_size
        self.graph_number = graph_number

        self.generate_test_batches()

    def generate_test_batches(self):
        encoder_batch, decoder_batch, target_batch, decoder_full_length, task_graph_list = \
                        generate_point_batch_for_multi_graphs(100, 100,
                                                         graph_file_path=self.graph_file_path,
                                                         start_symbol=self.start_symbol,
                                                         time_major=self.time_major,
                                                         is_xml=self.is_xml)
        self.test_encoder_batch = encoder_batch
        self.test_decoder_full_length = decoder_full_length
        self.test_graph_list = task_graph_list

    def get_initial_state_batch(self):
        return self.encoder_batchs

    def score_func(self, cost, max_running_time_batch, min_running_time_batch):
        return -np.array(cost) / ( max_running_time_batch - min_running_time_batch )

    def get_speedup_ratio(self, cost, task_graph):
        sequential_time_batch = []
        for task in task_graph:
            sequential_time_batch.append(task.sequential_time)

        sequential_time_batch = np.array(sequential_time_batch)

        return sequential_time_batch / cost

    def get_heft_base_line_reward(self):
        base_line_reward = []
        for i in range(self.graph_number):
            self.resource_cluster.reset()
            task_graph = self.task_graphs[i]
            plan, _ = heft_policy(cluster=self.resource_cluster, task_graph=task_graph)
            cost = self.resource_cluster.get_cost_through_step_by_step_schedule(plan, task_graph)
            score = -cost

            for j in range( int(self.batch_size / self.graph_number)):
                base_line_reward.append(score)

        base_line_reward = np.array(base_line_reward)

        return base_line_reward

    def get_running_cost(self, action_sequence_batch, task_graph_batch):
        cost_batch = []
        for plan, task_graph in zip(action_sequence_batch, task_graph_batch):
            self.resource_cluster.reset()
            cost = self.resource_cluster.get_running_time_through_schedule_plan(plan, task_graph)
            score = -cost
            cost_batch.append(score)

        cost_batch = np.array(cost_batch)
        return np.mean(cost_batch)

    def get_running_cost_batch(self, action_sequence_batch, task_graph_batch):
        cost_batch = []
        for plan, task_graph in zip(action_sequence_batch, task_graph_batch):
            self.resource_cluster.reset()

            real_plan = [(task_index, machine) for task_index, machine in zip(task_graph.heft_orders, plan)]

            cost = self.resource_cluster.get_running_time_through_schedule_plan(real_plan, task_graph)
            cost_batch.append(cost)

        cost_batch = np.array(cost_batch)
        return cost_batch

    def get_target_batch(self, action_sequence_batch, gamma=1.0):
        target_batch = []
        action_length = action_sequence_batch.shape[1]

        for i in range(self.graph_number):
            task_graph = self.task_graphs[i]

            for j in range( int(self.batch_size / self.graph_number) ):
                self.resource_cluster.reset()
                plan = action_sequence_batch[ i*int(self.batch_size / self.graph_number) + j ]
                cost = self.resource_cluster.get_running_time_through_schedule_plan(plan, task_graph)
                score = self.score_func(cost)

                target_batch.append(score)

        target_batch = np.array(target_batch)
        target_batch = target_batch.reshape(target_batch.shape[0], 1)
        target_goal_list = target_batch

        for i in range(1, action_length):
            target_goal_list = np.concatenate((target_goal_list, gamma ** i * target_batch), axis=-1)

        return target_goal_list

    def get_target_batch_step_by_step(self, action_sequence_batch):
        target_batch = []

        for i in range(self.graph_number):
            task_graph = self.task_graphs[i]

            for j in range( int(self.batch_size/ self.graph_number)):
                self.resource_cluster.reset()
                plan = action_sequence_batch[i * int(self.batch_size / self.graph_number) + j]
                cost = self.resource_cluster.get_cost_through_step_by_step_schedule(plan, task_graph)
                score = self.score_func(cost)

                target_batch.append(score)
        return target_batch

    def get_reward_batch_step_by_step(self, action_sequence_batch, task_graph_batch, max_running_time_batch, min_running_time_batch):
        target_batch = []
        for i in range(action_sequence_batch.shape[0]):
            task_graph = task_graph_batch[i]
            self.resource_cluster.reset()
            plan = action_sequence_batch[i]

            schedule_plan = [(task_index, machine) for task_index, machine in zip(task_graph.heft_orders, plan)]

            cost = self.resource_cluster.get_cost_through_step_by_step_schedule(schedule_plan, task_graph)
            #cost = self.resource_cluster.get_norm_cost_through_step_by_step_schedule(plan, task_graph)
            score = self.score_func(cost, max_running_time_batch[i], min_running_time_batch[i])
            target_batch.append(score)

        target_batch = np.array(target_batch)
        return target_batch

    def step(self, action_sequence_batch, task_graph_batch, max_running_time_batch, min_running_time_batch):
        return self.get_reward_batch_step_by_step(action_sequence_batch, task_graph_batch,
                                                  max_running_time_batch, min_running_time_batch)

    def get_goal_batch_step_by_step(self, action_sequence_batch, task_graph_batch):
        target_batch = []
        for i in range(action_sequence_batch.shape[0]):
            task_graph = task_graph_batch[i]
            self.resource_cluster.reset()
            plan = action_sequence_batch[i]
            cost = self.resource_cluster.get_final_cost_time_through_step_by_step_schedule(plan, task_graph)
            score = self.score_func(cost)

            target_batch.append(score)

        return target_batch

    def get_sparse_reward_batch(self, action_sequence_batch, task_graph_batch):
        target_batch = []
        time_steps = action_sequence_batch.shape[1]
        for i in range(action_sequence_batch.shape[0]):
            task_graph = task_graph_batch[i]
            self.resource_cluster.reset()
            plan = action_sequence_batch[i]
            cost = self.resource_cluster.get_running_time_through_schedule_plan(schedule_plan=plan, task_graph=task_graph)
            score = self.score_func(cost)
            reward = [0] * (time_steps -1)
            reward.append(score)
            target_batch.append(reward)

        target_batch = np.array(target_batch)
        return target_batch

    def get_label_cost(self):
        target_label = self.target_batchs.swapaxes(0,1)
        score_list = []
        for i in range(self.graph_number):
            self.resource_cluster.reset()
            plan = target_label[i]
            cost = self.resource_cluster.get_running_time_through_schedule_plan(plan, self.task_graphs[i])
            score = -(cost)
            score_list.append(score)

        return np.mean(score_list)

    def get_heft_cost_by_graph_batch(self, graph_batch):
        cost_batch = []
        for task_graph in graph_batch:
            self.resource_cluster.reset()
            heft_plan = heft_policy(self.resource_cluster, task_graph)
            self.resource_cluster.reset()

            cost = self.resource_cluster.get_running_time_through_schedule_plan(heft_plan, task_graph)

            cost_batch.append(cost)

        cost_batch = np.array(cost_batch)
        return cost_batch

    def get_heft_cost(self):
        score_list = []
        for task_graph_batches in self.task_graphs:
            for task_graph in task_graph_batches:
                self.resource_cluster.reset()
                heft_plan = heft_policy(cluster=self.resource_cluster, task_graph=task_graph)
                self.resource_cluster.reset()

                cost = self.resource_cluster.get_running_time_through_schedule_plan(heft_plan, task_graph)
                score = -(cost)
                score_list.append(score)

        return np.mean(score_list)

    def get_peft_cost(self):
        score_list = []
        for task_graph_batches in self.task_graphs:
            for task_graph in task_graph_batches:
                self.resource_cluster.reset()
                peft_plan = peft_policy(cluster=self.resource_cluster, task_graph=task_graph)

                self.resource_cluster.reset()

                cost = self.resource_cluster.get_running_time_through_schedule_plan(peft_plan, task_graph)
                score = -(cost)
                score_list.append(score)

        return np.mean(score_list)

    def get_greedy_cost(self):
        score_list = []
        for task_graph_batches in self.task_graphs:
            for task_graph in task_graph_batches:
                self.resource_cluster.reset()
                plan, _ = greedy_policy(cluster=self.resource_cluster, task_graph=task_graph)

                greedy_plan = [(task_index, resource) for task_index, resource in enumerate(plan)]
                self.resource_cluster.reset()

                cost = self.resource_cluster.get_running_time_through_schedule_plan(greedy_plan, task_graph)
                score = -(cost)
                score_list.append(score)

        return np.mean(score_list)

    def get_min_min_cost(self):
        score_list = []
        for task_graph_batches in self.task_graphs:
            for task_graph in task_graph_batches:
                self.resource_cluster.reset()
                plan= min_min_policy(cluster=self.resource_cluster, task_graph=task_graph)
                self.resource_cluster.reset()

                min_min_plan = [(task_index, resource) for task_index, resource in enumerate(plan)]
                cost = self.resource_cluster.get_running_time_through_schedule_plan(min_min_plan, task_graph)
                score = -(cost)
                score_list.append(score)

        return np.mean(score_list)

    def get_max_min_cost(self):
        score_list = []
        for task_graph_batches in self.task_graphs:
            for task_graph in task_graph_batches:
                self.resource_cluster.reset()
                plan = max_min_policy(cluster=self.resource_cluster, task_graph=task_graph)
                max_min_plan = [(task_index, resource) for task_index, resource in enumerate(plan)]
                self.resource_cluster.reset()

                cost = self.resource_cluster.get_running_time_through_schedule_plan(max_min_plan, task_graph)
                score = -(cost)
                score_list.append(score)

        return np.mean(score_list)



if __name__ == "__main__":
    workflow_env = WorkflowEnvironment(100, 100, '../data/random20/random.20.', is_xml=False)
    print("random 20")
    print("greedy cost is {}".format(workflow_env.get_greedy_cost()))
    print("heft cost is {}".format(workflow_env.get_heft_cost()))
    print("peft cost is {}".format(workflow_env.get_peft_cost()))
    print("min min cost is {}".format(workflow_env.get_min_min_cost()))
    print("max min cost is {}".format(workflow_env.get_max_min_cost()))
    '''
    workflow_env = WorkflowEnvironment(100, 100, '../data/CCR_DATA/CCR1.1/random.30.', is_xml=False)
    print("random 30")
    print("greedy cost is {}".format(workflow_env.get_greedy_cost()))
    print("heft cost is {}".format(workflow_env.get_heft_cost()))
    print("peft cost is {}".format(workflow_env.get_peft_cost()))
    print("min min cost is {}".format(workflow_env.get_min_min_cost()))
    print("max min cost is {}".format(workflow_env.get_max_min_cost()))

    
    workflow_env = WorkflowEnvironment(100, 100, '../data/random30/random.30.', is_xml=False)
    print("random 30")
    print("greedy cost is {}".format(workflow_env.get_greedy_cost()))
    print("heft cost is {}".format(workflow_env.get_heft_cost()))
    print("peft cost is {}".format(workflow_env.get_peft_cost()))
    print("min min cost is {}".format(workflow_env.get_min_min_cost()))
    print("max min cost is {}".format(workflow_env.get_max_min_cost()))
    

    workflow_env = WorkflowEnvironment(100, 100, '../data/CCR_DATA_20/CCR0.3/random.20.', is_xml=False)
    print("random 20 CCR 0.3")
    print("greedy cost is {}".format(workflow_env.get_greedy_cost()))
    print("heft cost is {}".format(workflow_env.get_heft_cost()))
    print("peft cost is {}".format(workflow_env.get_peft_cost()))
    print("min min cost is {}".format(workflow_env.get_min_min_cost()))
    print("max min cost is {}".format(workflow_env.get_max_min_cost()))

    workflow_env = WorkflowEnvironment(100, 100, '../data/CCR_DATA_20/CCR0.4/random.20.', is_xml=False)
    print("random 20 CCR 0.4")
    print("greedy cost is {}".format(workflow_env.get_greedy_cost()))
    print("heft cost is {}".format(workflow_env.get_heft_cost()))
    print("peft cost is {}".format(workflow_env.get_peft_cost()))
    print("min min cost is {}".format(workflow_env.get_min_min_cost()))
    print("max min cost is {}".format(workflow_env.get_max_min_cost()))

    workflow_env = WorkflowEnvironment(100, 100, '../data/CCR_DATA_20/CCR0.5/random.20.', is_xml=False)
    print("random 20 CCR 0.5")
    print("greedy cost is {}".format(workflow_env.get_greedy_cost()))
    print("heft cost is {}".format(workflow_env.get_heft_cost()))
    print("peft cost is {}".format(workflow_env.get_peft_cost()))
    print("min min cost is {}".format(workflow_env.get_min_min_cost()))
    print("max min cost is {}".format(workflow_env.get_max_min_cost()))

    workflow_env = WorkflowEnvironment(100, 100, '../data/CCR_DATA_20/CCR0.6/random.20.', is_xml=False)
    print("random 20 CCR 0.6")
    print("greedy cost is {}".format(workflow_env.get_greedy_cost()))
    print("heft cost is {}".format(workflow_env.get_heft_cost()))
    print("peft cost is {}".format(workflow_env.get_peft_cost()))
    print("min min cost is {}".format(workflow_env.get_min_min_cost()))
    print("max min cost is {}".format(workflow_env.get_max_min_cost()))

    workflow_env = WorkflowEnvironment(100, 100, '../data/CCR_DATA_20/CCR0.7/random.20.', is_xml=False)
    print("random 20 CCR 0.7")
    print("greedy cost is {}".format(workflow_env.get_greedy_cost()))
    print("heft cost is {}".format(workflow_env.get_heft_cost()))
    print("peft cost is {}".format(workflow_env.get_peft_cost()))
    print("min min cost is {}".format(workflow_env.get_min_min_cost()))
    print("max min cost is {}".format(workflow_env.get_max_min_cost()))

    workflow_env = WorkflowEnvironment(100, 100, '../data/CCR_DATA_20/CCR0.8/random.20.', is_xml=False)
    print("random 20 CCR 0.8")
    print("greedy cost is {}".format(workflow_env.get_greedy_cost()))
    print("heft cost is {}".format(workflow_env.get_heft_cost()))
    print("peft cost is {}".format(workflow_env.get_peft_cost()))
    print("min min cost is {}".format(workflow_env.get_min_min_cost()))
    print("max min cost is {}".format(workflow_env.get_max_min_cost()))

    workflow_env = WorkflowEnvironment(100, 100, '../data/CCR_DATA_20/CCR0.9/random.20.', is_xml=False)
    print("random 20 CCR 0.9")
    print("greedy cost is {}".format(workflow_env.get_greedy_cost()))
    print("heft cost is {}".format(workflow_env.get_heft_cost()))
    print("peft cost is {}".format(workflow_env.get_peft_cost()))
    print("min min cost is {}".format(workflow_env.get_min_min_cost()))
    print("max min cost is {}".format(workflow_env.get_max_min_cost()))

    workflow_env = WorkflowEnvironment(100, 100, '../data/CCR_DATA_20/CCR1.0/random.20.', is_xml=False)
    print("random 20 CCR 1.0")
    print("greedy cost is {}".format(workflow_env.get_greedy_cost()))
    print("heft cost is {}".format(workflow_env.get_heft_cost()))
    print("peft cost is {}".format(workflow_env.get_peft_cost()))
    print("min min cost is {}".format(workflow_env.get_min_min_cost()))
    print("max min cost is {}".format(workflow_env.get_max_min_cost()))

    workflow_env = WorkflowEnvironment(100, 100, '../data/CCR_DATA_20/CCR1.1/random.20.', is_xml=False)
    print("random 20 CCR 1.1")
    print("greedy cost is {}".format(workflow_env.get_greedy_cost()))
    print("heft cost is {}".format(workflow_env.get_heft_cost()))
    print("peft cost is {}".format(workflow_env.get_peft_cost()))
    print("min min cost is {}".format(workflow_env.get_min_min_cost()))
    print("max min cost is {}".format(workflow_env.get_max_min_cost()))

    workflow_env = WorkflowEnvironment(100, 100, '../data/CCR_DATA_20/CCR1.2/random.20.', is_xml=False)
    print("random 20 CCR 1.2")
    print("greedy cost is {}".format(workflow_env.get_greedy_cost()))
    print("heft cost is {}".format(workflow_env.get_heft_cost()))
    print("peft cost is {}".format(workflow_env.get_peft_cost()))
    print("min min cost is {}".format(workflow_env.get_min_min_cost()))
    print("max min cost is {}".format(workflow_env.get_max_min_cost()))

    workflow_env = WorkflowEnvironment(100, 100, '../data/CCR_DATA_20/CCR1.3/random.20.', is_xml=False)
    print("random 20 CCR 1.3")
    print("greedy cost is {}".format(workflow_env.get_greedy_cost()))
    print("heft cost is {}".format(workflow_env.get_heft_cost()))
    print("peft cost is {}".format(workflow_env.get_peft_cost()))
    print("min min cost is {}".format(workflow_env.get_min_min_cost()))
    print("max min cost is {}".format(workflow_env.get_max_min_cost()))

    workflow_env = WorkflowEnvironment(100, 100, '../data/CCR_DATA_20/CCR1.9/random.20.', is_xml=False)
    print("random 20 CCR 1.9")
    print("greedy cost is {}".format(workflow_env.get_greedy_cost()))
    print("heft cost is {}".format(workflow_env.get_heft_cost()))
    print("peft cost is {}".format(workflow_env.get_peft_cost()))
    print("min min cost is {}".format(workflow_env.get_min_min_cost()))
    print("max min cost is {}".format(workflow_env.get_max_min_cost()))

    workflow_env = WorkflowEnvironment(100, 100, '../data/CCR_DATA_20/CCR2.4/random.20.', is_xml=False)
    print("random 20 CCR 2.4")
    print("greedy cost is {}".format(workflow_env.get_greedy_cost()))
    print("heft cost is {}".format(workflow_env.get_heft_cost()))
    print("peft cost is {}".format(workflow_env.get_peft_cost()))
    print("min min cost is {}".format(workflow_env.get_min_min_cost()))
    print("max min cost is {}".format(workflow_env.get_max_min_cost()))

    workflow_env = WorkflowEnvironment(100, 100, '../data/CCR_DATA_20/CCR3.0/random.20.', is_xml=False)
    print("random 20 CCR 3.0")
    print("greedy cost is {}".format(workflow_env.get_greedy_cost()))
    print("heft cost is {}".format(workflow_env.get_heft_cost()))
    print("peft cost is {}".format(workflow_env.get_peft_cost()))
    print("min min cost is {}".format(workflow_env.get_min_min_cost()))
    print("max min cost is {}".format(workflow_env.get_max_min_cost()))

   
    
    workflow_env = WorkflowEnvironment(100, 100, "../data/random100/random.100.", is_xml=False)
    print("random 100")
    print("greedy cost is {}".format(workflow_env.get_greedy_cost()))
    print("heft cost is {}".format(workflow_env.get_heft_cost()))
    print("peft cost is {}".format(workflow_env.get_peft_cost()))
    print("min min cost is {}".format(workflow_env.get_min_min_cost()))
    print("max min cost is {}".format(workflow_env.get_max_min_cost()))

    workflow_env = WorkflowEnvironment(100, 100, "../data/random90/random.90.", is_xml=False)
    print("random 90")
    print("greedy cost is {}".format(workflow_env.get_greedy_cost()))
    print("heft cost is {}".format(workflow_env.get_heft_cost()))
    print("peft cost is {}".format(workflow_env.get_peft_cost()))
    print("min min cost is {}".format(workflow_env.get_min_min_cost()))
    print("max min cost is {}".format(workflow_env.get_max_min_cost()))

    workflow_env = WorkflowEnvironment(100, 100, "../data/random80/random.80.", is_xml=False)
    print("random 80")
    print("greedy cost is {}".format(workflow_env.get_greedy_cost()))
    print("heft cost is {}".format(workflow_env.get_heft_cost()))
    print("peft cost is {}".format(workflow_env.get_peft_cost()))
    print("min min cost is {}".format(workflow_env.get_min_min_cost()))
    print("max min cost is {}".format(workflow_env.get_max_min_cost()))
    
    

    workflow_env = WorkflowEnvironment(100, 100, "../data/random70/random.70.", is_xml=False)
    print("random 70")
    print("greedy cost is {}".format(workflow_env.get_greedy_cost()))
    print("heft cost is {}".format(workflow_env.get_heft_cost()))
    print("peft cost is {}".format(workflow_env.get_peft_cost()))
    print("min min cost is {}".format(workflow_env.get_min_min_cost()))
    print("max min cost is {}".format(workflow_env.get_max_min_cost()))

    workflow_env = WorkflowEnvironment(100, 100, "../data/random60/random.60.", is_xml=False)
    print("random 60")
    print("greedy cost is {}".format(workflow_env.get_greedy_cost()))
    print("heft cost is {}".format(workflow_env.get_heft_cost()))
    print("peft cost is {}".format(workflow_env.get_peft_cost()))
    print("min min cost is {}".format(workflow_env.get_min_min_cost()))
    print("max min cost is {}".format(workflow_env.get_max_min_cost()))

    workflow_env = WorkflowEnvironment(100, 100, "../data/random50/random.50.", is_xml=False)
    print("random 50")
    print("greedy cost is {}".format(workflow_env.get_greedy_cost()))
    print("heft cost is {}".format(workflow_env.get_heft_cost()))
    print("peft cost is {}".format(workflow_env.get_peft_cost()))
    print("min min cost is {}".format(workflow_env.get_min_min_cost()))
    print("max min cost is {}".format(workflow_env.get_max_min_cost()))

    workflow_env = WorkflowEnvironment(100, 100, "../data/random40/random.40.", is_xml=False)
    print("random 40")
    print("greedy cost is {}".format(workflow_env.get_greedy_cost()))
    print("heft cost is {}".format(workflow_env.get_heft_cost()))
    print("peft cost is {}".format(workflow_env.get_peft_cost()))
    print("min min cost is {}".format(workflow_env.get_min_min_cost()))
    print("max min cost is {}".format(workflow_env.get_max_min_cost()))

    workflow_env = WorkflowEnvironment(100, 100, "../data/random30/random.30.", is_xml=False)
    print("random 30")
    print("greedy cost is {}".format(workflow_env.get_greedy_cost()))
    print("heft cost is {}".format(workflow_env.get_heft_cost()))
    print("peft cost is {}".format(workflow_env.get_peft_cost()))
    print("min min cost is {}".format(workflow_env.get_min_min_cost()))
    print("max min cost is {}".format(workflow_env.get_max_min_cost()))

    workflow_env = WorkflowEnvironment(100, 100, "../data/random20/random.20.", is_xml=False)
    print("random 20")
    print("greedy cost is {}".format(workflow_env.get_greedy_cost()))
    print("heft cost is {}".format(workflow_env.get_heft_cost()))
    print("peft cost is {}".format(workflow_env.get_peft_cost()))
    print("min min cost is {}".format(workflow_env.get_min_min_cost()))
    print("max min cost is {}".format(workflow_env.get_max_min_cost()))
    '''