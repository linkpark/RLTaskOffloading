import numpy as np
import os

from rltaskoffloading.environment.offloading_task_graph import OffloadingTaskGraph
"""
System bandwidth B 20MHz
UE Bandwidth W 1 MHz
PTx = 1.258W
PRx = 1.181W
Noise = -174dbm/Hz
"""

"""
    Also we need to prioritize the tasks.
"""


class Resources(object):
    def __init__(self, mec_process_capable,
                  mobile_process_capable, bandwith_up = 7.0, bandwith_dl = 7.0):
        self.mec_process_capble = mec_process_capable
        self.mobile_process_capable = mobile_process_capable
        self.mobile_process_avaliable_time = 0.0
        self.mec_process_avaliable_time = 0.0

        self.bandwith_up = bandwith_up
        self.bandwith_dl = bandwith_dl

        self.Pap = 1.258
        self.Ptx = 1.258

        self.omega0 = 1.0
        self.optimal_qoe = 0.0

    def up_transmission_cost(self, data, distance=0.0):
        #PLDbm = 128.1 + 37.6 * np.log10(distance / 1000.0)
        #PLw = 10.0 ** ((PLDbm) / 10.0)

        #rate = self.bandwith_up * np.log2( 1 + self.Pap * PLw / (self.bandwith_up * self.omega0))

        # rate = 7.0 * (1024.0 * 1024.0 / 8.0)
        rate = self.bandwith_up * (1024.0 * 1024.0 / 8.0)

        transmission_time = data / rate

        return transmission_time

    def reset(self):
        self.mec_process_avaliable_time = 0.0
        self.mobile_process_avaliable_time = 0.0

    def dl_transmission_cost(self, data, distance=0.0):
        #PLDbm = 128.1 + 37.6 * np.log10( distance / 1000.0)
        #PLw = 10.0 ** ((PLDbm) / 10.0)

        #rate = self.bandwith_dl * np.log2(1 + self.Pap * PLw / (self.bandwith_dl * self.omega0))

        rate = self.bandwith_dl * (1024.0 * 1024.0 / 8.0)
        transmission_time = data / rate

        return transmission_time

    def locally_execution_cost(self, data):
        return self._computation_cost(data, self.mobile_process_capable)

    def mec_execution_cost(self, data):
        return self._computation_cost(data, self.mec_process_capble)

    def _computation_cost(self, data, processing_power):
        computation_time = data / processing_power

        return computation_time


class OffloadingEnvironment(object):
    def __init__(self, resource_cluster, batch_size, graph_number, graph_file_paths, time_major, lambda_t=1.0, lambda_e=0.0, encode_dependencies=True):
        self.resource_cluster = resource_cluster
        self.task_graphs = []
        self.encoder_batchs = []
        self.encoder_lengths = []
        self.decoder_full_lengths = []
        self.max_running_time_batchs = []
        self.min_running_time_batchs = []
        self.optimal_solution = -1
        self.optimal_energy = -1
        self.optimal_qoe = -1

        self.heft_avg_run_time = -1
        self.heft_avg_energy = -1
        self.heft_avg_qoe = -1

        self.optimal_qoe_energy = -1
        self.optimal_qoe_latency = -1
        self.encode_dependencies = encode_dependencies


        for graph_file_path in graph_file_paths:
            encoder_batchs, encoder_lengths, task_graph_batchs, decoder_full_lengths, max_running_time_batchs, min_running_time_batchs = \
                self.generate_point_batch_for_random_graphs(batch_size, graph_number, graph_file_path, time_major)

            self.encoder_batchs += encoder_batchs
            self.encoder_lengths += encoder_lengths
            self.task_graphs += task_graph_batchs
            self.decoder_full_lengths += decoder_full_lengths
            self.max_running_time_batchs += max_running_time_batchs
            self.min_running_time_batchs += min_running_time_batchs

        self.input_dim = np.array(encoder_batchs[0]).shape[-1]
        self.start_symbol = 0

        # these 3 parameters are used to calculate the processing energy consumption
        self.rho = 1.25 * 10 ** -26
        self.f_l = 0.8 * 10 ** 9
        self.zeta = 3

        # these 2 parameters are used to calculate the transmission energy consumption
        self.ptx = 1.258
        self.prx = 1.181

        # control the trade off between latency and energy consumption
        self.lambda_t = lambda_t
        self.lambda_e = lambda_e

        self.all_locally_execute, self.all_locally_energy = self.get_all_locally_execute_time()
        self.all_mec_execute, self.all_mec_energy = self.get_all_mec_execute_time()

    def calculate_optimal_solution(self):
        def exhaustion_plans(n):
            plan_batch = []

            for i in range(2**n):
                plan_str = bin(i)
                plan = []

                for x in plan_str[2:]:
                    plan.append(int(x))

                while len(plan) < n:
                    plan.insert(0, 0)
                plan_batch.append(plan)
            return plan_batch

        n = self.task_graphs[0][0].task_number
        plan_batch = exhaustion_plans(n)

        print("exhausted plan size: ", len(plan_batch))

        task_graph_optimal_costs = []
        task_graph_optimal_energys = []
        optimal_plan = []
        optimal_makespan_plan_energy_cost = []
        task_graph_optimal_makespan_energy= []
        optimal_plan_e = []


        for task_graph_batch in self.task_graphs:
            task_graph_batch_cost = []
            task_graph_batch_energy = []
            for task_graph in task_graph_batch:
                plans_costs = []
                plans_energy = []

                for plan in plan_batch:
                    plan_sequence = []
                    for action, task_id in zip(plan, task_graph.prioritize_sequence):
                        plan_sequence.append((task_id, action))

                    cost, energy, task_finish_time, energy_cost = self.get_scheduling_cost_step_by_step(plan_sequence, task_graph)
                    plans_costs.append(task_finish_time)
                    plans_energy.append(energy_cost)

                graph_min_cost = min(plans_costs)
                graph_min_energy = min(plans_energy)

                optimal_plan.append(plan_batch[np.argmin(plans_costs)])
                optimal_plan_e.append(plan_batch[np.argmin(plans_energy)])
                optimal_makespan_plan_energy_cost.append(plans_energy[np.argmin(plans_costs)])

                task_graph_batch_cost.append(graph_min_cost)
                task_graph_batch_energy.append(graph_min_energy)

            print("task_graph_batch cost shape is {}".format(np.array(task_graph_batch_cost).shape))
            avg_minimal_cost = np.mean(task_graph_batch_cost)
            avg_energy = np.mean(optimal_makespan_plan_energy_cost)
            avg_minimal_energy = np.mean(task_graph_batch_energy)

            task_graph_optimal_costs.append(avg_minimal_cost)
            task_graph_optimal_makespan_energy.append(avg_energy)
            task_graph_optimal_energys.append(avg_minimal_energy)

        self.optimal_solution = task_graph_optimal_costs
        self.optimal_energy =task_graph_optimal_energys
        self.optimal_makespan_energy = task_graph_optimal_makespan_energy
        print("energy consumption for optimal plan:", task_graph_optimal_makespan_energy)
        return task_graph_optimal_costs

    def calculate_optimal_qoe(self):
        def exhaustion_plans(n):
            plan_batch = []

            for i in range(2**n):
                plan_str = bin(i)
                plan = []

                for x in plan_str[2:]:
                    plan.append(int(x))

                while len(plan) < n:
                    plan.insert(0, 0)
                plan_batch.append(plan)
            return plan_batch

        n = self.task_graphs[0][0].task_number
        plan_batch = exhaustion_plans(n)

        print("exhausted plan size: ", len(plan_batch))

        task_graph_optimal_costs = []
        task_graph_optimal_energys = []
        optimal_plan = []
        optimal_makespan_plan_energy_cost = []
        task_graph_optimal_makespan_energy= []
        task_graph_optimal_qoe = []
        task_graph_optimal_qoe_energy = []
        task_graph_optimal_qoe_latency = []

        optimal_plan_e = []
        optimal_plan_qoe = []

        optimal_qoe_energy = []
        optimal_qoe_latency = []


        for task_graph_batch in self.task_graphs:
            task_graph_batch_cost = []
            task_graph_batch_energy = []
            task_graph_batch_qoe = []

            for task_graph in task_graph_batch:
                plans_costs = []
                plans_energy = []
                plans_qoe = []

                all_local_time, all_local_energy = self.get_all_local_cost_for_one_graph(task_graph)

                for plan in plan_batch:
                    plan_sequence = []
                    for action, task_id in zip(plan, task_graph.prioritize_sequence):
                        plan_sequence.append((task_id, action))

                    cost, energy, task_finish_time, energy_cost = self.get_scheduling_cost_step_by_step(plan_sequence, task_graph)
                    plans_costs.append(task_finish_time)
                    plans_energy.append(energy_cost)

                    qoe = (self.lambda_t * (task_finish_time - all_local_time) / all_local_time) + (self.lambda_e * (energy_cost - all_local_energy) / all_local_energy)
                    plans_qoe.append(qoe)

                graph_min_cost = min(plans_costs)
                graph_min_energy = min(plans_energy)
                graph_min_qoe = min(plans_qoe)

                optimal_plan.append(plan_batch[np.argmin(plans_costs)])
                optimal_plan_e.append(plan_batch[np.argmin(plans_energy)])
                optimal_makespan_plan_energy_cost.append(plans_energy[np.argmin(plans_costs)])

                optimal_plan_qoe.append(plan_batch[np.argmin(plans_qoe)])
                optimal_qoe_energy.append(plans_energy[np.argmin(plans_qoe)])
                optimal_qoe_latency.append(plans_costs[np.argmin(plans_qoe)])

                task_graph_batch_cost.append(graph_min_cost)
                task_graph_batch_energy.append(graph_min_energy)
                task_graph_batch_qoe.append(graph_min_qoe)

            print("task_graph_batch cost shape is {}".format(np.array(task_graph_batch_cost).shape))
            avg_minimal_cost = np.mean(task_graph_batch_cost)
            avg_energy = np.mean(optimal_makespan_plan_energy_cost)
            avg_qoe_energy = np.mean(optimal_qoe_energy)
            avg_qoe_latency = np.mean(optimal_qoe_latency)

            avg_minimal_energy = np.mean(task_graph_batch_energy)

            task_graph_optimal_costs.append(avg_minimal_cost)
            task_graph_optimal_makespan_energy.append(avg_energy)
            task_graph_optimal_energys.append(avg_minimal_energy)
            task_graph_optimal_qoe.append(task_graph_batch_qoe)
            task_graph_optimal_qoe_energy.append(avg_qoe_energy)
            task_graph_optimal_qoe_latency.append(avg_qoe_latency)

        self.optimal_solution = task_graph_optimal_costs
        self.optimal_energy =task_graph_optimal_energys
        self.optimal_makespan_energy = task_graph_optimal_makespan_energy
        self.optimal_qoe= task_graph_optimal_qoe
        self.optimal_qoe_energy = task_graph_optimal_qoe_energy
        self.optimal_qoe_latency = task_graph_optimal_qoe_latency
        print("energy consumption for optimal plan:", task_graph_optimal_makespan_energy)
        return task_graph_optimal_costs

    def random_solution(self):
        running_cost = []
        energy_cost = []
        for task_graph_batch, encode_batch in zip(self.task_graphs, self.encoder_batchs):
            batch_size = encode_batch.shape[0]
            sequence_length = encode_batch.shape[1]
            plan = np.random.randint(2, size=(batch_size, sequence_length))
            running_cost_batch, energy_consumption_batch = self.get_running_cost(plan, task_graph_batch)
            running_cost.append(running_cost_batch)
            energy_cost.append(energy_consumption_batch)

        return running_cost, energy_cost

    def round_robin_solution(self):
        running_cost = []
        energy_cost = []

        for task_graph_batch, encode_batch in zip(self.task_graphs, self.encoder_batchs):
            batch_size = encode_batch.shape[0]
            sequence_length = encode_batch.shape[1]
            plan = np.zeros(shape=(batch_size, sequence_length), dtype=np.int32)
            plan[:, ::2] = 1
            running_cost_batch, energy_consumption_batch = self.get_running_cost(plan, task_graph_batch)
            running_cost.append(running_cost_batch)
            energy_cost.append(energy_consumption_batch)

        return running_cost, energy_cost

    def all_offloading_solution(self):
        running_cost = []
        energy_cost = []

        for task_graph_batch, encode_batch in zip(self.task_graphs, self.encoder_batchs):
            batch_size = encode_batch.shape[0]
            sequence_length = encode_batch.shape[1]
            plan = np.ones(shape=(batch_size, sequence_length), dtype=np.int32)
            running_cost_batch, energy_consumption_batch = self.get_running_cost(plan, task_graph_batch)
            running_cost.append(running_cost_batch)
            energy_cost.append(energy_consumption_batch)

        return running_cost, energy_cost

    def all_local_solution(self):
        running_cost = []
        energy_cost = []

        for task_graph_batch, encode_batch in zip(self.task_graphs, self.encoder_batchs):
            batch_size = encode_batch.shape[0]
            sequence_length = encode_batch.shape[1]
            plan = np.zeros(shape=(batch_size, sequence_length), dtype=np.int32)
            running_cost_batch, energy_consumption_batch = self.get_running_cost(plan, task_graph_batch)
            running_cost.append(running_cost_batch)
            energy_cost.append(energy_consumption_batch)

        return running_cost, energy_cost

    def greedy_solution(self, heft=True):
        result_plan = []
        finish_time_batchs = []
        for task_graph_batch in self.task_graphs:
            plan_batchs = []
            finish_time_plan = []
            for task_graph in task_graph_batch:
                cloud_avaliable_time = 0.0
                ws_avaliable_time = 0.0
                local_avaliable_time = 0.0

                # finish time on cloud for each task
                FT_cloud = [0] * task_graph.task_number
                # finish time on sending channel for each task
                FT_ws = [0] * task_graph.task_number
                # finish time locally for each task
                FT_locally = [0] * task_graph.task_number
                # finish time recieving channel for each task
                FT_wr = [0] * task_graph.task_number
                plan = []

                if heft==True:
                    task_index = task_graph.prioritize_sequence
                else:
                    task_index = np.arange(task_graph.task_number)

                for i in task_index:
                    task = task_graph.task_list[i]

                    # calculate the local finish time
                    if len(task_graph.pre_task_sets[i]) != 0:
                        start_time = max(local_avaliable_time,
                                         max([max(FT_locally[j], FT_wr[j]) for j in task_graph.pre_task_sets[i]]))
                    else:
                        start_time = local_avaliable_time

                    local_running_time = self.resource_cluster.locally_execution_cost(task.processing_data_size)
                    FT_locally[i] = start_time + local_running_time

                    # calculate the remote finish time
                    if len(task_graph.pre_task_sets[i]) != 0:
                        ws_start_time = max(ws_avaliable_time,
                                            max([max(FT_locally[j], FT_ws[j]) for j in task_graph.pre_task_sets[i]]))
                        FT_ws[i] = ws_start_time + self.resource_cluster.up_transmission_cost(task.processing_data_size)
                        cloud_start_time = max(cloud_avaliable_time,
                                               max([max(FT_ws[i], FT_cloud[j]) for j in task_graph.pre_task_sets[i]]))
                        cloud_finish_time = cloud_start_time + self.resource_cluster.mec_execution_cost(
                            task.processing_data_size)
                        FT_cloud[i] = cloud_finish_time
                        # print("task {}, Cloud finish time {}".format(i, FT_cloud[i]))
                        wr_start_time = FT_cloud[i]
                        wr_finish_time = wr_start_time + self.resource_cluster.dl_transmission_cost(task.transmission_data_size)
                        FT_wr[i] = wr_finish_time
                    else:
                        ws_start_time = ws_avaliable_time
                        ws_finish_time = ws_start_time + self.resource_cluster.up_transmission_cost(task.processing_data_size)
                        FT_ws[i] = ws_finish_time

                        cloud_start_time = max(cloud_avaliable_time, FT_ws[i])
                        FT_cloud[i] = cloud_start_time + self.resource_cluster.mec_execution_cost(
                            task.processing_data_size)
                        FT_wr[i] = FT_cloud[i] + self.resource_cluster.dl_transmission_cost(task.transmission_data_size)

                    if FT_locally[i] < FT_wr[i]:
                        action = 0
                        local_avaliable_time = FT_locally[i]
                        FT_wr[i] = 0.0
                        FT_cloud[i] = 0.0
                        FT_ws[i] = 0.0
                    else:
                        action = 1
                        FT_locally[i] = 0.0
                        cloud_avaliable_time = FT_cloud[i]
                        ws_avaliable_time = FT_ws[i]
                    plan.append((i, action))

                finish_time = max( max(FT_wr), max(FT_locally) )
                plan_batchs.append(plan)
                finish_time_plan.append(finish_time)

            finish_time_batchs.append(finish_time_plan)
            result_plan.append(plan_batchs)

        return result_plan, finish_time_batchs

    def calculate_max_min_runningcost(self, max_data_size, min_data_size):
        max_time = max( [self.resource_cluster.up_transmission_cost(max_data_size),
                         self.resource_cluster.dl_transmission_cost(max_data_size),
                         self.resource_cluster.locally_execution_cost(max_data_size)] )

        min_time = self.resource_cluster.mec_execution_cost(min_data_size)

        return max_time, min_time

    def get_all_locally_execute_time(self):
        running_cost = []
        energy_cost = []
        for task_graph_batch, encode_batch in zip(self.task_graphs, self.encoder_batchs):
            batch_size = encode_batch.shape[0]
            sequence_length = encode_batch.shape[1]

            scheduling_action = np.zeros(shape=(batch_size, sequence_length), dtype=np.int32)
            running_cost_batch, energy_consumption_batch = self.get_running_cost(scheduling_action, task_graph_batch)
            running_cost.append(np.mean(running_cost_batch))
            energy_cost.append(np.mean(energy_consumption_batch))

        return running_cost, energy_cost

    def get_all_mec_execute_time(self):
        running_cost = []
        energy_cost = []

        for task_graph_batch, encode_batch in zip(self.task_graphs, self.encoder_batchs):
            batch_size = encode_batch.shape[0]
            sequence_length = encode_batch.shape[1]

            scheduling_action = np.ones(shape=(batch_size, sequence_length), dtype=np.int32)
            running_cost_batch, energy_consumption_batch = self.get_running_cost(scheduling_action, task_graph_batch)

            running_cost.append(np.mean(running_cost_batch))
            energy_cost. append(np.mean(energy_consumption_batch))

        return running_cost, energy_cost

    def get_all_mec_execute_time_batch(self):
        running_cost = []
        energy_cost = []

        for task_graph_batch, encode_batch in zip(self.task_graphs, self.encoder_batchs):
            batch_size = encode_batch.shape[0]
            sequence_length = encode_batch.shape[1]

            scheduling_action = np.ones(shape=(batch_size, sequence_length), dtype=np.int32)
            running_cost_batch, energy_consumption_batch = self.get_running_cost(scheduling_action, task_graph_batch)

            running_cost.append(running_cost_batch)
            energy_cost.append(energy_consumption_batch)

        return running_cost, energy_cost

    def get_all_locally_execute_time_batch(self):
        running_cost = []
        energy_cost = []
        for task_graph_batch, encode_batch in zip(self.task_graphs, self.encoder_batchs):
            batch_size = encode_batch.shape[0]
            sequence_length = encode_batch.shape[1]

            scheduling_action = np.zeros(shape=(batch_size, sequence_length), dtype=np.int32)
            running_cost_batch, energy_consumption_batch = self.get_running_cost(scheduling_action, task_graph_batch)
            running_cost.append(running_cost_batch)
            energy_cost.append(energy_consumption_batch)

        return running_cost, energy_cost

    def get_all_local_cost_for_one_graph(self, task_graph):
        scheduling_action = [0] * task_graph.task_number
        plan_sequence = []

        for action, task_id in zip(scheduling_action,
                                   task_graph.prioritize_sequence):
            plan_sequence.append((task_id, action))

        _, _, task_finish_time, total_energy = self.get_scheduling_cost_step_by_step(plan_sequence, task_graph)
        return task_finish_time, total_energy

    def generate_point_batch_for_random_graphs(self, batch_size, graph_number, graph_file_path, time_major):
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
            task_graph = OffloadingTaskGraph(graph_file_path + str(i) + '.gv', is_matrix=False)
            task_graph_list.append(task_graph)

            max_time, min_time = self.calculate_max_min_runningcost(task_graph.max_data_size,
                                                                    task_graph.min_data_size)
            max_running_time_vector.append(max_time)
            min_running_time_vector.append(min_time)

            # the scheduling sequence will also store in self.'prioritize_sequence'
            scheduling_sequence = task_graph.prioritize_tasks(self.resource_cluster)

            task_encode = np.array(task_graph.encode_point_sequence_with_ranking_and_cost(scheduling_sequence,
                                                                                          self.resource_cluster,
                                                                                          encode_dependencies=self.encode_dependencies))
            encoder_list.append(task_encode)

        for i in range(int(graph_number / batch_size)):
            start_batch_index = i * batch_size
            end_batch_index = (i + 1) * batch_size

            task_encode_batch = encoder_list[start_batch_index:end_batch_index]
            if time_major:
                task_encode_batch = np.array(task_encode_batch).swapaxes(0, 1)
                sequence_length = np.asarray([task_encode_batch.shape[0]] * task_encode_batch.shape[1])
            else:
                task_encode_batch = np.array(task_encode_batch)
                sequence_length = np.asarray([task_encode_batch.shape[1]] * task_encode_batch.shape[0])

            decoder_full_lengths.append(sequence_length)
            encoder_lengths.append(sequence_length)
            encoder_batchs.append(task_encode_batch)

            task_graph_batch = task_graph_list[start_batch_index:end_batch_index]
            task_graph_batchs.append(task_graph_batch)
            max_running_time_batchs.append(max_running_time_vector[start_batch_index:end_batch_index])
            min_running_time_batchs.append(min_running_time_vector[start_batch_index:end_batch_index])

        return encoder_batchs, encoder_lengths, task_graph_batchs, decoder_full_lengths, max_running_time_batchs, min_running_time_batchs

    def get_scheduling_cost_by_plan(self, plan, task_graph, heft=True):
        plan_sequence = []
        if heft == True:
            task_index = task_graph.prioritize_sequence
        else:
            task_index = np.arange(task_graph.task_number)

        for action, task_id in zip(plan, task_index):
            plan_sequence.append((task_id, action))

        _, _, task_finish_time, total_energy = self.get_scheduling_cost_step_by_step(plan_sequence, task_graph)

        return task_finish_time

    def get_scheduling_cost_step_by_step(self, plan, task_graph):
        cloud_avaliable_time = 0.0
        ws_avaliable_time =0.0
        local_avaliable_time = 0.0

        # running time on local processor
        T_l = [0] * task_graph.task_number
        # running time on sending channel
        T_ul = [0] * task_graph.task_number
        #running time on receiving channel
        T_dl = [0] * task_graph.task_number


        # finish time on cloud for each task
        FT_cloud = [0] * task_graph.task_number
        # finish time on sending channel for each task
        FT_ws = [0] * task_graph.task_number
        # finish time locally for each task
        FT_locally = [0] * task_graph.task_number
        # finish time recieving channel for each task
        FT_wr = [0] * task_graph.task_number
        current_FT = 0.0
        total_energy = 0.0
        return_latency = []
        return_energy = []

        for item in plan:
            i = item[0]
            task = task_graph.task_list[i]
            x = item[1]

            # locally scheduling
            if x == 0:
                if len(task_graph.pre_task_sets[i]) != 0:
                    start_time = max(local_avaliable_time,
                                     max([max(FT_locally[j], FT_wr[j]) for j in task_graph.pre_task_sets[i]]))
                else:
                    start_time = local_avaliable_time

                T_l[i] = self.resource_cluster.locally_execution_cost(task.processing_data_size)
                FT_locally[i] = start_time + T_l[i]
                local_avaliable_time = FT_locally[i]

                task_finish_time = FT_locally[i]

                # calculate the energy consumption
                energy_consumption = T_l[i] * self.rho * (self.f_l ** self.zeta)
            # mcc scheduling
            else:
                if len(task_graph.pre_task_sets[i]) != 0:
                    ws_start_time = max(ws_avaliable_time,
                                        max([max(FT_locally[j], FT_ws[j])  for j in task_graph.pre_task_sets[i]]))

                    T_ul[i] = self.resource_cluster.up_transmission_cost(task.processing_data_size)
                    ws_finish_time = ws_start_time + T_ul[i]
                    FT_ws[i] = ws_finish_time
                    ws_avaliable_time = ws_finish_time

                    cloud_start_time = max( cloud_avaliable_time,
                                            max([max(FT_ws[i], FT_cloud[j]) for j in task_graph.pre_task_sets[i]]))
                    cloud_finish_time = cloud_start_time + self.resource_cluster.mec_execution_cost(task.processing_data_size)
                    FT_cloud[i] = cloud_finish_time
                    # print("task {}, Cloud finish time {}".format(i, FT_cloud[i]))
                    cloud_avaliable_time = cloud_finish_time

                    wr_start_time = FT_cloud[i]
                    T_dl[i] = self.resource_cluster.dl_transmission_cost(task.transmission_data_size)
                    wr_finish_time = wr_start_time + T_dl[i]
                    FT_wr[i] = wr_finish_time

                    # calculate the energy consumption
                    energy_consumption = T_ul[i] * self.ptx + T_dl[i] * self.prx

                else:
                    ws_start_time = ws_avaliable_time
                    T_ul[i] = self.resource_cluster.up_transmission_cost(task.processing_data_size)
                    ws_finish_time = ws_start_time + T_ul[i]
                    FT_ws[i] = ws_finish_time

                    cloud_start_time = max(cloud_avaliable_time, FT_ws[i])
                    cloud_finish_time = cloud_start_time + self.resource_cluster.mec_execution_cost(task.processing_data_size)
                    FT_cloud[i] = cloud_finish_time
                    cloud_avaliable_time = cloud_finish_time

                    wr_start_time = FT_cloud[i]
                    T_dl[i] = self.resource_cluster.dl_transmission_cost(task.transmission_data_size)
                    wr_finish_time = wr_start_time + T_dl[i]
                    FT_wr[i] = wr_finish_time

                    # calculate the energy consumption
                    energy_consumption = T_ul[i] * self.ptx + T_dl[i] * self.prx

                task_finish_time = wr_finish_time

            # print("task  {} finish time is {}".format(i , task_finish_time))
            total_energy += energy_consumption
            delta_make_span = max(task_finish_time, current_FT) - current_FT
            delta_energy = energy_consumption

            current_FT = max(task_finish_time, current_FT)

            return_latency.append(delta_make_span)
            return_energy.append(delta_energy)

        return return_latency, return_energy, current_FT, total_energy

    def score_func(self, cost, max_time, min_time):
        return -(cost - min_time) / (max_time - min_time)

    def score_func_qoe(self, cost, all_local_cost, number_of_task):
        try:
            cost = np.array(cost)
            avg_all_local_cost = all_local_cost / float(number_of_task)
            score = -(cost - avg_all_local_cost) / all_local_cost
        except:
            print("exception all local cost: ", all_local_cost)
            print("exception cost: ", cost)
            raise ValueError("Un spported operation")

        return score

    def get_cost_batch_step_by_step(self, action_sequence_batch, task_graph_batch):
        target_batch = []
        for i in range(len(action_sequence_batch)):
            task_graph = task_graph_batch[i]
            self.resource_cluster.reset()
            plan = action_sequence_batch[i]
            latency, energy, task_finish_time, total_energy = self.get_scheduling_cost_step_by_step(plan, task_graph)
            score = latency
            target_batch.append(score)

        target_batch = np.array(target_batch)
        return target_batch

    def get_reward_batch_step_by_step(self, action_sequence_batch, task_graph_batch,
                                      max_running_time_batch, min_running_time_batch):
        target_batch = []
        for i in range(len(action_sequence_batch)):
            max_running_time = max_running_time_batch[i]
            min_running_time = min_running_time_batch[i]

            task_graph = task_graph_batch[i]
            self.resource_cluster.reset()
            plan = action_sequence_batch[i]
            cost, energy, task_finish_time, total_energy = self.get_scheduling_cost_step_by_step(plan, task_graph)
            # latency = self.score_func(cost, max_running_time, min_running_time)
            all_local_time, all_local_energy = self.get_all_local_cost_for_one_graph(task_graph)

            latency = self.score_func_qoe(cost, all_local_cost=all_local_time,
                                          number_of_task=task_graph.task_number)

            max_energy = max_running_time * max((self.rho * (self.f_l ** self.zeta)) , (self.ptx +self.prx) )
            min_energy = min_running_time * min((self.rho * (self.f_l ** self.zeta)) , (self.ptx +self.prx) )

            #print("max_energy", max_energy)
            #print("min_energy", min_energy)
            # energy = self.score_func(energy, max_energy, min_energy)
            energy = self.score_func_qoe(energy, all_local_cost=all_local_energy,
                                         number_of_task=task_graph.task_number)
            #print("energy score", energy)

            score = self.lambda_t * np.array(latency) + self.lambda_e * np.array(energy)
            #print("score is", score)
            target_batch.append(score)

        target_batch = np.array(target_batch)
        return target_batch

    def step(self, action_sequence_batch, task_graph_batch, max_running_time_batch, min_running_time_batch):
        plan_batch = []
        for action_sequence, task_graph in zip(action_sequence_batch, task_graph_batch):
            plan_sequence = []

            for action, task_id in zip(action_sequence, task_graph.prioritize_sequence):
                plan_sequence.append((task_id, action))

            plan_batch.append(plan_sequence)

        return self.get_reward_batch_step_by_step(plan_batch,
                                                  task_graph_batch,
                                                  max_running_time_batch,
                                                  min_running_time_batch)

    def get_running_cost(self, action_sequence_batch, task_graph_batch):
        cost_batch = []
        energy_batch = []
        for action_sequence, task_graph in zip(action_sequence_batch,
                                               task_graph_batch):
            plan_sequence = []

            for action, task_id in zip(action_sequence,
                                       task_graph.prioritize_sequence):
                plan_sequence.append((task_id, action))

            _, _, task_finish_time, total_energy = self.get_scheduling_cost_step_by_step(plan_sequence, task_graph)

            cost_batch.append(task_finish_time)
            energy_batch.append(total_energy)

        return cost_batch, energy_batch

    def get_running_cost_by_plan_batch(self, plan_batch, task_graph_batch):
        cost_batch = []
        energy_batch = []
        for plan, task_graph in zip(plan_batch, task_graph_batch):
            self.resource_cluster.reset()
            _, _, task_finish_time, total_energy = self.get_scheduling_cost_step_by_step(plan, task_graph)

            cost_batch.append(task_finish_time)
            energy_batch.append(total_energy)

        return cost_batch, energy_batch

    def calculate_qoe(self, latency_batch, energy_batch):
        all_local_time, all_local_energy = self.get_all_locally_execute_time_batch()
        all_local_time = np.squeeze(all_local_time)
        all_local_energy = np.squeeze(all_local_energy)
        latency_batch = np.squeeze(latency_batch)
        energy_batch = np.squeeze(energy_batch)
        qoe_batch = []

        for latency, energy, single_all_local_latency, single_all_local_energy in zip(latency_batch, energy_batch,
                                                                                      all_local_time, all_local_energy):
            qoe = self.lambda_t * ((latency - single_all_local_latency) / single_all_local_latency) + \
                  self.lambda_e * ((energy - single_all_local_energy) / single_all_local_energy)

            qoe = -qoe
            qoe_batch.append(qoe)

        return qoe_batch

    def calculate_heft_cost(self):
        plans, finish_time_batchs = self.greedy_solution(heft=True)
        heft_latency_batch, heft_energy_batch = self.get_running_cost_by_plan_batch(plans[0], self.task_graphs[0])
        latency_batch = np.array(heft_latency_batch)
        energy_batch = np.array(heft_energy_batch)

        qoe_batch = self.calculate_qoe(latency_batch, energy_batch)
        self.heft_avg_run_time = np.mean(latency_batch)
        self.heft_avg_energy = np.mean(energy_batch)
        self.heft_avg_qoe = np.mean(qoe_batch)
