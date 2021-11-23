from rltaskoffloading.environment.offloading_env import Resources
from rltaskoffloading.environment.offloading_env import OffloadingEnvironment

import numpy as np
import time
import logging

def calculate_qoe(latency_batch, energy_batch, env):
    all_local_time, all_local_energy = env.get_all_locally_execute_time_batch()
    all_local_time = np.squeeze(all_local_time)
    all_local_energy = np.squeeze(all_local_energy)
    latency_batch = np.squeeze(latency_batch)
    energy_batch = np.squeeze(energy_batch)
    qoe_batch = []

    for latency, energy, single_all_local_latency, single_all_local_energy in zip(latency_batch, energy_batch, all_local_time, all_local_energy):
        qoe = env.lambda_t * ((latency - single_all_local_latency) / single_all_local_latency) + \
              env.lambda_e * ((energy - single_all_local_energy) / single_all_local_energy)

        qoe = -qoe
        qoe_batch.append(qoe)

    return qoe_batch


def evaluate_different_number(graph_file_pahts, lambda_t=1.0, lambda_e=0.0, logpath="./log.txt"):

    logging.basicConfig(filename=logpath,level=logging.DEBUG, filemode='w')
    ch = logging.StreamHandler()
    logger = logging.getLogger()
    logger.addHandler(ch)

    logger.info("=============Test heurestic methods for different n. =============")
    logger.info("lambda_t: "+ str(lambda_t))
    logger.info("lambda_e: "+ str(lambda_e))

    for graph_file_path in graph_file_pahts:
        resource_cluster = Resources(mec_process_capable=(10.0 * 1024 * 1024),
                                 mobile_process_capable=(1.0 * 1024 * 1024), bandwith_up=7.0, bandwith_dl=7.0)

        env = OffloadingEnvironment(resource_cluster = resource_cluster,
                               batch_size=100,
                               graph_number=100,
                               graph_file_paths=[graph_file_path],
                               time_major=False,
                               lambda_t=lambda_t,
                               lambda_e=lambda_e)
        if env.task_graphs[0][0].task_number < 20:
            env.calculate_optimal_qoe()

        # Calculate the heft algorithms latency, energy and qoe
        plans, finish_time_batchs = env.greedy_solution(heft=True)
        heft_latency_batch, heft_energy_batch = env.get_running_cost_by_plan_batch(plans[0], env.task_graphs[0])
        latency_batch = np.array(heft_latency_batch)
        energy_batch = np.array(heft_energy_batch)
        heft_qoe_batch = calculate_qoe(latency_batch, energy_batch, env)

        # Calculate the greedy algorithms latency, energy and qoe
        plans, finish_time_batchs = env.greedy_solution(heft=False)
        greedy_latency_batch, greedy_energy_batch = env.get_running_cost_by_plan_batch(plans[0], env.task_graphs[0])
        latency_batch = np.array(greedy_latency_batch)
        energy_batch = np.array(greedy_energy_batch)
        greedy_qoe_batch = calculate_qoe(latency_batch, energy_batch, env)

        # Calculate the round robin latency, energy and qoe
        rrb_latency_batch, rrb_energy_batch = env.round_robin_solution()
        rrb_qoe_batch = calculate_qoe(rrb_latency_batch, rrb_energy_batch, env)

        # Calculate the random latency latency, energy and qoe
        random_latency_batch, random_energy_batch = env.random_solution()
        random_qoe_batch = calculate_qoe(random_latency_batch, random_energy_batch, env)

        # Calculate the all local latency, energy and qoe
        all_local_latency_batch, all_local_energy_batch = env.get_all_locally_execute_time_batch()
        all_local_qoe_batch = calculate_qoe(all_local_latency_batch, all_local_energy_batch, env)

        # Calculate the all remote latency, energy and qoe
        all_remote_latency_batch, all_remote_energy_batch = env.get_all_mec_execute_time_batch()
        all_remote_qoe_batch = calculate_qoe(all_remote_latency_batch, all_remote_energy_batch, env)

        logger.info(graph_file_path)
        logger.info("HEFT algorighm result: ")
        logger.info("latency: " + str(np.mean(heft_latency_batch)))
        logger.info("energy: "+ str( np.mean(heft_energy_batch)))
        logger.info("qoe: "+ str(np.mean(heft_qoe_batch)))
        logger.info(" ")
        logger.info("Greedy algorighm result: ")
        logger.info("latency: "+ str(np.mean(greedy_latency_batch)))
        logger.info("energy: "+ str(np.mean(greedy_energy_batch)))
        logger.info("qoe: "+ str(np.mean(greedy_qoe_batch)))
        logger.info(" ")
        logger.info("round roubin algorighm result: ")
        logger.info("latency: "+ str(np.mean(rrb_latency_batch)))
        logger.info("energy: "+ str(np.mean(rrb_energy_batch)))
        logger.info("qoe: " + str(np.mean(rrb_qoe_batch)))
        logger.info(" ")
        logger.info("random algorighm result: ")
        logger.info("latency: " + str(np.mean(random_latency_batch)))
        logger.info("energy: " + str(np.mean(random_energy_batch)))
        logger.info("qoe: " + str(np.mean(random_qoe_batch)))
        logger.info(" ")
        logger.info("all local algorighm result: ")
        logger.info("latency: " + str(np.mean(all_local_latency_batch)))
        logger.info("energy: " + str(np.mean(all_local_energy_batch)))
        logger.info("qoe: " + str(np.mean(all_local_qoe_batch)))
        logger.info(" ")
        logger.info("all remote algorigthm result: ")
        logger.info("latency: " + str(np.mean(all_remote_latency_batch)))
        logger.info("energy: " + str(np.mean(all_remote_energy_batch)))
        logger.info("qoe: " + str( np.mean(all_remote_qoe_batch)))

        logger.info("optimal qoe algorithm result: ")
        logger.info("optimal qoe: " + str(np.mean(env.optimal_qoe)))
        logger.info("optimal qoe latency: " + str(np.mean(env.optimal_qoe_latency)))
        logger.info("optimal qoe energy: "+ str( np.mean(env.optimal_qoe_energy)))


def evaluate_different_trans(graph_file_paths, lambda_t=1.0,
                             lambda_e=0.0, bandwidths=[3.0, 7.0, 11.0, 15.0, 19.0], logpath="./log.txt"):
    logging.basicConfig(filename=logpath, level=logging.DEBUG, filemode='w')
    ch = logging.StreamHandler()
    logger = logging.getLogger()
    logger.addHandler(ch)

    def test_case(trans_rate, graph_file_path, lambda_t=0.5, lambda_e=0.5):
        resource_cluster = Resources(mec_process_capable=(8.0 * 1024 * 1024),
                                     mobile_process_capable=(1.0 * 1024 * 1024), bandwith_up=trans_rate,
                                     bandwith_dl=trans_rate)

        logger.info("========= Testing the transmission rate {}Mbps. ============".format(trans_rate))
        env = OffloadingEnvironment(resource_cluster=resource_cluster,
                                    batch_size=100,
                                    graph_number=100,
                                    graph_file_paths=[graph_file_path],
                                    time_major=False,
                                    lambda_t=lambda_t,
                                    lambda_e=lambda_e)

        env.calculate_optimal_qoe()

        # Calculate the heft algorithms latency, energy and qoe
        plans, finish_time_batchs = env.greedy_solution(heft=True)
        heft_latency_batch, heft_energy_batch = env.get_running_cost_by_plan_batch(plans[0], env.task_graphs[0])
        latency_batch = np.array(heft_latency_batch)
        energy_batch = np.array(heft_energy_batch)
        heft_qoe_batch = calculate_qoe(latency_batch, energy_batch, env)

        # Calculate the greedy algorithms latency, energy and qoe
        plans, finish_time_batchs = env.greedy_solution(heft=False)
        greedy_latency_batch, greedy_energy_batch = env.get_running_cost_by_plan_batch(plans[0], env.task_graphs[0])
        latency_batch = np.array(greedy_latency_batch)
        energy_batch = np.array(greedy_energy_batch)
        greedy_qoe_batch = calculate_qoe(latency_batch, energy_batch, env)

        # Calculate the round robin latency, energy and qoe
        rrb_latency_batch, rrb_energy_batch = env.round_robin_solution()
        rrb_qoe_batch = calculate_qoe(rrb_latency_batch, rrb_energy_batch, env)

        # Calculate the random latency latency, energy and qoe
        random_latency_batch, random_energy_batch = env.random_solution()
        random_qoe_batch = calculate_qoe(random_latency_batch, random_energy_batch, env)

        # Calculate the all local latency, energy and qoe
        all_local_latency_batch, all_local_energy_batch = env.get_all_locally_execute_time_batch()
        all_local_qoe_batch = calculate_qoe(all_local_latency_batch, all_local_energy_batch, env)

        # Calculate the all remote latency, energy and qoe
        all_remote_latency_batch, all_remote_energy_batch = env.get_all_mec_execute_time_batch()
        all_remote_qoe_batch = calculate_qoe(all_remote_latency_batch, all_remote_energy_batch, env)

        logger.info(graph_file_path)
        logger.info("HEFT algorighm result: ")
        logger.info("latency: "+str(np.mean(heft_latency_batch)))
        logger.info("energy: "+str(np.mean(heft_energy_batch)))
        logger.info("qoe: "+str(np.mean(heft_qoe_batch)))
        logger.info(" ")
        logger.info("Greedy algorighm result: ")
        logger.info("latency: "+str(np.mean(greedy_latency_batch)))
        logger.info("energy: "+str(np.mean(greedy_energy_batch)))
        logger.info("qoe: "+str(np.mean(greedy_qoe_batch)))
        logger.info(" ")
        logger.info("round roubin algorighm result: ")
        logger.info("latency: "+str(np.mean(rrb_latency_batch)))
        logger.info("energy: "+str(np.mean(rrb_energy_batch)))
        logger.info("qoe: "+str(np.mean(rrb_qoe_batch)))
        logger.info(" ")
        logger.info("random algorighm result: ")
        logger.info("latency: "+str(np.mean(random_latency_batch)))
        logger.info("energy: "+str( np.mean(random_energy_batch)))
        logger.info("qoe: "+str( np.mean(random_qoe_batch)))
        logger.info(" ")
        logger.info("all local algorighm result: ")
        logger.info("latency: "+str(np.mean(all_local_latency_batch)))
        logger.info("energy: "+str(np.mean(all_local_energy_batch)))
        logger.info("qoe: "+str( np.mean(all_local_qoe_batch)))
        logger.info(" ")
        logger.info("all remote algorigthm result: ")
        logger.info("latency: "+str(np.mean(all_remote_latency_batch)))
        logger.info("energy: "+str(np.mean(all_remote_energy_batch)))
        logger.info("qoe: "+str(np.mean(all_remote_qoe_batch)))
        logger.info(" ")
        logger.info("optimal qoe algorithm result: ")
        logger.info("optimal qoe: "+str(np.mean(env.optimal_qoe)))
        logger.info("optimal qoe latency: "+str(np.mean(env.optimal_qoe_latency)))
        logger.info("optimal qoe energy: "+str(np.mean(env.optimal_qoe_energy)))

    for bandwidth in bandwidths:
        test_case(trans_rate=bandwidth, lambda_t=lambda_t,
                  lambda_e=lambda_e, graph_file_path=graph_file_paths)



