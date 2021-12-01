import argparse

import os, os.path
from rltaskoffloading.environment.offloading_env_test_heuristics import evaluate_different_number, evaluate_different_trans

def evluate(args):
    graph_paths_test_for_number = ["./rltaskoffloading/offloading_data/offload_random10_test/random.10.",
                                   "./rltaskoffloading/offloading_data/offload_random15_test/random.15.",
                                   "./rltaskoffloading/offloading_data/offload_random20_test/random.20.",
                                   "./rltaskoffloading/offloading_data/offload_random25_test/random.25.",
                                   "./rltaskoffloading/offloading_data/offload_random30_test/random.30.",
                                   "./rltaskoffloading/offloading_data/offload_random35_test/random.35.",
                                   "./rltaskoffloading/offloading_data/offload_random40_test/random.40.",
                                   "./rltaskoffloading/offloading_data/offload_random45_test/random.45.",
                                   "./rltaskoffloading/offloading_data/offload_random50_test/random.50."
                                   ]

    graph_paths_test_for_trans = "./rltaskoffloading/offloading_data/offload_random15/random.15."

    if not os.path.exists(args.logpath):
        os.makedirs(args.logpath)

    logpath = args.logpath+"/heuristic-evaluate-"+args.scenario+"-"+args.goal + ".txt"

    if args.scenario == "Number":
        if args.goal == "LO":
            evaluate_different_number(graph_paths_test_for_number, lambda_t=1.0, lambda_e=0.0, logpath=logpath)
        elif args.goal == "EE":
            evaluate_different_number(graph_paths_test_for_number, lambda_t=0.5, lambda_e=0.5, logpath=logpath)
    elif args.scenario == "Trans":
        if args.goal == "LO":
            evaluate_different_trans(graph_paths_test_for_trans, lambda_t=1.0,
                                     lambda_e=0.0, bandwidths=[3.0, 7.0, 11.0, 15.0, 19.0],logpath=logpath)
        elif args.goal == "EE":
            evaluate_different_trans(graph_paths_test_for_trans, lambda_t=0.5,
                                     lambda_e=0.5, bandwidths=[3.0, 7.0, 11.0, 15.0, 19.0],logpath=logpath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--scenario", type=str, default="Trans", choices=["Number", "Trans"])
    parser.add_argument("--goal", type=str, default="LO", choices=["EE", "LO"])
    parser.add_argument("--logpath", type=str, default="./log")
    args = parser.parse_args()

    evluate(args)