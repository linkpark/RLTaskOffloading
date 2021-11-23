import argparse

from rltaskoffloading.offloading_ddqn.lstm_ddqn import DDQNTO_number, DDQNTO_trans
from rltaskoffloading.offloading_ppo.offloading_ppo import DRLTO_number, DRLTO_trans

def train(args):
    # Here is some global configuration for the datapath
    graph_paths_train_for_number = ["./rltaskoffloading/offloading_data/offload_random10/random.10.",
                              "./rltaskoffloading/offloading_data/offload_random15/random.15.",
                              "./rltaskoffloading/offloading_data/offload_random20/random.20.",
                              "./rltaskoffloading/offloading_data/offload_random25/random.25.",
                              "./rltaskoffloading/offloading_data/offload_random30/random.30.",
                              "./rltaskoffloading/offloading_data/offload_random35/random.35.",
                              "./rltaskoffloading/offloading_data/offload_random40/random.40.",
                              "./rltaskoffloading/offloading_data/offload_random45/random.45.",
                              "./rltaskoffloading/offloading_data/offload_random50/random.50.",
                              ]
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

    graph_paths_train_for_trans = ["./rltaskoffloading/offloading_data/offload_random15/random.15."]
    graph_paths_test_for_trans = ["./rltaskoffloading/offloading_data/offload_random15/random.15."]

    logpath = args.logpath+"-"+args.algo +"-"+args.scenario+"-"+args.goal +"-dependency-" + str(args.dependency)
    if args.algo == "DDQNTO":
        if args.scenario == "Number":
            if args.goal == "LO":
                DDQNTO_number(lambda_t = 1.0, lambda_e = 0.0, logpath=logpath, encode_dependencies=args.dependency,
                              train_graph_file_paths = graph_paths_train_for_number,
                              test_graph_file_paths= graph_paths_test_for_number)
            elif args.goal == "EE":
                DDQNTO_number(lambda_t=0.5, lambda_e=0.5, logpath=logpath, encode_dependencies=args.dependency,
                              train_graph_file_paths=graph_paths_train_for_number,
                              test_graph_file_paths=graph_paths_test_for_number)
        if args.scenario == "Trans":
            if args.goal == "LO":
                DDQNTO_trans(lambda_t = 1.0, lambda_e = 0.0, logpath=logpath, encode_dependencies=args.dependency,
                             train_graph_file_paths=graph_paths_train_for_trans,
                             test_graph_file_paths=graph_paths_test_for_trans,
                             bandwidths=[3.0, 7.0, 11.0, 15.0, 19.0])
            elif args.goal == "EE":
                DDQNTO_trans(lambda_t=0.5, lambda_e=0.5, logpath=logpath, encode_dependencies=args.dependency,
                             train_graph_file_paths=graph_paths_train_for_trans,
                             test_graph_file_paths=graph_paths_test_for_trans,
                             bandwidths=[3.0, 7.0, 11.0, 15.0, 19.0])
    elif args.algo == "DRLTO":
        if args.scenario == "Number":
            if args.goal == "LO":
                DRLTO_number(lambda_t = 1.0, lambda_e = 0.0, logpath=logpath, encode_dependencies=args.dependency,
                              train_graph_file_paths = graph_paths_train_for_number,
                              test_graph_file_paths= graph_paths_test_for_number)
            elif args.goal == "EE":
                DRLTO_number(lambda_t=0.5, lambda_e=0.5, logpath=logpath, encode_dependencies=args.dependency,
                             train_graph_file_paths=graph_paths_train_for_number,
                             test_graph_file_paths=graph_paths_test_for_number)
        if args.scenario == "Trans":
            if args.goal == "LO":
                DRLTO_trans(lambda_t=1.0, lambda_e=0.0, logpath=logpath, encode_dependencies=args.dependency,
                             train_graph_file_paths=graph_paths_train_for_trans,
                             test_graph_file_paths=graph_paths_test_for_trans,
                             bandwidths=[3.0, 7.0, 11.0, 15.0, 19.0])
            elif args.goal == "EE":
                DRLTO_trans(lambda_t=0.5, lambda_e=0.5, logpath=logpath, encode_dependencies=args.dependency,
                             train_graph_file_paths=graph_paths_train_for_trans,
                             test_graph_file_paths=graph_paths_test_for_trans,
                             bandwidths=[3.0, 7.0, 11.0, 15.0, 19.0])
    else:
        raise Exception("No defined algorithm")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--algo", type=str, default="DRLTO", choices=["DDQNTO", "DRLTO"])
    parser.add_argument("--scenario", type=str, default="Trans", choices=["Number", "Trans"])
    parser.add_argument("--goal", type=str, default="EE", choices=["EE", "LO"])
    parser.add_argument("--logpath", type=str, default="./log/Result")
    parser.add_argument("--dependency", type=bool, default=True)
    args = parser.parse_args()

    train(args)