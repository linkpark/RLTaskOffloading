from rltaskoffloading.offloading_ppo.offloading_ppo import S2SModel
from rltaskoffloading.offloading_ppo.offloading_ppo import Runner
from rltaskoffloading.environment.offloading_env import Resources
from rltaskoffloading.environment.offloading_env import OffloadingEnvironment
import tensorflow as tf
import numpy as np

gamma=0.99
lam=0.95
ent_coef=0.01
vf_coef=0.5
max_grad_norm=0.5
# load_path = "./checkpoint/model.ckpt"
load_path = None

resource_cluster = Resources(mec_process_capable=(10.0*1024*1024),
                                 mobile_process_capable=(1.0*1024*1024),  bandwith_up=3.0, bandwith_dl=3.0)

env = OffloadingEnvironment(resource_cluster = resource_cluster,
                           batch_size=100,
                           graph_number=100,
                           graph_file_paths=["../offloading_data/offload_random15_test/random.15."],
                           time_major=False)

ob = tf.placeholder(dtype=tf.float32, shape=[None, None, env.input_dim])
ob_length = tf.placeholder(dtype=tf.int32, shape=[None])

make_model = lambda: S2SModel(ob=ob, ob_length=ob_length, ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm)
model = make_model()
# model.load(load_path)

eval_runner = Runner(env=env, model=model, nepisode=1, gamma=gamma, lam=lam)

Tc, Ec = eval_runner.sample_eval()
greedy_Tc, greedy_Ec = eval_runner.greedy_eval()

greedy_Tc = np.mean(greedy_Tc)
greedy_Ec = np.mean(greedy_Ec)

print("greedy run time cost: ", greedy_Tc)
print("greedy energy consumption: ", greedy_Ec)