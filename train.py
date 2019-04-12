import config
import time
import copy
from multiprocessing import Process
import pickle
import random
import numpy as np
import tensorflow as tf
from script import parse, choose_traffic_file, config_all, parse_roadnet, write_summary
import os
from config import DIC_ENVS
from keras.layers import Input, Dense, Flatten, Reshape, Layer, Lambda, RepeatVector, Activation, Embedding, Conv2D
from keras.models import Model, model_from_json, load_model
from keras.optimizers import RMSprop, Adam
from keras.layers.merge import concatenate, add, dot, maximum, multiply
from keras import backend as K
from keras.initializers import RandomNormal, Constant


def loss(target, output):
    _epsilon =  tf.convert_to_tensor(10e-8, dtype=output.dtype.base_dtype)
    output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
    return - target * tf.log(output)


class Actor(object):
    def __init__(self, n_features, n_actions, lr=0.001):
        self.n_features = n_features
        self.n_actions = n_actions
        self.lr = lr
        self.network = self.build_network()
        self.dummy_act_picked = np.zeros((1, self.n_actions))

    def build_network(self):
        features = Input(shape=(self.n_features, ), name="features")
        action_picked = Input(shape=(self.n_actions, ), name="action_picked")

        hidden_1 = Dense(20, activation='relu', name="dense_1", kernel_initializer=RandomNormal(mean=0., stddev=.1),
                         bias_initializer=Constant(0.1))(features)
        act_prob = Dense(self.n_actions, activation='softmax', name="output_layer",
                    kernel_initializer=RandomNormal(mean=0., stddev=.1), bias_initializer=Constant(0.1))(hidden_1)
        selected_act_prob = multiply([act_prob, action_picked])
        selected_act_prob = Lambda(lambda x: K.sum(x, axis=-1, keepdims=True), output_shape=(1,))(selected_act_prob)

        # actor
        model = Model(inputs=[features, action_picked], outputs=[act_prob, selected_act_prob])
        opt = Adam(lr=self.lr)
        model.compile(loss=['mse', loss], loss_weights=[0.0, 1.0], optimizer=opt)
        model.summary()
        return model

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        self.network.train_on_batch([s, a], [self.dummy_act_picked, td])

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.network.predict([s, self.dummy_act_picked])[0].flatten()
        return np.random.choice(np.arange(self.n_actions), p=probs)  # return a int


class Critic(object):
    def __init__(self, n_features, lr=0.01):
        self.n_features = n_features
        self.lr = lr
        # self.dic_traffic_env_conf = dic_traffic_env_conf
        self.network = self.build_network()

    def build_network(self):
        features = Input(shape=(self.n_features,), name="features")
        hidden_1 = Dense(20, activation='relu', name="dense_1_critic", kernel_initializer=RandomNormal(mean=0., stddev=.1),
                         bias_initializer=Constant(0.1))(features)
        value = Dense(1, activation="linear", name="state_value", kernel_initializer=RandomNormal(mean=0., stddev=.1),
                      bias_initializer=Constant(0.1))(hidden_1)

        # critic
        model = Model(inputs=features, outputs=value)
        opt = Adam(lr=self.lr)
        model.compile(loss='mse', optimizer=opt)
        model.summary()
        return model

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]
        v = self.network.predict(s)[0]
        v_ = self.network.predict(s_)[0]
        target = r + 0.8 * v_
        td_error = target - v
        target = np.array([target])
        self.network.train_on_batch(s, target)
        return np.array([td_error])


def main(args):
    traffic_file_list = choose_traffic_file(args)

    process_list = []
    for traffic_file in traffic_file_list:

        traffic_of_tasks = [traffic_file]

        ### *** exp, agent, traffic_env, path_conf
        dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path = config_all(args)
        # path
        _time = time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))
        postfix = ""
        dic_path.update({
            "PATH_TO_MODEL": os.path.join(dic_path["PATH_TO_MODEL"], traffic_file + "_" + _time + postfix),
            "PATH_TO_WORK_DIRECTORY": os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"],
                                                   traffic_file + "_" + _time + postfix),
            "PATH_TO_GRADIENT": os.path.join(dic_path["PATH_TO_GRADIENT"], traffic_file + "_" + _time + postfix,
                                             "gradient")
        })
        # traffic env
        dic_traffic_env_conf["TRAFFIC_FILE"] = traffic_file
        dic_traffic_env_conf["TRAFFIC_IN_TASKS"] = [traffic_file]
        if ".json" in traffic_file:
            dic_traffic_env_conf.update({"SIMULATOR_TYPE": "anon"})
        elif ".xml" in traffic_file:
            dic_traffic_env_conf.update({"SIMULATOR_TYPE": "sumo"})
        else:
            raise (ValueError)
        # parse roadnet
        roadnet_path = os.path.join(dic_path['PATH_TO_DATA'], dic_traffic_env_conf['ROADNET_FILE'])
        lane_phase_info = parse_roadnet(roadnet_path)
        dic_traffic_env_conf["LANE_PHASE_INFO"] = lane_phase_info["intersection_1_1"]
        dic_traffic_env_conf["num_lanes"] = int(len(lane_phase_info["intersection_1_1"]["start_lane"]) / 4) # num_lanes per direction
        dic_traffic_env_conf["num_phases"] = len(lane_phase_info["intersection_1_1"]["phase"])

        dic_exp_conf.update({
            "TRAFFIC_FILE": traffic_file,  # Todo
            "TRAFFIC_IN_TASKS": traffic_of_tasks})

        single_process = False
        if single_process:
            _train(copy.deepcopy(dic_exp_conf),
                   copy.deepcopy(dic_agent_conf),
                   copy.deepcopy(dic_traffic_env_conf),
                   copy.deepcopy(dic_path))
        else:
            p = Process(target=_train, args=(copy.deepcopy(dic_exp_conf),
                                             copy.deepcopy(dic_agent_conf),
                                             copy.deepcopy(dic_traffic_env_conf),
                                             copy.deepcopy(dic_path)))

            process_list.append(p)

    num_process = args.num_process
    if not single_process:
        i = 0
        list_cur_p = []
        for p in process_list:
            if len(list_cur_p) < num_process:
                print(i)
                p.start()
                list_cur_p.append(p)
                i += 1
            if len(list_cur_p) < num_process:
                continue

            idle = check_all_workers_working(list_cur_p)

            while idle == -1:
                time.sleep(1)
                idle = check_all_workers_working(
                    list_cur_p)
            del list_cur_p[idle]

        for i in range(len(list_cur_p)):
            p = list_cur_p[i]
            p.join()


def check_all_workers_working(list_cur_p):
    for i in range(len(list_cur_p)):
        if not list_cur_p[i].is_alive():
            return i

    return -1


def convert_to_input(state, dic_traffic_env_conf):
    inputs = []
    all_start_lane = dic_traffic_env_conf["LANE_PHASE_INFO"]["start_lane"]
    phase = [0] * len(all_start_lane)
    inputs.extend(state['lane_num_vehicle'])
    start_lane = dic_traffic_env_conf["LANE_PHASE_INFO"]["phase_startLane_mapping"][state["cur_phase"][0]]
    for lane in start_lane:
        phase[all_start_lane.index(lane)] = 1
    inputs.extend(phase)
    inputs = np.array(inputs)

    return inputs


def _train(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path):
    random.seed(dic_agent_conf['SEED'])
    np.random.seed(dic_agent_conf['SEED'])
    tf.set_random_seed(dic_agent_conf['SEED'])


    actor = Actor(n_features=16, n_actions=8, lr=1e-3)  # 初始化Actor
    critic = Critic(n_features=16, lr=1e-3)  # 初始化Critic


    dic_path["PATH_TO_LOG"] = os.path.join(dic_path['PATH_TO_WORK_DIRECTORY'], "train_round")

    if not os.path.exists(dic_path['PATH_TO_LOG']):
        os.makedirs(dic_path['PATH_TO_LOG'])

    if dic_traffic_env_conf['SIMULATOR_TYPE'] == 'sumo':
        path_to_work_directory = dic_path["PATH_TO_SUMO_CONF"]
        env = DIC_ENVS[dic_traffic_env_conf["SIMULATOR_TYPE"]](
            path_to_log=dic_path["PATH_TO_LOG"],
            path_to_work_directory=path_to_work_directory,
            dic_traffic_env_conf=dic_traffic_env_conf)

    elif dic_traffic_env_conf['SIMULATOR_TYPE'] == 'anon':
        env = DIC_ENVS[dic_traffic_env_conf["SIMULATOR_TYPE"]](
            path_to_log=dic_path["PATH_TO_LOG"],
            path_to_work_directory=dic_path["PATH_TO_DATA"],
            dic_traffic_env_conf=dic_traffic_env_conf)

    for i in range(200):

        done = False
        state = env.reset()
        step_num = 0
        stop_cnt = 0
        while not done and step_num < int(dic_exp_conf["EPISODE_LEN"] / dic_traffic_env_conf["MIN_ACTION_TIME"]):
            action_list = []
            for one_state in state:
                s = convert_to_input(one_state, dic_traffic_env_conf)
                action = actor.choose_action(s)  # one for multi-state, the other for multi-intersection
                action_list.append(action)  # for multi-state

            next_state, reward, done, _ = env.step(action_list)

            s = convert_to_input(state[0], dic_traffic_env_conf)
            s_ = convert_to_input(next_state[0], dic_traffic_env_conf)

            td_error = critic.learn(s, reward[0], s_)
            act_one_hot = np.zeros((1, 8))
            act_one_hot[0, action] = 1.0
            actor.learn(s, act_one_hot, td_error)

            state = next_state
            step_num += 1
            stop_cnt += 1
        env.bulk_log(i)
        write_summary(dic_path, dic_exp_conf, i)


if __name__ == '__main__':
    args = parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpu
    main(args)
