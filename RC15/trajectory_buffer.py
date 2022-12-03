import os

import numpy as np
import pandas as pd
import tensorflow as tf
from utility import to_pickled_df, pad_history

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('history_length', 10, 'uniform history length')
flags.DEFINE_integer('max_length', 50, 'maximum session length')
flags.DEFINE_float('reward_click', 0.2, 'reward for click')
flags.DEFINE_float('reward_buy', 1.0, 'reward for purchase')
flags.DEFINE_float('discount', 1.0, 'discount factor for RL')

if __name__ == '__main__':

    data_directory = 'data'

    length = FLAGS.history_length
    reward_click = FLAGS.reward_click
    reward_purchase = FLAGS.reward_buy
    discount = FLAGS.discount
    max_length = FLAGS.max_length

    # trajectory=pd.read_pickle(os.path.join(data_directory, 'trajectory_buffer.df'))
    # reply_buffer = pd.DataFrame(columns=['state','action','reward','next_state','is_done'])

    sampled_sessions = pd.read_pickle(os.path.join(data_directory, 'sampled_sessions.df'))
    item_ids = sampled_sessions.item_id.unique()
    pad_item = len(item_ids)

    sampled_sessions['valid_session'] = sampled_sessions.session_id.map(
        sampled_sessions.groupby('session_id')['item_id'].size() < max_length + 1)
    sampled_sessions = sampled_sessions.loc[sampled_sessions.valid_session].drop('valid_session', axis=1)

    ids = sampled_sessions.session_id.unique()
    groups = sampled_sessions.groupby('session_id')
    # reward_session = []
    reward_step = []
    reward_mean = []
    reward_max = []
    for step in range(0, max_length):
        crs = []
        reward_step.append(crs)
    # step_session=[]
    for id in ids:
        group = groups.get_group(id)
        cumulative_reward = 0
        step = 0
        for index, row in group.iterrows():
            is_b = row['is_buy']
            if is_b == 1:
                cumulative_reward += reward_purchase * pow(discount, step)
            else:
                cumulative_reward += reward_click * pow(discount, step)
            step += 1
        # reward_session.append(cumulative_reward)

        step = 0
        reward_step[step].append(cumulative_reward)
        for index, row in group.iterrows():
            is_b = row['is_buy']
            if is_b == 1:
                cumulative_reward -= reward_purchase * pow(discount, step)
            else:
                cumulative_reward -= reward_click * pow(discount, step)
            step += 1
            if step == group.shape[0]:
                break
            reward_step[step].append(cumulative_reward)
    for step in range(0, max_length):
        reward_mean.append(np.mean(reward_step[step]))
        reward_max.append(np.max(reward_step[step]))
    # reward_mean = np.mean(reward_session)
    # reward_max = np.max(reward_session)
    # step_mean=np.mean(step_session)
    # step_max=np.max(step_session)
    # print(reward_mean)
    # print("-------")
    # print(reward_max)

    train_sessions = pd.read_pickle(os.path.join(data_directory, 'sampled_train.df'))
    train_sessions['valid_session'] = train_sessions.session_id.map(
        train_sessions.groupby('session_id')['item_id'].size() < max_length + 1)
    train_sessions = train_sessions.loc[train_sessions.valid_session].drop('valid_session', axis=1)

    groups = train_sessions.groupby('session_id')
    ids = train_sessions.session_id.unique()

    timestamp, state, len_state, action, reward_to_go, is_buy = [], [], [], [], [],[]
    for id in ids:
        group=groups.get_group(id)
        cumulative_reward = 0
        step = 0
        for index, row in group.iterrows():
            is_b = row['is_buy']
            if is_b == 1:
                cumulative_reward += reward_purchase * pow(discount, step)
            else:
                cumulative_reward += reward_click * pow(discount, step)
            step += 1

        history = []
        step = 0
        for index, row in group.iterrows():
            reward_to_go.append(cumulative_reward)
            timestamp.append(step)
            s=list(history)
            len_state.append(length if len(s)>=length else 1 if len(s)==0 else len(s))
            s=pad_history(s,length,pad_item)
            a=row['item_id']

            state.append(s)
            action.append(a)

            is_b = row['is_buy']
            is_buy.append(is_b)
            if is_b == 1:
                cumulative_reward -= reward_purchase * pow(discount, step)
            else:
                cumulative_reward -= reward_click * pow(discount, step)
            step += 1
            history.append(row['item_id'])

    dic={'timestamp':timestamp, 'state':state,'len_state':len_state,'action':action,'reward_to_go':reward_to_go,'is_buy':is_buy}
    trajectory_buffer=pd.DataFrame(data=dic)
    to_pickled_df(data_directory, trajectory_buffer=trajectory_buffer)

    dic={'state_size':[length],'item_num':[pad_item],'reward_mean':[reward_mean],'reward_max':[reward_max]}
    data_statis=pd.DataFrame(data=dic)
    to_pickled_df(data_directory,data_statis=data_statis)
