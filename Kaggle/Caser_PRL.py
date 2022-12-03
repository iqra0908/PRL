import tensorflow as tf
import numpy as np
import pandas as pd
import os
import argparse
# import trfl
from utility import *
from SASRecModules import *
import logging
import time as Time

logging.getLogger().setLevel(logging.INFO)



def parse_args():
    parser = argparse.ArgumentParser(description="Run Caser with PRL.")

    parser.add_argument('--epoch', type=int, default=30,
                        help='Number of max epochs.')
    parser.add_argument('--data', nargs='?', default='data',
                        help='data directory')
    # parser.add_argument('--pretrain', type=int, default=1,
    #                     help='flag for pretrain. 1: initialize from pretrain; 0: randomly initialize; -1: save the model to pretrain file')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--r_click', type=float, default=0.2,
                        help='reward for the click behavior.')
    parser.add_argument('--r_buy', type=float, default=1.0,
                        help='reward for the purchase behavior.')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate.')
    parser.add_argument('--max_step', type=int, default=50,
                        help='maximum step')
    parser.add_argument('--num_filters', type=int, default=16,
                        help='Number of filters per filter size (default: 128)')
    parser.add_argument('--filter_sizes', nargs='?', default='[2,3,4]',
                        help='Specify the filter_size')
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--num_blocks', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.3, type=float)
    parser.add_argument('--reward_to_go', default=2.0, type=float)
    parser.add_argument('--std', default=0.0, type=float)

    return parser.parse_args()


class Caser_PRLnetwork:
    def __init__(self, hidden_size, learning_rate, item_num, state_size, max_step):
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.max_step = max_step

        all_embeddings = self.initialize_embeddings()

        self.inputs = tf.placeholder(tf.int32, [None, state_size], name='inputs')
        # self.cls=tf.placeholder(tf.int32, [None], name='cls')
        self.len_state = tf.placeholder(tf.int32, [None], name='len_state')
        self.target = tf.placeholder(tf.int32, [None], name='target')  # target item, to calculate ce loss
        self.reward_to_go = tf.placeholder(tf.float32, [None], name='reward_to_go')
        self.timestamp = tf.placeholder(tf.int32, [None], name='timestamp')
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.weight=tf.placeholder(tf.float32, [None], name='weight')


        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.inputs, item_num)), -1)

        self.input_emb = tf.nn.embedding_lookup(all_embeddings['state_embeddings'], self.inputs)
        self.input_emb *= mask
        self.embedded_chars_expanded = tf.expand_dims(self.input_emb, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        num_filters = args.num_filters
        filter_sizes = eval(args.filter_sizes)
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.hidden_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")

                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                # new shape after max_pool[?, 1, 1, num_filters]
                # be carefyul, the  new_sequence_length has changed because of wholesession[:, 0:-1]
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, state_size - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])  # shape=[batch_size, 384]
        # design the veritcal cnn
        with tf.name_scope("conv-verical"):
            filter_shape = [self.state_size, 1, 1, 1]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[1]), name="b")
            conv = tf.nn.conv2d(
                self.embedded_chars_expanded,
                W,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        self.vcnn_flat = tf.reshape(h, [-1, self.hidden_size])
        self.final = tf.concat([self.h_pool_flat, self.vcnn_flat], 1)  # shape=[batch_size, 384+100]

        # Add dropout
        with tf.name_scope("dropout"):
            self.state_hidden = tf.layers.dropout(self.final,
                                                  rate=args.dropout_rate,
                                                  training=tf.convert_to_tensor(self.is_training))

        self.step_embeddings = tf.nn.embedding_lookup(all_embeddings['step_embeddings'], self.timestamp)
        self.reward_embedding = tf.matmul(tf.expand_dims(self.reward_to_go, axis=-1),
                                          all_embeddings['reward_embedding'])
        # self.cls_embedding=tf.nn.embedding_lookup(all_embeddings['cls_embedding'], self.cls)

        self.seq = tf.stack([self.final, self.step_embeddings, self.reward_embedding], axis=1)

        # Dropout
        self.seq = tf.layers.dropout(self.seq,
                                     rate=args.dropout_rate,
                                     training=tf.convert_to_tensor(self.is_training))

        # Build blocks

        for i in range(args.num_blocks):
            with tf.variable_scope("num_blocks_%d" % i):
                # Self-attention
                self.seq = multihead_attention(queries=normalize(self.seq),
                                               keys=self.seq,
                                               num_units=112,
                                               num_heads=args.num_heads,
                                               dropout_rate=args.dropout_rate,
                                               is_training=self.is_training,
                                               causality=False,
                                               scope="self_attention")

                # Feed forward
                # self.seq = feedforward(normalize(self.seq), num_units=[self.hidden_size, self.hidden_size],
                #                        dropout_rate=args.dropout_rate,
                #                        is_training=self.is_training)

        self.seq = normalize(self.seq)

        s, t, r = tf.unstack(self.seq, axis=1)

        self.state_final = s
        # self.state_final=tf.reduce_mean(self.seq,axis=1)

        self.output = tf.contrib.layers.fully_connected(self.state_final, self.item_num, activation_fn=None,
                                                        scope='fc')

        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target, logits=self.output)
        self.loss = tf.multiply(self.loss,self.weight)
        self.loss = tf.reduce_mean(self.loss)
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def initialize_embeddings(self):
        all_embeddings = dict()
        state_embeddings = tf.Variable(tf.random_normal([self.item_num + 1, self.hidden_size], 0.0, 0.01),
                                       name='state_embeddings')
        step_embeddings = tf.Variable(tf.random_normal([self.max_step, 112], 0.0, 0.01),
                                      name='step_embeddings')
        reward_embedding = tf.Variable(tf.random_normal([1, 112], 0.0, 0.01),
                                       name='reward_embedding')
        # cls_embedding = tf.Variable(tf.random_normal([1, self.hidden_size], 0.0, 0.01),
        #                                name='cls_embedding')
        all_embeddings['state_embeddings'] = state_embeddings
        all_embeddings['step_embeddings'] = step_embeddings
        all_embeddings['reward_embedding'] = reward_embedding
        # all_embeddings['cls_embedding']=cls_embedding
        return all_embeddings


def evaluate(sess):
    eval_sessions = pd.read_pickle(os.path.join(data_directory, 'sampled_val.df'))

    eval_sessions['valid_session'] = eval_sessions.session_id.map(
        eval_sessions.groupby('session_id')['item_id'].size() < max_step + 1)
    eval_sessions = eval_sessions.loc[eval_sessions.valid_session].drop('valid_session', axis=1)

    eval_ids = eval_sessions.session_id.unique()
    groups = eval_sessions.groupby('session_id')
    batch = 100
    evaluated = 0
    total_clicks = 0.0
    total_purchase = 0.0
    total_reward = [0, 0, 0, 0]
    hit_clicks = [0, 0, 0, 0]
    ndcg_clicks = [0, 0, 0, 0]
    hit_purchase = [0, 0, 0, 0]
    ndcg_purchase = [0, 0, 0, 0]
    while evaluated < len(eval_ids):
        states, len_states, actions, rewards = [], [], [], []
        timestamp, reward_to_go = [], []
        # cls=[]
        for i in range(batch):
            # if evaluated in eval_ids:
            id = eval_ids[evaluated]
            group = groups.get_group(id)
            history = []
            step = 0
            for index, row in group.iterrows():
                state = list(history)
                len_states.append(state_size if len(state) >= state_size else 1 if len(state) == 0 else len(state))
                state = pad_history(state, state_size, item_num)
                states.append(state)
                action = row['item_id']
                is_buy = row['is_buy']
                reward = reward_buy if is_buy == 1 else reward_click
                if is_buy == 1:
                    total_purchase += 1.0
                else:
                    total_clicks += 1.0
                actions.append(action)
                rewards.append(reward)
                timestamp.append(step)
                # r_t = r_t_g
                reward_to_go.append(np.random.normal(r_t_g, args.std))
                history.append(row['item_id'])
                # cls.append(0)
                step += 1
            evaluated += 1
            if evaluated == len(eval_ids):
                break
        prediction = sess.run(Caser_PRLnet.output, feed_dict={Caser_PRLnet.inputs: states, Caser_PRLnet.len_state: len_states,
                                                            Caser_PRLnet.reward_to_go: reward_to_go,
                                                            Caser_PRLnet.timestamp: timestamp,
                                                            # Caser_PRLnet.cls: cls,
                                                            Caser_PRLnet.is_training: False
                                                            })
        sorted_list = np.argsort(prediction)
        calculate_hit(sorted_list, topk, actions, rewards, reward_click, total_reward, hit_clicks, ndcg_clicks,
                      hit_purchase, ndcg_purchase)
    logging.info('#############################################################')
    logging.info('total clicks: %d, total purchase:%d' % (total_clicks, total_purchase))
    for i in range(len(topk)):
        hr_click = hit_clicks[i] / total_clicks
        hr_purchase = hit_purchase[i] / total_purchase
        ng_click = ndcg_clicks[i] / total_clicks
        ng_purchase = ndcg_purchase[i] / total_purchase
        logging.info('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        logging.info('cumulative reward @ %d: %f' % (topk[i], total_reward[i]))
        logging.info('clicks hr ndcg @ %d : %f, %f' % (topk[i], hr_click, ng_click))
        logging.info('purchase hr and ndcg @%d : %f, %f' % (topk[i], hr_purchase, ng_purchase))
    logging.info('#############################################################')


if __name__ == '__main__':
    # Network parameters
    args = parse_args()

    data_directory = args.data
    data_statis = pd.read_pickle(
        os.path.join(data_directory,
                     'data_statis.df'))  # read data statistics, includeing state_size and item_num, mean_reward and max_reward
    state_size = data_statis['state_size'][0]  # the length of history to define the state
    item_num = data_statis['item_num'][0]  # total number of items
    reward_mean = data_statis['reward_mean'][0]
    reward_max = data_statis['reward_max'][0]
    reward_click = args.r_click
    reward_buy = args.r_buy
    topk = [5, 10, 15, 20]
    max_step = args.max_step
    r_t_g = args.reward_to_go
    # save_file = 'pretrain-Caser/%d' % (hidden_size)

    tf.reset_default_graph()

    logging.basicConfig(
        filename="./log/Caser/head:{}_block:{}_drop:{}_rtg:{}_std:{}_lr:{}_{}".format(
                                                            args.num_heads, args.num_blocks, args.dropout_rate,
                                                            args.reward_to_go, args.std,args.lr,Time.strftime("%m-%d %H:%M:%S", Time.localtime())))

    Caser_PRLnet = Caser_PRLnetwork(hidden_size=args.hidden_factor, learning_rate=args.lr, item_num=item_num,
                                state_size=state_size, max_step=max_step)

    trajectory_buffer = pd.read_pickle(os.path.join(data_directory, 'trajectory_buffer.df'))
    saver = tf.train.Saver()

    total_step = 0
    with tf.Session() as sess:
        # Initialize variables
        sess.run(tf.global_variables_initializer())
        # evaluate(sess)
        num_rows = trajectory_buffer.shape[0]
        num_batches = int(num_rows / args.batch_size)
        for i in range(args.epoch):
            for j in range(num_batches):
                batch = trajectory_buffer.sample(n=args.batch_size).to_dict()
                state = list(batch['state'].values())
                len_state = list(batch['len_state'].values())
                target = list(batch['action'].values())
                timestamp = list(batch['timestamp'].values())
                reward_to_go = list(batch['reward_to_go'].values())
                is_buy=list(batch['is_buy'].values())
                weight=[]
                for k in range(args.batch_size):
                    reward_to_go[k] = reward_to_go[k] / reward_mean[timestamp[k]]
                    if is_buy[k]==1:
                        weight.append(reward_buy)
                    if is_buy[k]==0:
                        weight.append(reward_click)
                # cls=[0]*len(timestamp)
                loss, _ = sess.run([Caser_PRLnet.loss, Caser_PRLnet.opt],
                                   feed_dict={Caser_PRLnet.inputs: state,
                                              Caser_PRLnet.len_state: len_state,
                                              Caser_PRLnet.target: target,
                                              Caser_PRLnet.reward_to_go: reward_to_go,
                                              Caser_PRLnet.timestamp: timestamp,
                                              Caser_PRLnet.weight:weight,
                                              # Caser_PRLnet.cls:cls,
                                              Caser_PRLnet.is_training: True})
                total_step += 1
                if total_step % 200 == 0:
                    # print("the loss in %dth batch is: %f" % (total_step, loss))
                    logging.info("the loss in %dth batch is: %f" % (total_step, loss))
                if total_step % 2000 == 0:
                    evaluate(sess)
        # saver.save(sess, save_file)
