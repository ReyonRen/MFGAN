from gen_modules import *
import tensorflow as tf
import numpy as np
from tqdm import tqdm


class Gen():
    def __init__(self, usernum, itemnum, args, reuse=None):
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.u = tf.placeholder(tf.int32, shape=(None))
        self.input_seq = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.pos = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.neg = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        pos = self.pos
        neg = self.neg
        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_seq, 0)), -1)

        with tf.variable_scope("SA_gen", reuse=reuse):
            # sequence embedding, item embedding table
            self.seq, self.item_emb_table = embedding(self.input_seq,
                                                 vocab_size=itemnum + 1,
                                                 num_units=args.hidden_units,
                                                 zero_pad=True,
                                                 scale=True,
                                                 l2_reg=args.l2_emb,
                                                 scope="input_embeddings_gen",
                                                 with_t=True,
                                                 reuse=reuse
                                                 )

            # Positional Encoding
            t, pos_emb_table = embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]), 0), [tf.shape(self.input_seq)[0], 1]),
                vocab_size=args.maxlen,
                num_units=args.hidden_units,
                zero_pad=False,
                scale=False,
                l2_reg=args.l2_emb,
                scope="dec_pos_gen",
                reuse=reuse,
                with_t=True
            )
            self.seq += t

            # Dropout
            self.seq = tf.layers.dropout(self.seq,
                                         rate=args.gen_dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training))
            self.seq *= mask

            # Build blocks
            for i in range(args.gen_num_blocks):
                with tf.variable_scope("num_blocks_gen_%d" % i):
                    # Feed-forward1
                    self.seq = feedforward(normalize(self.seq), num_units=[args.hidden_units, args.hidden_units],
                                           dropout_rate=args.gen_dropout_rate, is_training=self.is_training)
                    # Self-attention
                    self.seq = multihead_attention(queries=normalize(self.seq),
                                                   keys=self.seq,
                                                   num_units=args.hidden_units,
                                                   num_heads=args.gen_num_heads,
                                                   dropout_rate=args.gen_dropout_rate,
                                                   is_training=self.is_training,
                                                   causality=True,
                                                   scope="self_attention_gen")

                    # Feed forward2
                    self.seq = feedforward2(normalize(self.seq), num_units=[args.hidden_units, args.hidden_units],
                                           dropout_rate=args.gen_dropout_rate, is_training=self.is_training)
                    self.seq *= mask

            self.seq = normalize(self.seq)

        self.rewards = tf.placeholder(tf.float32, shape=(args.gen_batch_size * args.maxlen))

        pos = tf.reshape(pos, [tf.shape(self.input_seq)[0] * args.maxlen])
        neg = tf.reshape(neg, [tf.shape(self.input_seq)[0] * args.maxlen])
        pos_emb = tf.nn.embedding_lookup(self.item_emb_table, pos)
        neg_emb = tf.nn.embedding_lookup(self.item_emb_table, neg)
        seq_emb = tf.reshape(self.seq, [tf.shape(self.input_seq)[0] * args.maxlen, args.hidden_units])

        self.test_item = tf.placeholder(tf.int32, shape=(101))
        test_item_emb = tf.nn.embedding_lookup(self.item_emb_table, self.test_item)
        self.test_logits = tf.matmul(seq_emb, tf.transpose(test_item_emb))
        self.test_logits = tf.reshape(self.test_logits, [tf.shape(self.input_seq)[0], args.maxlen, 101])
        self.test_logits = self.test_logits[:, -1, :]

        # self.position = tf.placeholder(tf.int32, shape=None)
        # self.seq_emb_i = self.seq[:, self.position, :]
        # self.seq_emb_i = tf.reshape(self.seq_emb_i, [tf.shape(self.input_seq)[0], args.hidden_units])
        # self.item_logits = tf.matmul(seq_emb, tf.transpose(self.item_emb_table))
        # self.item_logits = tf.reshape(self.item_logits, [tf.shape(self.input_seq)[0], args.maxlen, tf.shape(self.item_emb_table)[0]])
        # self.item_logits = self.item_logits[:, -1, :]

        self.seq_emb_i = self.seq[:, -1, :]
        self.last_item_logits = tf.matmul(self.seq_emb_i, tf.transpose(self.item_emb_table))

        # prediction layer
        self.pos_logits = tf.reduce_sum(pos_emb * seq_emb, -1)
        self.neg_logits = tf.reduce_sum(neg_emb * seq_emb, -1)

        # ignore padding items (0)
        istarget = tf.reshape(tf.to_float(tf.not_equal(pos, 0)), [tf.shape(self.input_seq)[0] * args.maxlen])
        self.pre_loss = tf.reduce_sum(
            - tf.log(tf.sigmoid(self.pos_logits) + 1e-24) * istarget -
            tf.log(1 - tf.sigmoid(self.neg_logits) + 1e-24) * istarget
        ) / tf.reduce_sum(istarget)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.pre_loss += sum(reg_losses)

        self.pre_global_step = tf.Variable(0, name='global_step_gen', trainable=False)
        self.pre_optimizer = tf.train.AdamOptimizer(learning_rate=args.gen_lr, beta2=0.98)
        self.pre_train_op = self.pre_optimizer.minimize(self.pre_loss, global_step=self.pre_global_step)

        self.gen_loss = tf.reduce_sum(
            (- tf.log(tf.sigmoid(self.pos_logits) + 1e-24) * istarget * self.rewards -
             tf.log(1 - tf.sigmoid(self.neg_logits) + 1e-24) * istarget)
        ) / tf.reduce_sum(istarget)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.gen_loss += sum(reg_losses)

        self.gen_global_step = tf.Variable(0, name='global_step_gen', trainable=False)
        self.gen_optimizer = tf.train.AdamOptimizer(learning_rate=args.gen_lr, beta2=0.98)
        self.gen_train_op = self.gen_optimizer.minimize(self.gen_loss, global_step=self.gen_global_step)

        self.merged = tf.summary.merge_all()

    def predict(self, sess, u, seq, item_idx):
        return sess.run(self.test_logits,
                        {self.u: u, self.input_seq: seq, self.test_item: item_idx, self.is_training: False})

    def generate_position_k(self, sess, u, seq, k, args, batch=6040):
        print("sampling")
        sampled_item = np.zeros([len(u), args.maxlen, k], dtype=np.int32)
        for i in tqdm(range(batch), total=batch, ncols=70, leave=False, unit='u'):
            logit = sess.run(self.item_logits,
                             {self.u: u, self.input_seq: [seq[i]], self.is_training: False})
            logit = -logit
            index = logit.argsort()
            for position in range(args.maxlen):
                if seq[i][args.maxlen - 1 - position] == 0:
                    break
                cnt = 0
                for j in range(k):
                    if index[args.maxlen - 1 - position][cnt] == seq[i][args.maxlen - 1 - position]:
                        cnt += 1
                    sampled_item[i][args.maxlen - 1 - position][j] = index[args.maxlen - 1 - position][cnt]
                    cnt += 1

        return sampled_item  # user * maxlen * k

    def generate_k(self, sess, u, seq, pos, k):
        batch = 10
        interval = int(len(seq) / batch)
        begin = 0
        sampled_item = np.zeros([len(u), k], dtype=np.int32)
        global_pos = 0
        for i in tqdm(range(batch-1), total=batch - 1, ncols=70, leave=False, unit='u'):
            logit = sess.run(self.last_item_logits,
                             {self.u: u, self.input_seq: seq[begin:begin + interval], self.is_training: False})
            begin += interval
            logit = -logit
            index = logit.argsort()
            for line in range(len(logit)):
                cnt = 0
                for rank in range(k):
                    if index[line][cnt] == pos[global_pos]: cnt += 1
                    sampled_item[global_pos][rank] = index[line][cnt]
                    cnt += 1
                global_pos += 1

        logit = sess.run(self.last_item_logits,
                         {self.u: u, self.input_seq: seq[begin:len(seq)], self.is_training: False})
        for line in range(len(logit)):
            cnt = 0
            for rank in range(k):
                if index[line][cnt] == pos[global_pos]: cnt += 1
                sampled_item[global_pos][rank] = index[line][cnt]
                cnt += 1
            global_pos += 1

        return sampled_item

    def generate_last_item(self, sess, u, seq):
        batch = 1
        interval = int(len(seq) / batch)
        begin = 0
        global_pos = 0
        top_item = np.zeros([len(u)])
        for i in range(batch-1):
            logit = sess.run(self.last_item_logits,
                             {self.u: u, self.input_seq: seq[begin:begin + interval], self.is_training: False})
            begin += interval
            logit = -logit
            index = logit.argsort()
            for line in range(len(logit)):
                top_item[global_pos] = index[line][0]
                global_pos += 1
        logit = sess.run(self.last_item_logits,
                         {self.u: u, self.input_seq: seq[begin:len(seq)], self.is_training: False})
        logit = -logit
        index = logit.argsort()
        for line in range(len(logit)):
            top_item[global_pos] = index[line][0]
            global_pos += 1
        return top_item
