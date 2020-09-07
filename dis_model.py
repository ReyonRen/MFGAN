from dis_modules import *


class Dis():
    def __init__(self, usernum, itemnum, args, reuse=None):
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.u = tf.placeholder(tf.int32, shape=(None))
        self.input_seq = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.label = tf.placeholder(tf.float32, shape=(None, 2))

        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_seq, 0)), -1)

        with tf.variable_scope("discriminator", reuse=reuse):
            # -------------------------------------- ID ---------------------------------------
            # sequence embedding
            self.seq = embedding(self.input_seq,
                                 vocab_size=itemnum + 1,
                                 num_units=args.hidden_units,
                                 zero_pad=True,
                                 scale=True,
                                 l2_reg=args.l2_emb,
                                 scope="input_embeddings_dis",
                                 with_t=True,
                                 reuse=reuse
                                 )
            # Positional Encoding
            t = embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]), 0), [tf.shape(self.input_seq)[0], 1]),
                vocab_size=args.maxlen,
                num_units=args.hidden_units,
                zero_pad=False,
                scale=False,
                l2_reg=args.l2_emb,
                scope="dec_pos_dis",
                reuse=reuse,
                with_t=True
            )
            self.seq += t
            # Dropout
            self.seq = tf.layers.dropout(self.seq, rate=args.dis_dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training))
            self.seq *= mask
            # Build blocks
            for i in range(args.dis_num_blocks):
                with tf.variable_scope("num_blocks_dis_%d" % i):
                    # Self-attention
                    self.seq = multihead_attention(queries=normalize(self.seq),
                                                   keys=self.seq,
                                                   num_units=args.hidden_units,
                                                   num_heads=args.dis_num_heads,
                                                   dropout_rate=args.dis_dropout_rate,
                                                   is_training=self.is_training,
                                                   causality=False,
                                                   scope="self_attention_dis")

                    # Feed forward
                    self.seq = feedforward(normalize(self.seq), num_units=[args.hidden_units, args.hidden_units],
                                           dropout_rate=args.dis_dropout_rate, is_training=self.is_training)
                    self.seq *= mask
            self.seq = normalize(self.seq)
            self.seq = self.seq[:, -1, :]

            # -------------------------------------- KG ---------------------------------------
            # KG Embedding
            self.kg_seq = kg_embedding(self.input_seq,
                                       num_units=args.hidden_units_kg,
                                       scope="kg_embeddings",
                                       reuse=reuse
                                       )
            # Positional Encoding
            t = embedding(
               tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]), 0), [tf.shape(self.input_seq)[0], 1]),
               vocab_size=args.maxlen,
               num_units=args.hidden_units_kg,
               zero_pad=False,
               scale=False,
               l2_reg=args.l2_emb,
               scope="dec_pos_dis_kg",
               reuse=reuse,
               with_t=True
            )
            self.kg_seq += t
            # Dropout
            self.kg_seq = tf.layers.dropout(self.kg_seq, rate=args.dis_dropout_rate,
                                            training=tf.convert_to_tensor(self.is_training))
            self.kg_seq *= mask
            # Build blocks
            for i in range(args.dis_num_blocks):
                with tf.variable_scope("num_blocks_dis_kg_%d" % i):
                    # Self-attention
                    self.kg_seq = multihead_attention(queries=normalize(self.kg_seq),
                                                      keys=self.kg_seq,
                                                      num_units=args.hidden_units_kg,
                                                      num_heads=args.dis_num_heads,
                                                      dropout_rate=args.dis_dropout_rate,
                                                      is_training=self.is_training,
                                                      causality=False,
                                                      scope="self_attention_dis_kg")
                    # Feed forward
                    self.kg_seq = feedforward(normalize(self.kg_seq),
                                              num_units=[args.hidden_units_kg, args.hidden_units_kg],
                                              dropout_rate=args.dis_dropout_rate, is_training=self.is_training,
                                              scope="self_attention_dis_kg")
                    self.kg_seq *= mask
            self.kg_seq = normalize(self.kg_seq)
            self.kg_seq = self.kg_seq[:, -1, :]

            # -------------------------------------- category ---------------------------------------
            # sequence embedding, item embedding table
            self.input_cat, cat_num = cat_match(self.input_seq)
            self.cat_seq = embedding(self.input_cat,
                                     vocab_size=cat_num + 1,
                                     num_units=args.hidden_units,
                                     zero_pad=True,
                                     scale=True,
                                     l2_reg=args.l2_emb,
                                     scope="category_embeddings_dis",
                                     with_t=True,
                                     reuse=reuse
                                     )
            # Positional Encoding
            t = embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]), 0), [tf.shape(self.input_seq)[0], 1]),
                vocab_size=args.maxlen,
                num_units=args.hidden_units,
                zero_pad=False,
                scale=False,
                l2_reg=args.l2_emb,
                scope="dec_pos_dis_cat",
                reuse=reuse,
                with_t=True
            )
            self.cat_seq += t
            # Dropout
            self.cat_seq = tf.layers.dropout(self.cat_seq, rate=args.dis_dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training))
            self.cat_seq *= mask
            # Build blocks
            for i in range(args.dis_num_blocks):
                with tf.variable_scope("num_blocks_dis_cat_%d" % i):
                    # Self-attention
                    self.cat_seq = multihead_attention(queries=normalize(self.cat_seq),
                                                   keys=self.cat_seq,
                                                   num_units=args.hidden_units_cat,
                                                   num_heads=args.dis_num_heads,
                                                   dropout_rate=args.dis_dropout_rate,
                                                   is_training=self.is_training,
                                                   causality=False,
                                                   scope="self_attention_cat_dis")

                    # Feed forward
                    self.cat_seq = feedforward(normalize(self.cat_seq),
                                               num_units=[args.hidden_units_cat, args.hidden_units_cat],
                                               dropout_rate=args.dis_dropout_rate, is_training=self.is_training)
                    self.cat_seq *= mask
            self.cat_seq = normalize(self.cat_seq)
            self.cat_seq = self.cat_seq[:, -1, :]

            # -------------------------------------- popularity ---------------------------------------
            # sequence embedding
            self.input_pop, pop_num = pop_match(self.input_seq)
            self.pop_seq = embedding(self.input_pop,
                                     vocab_size=pop_num + 1,
                                     num_units=args.hidden_units,
                                     zero_pad=True,
                                     scale=True,
                                     l2_reg=args.l2_emb,
                                     scope="popularity_embeddings_dis",
                                     with_t=True,
                                     reuse=reuse
                                     )
            # Positional Encoding
            t = embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]), 0), [tf.shape(self.input_seq)[0], 1]),
                vocab_size=args.maxlen,
                num_units=args.hidden_units,
                zero_pad=False,
                scale=False,
                l2_reg=args.l2_emb,
                scope="dec_pos_dis_pop",
                reuse=reuse,
                with_t=True
            )
            self.pop_seq += t
            # Dropout
            self.pop_seq = tf.layers.dropout(self.pop_seq, rate=args.dis_dropout_rate,
                                             training=tf.convert_to_tensor(self.is_training))
            self.pop_seq *= mask
            # Build blocks
            for i in range(args.dis_num_blocks):
                with tf.variable_scope("num_blocks_dis_pop_%d" % i):
                    # Self-attention
                    self.pop_seq = multihead_attention(queries=normalize(self.pop_seq),
                                                       keys=self.pop_seq,
                                                       num_units=args.hidden_units_pop,
                                                       num_heads=args.dis_num_heads,
                                                       dropout_rate=args.dis_dropout_rate,
                                                       is_training=self.is_training,
                                                       causality=False,
                                                       scope="self_attention_pop_dis")

                    # Feed forward
                    self.pop_seq = feedforward(normalize(self.pop_seq),
                                               num_units=[args.hidden_units_pop, args.hidden_units_pop],
                                               dropout_rate=args.dis_dropout_rate, is_training=self.is_training)
                    self.pop_seq *= mask
            self.pop_seq = normalize(self.pop_seq)
            self.pop_seq = self.pop_seq[:, -1, :]

            # Final (unnormalized) scores and predictions
            l2_reg_lambda = 0.2
            l2_loss1 = tf.constant(0.0)
            with tf.name_scope("output1"):
                W1 = tf.Variable(tf.truncated_normal([args.hidden_units, 2], stddev=0.1), name="W1")
                b1 = tf.Variable(tf.constant(0.1, shape=[2]), name="b1")
                l2_loss1 += tf.nn.l2_loss(W1)
                l2_loss1 += tf.nn.l2_loss(b1)
                self.scores1 = tf.nn.xw_plus_b(self.seq, W1, b1, name="scores1")
                self.ypred_for_auc1 = tf.nn.softmax(self.scores1)

            l2_loss2 = tf.constant(0.0)
            with tf.name_scope("output2"):
                W2 = tf.Variable(tf.truncated_normal([args.hidden_units_kg, 2], stddev=0.1), name="W2")
                b2 = tf.Variable(tf.constant(0.1, shape=[2]), name="b2")
                l2_loss2 += tf.nn.l2_loss(W2)
                l2_loss2 += tf.nn.l2_loss(b2)
                self.scores2 = tf.nn.xw_plus_b(self.kg_seq, W2, b2, name="scores2")
                self.ypred_for_auc2 = tf.nn.softmax(self.scores2)

            l2_loss3 = tf.constant(0.0)
            with tf.name_scope("output3"):
                W3 = tf.Variable(tf.truncated_normal([args.hidden_units_cat, 2], stddev=0.1), name="W3")
                b3 = tf.Variable(tf.constant(0.1, shape=[2]), name="b3")
                l2_loss3 += tf.nn.l2_loss(W3)
                l2_loss3 += tf.nn.l2_loss(b3)
                self.scores3 = tf.nn.xw_plus_b(self.cat_seq, W3, b3, name="scores3")
                self.ypred_for_auc3 = tf.nn.softmax(self.scores3)

            l2_loss4 = tf.constant(0.0)
            with tf.name_scope("output4"):
                W4 = tf.Variable(tf.truncated_normal([args.hidden_units_pop, 2], stddev=0.1), name="W4")
                b4 = tf.Variable(tf.constant(0.1, shape=[2]), name="b4")
                l2_loss4 += tf.nn.l2_loss(W4)
                l2_loss4 += tf.nn.l2_loss(b4)
                self.scores4 = tf.nn.xw_plus_b(self.pop_seq, W4, b4, name="scores4")
                self.ypred_for_auc4 = tf.nn.softmax(self.scores4)
                # self.predictions4 = tf.argmax(self.scores4, 1, name="predictions4")

            self.ypred_for_auc = (self.ypred_for_auc1 + self.ypred_for_auc2 + self.ypred_for_auc3 +
                                  self.ypred_for_auc4) / 4

            # Calculate mean cross-entropy loss
            with tf.name_scope("loss"):
                loss1 = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores1, labels=self.label)
                loss2 = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores2, labels=self.label)
                loss3 = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores3, labels=self.label)
                loss4 = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores4, labels=self.label)
                self.loss = tf.reduce_mean(loss1 + loss2 + loss3 + loss4) + \
                            l2_reg_lambda * (l2_loss1 + l2_loss2 + l2_loss3 + l2_loss4)

        if reuse is None:
            self.params = [param for param in tf.trainable_variables() if 'discriminator' in param.name]
            self.global_step = tf.Variable(0, name='global_step_dis', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=args.dis_lr, beta2=0.98)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

        self.merged = tf.summary.merge_all()


