import sys
import copy
import random
import numpy as np
from collections import defaultdict


def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    user_total = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split()
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_total[user] = User[user]
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_total[user] = User[user]
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_total, user_train, user_valid, user_test, usernum, itemnum]


def evaluate(model, dataset, args, sess):
    [total, train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    NDCG_sparse = 0.0
    HT = 0.0
    HT_sparse = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)### 注意user是从0开始的还是从1开始的

    P = 0.0;
    R = 0.0;
    MAP = 0.0;
    MRR = 0.0;
    MRR_sparse = 0.0;
    sparse_user = 0.0
    for u in users:
        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0) ### ？？？
        item_idx = [test[u][0]]
        for _ in range(100): # 100个负例
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(sess, [u], [seq], item_idx)
        # print(predictions)
        #
        # print(np.shape(predictions))
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0] # pos item rank

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
            MRR += 1 / (rank + 1)
        if len(train[u]) < 32:
            sparse_user += 1
            if rank < 10:
                NDCG_sparse += 1 / np.log2(rank + 2)
                HT_sparse += 1
                MRR_sparse += 1 / (rank + 1)


        top_k = 10

        # trueResult = [test[u][0]]
        # predictions = -predictions
        # total_pro = [(item_idx[i], predictions[i]) for i in range(101)]
        # total_pro.sort(key=lambda x: x[1], reverse=True)
        # rankedItem = [pair[0] for pair in total_pro]
        #
        # right_num = 0
        # trueNum = len(trueResult)
        # count = 0
        # for j in rankedItem:
        #     if count == top_k:
        #         P += 1.0 * right_num / count
        #         R += 1.0 * right_num / trueNum
        #     count += 1
        #     if j in trueResult:
        #         right_num += 1
        #         MAP = MAP + 1.0 * right_num / count
        #         if right_num == 1:
        #             MRR += 1.0 / count
        # if right_num != 0:
        #     MAP /= right_num

        if valid_user % 100 == 0:
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user, MRR / valid_user, \
           NDCG_sparse / sparse_user, HT_sparse / sparse_user, MRR_sparse / sparse_user


def evaluate_valid(model, dataset, args, sess):
    [total, train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)

    P = 0.0; R = 0.0; MAP = 0.0; MRR = 0.0;
    for u in users:
        # if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(sess, [u], [seq], item_idx)
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

        top_k = 10

        trueResult = [valid[u][0]]
        predictions = -predictions
        total_pro = [(item_idx[i], predictions[i]) for i in range(101)]
        total_pro.sort(key=lambda x: x[1], reverse=True)
        rankedItem = [pair[0] for pair in total_pro]

        right_num = 0
        trueNum = len(trueResult)
        count = 0
        for j in rankedItem:
            if count == top_k:
                P += 1.0 * right_num / count
                R += 1.0 * right_num / trueNum
            count += 1
            if j in trueResult:
                right_num += 1
                MAP = MAP + 1.0 * right_num / count
                if right_num == 1:
                    MRR += 1.0 / count
        if right_num != 0:
            MAP /= right_num

        if valid_user % 100 == 0:
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user, R / valid_user, MRR / valid_user
