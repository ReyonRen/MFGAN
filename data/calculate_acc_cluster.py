import sys
import numpy as np

f_answer = open('metric/test_answer_with_id_3610_nowordpiece.tsv', 'r')


print('reading text')
p_text = []
for i in range(8):
    f_p_text = open('wiki_corpus_by_fb/wiki_corpus.part%d' % i, 'r')
    for line in f_p_text:
        line = line.strip('\n').split('\t')
        p_text.append(line[2].strip().lower().replace(' ', ''))

topk = int(sys.argv[1])

print('reading query-para-score')
para = {}
score = {}
for i in range(8):
    f_qp = open('res.nq.top%s-part%s' % (topk, i), 'r')
    for line in f_qp:
        line = line.strip('\n').split('\t')
        q = line[0]
        p = line[1]
        s = line[2]
        if q not in para and q not in score:
            para[q] = []
            score[q] = []
        para[q].append(p)
        score[q].append(float(s))
    f_qp.close()

print('calculating acc')
right_num_r50 = 0.0
right_num_r100 = 0.0
right_num_r20 = 0.0
right_num_r10 = 0.0
right_num_r5 = 0.0
right_num_r1 = 0.0
query_num = 0.0
MRR = 0.0
out = open('output/res.nq.top%d' % topk, 'w')
for line in f_answer:
    query_num += 1
    line = line.strip('\n').split('\t')
    answer = line[1:]
    q = str(int(line[0])+1)
    data = list(zip(score[q], para[q]))
    data.sort()
    data.reverse()
    data = data[:topk]
    flag = 0
    for i in range(topk):
        out.write("%s\t%s\t%s\t%s\n" % (q, data[i][1], i+1, data[i][0]))
    for i in range(100):
        for j in range(len(answer)):
            if p_text[int(data[i][1])].find(answer[j].strip().lower().replace(' ', '')) != -1:
                right_num_r100 += 1
                flag = 1
                break
        if flag == 1:
            break
    flag = 0
    for i in range(50):
        for j in range(len(answer)):
            if p_text[int(data[i][1])].find(answer[j].strip().lower().replace(' ', '')) != -1:
                right_num_r50 += 1
                flag = 1
                break
        if flag == 1:
            break
    flag = 0
    for i in range(20):
        for j in range(len(answer)):
            if p_text[int(data[i][1])].find(answer[j].strip().lower().replace(' ', '')) != -1:
                right_num_r20 += 1
                flag = 1
                break
        if flag == 1:
            break
    flag = 0
    for i in range(10):
        for j in range(len(answer)):
            if p_text[int(data[i][1])].find(answer[j].strip().lower().replace(' ', '')) != -1:
                right_num_r10 += 1
                flag = 1
                break
        if flag == 1:
            break
    flag = 0
    for i in range(5):
        for j in range(len(answer)):
            if p_text[int(data[i][1])].find(answer[j].strip().lower().replace(' ', '')) != -1:
                right_num_r5 += 1
                flag = 1
                break
        if flag == 1:
            break
    flag = 0
    for i in range(1):
        for j in range(len(answer)):
            if p_text[int(data[i][1])].find(answer[j].strip().lower().replace(' ', '')) != -1:
                right_num_r1 += 1
                flag = 1
                break
        if flag == 1:
            break
    flag = 0
    for i in range(10):
        for j in range(len(answer)):
            if p_text[int(data[i][1])].find(answer[j].strip().lower().replace(' ', '')) != -1:
                MRR += 1.0 / (i+1)
                flag = 1
                break
        if flag == 1:
            break
out.close()
r100 = right_num_r100 / query_num
r50 = right_num_r50 / query_num
r20 = right_num_r20 / query_num
r10 = right_num_r10 / query_num
r5 = right_num_r5 / query_num
r1 = right_num_r1 / query_num
MRR = MRR / query_num

print('recall@100: ' +  str(r100))
print('recall@50: ' + str(r50))
print('recall@20: ' + str(r20))
print('recall@10: ' + str(r10))
print('recall@5: ' + str(r5))
print('recall@1: ' + str(r1))
print('MRR@10: ' + str(MRR))



