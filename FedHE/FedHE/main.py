import pickle
import torch
import numpy as np
from user import user
from server import server
from sklearn import metrics
import math
import argparse
import warnings
import sys
import faulthandler
from log import Logger
faulthandler.enable()
warnings.filterwarnings('ignore')
# torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description="args for FedGNN")
parser.add_argument('--embed_size', type=int, default=32)
parser.add_argument('--lr', type=float, default = 0.2)
parser.add_argument('--data', default='ciao')
parser.add_argument('--user_batch', type=int, default=256)
parser.add_argument('--clip', type=float, default = 0.3)
parser.add_argument('--laplace_lambda', type=float, default = 0.1)
parser.add_argument('--negative_sample', type = int, default = 10)
parser.add_argument('--valid_step', type = int, default = 20)
parser.add_argument('--weight_decay', type = float, default = 0.001)
parser.add_argument('--device', type = str, default = 'cuda:0')
args = parser.parse_args()

embed_size = args.embed_size
user_batch = args.user_batch
lr = args.lr
# device = torch.device('cpu')
if args.device != 'cpu':
    device = torch.device('cuda:0')

def processing_valid_data(valid_data):
    res = []
    for key in valid_data.keys():
        if len(valid_data[key]) > 0:
            for ratings in valid_data[key]:
                item, rate, _ = ratings
                res.append((int(key), int(item), rate))
    return np.array(res)

def loss(server, valid_data):
    label = valid_data[:, -1]
    predicted = server.predict(valid_data)
    mae = sum(abs(label - predicted)) / len(label)
    rmse = math.sqrt(sum((label - predicted) ** 2) / len(label))
    return mae, rmse

# read data
data_file = open('../data/' + args.data + '_FedMF.pkl', 'rb')
[train_data, valid_data, test_data, user_id_list, item_id_list, social] = pickle.load(data_file)
data_file.close()
valid_data = processing_valid_data(valid_data)
test_data = processing_valid_data(test_data)
sys.stdout = Logger("ciao_d_32_lr_0.2.log", sys.stdout)
print('数据集: {}, lr:{}, batch_size:{}, embedding_size:{}, laplace_lambda:{}, negative_sample:{}, valid_step:{}'.format(args.data, args.lr, args.user_batch,
                                                                                                                         args.embed_size, args.laplace_lambda, args.negative_sample, args.valid_step))

# build user_list
rating_max = -9999
rating_min = 9999
user_list = []
for u in user_id_list:
    ratings = train_data[u]
    items = []
    rating = []
    for i in range(len(ratings)):
        item, rate, _  = ratings[i]
        items.append(item)
        rating.append(rate)

    if len(rating) > 0:
        rating_max = max(rating_max, max(rating))
        rating_min = min(rating_min, min(rating))
    user_list.append(user(u, items, rating, list(social[u]), embed_size, args.clip, args.laplace_lambda, args.negative_sample))

# build server
server = server(user_list, user_batch, user_id_list, item_id_list, embed_size, lr, device, rating_max, rating_min, args.weight_decay)
count = 0

# train and evaluate
rmse_best = 9999
while 1:
    for i in range(args.valid_step):
        server.train()
    print('valid')
    mae, rmse = loss(server, valid_data)
    print('valid mae: {}, valid rmse:{}'.format(mae, rmse))
    if rmse < rmse_best:
        rmse_best = rmse
        count = 0
        mae_test, rmse_test = loss(server, test_data)
    else:
        count += 1
    if count > 5:
        print('not improved for 5 epochs, stop trianing')
        break
    # if count == 3:
    #     print('not improved for 3 epochs, set lr=0.1')
    #     lr = 0.1
    # elif count == 4:
    #     print('not improved for 4 epochs, set lr=0.05')
    #     lr = 0.05
    # elif count == 5:
    #     print('not improved for 5 epochs, set lr=0.01')
    #     lr = 0.01
    # elif count > 5:
    #     print('not improved for 5 epochs, stop trianing')
    #     break
print('final test mae: {}, test rmse: {}'.format(mae_test, rmse_test))


# if __name__ == "__main__":
#     train_data = {0: [(0, 2, -1), (2, 2, -1)], 1: [(0, 1, -1), (2, 1, -1), (4, 1, -1), (5, 2, -1)],
#                   2: [(0, 5, -1)], 3: [(0, 3, -1), (5, 3, -1)], 4: [(1, 4, -1)], 5: [(2, 5, -1)],
#                   6: [(2, 3, -1), (4, 2, -1)]}
#     valid_data = {2: [(1, 4, -1)], 4: [(1, 4, -1)], 6: [(5, 3, -1)]}
#     test_data = {0: [(3, 3, -1)], 5: [(3, 4, -1)]}
#     social = {0: {2, 3, 4, 5}, 1: {2, 3}, 2: {0, 1, 3}, 3: {0, 1, 2}, 4: {0, 5}, 5: {0, 4, 6}, 6: {5}}
#     valid_data = processing_valid_data(valid_data)
#     test_data = processing_valid_data(test_data)
#     user_id_list = [0, 1, 2, 3, 4, 5, 6]
#     item_id_list = [0, 1, 2, 3, 4, 5]
#
#     # build user_list
#     rating_max = -9999
#     rating_min = 9999
#     user_list = []
#     for u in user_id_list:
#         ratings = train_data[u]
#         items = []
#         rating = []
#         for i in range(len(ratings)):
#             item, rate, _ = ratings[i]
#             items.append(item)
#             rating.append(rate)
#
#         if len(rating) > 0:
#             rating_max = max(rating_max, max(rating))
#             rating_min = min(rating_min, min(rating))
#         user_list.append(
#             user(u, items, rating, list(social[u]), embed_size, args.clip, args.laplace_lambda, args.negative_sample))
#
#     # build server
#     server = server(user_list, user_batch, user_id_list, item_id_list, embed_size, lr, device, rating_max, rating_min, args.weight_decay)
#     count = 0
#
#     # train and evaluate
#     rmse_best = 9999
#     while 1:
#         for i in range(args.valid_step):
#             server.train()
#         print('valid')
#         mae, rmse = loss(server, valid_data)
#         print('valid mae: {}, valid rmse:{}'.format(mae, rmse))
#         if rmse < rmse_best:
#             rmse_best = rmse
#             count = 0
#             mae_test, rmse_test = loss(server, test_data)
#         else:
#             count += 1
#         if count > 5:
#             print('not improved for 5 epochs, stop trianing')
#             break
#     print('final test mae: {}, test rmse: {}'.format(mae_test, rmse_test))
