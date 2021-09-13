# -*- encoding: utf-8 -*-

import math
import pickle
import argparse
from loguru import logger
import torch
from torch.utils.data import Dataset, DataLoader

from model import GMF, EHCF

class Read_Data(Dataset):
    def __init__(self, num_whole_users):
        self.num_whole_users = num_whole_users
    def __getitem__(self, index):
        return index
    def __len__(self):
        return self.num_whole_users

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train GMF or EHCF')
    parser.add_argument('--name', default='Beibei', type=str)
    parser.add_argument('--module', default='GMF', type=str)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=2021, type=str)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr', default=0.05, type=float)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--dim_embedding', default=64, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--weight_negative', default=0.1, type=int)
    parser.add_argument('--lambdas', default={'view':0.05, 'cart':0.8, 'buy':0.15}, type=dict)
    parser.add_argument('--mu', default={'para':1e-4, 'embed':1e-3}, type=dict)

    args = parser.parse_args()
    if args.name == 'Taobao':
        args.batch_size = 256
        args.weight_negative = 0.01
        args.lambdas = {'view':1./6, 'cart':4./6, 'buy':1./6}

    torch.cuda.set_device(args.gpu)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if os.path.isdir('./result/main/log/'.format(args.name)) == False:
        os.makedirs('./result/main/log/'.format(args.name))

    logger.add('./result/main/log/{}_{}.log'.format(args.name, args.module), rotation="500 MB", level="INFO")
    logger.info(args)

    logger.info('{} {} Running... ...'.format(args.name, args.module))
    with open('./result/pre/{}_view.pkl'.format(args.name),'rb') as load1:
        view_U_I = torch.Tensor(pickle.load(load1))
    with open('./result/pre/{}_cart.pkl'.format(args.name),'rb') as load2:
        cart_U_I = torch.Tensor(pickle.load(load2))
    with open('./result/pre/{}_buy.pkl'.format(args.name),'rb') as load3:
        buy_U_I = torch.Tensor(pickle.load(load3))
    with open('./result/pre/{}_test.pkl'.format(args.name),'rb') as load4:
        test = pickle.load(load4)
    logger.info('{} {} Data Loaded Successfully!'.format(args.name, args.module))

    num_whole_users = view_U_I.shape[0]
    num_whole_items = view_U_I.shape[1]

    dataset = Read_Data(num_whole_users=num_whole_users)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size, pin_memory=True)

    ######################################## Train (GPU) ########################################
    if args.module == 'GMF':
        model = GMF(num_users=num_whole_users, num_items=num_whole_items, dim_embedding=args.dim_embedding).cuda()
    if args.module == 'EHCF':
        model = EHCF(num_users=num_whole_users, num_items=num_whole_items, dim_embedding=args.dim_embedding).cuda()

    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, initial_accumulator_value=1e-8)
    for epoch in range(1, args.epochs+1):
        for batch_data in dataloader:
            batch_users = batch_data
            model.forward(batch_users=batch_users.cuda(), \
                          whole_items=torch.LongTensor(range(num_whole_items)).cuda(), \
                          dropout_ration=args.dropout)
            batch_loss = model.compute_loss(batch_view=view_U_I[batch_users].cuda(), \
                                            batch_cart=cart_U_I[batch_users].cuda(), \
                                            batch_buy=buy_U_I[batch_users].cuda(), \
                                            weight_negative=args.weight_negative, lambdas=args.lambdas, mu=args.mu)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
        logger.info('{} {} Epoch [{}/{}]'.format(args.name, args.module, epoch, args.epochs))

    ######################################## Test (CPU) ########################################
        if epoch % 10 == 0:
            scores = []
            for step in range(0, int(num_whole_users/args.batch_size)+1):
                start = step * args.batch_size
                end = (step+1) * args.batch_size
                if end >= num_whole_users:
                    end = num_whole_users
                with torch.no_grad():
                    model.forward(batch_users=torch.LongTensor(range(start,end)).cuda(), \
                                  whole_items=torch.LongTensor(range(num_whole_items)).cuda(), \
                                  dropout_ration=0)
                    scores += model.likelihood_buy.cpu().tolist()
            logger.info('{} {} Prediction Completed!'.format(args.name, args.module))
            
            scores_ = torch.where(buy_U_I==0, torch.Tensor(scores), torch.ones_like(torch.Tensor(scores))*-1e10)
            scores = []
            topk_indices = torch.topk(input=scores_, k=200, dim=1, largest=True, sorted=True)[1].tolist()
            count_hr_200 = 0
            count_ndcg_200 = 0
            count_hr_10 = 0
            count_ndcg_10 = 0
            count_hr_50 = 0
            count_ndcg_50 = 0
            count_hr_100 = 0
            count_ndcg_100 = 0
            for user_topk_indices, positive_item_id in zip(topk_indices, test):
                if positive_item_id in user_topk_indices:
                    count_hr_200 += 1
                    idx_200 = user_topk_indices.index(positive_item_id)+1
                    count_ndcg_200 += math.log(2) / math.log(1 + idx_200)
                    if idx_200 <= 10:
                        count_hr_10 += 1
                        idx_10 = idx_200
                        count_ndcg_10 += math.log(2) / math.log(1 + idx_10)
                    if idx_200 <= 50:
                        count_hr_50 += 1
                        idx_50 = idx_200
                        count_ndcg_50 += math.log(2) / math.log(1 + idx_50)
                    if idx_200 <= 100:
                        count_hr_100 += 1
                        idx_100 = idx_200
                        count_ndcg_100 += math.log(2) / math.log(1 + idx_100)
            HR_10 = count_hr_10 / num_whole_users
            NDCG_10 = count_ndcg_10 / num_whole_users
            HR_50 = count_hr_50 / num_whole_users
            NDCG_50 = count_ndcg_50 / num_whole_users
            HR_100 = count_hr_100 / num_whole_users
            NDCG_100 = count_ndcg_100 / num_whole_users
            HR_200 = count_hr_200 / num_whole_users
            NDCG_200 = count_ndcg_200 / num_whole_users

            scores_ = []
            topk_indices = []
            logger.info('{} {} HR@10:{:.4f}, HR@50:{:.4f}, HR@100:{:.4f}, HR@200:{:.4f}'.format(args.name, args.module, HR_10, HR_50, HR_100, HR_200))
            logger.info('{} {} NDCG@10:{:.4f}, NDCG@50:{:.4f}, NDCG@100:{:.4f}, NDCG@200:{:.4f}'.format(args.name, args.module, NDCG_10, NDCG_50, NDCG_100, NDCG_200))