# -*- encoding: utf-8 -*-

import os
import argparse
from loguru import logger
import pickle
import pandas as pd
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='data preprocess')
    parser.add_argument('--name', default='Beibei', type=str)
    args = parser.parse_args()

    if os.path.isdir('./result/pre/log/'.format(args.name)) == False:
        os.makedirs('./result/pre/log/'.format(args.name))
  
    logger.add('./result/log/{}.log'.format(args.name), rotation="500 MB", level="INFO")
    logger.info(args)

    view = pd.read_csv('./data/{}/view.csv'.format(args.name), sep='\t').rename(columns={'uid':'user_id', 'sid':'item_id'})[['user_id', 'item_id']]
    cart = pd.read_csv('./data/{}/cart.csv'.format(args.name), sep='\t').rename(columns={'uid':'user_id', 'sid':'item_id'})[['user_id', 'item_id']]
    buy_train = pd.read_csv('./data/{}/buy.train.txt'.format(args.name), sep='\t').rename(columns={'uid':'user_id', 'sid':'item_id'})
    buy_test = pd.read_csv('./data/{}/buy.test.txt'.format(args.name), sep='\t').rename(columns={'uid':'user_id', 'sid':'item_id'})
    logger.info('{} Data Loaded Successfully!'.format(args.name))

    buy = buy_train.append(buy_test)
    whole_users = set(buy['user_id'].unique())
    whole_items = set(buy['item_id'].unique())
    num_whole_users = max(whole_users) + 1
    num_whole_items = max(whole_items) + 1
    logger.info('{} Number of Users: {}'.format(args.name, num_whole_users))
    logger.info('{} Number of Items: {}'.format(args.name, num_whole_items))

    view_positive_item = pd.merge(pd.DataFrame(np.array(range(num_whole_users)), columns=['user_id']), view.groupby('user_id')['item_id'].apply(set).reset_index().rename(columns={'item_id':'positives'}), how='outer', on='user_id')
    user_nan_view = list(view_positive_item[view_positive_item['positives'].isnull().T]['user_id'])
    view_positive_item_list = []
    for i in range(num_whole_users):
        if i in user_nan_view:
            view_positive_item_list.append([])
        else:
            view_positive_item_list.append(list(view_positive_item.iloc[i].tolist()[1]))
    logger.info('{} View Train Data Completed!'.format(args.name))

    cart_positive_item = pd.merge(pd.DataFrame(np.array(range(num_whole_users)), columns=['user_id']), cart.groupby('user_id')['item_id'].apply(set).reset_index().rename(columns={'item_id':'positives'}), how='outer', on='user_id')
    user_nan_cart = list(cart_positive_item[cart_positive_item['positives'].isnull().T]['user_id'])
    cart_positive_item_list = []
    for i in range(num_whole_users):
        if i in user_nan_cart:
            cart_positive_item_list.append([])
        else:
            cart_positive_item_list.append(list(cart_positive_item.iloc[i].tolist()[1]))
    logger.info('{} Cart Train Data Completed!'.format(args.name))

    buy_train_positive_item = buy_train.groupby('user_id')['item_id'].apply(set).reset_index().rename(columns={'item_id':'positives'})
    buy_train_positive_item_list = [list(buy_train_positive_item.iloc[i].tolist()[1]) for i in range(num_whole_users)]
    logger.info('{} Buy Train Completed!'.format(args.name))

    buy_test_positive_item = [buy_test.iloc[i].tolist()[1] for i in range(num_whole_users)]
    with open('./result/pre/{}_test.pkl'.format(args.name),'wb') as save1:
        pickle.dump(buy_test_positive_item, save1, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info('{} Buy Test Completed!'.format(args.name))

    ######################################## Adjacent Matrix ########################################
    view_U_I = np.zeros((num_whole_users, num_whole_items))
    cart_U_I = np.zeros((num_whole_users, num_whole_items))
    buy_U_I = np.zeros((num_whole_users, num_whole_items))

    for i in range(num_whole_users):
        for j in view_positive_item_list[i]:
            view_U_I[i][j] = 1
        for j in cart_positive_item_list[i]:
            cart_U_I[i][j] = 1
        for j in buy_train_positive_item_list[i]:
            buy_U_I[i][j] = 1
    with open('./result/pre/{}_view.pkl'.format(args.name),'wb') as save2:
        pickle.dump(view_U_I, save2, protocol=pickle.HIGHEST_PROTOCOL)
    with open('./result/pre/{}_cart.pkl'.format(args.name),'wb') as save3:
        pickle.dump(cart_U_I, save3, protocol=pickle.HIGHEST_PROTOCOL)
    with open('./result/pre/{}_buy.pkl'.format(args.name),'wb') as save4:
        pickle.dump(buy_U_I, save4, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info('{} User-Item Interaction Adjacent Matrix Completed!'.format(args.name))