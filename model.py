# -*- encoding: utf-8 -*-

import math
import torch

######################################## GMF ########################################
class GMF(torch.nn.Module):
    def __init__(self, num_users, num_items, dim_embedding):
        super(GMF,self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.dim_embedding = dim_embedding

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.dim_embedding)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.dim_embedding)
        torch.nn.init.normal_(self.embedding_user.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.embedding_item.weight, mean=0, std=0.01)

        stdv = 1./math.sqrt(self.dim_embedding)
        self.weight_view = torch.nn.Parameter(torch.FloatTensor(self.dim_embedding, 1))
        torch.nn.init.uniform_(self.weight_view, a=-stdv, b=stdv)
        self.weight_cart = torch.nn.Parameter(torch.FloatTensor(self.dim_embedding, 1))
        torch.nn.init.uniform_(self.weight_cart, a=-stdv, b=stdv)
        self.weight_buy = torch.nn.Parameter(torch.FloatTensor(self.dim_embedding, 1))
        torch.nn.init.uniform_(self.weight_buy, a=-stdv, b=stdv)

    def forward(self, batch_users, whole_items, dropout_ration):
        self.batch_user_embeddings = torch.nn.functional.dropout(self.embedding_user(batch_users), p=dropout_ration, training=True)
        self.whole_item_embeddings = self.embedding_item(whole_items)

        # (batch_size, num_items, dim_embedding)
        element_product = torch.mul(self.batch_user_embeddings.unsqueeze(1), self.whole_item_embeddings.unsqueeze(0))
        # (batch_size, num_items)
        self.likelihood_view = torch.squeeze(torch.tensordot(element_product, self.weight_view, dims=([2],[0])))
        self.likelihood_cart = torch.squeeze(torch.tensordot(element_product, self.weight_cart, dims=([2],[0])))
        self.likelihood_buy = torch.squeeze(torch.tensordot(element_product, self.weight_buy, dims=([2],[0])))

    def compute_loss(self, batch_view, batch_cart, batch_buy, weight_negative, lambdas, mu):
        # positive loss
        loss_positive_view = torch.sum(torch.mul(torch.pow(1 - self.likelihood_view, 2), batch_view))
        loss_positive_cart = torch.sum(torch.mul(torch.pow(1 - self.likelihood_cart, 2), batch_cart))
        loss_positive_buy = torch.sum(torch.mul(torch.pow(1 - self.likelihood_buy, 2), batch_buy))
        # negative loss
        loss_negative_view = torch.sum(torch.mul(torch.pow(self.likelihood_view, 2), 1-batch_view))
        loss_negative_cart = torch.sum(torch.mul(torch.pow(self.likelihood_cart, 2), 1-batch_cart))
        loss_negative_buy = torch.sum(torch.mul(torch.pow(self.likelihood_buy, 2), 1-batch_buy))

        loss_view = loss_positive_view + weight_negative * loss_negative_view
        loss_cart = loss_positive_cart + weight_negative * loss_negative_cart
        loss_buy = loss_positive_buy + weight_negative * loss_negative_buy
        
        regular_para = torch.norm(self.weight_view) + torch.norm(self.weight_cart) + torch.norm(self.weight_buy)
        regular_embed = torch.norm(self.batch_user_embeddings) + torch.norm(self.whole_item_embeddings)

        loss = lambdas['view'] * loss_view + lambdas['cart'] * loss_cart + lambdas['buy'] * loss_buy + mu['para'] * regular_para + mu['embed'] * regular_embed
        return loss

######################################## EHCF ########################################
class EHCF(torch.nn.Module):
    def __init__(self, num_users, num_items, dim_embedding):
        super(EHCF,self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.dim_embedding = dim_embedding

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.dim_embedding)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.dim_embedding)
        torch.nn.init.normal_(self.embedding_user.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.embedding_item.weight, mean=0, std=0.01)

        # view prediction layer
        self.weight_view = torch.nn.Parameter(torch.FloatTensor(self.dim_embedding, 1))
        torch.nn.init.uniform_(self.weight_view, a=-math.sqrt(1./self.dim_embedding), b=math.sqrt(1./self.dim_embedding))
        # relation transfer weight
        stdv = 3./math.sqrt(self.dim_embedding)
        self.weight_view_to_cart = torch.nn.Parameter(torch.FloatTensor(self.dim_embedding, self.dim_embedding))
        torch.nn.init.uniform_(self.weight_view_to_cart, a=-stdv, b=stdv)
        self.weight_view_to_buy = torch.nn.Parameter(torch.FloatTensor(self.dim_embedding, self.dim_embedding))
        torch.nn.init.uniform_(self.weight_view_to_buy, a=-stdv, b=stdv)
        self.weight_cart_to_buy = torch.nn.Parameter(torch.FloatTensor(self.dim_embedding, self.dim_embedding))
        torch.nn.init.uniform_(self.weight_cart_to_buy, a=-stdv, b=stdv)
        # relation transfer bias
        self.bias_view_to_cart = torch.nn.Parameter(torch.FloatTensor(self.dim_embedding, 1))
        torch.nn.init.uniform_(self.bias_view_to_cart, a=-stdv, b=stdv)
        self.bias_view_to_buy = torch.nn.Parameter(torch.FloatTensor(self.dim_embedding, 1))
        torch.nn.init.uniform_(self.bias_view_to_buy, a=-stdv, b=stdv)
        self.bias_cart_to_buy = torch.nn.Parameter(torch.FloatTensor(self.dim_embedding, 1))
        torch.nn.init.uniform_(self.bias_cart_to_buy, a=-stdv, b=stdv)

    def forward(self, batch_users, whole_items, dropout_ration):
        self.batch_user_embeddings = torch.nn.functional.dropout(self.embedding_user(batch_users), p=dropout_ration, training=True)
        self.whole_item_embeddings = self.embedding_item(whole_items)

        # (batch_size, num_items, dim_embedding)
        element_product = torch.mul(self.batch_user_embeddings.unsqueeze(1), self.whole_item_embeddings.unsqueeze(0))
        # cart & buy prediction layer
        self.weight_cart = torch.mm(self.weight_view_to_cart, self.weight_view) + self.bias_view_to_cart
        self.weight_buy = torch.mm(self.weight_view_to_buy, self.weight_view) + self.bias_view_to_buy \
                        + torch.mm(self.weight_cart_to_buy, self.weight_cart) + self.bias_cart_to_buy
        # (batch_size, num_items)
        self.likelihood_view = torch.squeeze(torch.tensordot(element_product, self.weight_view, dims=([2],[0])))
        self.likelihood_cart = torch.squeeze(torch.tensordot(element_product, self.weight_cart, dims=([2],[0])))
        self.likelihood_buy = torch.squeeze(torch.tensordot(element_product, self.weight_buy, dims=([2],[0])))

    def compute_loss(self, batch_view, batch_cart, batch_buy, weight_negative, lambdas, mu):
        # positive loss
        loss_positive_view = torch.sum(torch.mul(torch.pow(1 - self.likelihood_view, 2), batch_view))
        loss_positive_cart = torch.sum(torch.mul(torch.pow(1 - self.likelihood_cart, 2), batch_cart))
        loss_positive_buy = torch.sum(torch.mul(torch.pow(1 - self.likelihood_buy, 2), batch_buy))
        # negative loss
        loss_negative_view = torch.sum(torch.mul(torch.pow(self.likelihood_view, 2), 1-batch_view))
        loss_negative_cart = torch.sum(torch.mul(torch.pow(self.likelihood_cart, 2), 1-batch_cart))
        loss_negative_buy = torch.sum(torch.mul(torch.pow(self.likelihood_buy, 2), 1-batch_buy))

        loss_view = loss_positive_view + weight_negative * loss_negative_view
        loss_cart = loss_positive_cart + weight_negative * loss_negative_cart
        loss_buy = loss_positive_buy + weight_negative * loss_negative_buy

        regular_para = torch.norm(self.weight_view_to_cart) + torch.norm(self.weight_view_to_buy) + torch.norm(self.weight_cart_to_buy)
        regular_embed = torch.norm(self.batch_user_embeddings) + torch.norm(self.whole_item_embeddings)

        loss = lambdas['view'] * loss_view + lambdas['cart'] * loss_cart + lambdas['buy'] * loss_buy + mu['para'] * regular_para + mu['embed'] * regular_embed
        return loss