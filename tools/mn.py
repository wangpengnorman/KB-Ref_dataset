import torch
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F
import pdb
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
from torch.autograd import Variable
import argparse


class Basenet(nn.Module):
    def __init__(self, config):
        super(Basenet, self).__init__()
        self.w_embed_size = config.vocab_size
        self.text_len = config.text_len
        self.cand_len = config.cand_len
        self.fact_len = config.fact_len
        self.max_episodic = config.max_episodic
        self.q_lstm_dim = config.q_lstm_dim
        self.s_lstm_dim = config.s_lstm_dim
        self.g_dim = 512
        self.g_attnd_dim = 512
        self.o_dim = config.o_dim
        self.l_dim = config.l_dim
        self.s_attnd_dim = 512
        

        self.f_global = models.vgg16(pretrained=True).features
        #self.global = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])

        self.w_embedding = nn.Embedding(self.w_embed_size, self.q_lstm_dim)

        self.f_expression = nn.LSTM(self.q_lstm_dim, self.q_lstm_dim, batch_first=True)
        self.exp_bn = nn.BatchNorm1d(self.q_lstm_dim)
        self.f_fact = nn.LSTM(self.q_lstm_dim, self.s_lstm_dim, batch_first=True)
        self.fact_bn = nn.BatchNorm1d(self.s_lstm_dim)

        self.g_attnd_q = nn.Linear(self.q_lstm_dim, self.g_attnd_dim)
        self.g_attnd_g = nn.Linear(self.g_dim, self.g_attnd_dim)
        self.g_fc = nn.Linear(self.g_attnd_dim, 1)

        self.f_local = nn.Sequential(
            nn.Linear(self.g_dim, self.o_dim),
            nn.BatchNorm1d(self.o_dim),
            nn.ReLU(inplace=True),
        )

        self.f_location = nn.Sequential(
            nn.Linear(5, self.l_dim),
            nn.BatchNorm1d(self.l_dim),
            nn.ReLU(inplace=True),
        )
        self.g1 = nn.Sequential(
            nn.Linear(self.s_lstm_dim*4, config.g1_dim),
        )
        self.g2 = nn.Linear(config.g1_dim, 1)
        self.mn = nn.LSTMCell(self.s_lstm_dim, self.s_lstm_dim)
        self.m_Cell = nn.LSTMCell(self.s_lstm_dim, self.s_lstm_dim)
        self.f_final = nn.Sequential(
            nn.Linear(self.g_dim+self.s_lstm_dim+self.o_dim+self.l_dim, self.q_lstm_dim),
        )

    def attention_image(self, whole, f_expression):
        with torch.no_grad():
            f_global = self.f_global(whole).view(-1, self.g_dim, 7*7).permute(0, 2, 1).contiguous()
        g_attnd_g = self.g_attnd_g(f_global)
        g_attnd_q = self.g_attnd_q(f_expression).unsqueeze(1).expand_as(g_attnd_g)
        g_attnd = self.g_fc(F.tanh(g_attnd_g+g_attnd_q))
        weight = F.softmax(g_attnd, dim=1).unsqueeze(1).squeeze(3)
        return torch.bmm(weight, f_global).squeeze(1)
        
    def candidate_visual(self, locals):
        bs = locals.size()[0]
        locals = locals.view(bs, -1, 3, 224, 224)
        locals = locals.view(-1, 3, 224, 224).contiguous()
        with torch.no_grad():
            f_local = self.f_global(locals)
        f_local = F.avg_pool2d(f_local, kernel_size=(7, 7)).squeeze(2).squeeze(2)
        f_local = self.f_local(f_local).view(bs, -1, self.g_dim)
        return f_local

    def episodic_memory(self, f_facts, f_expression, m, f_mask, ff_mask):
        f_expression = f_expression.unsqueeze(1).expand_as(f_facts)
        m = m.unsqueeze(1).expand_as(f_facts)
        z = torch.cat([f_facts*f_expression, f_facts*m, torch.abs(f_facts-f_expression), torch.abs(f_facts-m)], dim=2).view(-1, self.s_lstm_dim*4)
        Z = self.g2(F.tanh(self.g1(z))).view(-1, self.fact_len).masked_fill_(f_mask, -9999999)
        weights = F.softmax(Z, 1)
        h_pre = Variable(torch.zeros(Z.size()[0], self.s_lstm_dim).cuda())
        c_pre = Variable(torch.zeros(Z.size()[0], self.s_lstm_dim).cuda())
        hs = Variable(torch.zeros(Z.size()[0], 1, self.s_lstm_dim).cuda())
        cs = Variable(torch.zeros(Z.size()[0], 1, self.s_lstm_dim).cuda())
        for i in range(self.fact_len):
            h, c = self.mn(f_facts[:, i, :].squeeze(1), (h_pre, c_pre))
            h_pre =  weights[:, i].unsqueeze(1) * h + (1-weights[:, i]).unsqueeze(1) * h_pre
            c_pre =  c
            hs = torch.cat((hs, h_pre.unsqueeze(1)), 1)
            cs = torch.cat((cs, c_pre.unsqueeze(1)), 1)
        return torch.bmm(ff_mask.view(-1, self.fact_len).unsqueeze(1), hs[:,1:,:]).squeeze(1), torch.bmm(ff_mask.view(-1, self.fact_len).unsqueeze(1), cs[:,1:,:]).squeeze(1)

    def forward(self, whole, expression, e_mask, locals, locations, facts, mask, f_mask, ff_mask, c_mask):
        #f_expression
        bs = expression.size()[0]
        f_expression = F.dropout(self.w_embedding(expression), 0.1)
        self.f_expression.flatten_parameters()
        x, _ = self.f_expression(f_expression)
        f_expression = torch.bmm(e_mask.unsqueeze(1), x).squeeze(1)
        f_expression = self.exp_bn(f_expression)

        #top-down attention
        f_global = self.attention_image(whole, f_expression)

        #f_local
        f_local = self.candidate_visual(locals)	

        #f_location
        f_location = self.f_location(locations.view(-1, 5)).view(bs, -1, 128)

        #f_facts
        f_facts = F.dropout(self.w_embedding(facts), 0.1)
        #pdb.set_trace()
        self.f_fact.flatten_parameters()
        x, _ = self.f_fact(f_facts.contiguous().view(-1, self.text_len, self.s_lstm_dim))
        f_facts = torch.bmm(mask.view(-1, self.text_len).unsqueeze(1), x.view(-1, self.text_len, self.s_lstm_dim))
        f_facts = self.fact_bn(f_facts.squeeze(1).contiguous()).view(-1, self.cand_len*self.fact_len, self.s_lstm_dim).view(-1, self.cand_len, self.fact_len, self.s_lstm_dim)

        #memory network
        f_m = f_expression.unsqueeze(1).expand(bs, self.cand_len, self.q_lstm_dim).contiguous().view(-1, self.q_lstm_dim)
        m = f_m
        f_mask = torch.eq(f_mask.view(-1, self.fact_len), 0)
        for i in range(self.max_episodic):
            h, c = self.episodic_memory(f_facts.view(-1, self.fact_len, self.s_lstm_dim), f_m, m, f_mask, ff_mask)
            m, _ = self.m_Cell(h, (m, c))  
        m = m.view(-1, self.cand_len, self.s_lstm_dim)

        #prediction
        f_global = f_global.unsqueeze(1).expand(bs, self.cand_len, self.g_dim)
        f = self.f_final(torch.cat((m, f_global, f_local, f_location), 2).contiguous().view(-1, self.g_dim+self.s_lstm_dim+self.o_dim+self.l_dim)).contiguous().view(bs, -1, self.q_lstm_dim)
        f_expression = f_expression.unsqueeze(1).expand(bs, self.cand_len, self.q_lstm_dim)
        scores = torch.sum(f_expression * f, dim=2)
        c_mask = torch.eq(c_mask.view(-1, self.cand_len), 0)
        scores = F.softmax(scores.masked_fill_(c_mask, -9999999), dim=1)
        return scores


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--vocab_size', type=int, default=15733)
    argparser.add_argument('--cand_len', type=int, default=10)
    argparser.add_argument('--fact_len', type=int, default=50)
    argparser.add_argument('--text_len', type=int, default=50)
    argparser.add_argument('--max_episodic', type=int, default=5)
    argparser.add_argument('--q_lstm_dim', type=int, default=2048)
    argparser.add_argument('--s_lstm_dim', type=int, default=2048)
    argparser.add_argument('--o_dim', type=int, default=512)
    argparser.add_argument('--l_dim', type=int, default=128)
    argparser.add_argument('--g1_dim', type=int, default=512)
    args = argparser.parse_args()
    net = Basenet(args)
    print(net)
