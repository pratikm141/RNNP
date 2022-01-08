import numpy as np
import torch
import argparse
import os
import pprint
from base_trainer import Trainer
import time
import os.path as osp
from tqdm import tqdm
import torch.nn.functional as F

_utils_pp = pprint.PrettyPrinter()


def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)

def pprint(x):
    _utils_pp.pprint(x)

def postprocess_args(args):            
    args.save_path = args.save_dir
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    return args


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

def get_command_line_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--episodes_per_epoch', type=int, default=100)
    parser.add_argument('--num_eval_episodes', type=int, default=600)
    parser.add_argument('--model_class', type=str, default='ProtoNet') # None for MatchNet or ProtoNet
    parser.add_argument('--use_euclidean', action='store_true', default=False)    
    parser.add_argument('--backbone_class', type=str, default='Res12')
    parser.add_argument('--dataset', type=str, default='MiniImageNet')
    
    parser.add_argument('--eval_way', type=int, default=5)
    parser.add_argument('--eval_shot', type=int, default=1)
    parser.add_argument('--eval_query', type=int, default=15)
    parser.add_argument('--temperature', type=float, default=1)
     
    # optimization parameters
    parser.add_argument('--orig_imsize', type=int, default=-1) 
    parser.add_argument('--step_size', type=str, default='20')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--init_weights', type=str, default=None)
    
    # usually untouched parameters
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')

    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--nummix', type=int, default=4)
    
    return parser


import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from samplers import CategoriesSampler, RandomSampler, ClassSampler





from protonet import ProtoNet


def get_dataloader(args):


    from mini_imagenet import MiniImageNet as Dataset


    num_device = torch.cuda.device_count()
    num_workers=args.num_workers

    
    testset = Dataset('test', args)
    test_sampler = CategoriesSampler(testset.label,
                             args.num_eval_episodes,#10000
                            args.eval_way, args.eval_shot + args.eval_query)
    test_loader = DataLoader(dataset=testset,
                            batch_sampler=test_sampler,
                            num_workers=args.num_workers,
                            pin_memory=True)    
    args.num_class = testset.num_class
    return test_loader

def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()

def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm

def prepare_model(args):
    model = ProtoNet(args)
    # load pre-trained model (no FC weights)
    if args.init_weights is not None:
        model_dict = model.state_dict()        
        pretrained_dict = torch.load(args.init_weights)['params']
        if args.backbone_class == 'ConvNet':
            pretrained_dict = {'encoder.'+k: v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    return model




def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits

class Eval(Trainer):
    def __init__(self, args):
        super().__init__(args)

        self.test_loader = get_dataloader(args)
        self.model = prepare_model(args) 


    def train(self):
        pass


    def evaluate(self, data_loader):
        pass
    

    def evaluate_test(self, data_loader):
        pass  

    def count_acc_em_soft(self,label, p,shot,way, embed_query,query,gen=2):
        em_dim = embed_query.size(-1)
        new_mean = p

        logits_test = euclidean_metric(embed_query, new_mean)
        lt = []
        nproto = []

        nproto.append(p.clone())
        for j in range(gen):

            sum_exem = 0*p
            cnt_tot = torch.zeros(way,1).cuda()

            for i in range(len(logits_test)):
                lgt = logits_test[i]
                pb = F.softmax(lgt)
                for lb in range(way):
                    sum_exem[lb] += pb[lb].item()*embed_query[i]
                    cnt_tot[lb,0]+= pb[lb].item()

            new_mean = sum_exem/cnt_tot


            nproto.append(new_mean.clone())
            logits_test = euclidean_metric(embed_query, new_mean)


        return lt,nproto

    def count_acc_per_query(self,label,p,shot,way,embed_query,support,logits,extra,alpha=0.5,beta=3,gen=2):

        lt = []

        em_dim = support.size(-1)
        sup_c = support.view(shot,way,em_dim)
        for i in range(way):
            for j in range(shot): 
                tt=torch.randperm(shot)
                for b in range(beta):
                    if(tt[b].item!=j):
                        cand=alpha*sup_c[j,i]+(1-alpha)*sup_c[tt[b].item(),i]
                        lt.append(cand)
                    else:
                        cand=alpha*sup_c[j,i]+(1-alpha)*sup_c[tt[beta].item(),i]
                        lt.append(cand) 

        extra = torch.stack(lt)


        emb_mod = torch.cat((support,extra),dim=0)
        lt,newp= self.count_acc_em_soft(label,p,shot,way,emb_mod,embed_query,gen=2)


        logits_test = euclidean_metric(embed_query, newp[2])
        acc_cor = count_acc(logits_test, label)



        return acc_cor

    def evaluate_test_em_ext_corrupt_40pt_rnnp(self,alpha,beta):

        print('Alpha=',alpha)
        print('Num per sample=',beta)
        ave_acc1 = Averager()

        args = self.args
        # evaluation mode
        self.model.eval()
        record = np.zeros((args.num_eval_episodes, 2)) # loss and acc
        label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)
        label = label.type(torch.LongTensor)
        avg_acc = Averager()
        avg_acc_cor = Averager()
        if torch.cuda.is_available():
            label = label.cuda()

        with torch.no_grad():
            pbar = tqdm(enumerate(self.test_loader, 1))
            for i, batch in pbar:
                if torch.cuda.is_available():
                    data, _ = [_ for _ in batch]
                else:
                    data = batch[0]

                data = data.cuda()
                logits = self.model(data)
                loss = torch.zeros(1)
                acc = count_acc(logits, label)
                avg_acc.add(acc)

                emb = self.model(data,get_feature=True).detach()

                support = emb[:args.eval_shot*args.eval_way]
                query = emb[args.eval_shot*args.eval_way:]

                proto = support.reshape(args.eval_shot, -1, support.shape[-1]).mean(dim=0)


                sup_cor = support.clone()
                lt_cor = []

                num = args.eval_way
                times = int(0.4 * args.eval_shot)
                for tim in range(times):
                    add = torch.randint(1,args.eval_way,(1,)).item()
                    for iii in range(num):
                        tg = (iii+add)%num
                        lt_cor.append(sup_cor[(tim)*args.eval_way + tg])
                lt_cor = torch.stack(lt_cor)
                sup_cor[:times*num] = lt_cor

                proto_cor = sup_cor.reshape(args.eval_shot, -1, sup_cor.shape[-1]).mean(dim=0)
                proto_cor_test = proto_cor
                logits_test = euclidean_metric(query, proto_cor_test)
                acc_cor = count_acc(logits_test, label)
                avg_acc_cor.add(acc_cor)



                extra = 0

                acc_lt = self.count_acc_per_query( label, proto_cor,args.eval_shot,args.eval_way, query,sup_cor,logits,extra,alpha,beta)                    
                ave_acc1.add(acc_lt)


                record[i-1, 0] = loss.item()
                record[i-1, 1] = acc_lt
                pbar.set_description("avg acc orig {:.4f} avg acc corrupt {:.4f} rnnp acc 2 {:.4f}".format(avg_acc.item(),avg_acc_cor.item(),ave_acc1.item()))

        assert(i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:,0])
        va, vap = compute_confidence_interval(record[:,1])

        self.trlog['test_acc'] = va
        self.trlog['test_acc_interval'] = vap


        print('Test acc={:.4f} + {:.4f}\n'.format(
                self.trlog['test_acc'],
                self.trlog['test_acc_interval']))

        return vl, va, vap

    def final_record(self):
        pass



if __name__ == '__main__':
    parser = get_command_line_parser()
    args = postprocess_args(parser.parse_args())

    set_gpu(args.gpu)
    tester = Eval(args)
    tester.evaluate_test_em_ext_corrupt_40pt_rnnp(args.alpha,args.nummix)




