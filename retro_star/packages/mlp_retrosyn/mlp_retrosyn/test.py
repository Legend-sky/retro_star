import numpy as np
from mlp_retrosyn.mlp_policies import load_parallel_model , preprocess
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import torch
import torch.nn.functional as F
import os
from rdchiral.main import rdchiralRunText, rdchiralRun

fp_dim=2048
dirpath = 'E:/Learning_materials/Yanyi_xia/CSAI/retro_star-master/retro_star'
state_path=dirpath+'/one_step_model/saved_rollout_state_1_2048.ckpt'
template_path=dirpath+'/one_step_model/template_rules_1.dat'


x='COC1=CC=C(C=C1)N2C3=C(CCN(C3=O)C4=CC=C(C=C4)N5CCCCC5=O)C(=N2)C(=O)N'
database=[]

arr = preprocess(x, fp_dim)    #计算分子指纹

arr = np.reshape(arr,[-1, arr.shape[0]])    #arr.shape[0]=2048

arr = torch.tensor(arr, dtype=torch.float32)

net, idx2rules = load_parallel_model(state_path,template_path, fp_dim)

net.eval()
preds = net(arr)               #通过神经网络的预测

preds = F.softmax(preds,dim=1)      #按行进行softmax函数

probs, idx = torch.topk(preds,k=3)   #preds中前k=3大的预测

rule_k = [idx2rules[id] for id in idx[0].numpy().tolist()] #预测出的前k个最有可能反应的规则

result=[]
result.append(x)
reactants = []
scores = []
templates = []
for i , rule in enumerate(rule_k):
    out1 = []
    try:
        out1 = rdchiralRunText(rule, x) #返回根据规则最有可能分解成的分子结果
        #print('第',i+1,'条规则:\n',rule)
        #print('第',i+1,'条规则下可能生成的结果：\n',out1,'\n')
        # out1 = rdchiralRunText(rule, Chem.MolToSmiles(Chem.MolFromSmarts(x)))
        if len(out1) == 0: continue     #没有结果，则跳出本次循环
        # if len(out1) > 1: print("more than two reactants."),print(out1)
        out1 = sorted(out1)
        for reactant in out1:
            reactants.append(reactant)
            result.append(reactant)
            scores.append(probs[0][i].item()/len(out1))
            templates.append(rule)
        result.append(scores[i])
        database.append(result)
        print(result,'\n')
        result=[]
        result.append(x)
    # out1 = rdchiralRunText(x, rule)
    except ValueError:
        pass
print(database)
