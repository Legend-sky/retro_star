import json
import os
from tqdm import tqdm
from retro_star.api import RSPlanner
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import mols2grid
import numpy as np
from mlp_retrosyn.mlp_policies import load_parallel_model , preprocess
import torch
import torch.nn.functional as F
from rdchiral.main import rdchiralRunText, rdchiralRun

def load_uspto_50k(file_path):
    """加载USPTO-50k数据集"""
    data=[]
    with open(file_path, 'r') as f:
        for line in f.readlines():
        # 分割每一行数据
            id_, class_, reaction = line.strip().split(',')
        # 将数据加入列表
            tmp = reaction.strip().split('>>')
            data.append(tmp)
    return data

def evaluate_top_k_accuracy(data, k=10):
    """评估top-K准确度"""
    fp_dim=2048
    dirpath = '/home/Flow/Learning/CASI/retro_star/retro_star'
    state_path=dirpath+'/one_step_model/saved_rollout_state_1_2048.ckpt'
    template_path=dirpath+'/one_step_model/template_rules_1.dat'
    net, idx2rules = load_parallel_model(state_path,template_path, fp_dim)

    correct = 0
    total = len(data)

    for reactants, products in tqdm(data):
        try:
            arr = preprocess(products, fp_dim)    #计算分子指纹
            arr = np.reshape(arr,[-1, arr.shape[0]])    #arr.shape[0]=2048
            arr = torch.tensor(arr, dtype=torch.float32)
            net.eval()
            preds = net(arr)               #通过神经网络的预测
            preds = F.softmax(preds,dim=1)      #按行进行softmax函数
            probs, idx = torch.topk(preds,k)    #preds中前k=10大的预测
            rule_k = [idx2rules[id] for id in idx[0].numpy().tolist()] #预测出的前k个最有可能反应的规则
            result=[]
            result.append(products)
            reactants_k = []
            scores = []
            templates = []
            for i , rule in enumerate(rule_k):
                out1 = []
                try:
                    out1 = rdchiralRunText(rule, products) #返回根据规则最有可能分解成的分子结果
                    #print('第',i+1,'条规则:\n',rule)
                    #print('第',i+1,'条规则下可能生成的结果：\n',out1,'\n')
                    # out1 = rdchiralRunText(rule, Chem.MolToSmiles(Chem.MolFromSmarts(x)))
                    if len(out1) == 0: continue     #没有结果，则跳出本次循环
                    # if len(out1) > 1: print("more than two reactants."),print(out1)
                    out1 = sorted(out1)
                    for reactant in out1:
                        reactants_k.append(reactant)
                        scores.append(probs[0][i].item()/len(out1))
                        templates.append(rule)
                except ValueError:
                    pass

            
            # 解析带有原子编号的SMILES字符串,reactants是带有原子编号的SMILES表示
            mol = Chem.MolFromSmiles(reactants, sanitize=False)
            # 删除原子编号
            for atom in mol.GetAtoms():
                atom.ClearProp('molAtomMapNumber')

            # 标准化SMILES字符串
            standard_smiles = Chem.MolToSmiles(mol)
            
            if any([standard_smiles in reactants_k]):
                correct += 1
            # print(standard_smiles,'\n')
            # print(reactants_k,'\n')
        except Exception as e:
            print(f"Failed to plan for {products}: {e}")

    accuracy = correct / total
    return accuracy

def main():
    # 路径配置
    uspto_50k_path = './uspto50k/raw_test.csv'
    
    # 加载USPTO-50k数据集
    data = load_uspto_50k(uspto_50k_path)
    data.pop(0)
    # for reactants, products in tqdm(data[:5]):
    #     print(reactants,'\n')
    #     reactant = reactants.split('.')
    #     print(reactant,'\n')
    # for i in range(3):
    #     print(data[i])
    
    # 评估top-10准确度
    top_k = 10
    accuracy = evaluate_top_k_accuracy(data, k=top_k)
    
    print(f"Top-{top_k} accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
