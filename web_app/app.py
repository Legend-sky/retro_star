from flask import Flask, render_template, request, send_file
from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np
from mlp_retrosyn.mlp_policies import load_parallel_model , preprocess
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import torch
import torch.nn.functional as F
import os
from rdchiral.main import rdchiralRunText, rdchiralRun


app = Flask(__name__, static_folder='static')

# load model
fp_dim=2048
dirpath = 'E:/Learning_materials/Yanyi_xia/CSAI/retro_star-master/retro_star'
state_path=dirpath+'/one_step_model/saved_rollout_state_1_2048.ckpt'
template_path=dirpath+'/one_step_model/template_rules_1.dat'

def load_model(smiles):
    x=smiles
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
            if len(out1) == 0:      #没有结果，则跳出本次循环
                scores.append(probs[0][i].item())
                database.append(result)
                continue
            # if len(out1) > 1: print("more than two reactants."),print(out1)
            out1 = sorted(out1)
            for reactant in out1:
                reactants.append(reactant)
                result.append(reactant)
                scores.append(probs[0][i].item()/len(out1))
                templates.append(rule)
            result.append(scores[i])
            database.append(result)
            #print(result,'\n')
            result=[]
            result.append(x)
        # out1 = rdchiralRunText(x, rule)
        except ValueError:
            pass
    return database




def getSmiles(database):
    target = database[0][0]
    smiles_list = [item[1:-1] for item in database]
    scores_list = [item[-1] for item in database]
    return target, smiles_list, scores_list


def getPictures(target, smiles_list):
    img_path1 = []
    img_path2 = []
    img_path3 = []
    Draw.MolToImage(Chem.MolFromSmiles(target)).save('./web_app/static/target.jpg')
    for i in range(len(smiles_list)):
        path = "./web_app/static/Solution" + str(i + 1)
        if not os.path.exists(path):
            os.makedirs(path)
            print("Folder created")
        else:
            print("Folder already exists")
        for j in range(len(smiles_list[i])):
            Draw.MolToImage(Chem.MolFromSmiles(smiles_list[i][j])).save(
                path + '/img' + str(j + 1) + '.jpg')
    for i in range(len(smiles_list)):
        for j in range(len(smiles_list[i])):
            if i == 0:
                img_path1.append('Solution1/img' + str(j + 1) + '.jpg')
            elif i == 1:
                img_path2.append('Solution2/img' + str(j + 1) + '.jpg')
            elif i == 2:
                img_path3.append('Solution3/img' + str(j + 1) + '.jpg')
    return img_path1, img_path2, img_path3


# target, smiles_list, scores_list = getSmiles(database)
# getPictures(target, smiles_list)






@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        Smiles = request.form.get('Smiles') # Capture text input data
        print(Smiles)
        # 调用模型得到database
        database=load_model(Smiles)
        print(database)
        
        if database:
            target, smiles_list, scores_list = getSmiles(database)
            print(target)
            print(smiles_list)
            print(scores_list)
            image_paths1, image_paths2, image_paths3 = getPictures(target, smiles_list)
            return render_template('result.html', target=target, smiles_list=smiles_list, scores_list=scores_list,
                                image_paths1=image_paths1, image_paths2=image_paths2, image_paths3=image_paths3)
        else:
            return render_template('wrong.html')
    return render_template('Index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
