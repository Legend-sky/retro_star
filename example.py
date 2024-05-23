from retro_star.api import RSPlanner
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import mols2grid
planner = RSPlanner(
    gpu=-1,
    use_value_fn=True,
    iterations=100,
    expansion_topk=50
)

#result = planner.plan('CCCC[C@@H](C(=O)N1CCC[C@H]1C(=O)O)[C@@H](F)C(=O)OC')
#print(result)

#result = planner.plan('CCOC(=O)c1nc(N2CC[C@H](NC(=O)c3nc(C(F)(F)F)c(CC)[nH]3)[C@H](OC)C2)sc1C')
#print(result)

#result = planner.plan('CC(C)c1ccc(-n2nc(O)c3c(=O)c4ccc(Cl)cc4[nH]c3c2=O)cc1')
#print(result)

#result1 = planner.plan('O=C1C2([H])C=C(CO)[C@@H](O)[C@]3(O)[C@@H](O)C(C)=C[C@@]31[C@H](C)C[C@]4([H])[C@@]2([H])C4(C)C')
#print('复杂的天然产物ingenol:',result1)

#result2 = planner.plan('C/C1=C/CC[C@H](C)C(O[C@H]1CC2=C(C)COC2=O)=O')
#print('中等难度：',result2)

#result3 = planner.plan('O=C1C=CC2=C(CC[C@H](CC)[C@@H]2O)O1')
#print('简单难度：',result3['best_routes'])

#result3 = planner.plan('CCC1=C(C=CC(=C1)C(=NOCC2=CC(=C(C=C2)C3CCCCC3)C(F)(F)F)C)CN4CC(C4)C(=O)O')
#result3 = planner.plan('C1CC(=O)NC(=O)C1N2CC3=C(C2=O)C=CC=C3N')
result3 = planner.plan('[CH3:1][C:2]([CH3:3])([CH3:4])[O:5][C:6](=[O:7])[n:15]1[c:14]2[cH:13][cH:12][c:11]([C:9]([CH3:8])=[O:10])[cH:19][c:18]2[cH:17][cH:16]1')
print(result3)
#
# print('分子最优路径：',result3['best_routes'])

# mols=[]
# for smi in result3['best_routes']:
#     mol = Chem.MolFromSmiles(smi)
#     mols.append(mol)
# #mols2grid.display(mols)
# img=Draw.MolsToGridImage(mols,molsPerRow=1,subImgSize=(400,400),legends=[x for x in result3['best_routes']])
# img.save('filename.png')

