import numpy as np
from queue import Queue
from graphviz import Digraph
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

class SynRoute:
    def __init__(self, target_mol, succ_value, search_status):
        self.target_mol = target_mol    #目标分子
        self.mols = [target_mol]
        self.values = [None]
        self.templates = [None]
        self.parents = [-1]
        self.children = [None]
        self.optimal = False
        self.costs = {}

        self.succ_value = succ_value    #成功找到时得value
        self.total_cost = 0
        self.length = 0
        self.search_status = search_status
        if self.succ_value <= self.search_status:
            self.optimal = True

    def _add_mol(self, mol, parent_id):
        self.mols.append(mol)
        self.values.append(None)
        self.templates.append(None)
        self.parents.append(parent_id)
        self.children.append(None)

        self.children[parent_id].append(len(self.mols)-1)

    def set_value(self, mol, value):
        assert mol in self.mols

        mol_id = self.mols.index(mol)   #找到该分子的id
        self.values[mol_id] = value     #赋值value

    def add_reaction(self, mol, value, template, reactants, cost):
        assert mol in self.mols

        self.total_cost += cost #累加总反应节点的cost
        self.length += 1        #反应路径长度

        parent_id = self.mols.index(mol)
        self.values[parent_id] = value
        self.templates[parent_id] = template
        self.children[parent_id] = []
        self.costs[parent_id] = cost

        for reactant in reactants:
            self._add_mol(reactant, parent_id)

    def viz_route(self, viz_file):          #绘制搜索路径图
        G = Digraph('G', filename=viz_file) #创建图G
        G.attr('node', shape='box')         #节点node属性，方形
        G.format = 'pdf'                    #导出文件为pdf格式
        mols=[]
        names = []
        for i in range(len(self.mols)):
            name = self.mols[i]
            # if self.templates[i] is not None:
            #     name += ' | %s' % self.templates[i]
            names.append(name)
        
        node_queue = Queue()
        node_queue.put((0,-1))   # target mol idx, and parent idx
        while not node_queue.empty():
            idx, parent_idx = node_queue.get()

            if parent_idx >= 0:
                m1=Chem.MolFromSmiles(names[parent_idx])
                m2=Chem.MolFromSmiles(names[idx])
                mols.append([m1,m2])
                #G.edge(m1, m2, label='cost')
                G.edge(names[parent_idx], names[idx], label='cost')

            if self.children[idx] is not None:
                for c in self.children[idx]:
                    node_queue.put((c, idx))
        
        G.render()

    def serialize_reaction(self, idx):
        s = self.mols[idx]
        if self.children[idx] is None:
            return s
        s += '>%.4f>' % np.exp(-self.costs[idx])
        s += self.mols[self.children[idx][0]]
        for i in range(1, len(self.children[idx])):
            s += '.'
            s += self.mols[self.children[idx][i]]

        return s

    def serialize(self):
        s = self.serialize_reaction(0)
        for i in range(1, len(self.mols)):
            if self.children[i] is not None:
                s += '|'
                s += self.serialize_reaction(i)

        return s