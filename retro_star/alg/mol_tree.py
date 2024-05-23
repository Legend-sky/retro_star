import numpy as np
from queue import Queue
import logging
import networkx as nx
from graphviz import Digraph
from retro_star.alg.mol_node import MolNode
from retro_star.alg.reaction_node import ReactionNode
from retro_star.alg.syn_route import SynRoute


class MolTree:
    def __init__(self, target_mol, known_mols, value_fn, zero_known_value=True):
        self.target_mol = target_mol                #目标分子
        self.known_mols = known_mols                #origin_dict.csv中的所有分子
        self.value_fn = value_fn                    #价值网络
        self.zero_known_value = zero_known_value    #判断节点价值是否已知
        self.mol_nodes = []                         #分子节点集合
        self.reaction_nodes = []                    #反应节点集合

        self.root = self._add_mol_node(target_mol, None)    #添加分子节点
        self.succ = target_mol in known_mols                #迭代结束条件，判断目标分子是否都在已知分子集合中
        self.search_status = 0                              #搜索状态，需要被扩展的节点的价值

        if self.succ:
            logging.info('Synthesis route found: target in starting molecules')

    def _add_mol_node(self, mol, parent):
        is_known = mol in self.known_mols   #判断添加进来的节点是否全都在已知库中

        init_value = self.value_fn(mol)     #使用价值网络计算出来的值作为分子节点的初始值

        mol_node = MolNode(                 #创建分子节点
            mol=mol,
            init_value=init_value,
            parent=parent,
            is_known=is_known,
            zero_known_value=self.zero_known_value
        )
        self.mol_nodes.append(mol_node)     #在整个分子节点的集合中加入这个分子节点
        mol_node.id = len(self.mol_nodes)   #分子节点的id，第几个加进去的id就是几

        return mol_node

    def _add_reaction_and_mol_nodes(self, cost, mols, parent, template, ancestors):
        assert cost >= 0

        for mol in mols:
            if mol in ancestors:    #子节点与父节点相同，返回
                return

        reaction_node = ReactionNode(parent, cost, template)    #创建反应节点，cost为反应的可能性，可能性越大，cost越小
        for mol in mols:
            self._add_mol_node(mol, reaction_node)              #添加分子节点，parent是反应节点
        reaction_node.init_values()
        self.reaction_nodes.append(reaction_node)
        reaction_node.id = len(self.reaction_nodes)

        return reaction_node

    def expand(self, mol_node, reactant_lists, costs, templates):
        assert not mol_node.is_known and not mol_node.children  #分子节点是否都在已知库中，是否有子节点；不都在已知库中并且没有子节点才运行

        if costs is None:      # No expansion results           #cost为空，scores的得分为1，不需要扩展，没有扩展结果
            assert mol_node.init_values(no_child=True) == np.inf
            if mol_node.parent:
                mol_node.parent.backup(np.inf, from_mol=mol_node.mol)   #？
            return self.succ

        assert mol_node.open    #分子节点未被扩展
        ancestors = mol_node.get_ancestors()
        for i in range(len(costs)):
            self._add_reaction_and_mol_nodes(costs[i], reactant_lists[i],
                                             mol_node, templates[i], ancestors) #添加反应节点和分子节点

        if len(mol_node.children) == 0:      #没有有效的扩展结果
            assert mol_node.init_values(no_child=True) == np.inf
            if mol_node.parent:
                mol_node.parent.backup(np.inf, from_mol=mol_node.mol)
            return self.succ

        v_delta = mol_node.init_values()
        if mol_node.parent:
            mol_node.parent.backup(v_delta, from_mol=mol_node.mol)

        if not self.succ and self.root.succ:
            logging.info('Synthesis route found!')
            self.succ = True

        return self.succ

    def get_best_route(self):   #得到最优路径
        if not self.succ:
            return None

        syn_route = SynRoute(
            target_mol=self.root.mol,
            succ_value=self.root.succ_value,
            search_status=self.search_status
        )

        mol_queue = Queue()
        mol_queue.put(self.root)        #创建分子队列mol_queue,首先加入根节点
        while not mol_queue.empty():    #队列不为空时，一直迭代下去
            mol = mol_queue.get()       #获得队列最前端的分子
            if mol.is_known:
                syn_route.set_value(mol.mol, mol.succ_value)
                continue

            best_reaction = None
            for reaction in mol.children:
                if reaction.succ:
                    if best_reaction is None or \
                            reaction.succ_value < best_reaction.succ_value:     #找到反应子节点中succ_value最小的点加入best_reaction
                        best_reaction = reaction
            assert best_reaction.succ_value == mol.succ_value

            reactants = []
            for reactant in best_reaction.children:     #队列中加入最有可能的反应下的所有分子节点
                mol_queue.put(reactant)
                reactants.append(reactant.mol)

            syn_route.add_reaction(     #添加反应节点
                mol=mol.mol,
                value=mol.succ_value,
                template=best_reaction.template,
                reactants=reactants,
                cost=best_reaction.cost
            )

        return syn_route

    def viz_search_tree(self, viz_file):        #绘制整个搜索树图
        G = Digraph('G', filename=viz_file)
        G.attr(rankdir='LR')
        G.attr('node', shape='box')
        G.format = 'pdf'

        node_queue = Queue()
        node_queue.put((self.root, None))
        while not node_queue.empty():
            node, parent = node_queue.get()

            if node.open:
                color = 'lightgrey'
            else:
                color = 'aquamarine'

            if hasattr(node, 'mol'):
                shape = 'box'
            else:
                shape = 'rarrow'

            if node.succ:
                color = 'lightblue'
                if hasattr(node, 'mol') and node.is_known:
                    color = 'lightyellow'

            G.node(node.serialize(), shape=shape, color=color, style='filled')

            label = ''
            if hasattr(parent, 'mol'):
                label = '%.3f' % node.cost
            if parent is not None:
                G.edge(parent.serialize(), node.serialize(), label=label)

            if node.children is not None:
                for c in node.children:
                    node_queue.put((c, node))

        G.render()
