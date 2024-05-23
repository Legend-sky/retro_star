import os
import numpy as np
import logging
from retro_star.alg.mol_tree import MolTree


def molstar(target_mol, target_mol_id, starting_mols, expand_fn, value_fn,  #expand_fn是MLP模型，返回预测的反应物，得分和模板
            iterations, viz=False, viz_dir=None):
    mol_tree = MolTree(
        target_mol=target_mol,
        known_mols=starting_mols,   #origin_dict.csv已知库中的所有分子
        value_fn=value_fn
    )

    i = -1

    if not mol_tree.succ:                       #结束条件，所有分子都在已知分子库中，mol in self.known_mols
        for i in range(iterations):
            scores = []
            for m in mol_tree.mol_nodes:        #遍历分子节点
                if m.open:                      #open默认为True，表示该节点还未被扩展
                    scores.append(m.v_target()) #如果父节点不为空，返回父节点的v_target,否则返回节点的self.value(默认为init_value)
                else:
                    scores.append(np.inf)       #节点已经被扩展，则返回无穷大
            scores = np.array(scores)

            if np.min(scores) == np.inf:        #全部分子节点都已扩展并且都在已知库中，跳出循环
                logging.info('No open nodes!')
                break

            metric = scores

            mol_tree.search_status = np.min(metric)         #需要被扩展的节点的价值
            m_next = mol_tree.mol_nodes[np.argmin(metric)]  #下一个需要被扩展的分子节点的索引，argmin返回索引值
            assert m_next.open                              #若该节点未被扩展

            result = expand_fn(m_next.mol)                  #扩展该节点，调用one_step函数，返回 {'reactants':reactants,'scores' : scores,'template' : templates}

            if result is not None and (len(result['scores']) > 0):
                reactants = result['reactants']
                scores = result['scores']
                costs = 0.0 - np.log(np.clip(np.array(scores), 1e-3, 1.0))  #np.clip，将scores中的值限定在1e-3到1.0之间，反应的花费，scores越低，costs越大
                # costs = 1.0 - np.array(scores)
                if 'templates' in result.keys():            #？
                    templates = result['templates']
                else:
                    templates = result['template']

                reactant_lists = []
                for j in range(len(scores)):
                    reactant_list = list(set(reactants[j].split('.')))
                    reactant_lists.append(reactant_list)

                assert m_next.open
                succ = mol_tree.expand(m_next, reactant_lists, costs, templates)    #扩展树，并返回是否找到的变量

                if succ:
                    break

                # found optimal route
                if mol_tree.root.succ_value <= mol_tree.search_status:              #？
                    break

            else:
                mol_tree.expand(m_next, None, None, None)
                logging.info('Expansion fails on %s!' % m_next.mol)

        logging.info('Final search status | success value | iter: %s | %s | %d'
                     % (str(mol_tree.search_status), str(mol_tree.root.succ_value), i+1))

    best_route = None
    if mol_tree.succ:
        best_route = mol_tree.get_best_route()
        assert best_route is not None

    mols=[]
    for i in range(len(best_route.mols)):
        mol = best_route.mols[i]
        mols.append(mol)

    if viz:
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)

        if mol_tree.succ:
            if best_route.optimal:
                f = '%s/mol_%d_route_optimal' % (viz_dir, target_mol_id)
            else:
                f = '%s/mol_%d_route' % (viz_dir, target_mol_id)
            best_route.viz_route(f)

        f = '%s/mol_%d_search_tree' % (viz_dir, target_mol_id)
        mol_tree.viz_search_tree(f)

    return mol_tree.succ, (best_route, i+1,mols)
