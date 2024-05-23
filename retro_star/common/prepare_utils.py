import pickle
import pandas as pd
import logging
from mlp_retrosyn.mlp_inference import MLPModel
from retro_star.alg import molstar

def prepare_starting_molecules(filename):
    logging.info('Loading starting molecules from %s' % filename)

    if filename[-3:] == 'csv':
        starting_mols = set(list(pd.read_csv(filename)['mol']))
    else:
        assert filename[-3:] == 'pkl'
        with open(filename, 'rb') as f:
            starting_mols = pickle.load(f)

    logging.info('%d starting molecules loaded' % len(starting_mols))
    return starting_mols

def prepare_mlp(templates, model_dump):
    logging.info('Templates: %s' % templates)
    logging.info('Loading trained mlp model from %s' % model_dump)
    one_step = MLPModel(model_dump, templates, device=-1)
    return one_step

def prepare_molstar_planner(one_step, value_fn, starting_mols, expansion_topk,
                            iterations, viz=False, viz_dir=None):
    expansion_handle = lambda x: one_step.run(x, topk=expansion_topk)   #运行one_step的MLPModel类下的run函数进行预测反应物，得分和模板，expansion_topk为前k个最大可能性的模板，默认k=10

    plan_handle = lambda x, y=0: molstar(
        target_mol=x,
        target_mol_id=y,
        starting_mols=starting_mols,    #origin_dict.csv中的所有分子
        expand_fn=expansion_handle,
        value_fn=value_fn,
        iterations=iterations,  #500
        viz=viz,                #False
        viz_dir=viz_dir         #'viz'
    )
    return plan_handle
