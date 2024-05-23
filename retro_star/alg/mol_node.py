import numpy as np
import logging


class MolNode:
    def __init__(self, mol, init_value, parent=None, is_known=False,
                 zero_known_value=True):
        self.mol = mol                  #分子
        self.pred_value = init_value    #分子节点的预测价值，就是用价值网络计算出来的初始值
        self.value = init_value         #节点的价值，初始化为价值网络计算出来的值
        self.succ_value = np.inf        # total cost for existing solution 现存方案的总价值
        self.parent = parent

        self.id = -1
        if self.parent is None:         #没有父节点，该节点为最终的目标分子
            self.depth = 0
        else:
            self.depth = self.parent.depth  # +1？

        self.is_known = is_known
        self.children = []              #该节点的子节点集合
        self.succ = is_known
        self.open = True                #扩展之前：True，拓展之后：False
        if is_known:                    #判断添加进来的节点是否全都在已知库中，全部在：True，不全部在：False
            self.open = False           #如果全部在已知库中了，此时将扩展标志置位已扩展
            if zero_known_value:        #已知节点分子的价值未知，则置位0
                self.value = 0
            self.succ_value = self.value

        if parent is not None:
            parent.children.append(self)

    def v_self(self):
        """
        :return: V_self(self | subtree)
        """
        return self.value

    def v_target(self):
        """
        :return: V_target(self | whole tree)
        """
        if self.parent is None:
            return self.value
        else:
            return self.parent.v_target()

    def init_values(self, no_child=False):
        assert self.open and (no_child or self.children)    #满足assert后的条件就往下运行，否则报错

        new_value = np.inf
        self.succ = False
        for reaction in self.children:
            new_value = np.min((new_value, reaction.v_self()))  #寻找反应子节点的最小value值
            self.succ |= reaction.succ

        v_delta = new_value - self.value
        self.value = new_value              #以反应子节点的最小value值代替自身的value值

        if self.succ:                       #如果已经成功找到了路径，更新succ_value
            for reaction in self.children:
                self.succ_value = np.min((self.succ_value,
                                          reaction.succ_value))

        self.open = False

        return v_delta

    def backup(self, succ):
        assert not self.is_known    #节点没有全都在已知库中

        new_value = np.inf
        for reaction in self.children:  #遍历分子节点的反应子节点，new_value为反应子节点中value最小的
            new_value = np.min((new_value, reaction.v_self()))
        new_succ = self.succ | succ     #按位运算只要有一个是1，结果就为1
        updated = (self.value != new_value) or (self.succ != new_succ)  #判断分子节点的价值与反应字节点的最小的价值是否相等。

        new_succ_value = np.inf
        if new_succ:
            for reaction in self.children:
                new_succ_value = np.min((new_succ_value, reaction.succ_value))
            updated = updated or (self.succ_value != new_succ_value)    #找到了更小的succ_value值得反应节点，需要backup更新

        v_delta = new_value - self.value
        self.value = new_value              #更新为反应子节点中value最小的值
        self.succ = new_succ
        self.succ_value = new_succ_value

        if updated and self.parent:
            return self.parent.backup(v_delta, from_mol=self.mol)

    def serialize(self):
        text = '%d | %s' % (self.id, self.mol)
        # text = '%d | %s | pred %.2f | value %.2f | target %.2f' % \
        #        (self.id, self.mol, self.pred_value, self.v_self(),
        #         self.v_target())
        return text

    def get_ancestors(self):
        if self.parent is None:
            return {self.mol}

        ancestors = self.parent.parent.get_ancestors()
        ancestors.add(self.mol)
        return ancestors