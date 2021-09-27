# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 19:07:41 2021

@author: Windwalker
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from quadtree import QuadTree, Rect
df = pd.read_csv("data/train.csv")
xy = np.array(list(set(tuple(zip(df.x.values,df.y.values)))))

domain = Rect(0,0,400,400)
qtree = QuadTree(domain,32)

for p in xy:
    qtree.insert(p)

fig,ax= plt.subplots()
qtree.draw(ax)
ax.scatter(xy[:,0], xy[:,1],s=1)
plt.show()


cells = []
def visit_qtree(node):
    if not node.divided:
        if len(node)>0:
            cells.append(node)
        return node
    else:
        visit_qtree(node.tr)
        visit_qtree(node.tl)
        visit_qtree(node.br)
        visit_qtree(node.bl)
visit_qtree(qtree)

point2cellid = {}
for i in range(len(cells)):
    for p in cells[i].points:
        point2cellid[(p[0],p[1])] = i

def generateCellID(row):
    return point2cellid[(row.x,row.y)]

df['cell1']=df.apply(lambda row: generateCellID(row), axis=1)


