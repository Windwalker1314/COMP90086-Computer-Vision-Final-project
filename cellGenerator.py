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


class CellPartition():
    def __init__(self,cx,cy,w,h,t1,coords):
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h
        self.t1 = t1
        self.coords = coords
        
        domain = Rect(self.cx, self.cy, self.w, self.h)
        self.qtree = QuadTree(domain,t1)
        for p in self.coords:
            self.qtree.insert(p)
    
    def draw(self,ax):
        self.qtree.draw(ax)
        ax.scatter(self.coords[:,0],self.coords[:,1],s=1)
    
    def cellsInfo(self):
        self.cells = []
        self.__visit_qtree(self.qtree)
        
        point2pointid = {}
        for i in range(len(self.coords)):
            point2pointid[(self.coords[i][0],self.coords[i][1])] = i
        
        point2cellid = {}
        cellid2cellcenter = {}
        cellid2pointids = {}
        for i in range(len(self.cells)):
            pids = []
            for p in self.cells[i].points:
                point2cellid[(p[0],p[1])] = i
                pids.append(point2pointid[(p[0],p[1])])
            cellid2pointids = pids
            cellid2cellcenter[i] = np.mean(self.cells[i].points,axis=0)
        return {"point2cellid": point2cellid, 
                "point2pointid":point2pointid,
                "cellid2cellcenter":cellid2cellcenter,
                "cellid2pointids":cellid2pointids,
                "num_cells":len(self.cells),
                "coords":self.coords}
    
    def __visit_qtree(self,node):
        if not node.divided:
            if len(node)>0:
                self.cells.append(node)
            return node
        else:
            self.__visit_qtree(node.tr)
            self.__visit_qtree(node.tl)
            self.__visit_qtree(node.br)
            self.__visit_qtree(node.bl)
        
partition = CellPartition(0,0,400,400,32,xy)
cellinfo = partition.cellsInfo()

# save and load cell information
import pickle
with open('cell120.pkl', 'wb') as fp:
    pickle.dump(cellinfo, fp, pickle.HIGHEST_PROTOCOL)

with open('cell120.pkl', 'rb') as fp:
    cellinfo = pickle.load(fp)
    print(cellinfo.keys())

# plot the partition
fig,ax = plt.subplots()
partition.draw(ax)
plt.show()

# map a (x,y) coordinate to its cell ID
point2cellid = cellinfo['point2cellid']
# map a (x,y) coordinate to its location ID
point2pointid = cellinfo['point2pointid']

def generateCellID(row):
    return point2cellid[(row.x,row.y)]

def generateLocationID(row):
    return point2pointid[(row.x,row.y)]

df['cells1']     = df.apply(lambda row: generateCellID(row), axis=1)
df['locationID'] = df.apply(lambda row: generateLocationID(row), axis=1)
df['filename']   = df.apply(lambda row: row.id+'.jpg',axis=1) 

df.to_csv("train_with_cells.csv",index=False)
