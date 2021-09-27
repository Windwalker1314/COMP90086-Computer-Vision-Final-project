# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 05:02:59 2021

@author: Windwalker
"""
import numpy as np

class Point:
    def __init__(self,x,y,payload=None):
        self.x, self.y = x, y
        self.payload = payload
    
    def __str__(self):
        return 'P({:,3f},{:.3f})'.format(self.x,self.y)
    
    def distance_to(self,p, dist_type = "E"):
        if isinstance(p,Point):
            to_x, to_y = p.x, p.y
        else:
            to_x, to_y = p
        
        if dist_type == "E":
            # Euclidean distance
            return np.hypot(to_x-self.x, to_y-self.y)
        else:
            # Manhattan distance
            return np.abs(to_x-self.x)+np.abs(to_y-self.y)
        
class Rect:
    def __init__(self,cx,cy,w,h):
        self.cx, self.cy, self.w, self.h = cx,cy,w,h
        self.left   = cx-w/2
        self.right  = cx+w/2
        self.top    = cy+h/2
        self.bottom = cy-h/2
    
    def __str__(self):
        return '({:.3f},{:.3f},{:.3f},{:.3f})'.format(self.right,self.top,self.left,self.bottom)
    
    def contains(self, p):
        if isinstance(p,Point):
            p_x, p_y = p.x, p.y
        else:
            p_x, p_y = p
        
        return (p_x <= self.right and 
                p_x >= self.left and 
                p_y <= self.top and 
                p_y >= self.bottom)
    
    def intersects(self, rect):
        assert isinstance(rect,Rect)
        return not (rect.left > self.right or
                    rect.right < self.left or
                    rect.top < self.bottom or
                    rect.bottom > self.top)
    
    def draw(self,ax, c='k',lw=1, **kwargs):
        xmin, ymin = self.left,self.bottom
        xmax, ymax = self.right,self.top
        ax.plot([xmin,xmax,xmax,xmin,xmin],[ymin,ymin,ymax,ymax,ymin],c=c,lw=lw,**kwargs)

class QuadTree:
    def __init__(self, boundary, t1=5, d=0):
        # t1: maximum number of points in the node can hold
        self.boundary = boundary
        self.t1 = t1
        self.d = d
        self.points = []
        self.divided = False
    
    def divide(self):
        x,y,w,h = self.boundary.cx,self.boundary.cy,self.boundary.w/2,self.boundary.h/2
        self.divided=True
        self.tr = QuadTree(Rect(x+w/2, y+h/2, w, h),self.t1,self.d+1)
        self.tl = QuadTree(Rect(x-w/2, y+h/2, w, h),self.t1,self.d+1)
        self.br = QuadTree(Rect(x+w/2, y-h/2, w, h),self.t1,self.d+1)
        self.bl = QuadTree(Rect(x-w/2, y-h/2, w, h),self.t1,self.d+1)
        for p in self.points:
            if self.tr.insert(p):
                continue
            if self.tl.insert(p):
                continue
            if self.br.insert(p):
                continue
            if self.bl.insert(p):
                continue

    def insert(self, p):
        if not self.boundary.contains(p):
            return False
        if len(self.points) < self.t1:
            self.points.append(p)
            return True
        
        if not self.divided:
            self.divide()
        
        return (self.tr.insert(p) or 
                self.tl.insert(p) or
                self.br.insert(p) or
                self.bl.insert(p))
  
    def __len__(self):
        if self.divided:
            return len(self.tr) + len(self.tl)+len(self.br) + len(self.bl)
        return len(self.points)
    
    def draw(self,ax):
        self.boundary.draw(ax)
        if self.divided:
            self.tr.draw(ax)
            self.tl.draw(ax)
            self.br.draw(ax)
            self.bl.draw(ax)

