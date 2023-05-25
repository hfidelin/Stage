import time
import numpy as np
import scipy as sp
from scipy.sparse.linalg import LinearOperator as linop
import matplotlib as plt
import matplotlib.pyplot
import copy
from numba import double, jit,typeof

#fast_C = jit(C,)
#@jit(arg_tupes = [object,int,int])
def C(self,row_ind_start,col_ind_start):
    #import ipdb; ipdb.set_trace()
    row_ind = row_ind_start
    col_ind = col_ind_start 
    test1 = 0
    test2 = 0
    for i in self.row_nl_list:  
        if(self.factor.col_coef[i].shape[1] != 0) :
            row_ind = row_ind_start
            for j in self.row_nl_list:
                if (j in self.factor.row_tree.far[i]) :
                    for counter in xrange(len(self.factor.row_tree.far[i])):
                        if self.factor.row_tree.far[i][counter] == j:
                            temp_1 =  counter
                    temp = self.factor.row_comp_far_matrix[i][temp_1].T                    
                    self.hyper[row_ind : (row_ind+temp.shape[0]),col_ind : (col_ind+temp.shape[1])]=temp
                    row_ind += temp.shape[0]
                else:
                    row_ind += self.factor.col_coef[j].shape[1]
            for j in self.row_l_list:
                if j in self.factor.row_tree.far[i]:
                    for counter in xrange(len(self.factor.row_tree.far[i])):
                        if self.factor.row_tree.far[i][counter] == j:
                            temp_1 =  counter
                    temp = self.factor.row_comp_far_matrix[i][temp_1].T                    
                    self.hyper[row_ind : (row_ind+temp.shape[0]),col_ind : (col_ind+temp.shape[1])]=temp
                    row_ind += temp.shape[0]
                else:
                    row_ind += self.factor.col_coef[j].shape[1]
            col_ind += self.factor.row_coef[i].shape[1]
    for i in self.row_l_list:
        if( self.factor.col_coef[i].shape[1] != 0) :
            row_ind = row_ind_start
            for j in self.row_nl_list:
                if (j in self.factor.row_tree.far[i]):
                    for counter in xrange(len(self.factor.row_tree.far[i])):
                        if self.factor.row_tree.far[i][counter] == j:
                            temp_1 =  counter
                    temp = self.factor.row_comp_far_matrix[i][temp_1].T                    
                    self.hyper[row_ind : (row_ind+temp.shape[0]),col_ind : (col_ind+temp.shape[1])]=temp
                    row_ind += temp.shape[0]
                else:
                    row_ind += self.factor.col_coef[j].shape[1]
            for j in self.row_l_list:
                if j in self.factor.row_tree.far[i]:
                    for counter in xrange(len(self.factor.row_tree.far[i])):
                        if self.factor.row_tree.far[i][counter] == j:
                            temp_1 =  counter
                    temp = self.factor.row_comp_far_matrix[i][temp_1].T                    
                    self.hyper[row_ind : (row_ind+temp.shape[0]),col_ind : (col_ind+temp.shape[1])]=temp
                    row_ind += temp.shape[0]
                    
                else:
                    row_ind += self.factor.col_coef[j].shape[1]
            col_ind +=  self.factor.row_coef[i].shape[1]
        
               
       
   
