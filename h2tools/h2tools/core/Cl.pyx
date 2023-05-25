import numpy as np
import scipy as sp
import copy

cimport numpy as np
#cimport scipy as sp



def get_child(self,int i):
    if  self.factor.col_tree.child[i] != []:
        child = get_child(self,self.factor.col_tree.child[i][0])+get_child(self,self.factor.col_tree.child[i][1])
    else:
        child = [i]
    return child
        
def  CL(self,row_ind_start,col_ind_start):
    cdef int i,i1,j
    my_cl_mat = copy.deepcopy(self.close_factor.row_matrix)
    my_cl = copy.deepcopy(self.close_factor.row_tree.close)
    row_ind = row_ind_start
    col_ind = col_ind_start
    for i in self.row_nl_list:
        if my_cl[i] != []:
            for i1 in xrange(len(my_cl[i])):
                p = 0
                for j in  get_child(self,i):
                    my_cl[j] .append(my_cl[i][i1])
                    if self.factor.col_coef[j].shape[0] != 0:
                        temp = my_cl_mat[i][i1][p:(p+self.factor.col_coef[j].shape[0]),:]
                        p += self.factor.col_coef[j].shape[0]
                        my_cl_mat[j].append(temp)
                    else:
                        temp = my_cl_mat[i][i1][p:(p+self.factor.col_coef[j].shape[1]),:]
                        p += self.factor.col_coef[j].shape[1]
                        my_cl_mat[j].append(temp)
    for i in self.row_l_list:
        for i1 in xrange(len(my_cl[i])):
            p = 0
            if my_cl[i][i1] in self.close_nl:
                my_cl[i] = my_cl[i] + get_child(self,my_cl[i][i1])
                for j in  get_child(self,my_cl[i][i1]):
                    if self.factor.col_coef[j].shape[0]!=0:
                        temp = my_cl_mat[i][i1][:,p:(p+self.factor.col_coef[j].shape[0])]
                        p += self.factor.col_coef[j].shape[0]
                        my_cl_mat[i].append(temp)
                    else:
                        temp = my_cl_mat[i][i1][:,p:(p+self.factor.col_coef[j].shape[1])]
                        p += self.factor.col_coef[j].shape[1]
                        my_cl_mat[i].append(temp)
    
    for i in self.row_l_list:
        row_ind  = row_ind_start
        if(len(my_cl_mat[i]) != 0):
            for j in self.col_l_list:
                if j in my_cl[i]:
                    
                    temp = my_cl_mat[i][my_cl[i].index(j)].T
                    self.hyper[row_ind : (row_ind+temp.shape[0]),col_ind : (col_ind+temp.shape[1])]=temp
                    row_ind += temp.shape[0]
                else:
                     row_ind += self.factor.col_coef[j].shape[0]
                     if self.factor.col_coef[j].shape[0] == 0:
                              row_ind += self.factor.col_coef[j].shape[1]
        col_ind += self.factor.row_coef[i].shape[0]
        if self.factor.row_coef[i].shape[0] == 0:
            col_ind += self.factor.row_coef[i].shape[1]
        
    
        
   
