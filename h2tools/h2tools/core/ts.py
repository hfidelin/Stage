import time
import numpy as np
import scipy as sp
from scipy.sparse.linalg import LinearOperator as linop
import matplotlib as plt
import matplotlib.pyplot
import copy


############################## HYPERMATRIX #################################
def hyp(self, block_prec=0):
    #print "WTF!!!"
    #import ipdb; ipdb.set_trace()
    
    giper_m = 0
    giper_n = 0
    Q_nl = 0
    Q_l = 0
    W_nl = 0
    W_l = 0
    Q_hat = 0
    W_hat = 0 
    Q_hat_nl = 0
    W_hat_nl = 0

    col_size = self.factor.col_tree.level[-1]
    row_size = self.factor.row_tree.level[-1]
        
     
        
        
    for i in xrange(col_size):
        if len(self.factor.row_tree.child[i]) == 0:
            if  self.factor.col_coef[i].shape[0] == 0:
                Q_hat += self.factor.col_coef[i].shape[1] 
            else:
                Q_hat += self.factor.col_coef[i].shape[0] 
            Q_l += self.factor.col_coef[i].shape[1] 
    for i in xrange(col_size):
        if len(self.factor.row_tree.child[i]) == 0:    
            if  self.factor.row_coef[i].shape[0] == 0:
                W_hat += self.factor.row_coef[i].shape[1] 
            else:
                W_hat += self.factor.row_coef[i].shape[0] 
            W_l += self.factor.row_coef[i].shape[1] 
    for i in xrange(col_size):
        if  self.col_tree.child[i]!=[] :
            Q_nl += self.factor.col_coef[i].shape[1] 
    for i in xrange(row_size):
        if   self.row_tree.child[i]!=[]:
            W_nl += self.factor.row_coef[i].shape[1] 
   	
   	
   	self.col_nl_list = []
    self.row_nl_list = []
    self.col_l_list = []
    self.row_l_list = []
    self.close_nl = []
        
    
    for i in xrange(col_size):
        if (self.factor.col_tree.child[i]!=[]):
            self.col_nl_list.append(i)
        else:
            self.col_l_list.append(i)
    for i in xrange(row_size):
        if (self.factor.row_tree.child[i]!=[]):
            self.row_nl_list.append(i)
        else:
            self.row_l_list.append(i)
    for i in self.col_nl_list:
        if (self.close_factor.row_tree.close[i]!=[]):
            self.close_nl.append(i)
    giper_m = Q_hat+Q_nl+Q_l+W_nl+W_l
    giper_n = W_hat+Q_nl+Q_l+W_nl+W_l

    
    
    if block_prec == 0: 
    	self.hyper = sp.sparse.lil_matrix((giper_m,giper_n), dtype = self.dtype)
        print "Size of original matrix: ", (W_hat, Q_hat)
        print "Size of SE matrix: ", self.hyper.shape                
    
        time1 = time.time()
    
        time0 = time.time()
        giper_row_ind = Q_hat 
        giper_col_ind =W_hat + W_nl + W_l
        sparse_B(self,giper_row_ind,giper_col_ind)
        print "B time", time.time() - time0
    
    
        time0 = time.time()
        giper_row_ind = Q_hat+Q_nl
        giper_col_ind = W_hat + W_nl + W_l + Q_nl
        size_I = Q_l
        insert_I(self,size_I,giper_row_ind,giper_col_ind)
        print "I time",time.time() - time0
        
        time0 = time.time()
        giper_row_ind = Q_hat+Q_l+Q_nl
        giper_col_ind = W_hat
        sparse_A(self,giper_row_ind,giper_col_ind)
        print "A time",time.time() - time0
        
        time0 = time.time()
        giper_row_ind = Q_hat
        giper_col_ind = W_hat
        from  h2py.core.C import C
        C(self,giper_row_ind,giper_col_ind)
        print "C time",time.time() - time0
        
        time0 = time.time()
        giper_row_ind = Q_hat+Q_nl+Q_l+W_nl
        giper_col_ind = W_hat + W_nl
        size_I = W_l
        insert_I(self,size_I,giper_row_ind,giper_col_ind)
        print "I time",time.time() - time0
        
        #self.for_prec = copy.deepcopy(self.hyper)
    #    insert_plus_I(self,Q_hat,0,0)
     #   sq_C = max(W_nl+W_l,Q_nl+Q_l)
        #insert_plus_I(self,self.hyper.shape[0]-sq_C-Q_hat,sq_C+Q_hat,sq_C+Q_hat)
     #   import ipdb; ipdb.set_trace()
    
        time0 = time.time()
        giper_row_ind = 0
        giper_col_ind = W_hat + W_nl + W_l + Q_nl
        sparse_E(self,giper_row_ind,giper_col_ind)
        print "E time",time.time() - time0
        
        time0 = time.time()
        giper_row_ind = 0
        giper_col_ind = 0
        from  h2py.core.Cl import CL
        CL(self,giper_row_ind,giper_col_ind)
        print "CL time",time.time() - time0
        
        time0 = time.time()
        giper_row_ind = Q_hat+Q_l+Q_nl+W_nl
        giper_col_ind = 0 
        sparse_D(self,giper_row_ind,giper_col_ind)
        print "D time",time.time() - time0
        
        print "Total time",time.time() - time1
        #import scipy.io
        #csr_hyp =  sp.sparse.csr_matrix(self.hyper.T)
        #scipy.io.mmwrite('hyper.mtx',csr_hyp)
        #import ipdb; ipdb.set_trace()
       
        
        return (self.hyper.shape[0], Q_hat)
    else:
    	#mat = copy.deepcopy(self.hyper)
    	self.hyper = sp.sparse.lil_matrix((giper_m,giper_n), dtype = self.dtype)
        print "Size of original matrix: ", (W_hat, Q_hat)
        print "Size of block preconditioner: ", self.hyper.shape                
    
        time1 = time.time()
    
        time0 = time.time()
        giper_row_ind = Q_hat 
        giper_col_ind =W_hat + W_nl + W_l
        sparse_B(self,giper_row_ind,giper_col_ind)
        print "B time", time.time() - time0
    
    
        time0 = time.time()
        giper_row_ind = Q_hat+Q_nl
        giper_col_ind = W_hat + W_nl + W_l + Q_nl
        size_I = Q_l
        insert_I(self,size_I,giper_row_ind,giper_col_ind)
        print "I time",time.time() - time0
        
        time0 = time.time()
        giper_row_ind = Q_hat+Q_l+Q_nl
        giper_col_ind = W_hat
        sparse_A(self,giper_row_ind,giper_col_ind)
        print "A time",time.time() - time0
        
        time0 = time.time()
        giper_row_ind = Q_hat
        giper_col_ind = W_hat
        from  h2py.core.C import C
        C(self,giper_row_ind,giper_col_ind)
        print "C time",time.time() - time0
        
        time0 = time.time()
        giper_row_ind = Q_hat+Q_nl+Q_l+W_nl
        giper_col_ind = W_hat + W_nl
        size_I = W_l
        insert_I(self,size_I,giper_row_ind,giper_col_ind)
        print "I time",time.time() - time0
        
        
        insert_I(self,Q_hat,0,0)

       

        print "Total time",time.time() - time1
        return "Perconditioner built"


def get_child(self,i):
    if  self.factor.col_tree.child[i] != []:
        child = get_child(self,self.factor.col_tree.child[i][0])+get_child(self,self.factor.col_tree.child[i][1])
    else:
        child = [i]
    return child
    
def l_nl_child(self,i):
    if self.factor.col_tree.child[ self.factor.col_tree.child[i][0]] != []:
        if self.factor.col_tree.child[ self.factor.col_tree.child[i][1]] == []:
            return True
    else:
        if self.factor.col_tree.child[ self.factor.col_tree.child[i][1]] != []:    
            return True
    return False
    
    
        


    
def sparse_B(self,row_ind_start,col_ind_start): 
    row_ind = row_ind_start
    col_ind = col_ind_start
    k=0
    rem_l = [[],[]]
    for i in self.row_nl_list:
        row_ind = row_ind_start
        if self.factor.col_coef[i].shape[0]!=0:
            for j in self.row_nl_list:
                if i == j:
                    insert_I(self,self.factor.col_coef[i].shape[1],row_ind,col_ind)
                    row_ind += self.factor.col_coef[i].shape[1]
                else:
                    if  j in self.factor.col_tree.child[i]:
                        if j%2 == 1 :
                            temp = self.factor.col_coef[i][:][0:self.factor.col_coef[j].shape[1]]
                            self.hyper[row_ind : (row_ind+temp.shape[0]),col_ind : (col_ind+temp.shape[1])] = temp
                            row_ind += temp.shape[0]
                        else:
                            temp = self.factor.col_coef[i][:][(self.factor.col_coef[i].shape[0]-self.factor.col_coef[j].shape[1]):self.factor.col_coef[i].shape[0]]
                            self.hyper[row_ind : (row_ind+temp.shape[0]),col_ind : (col_ind+temp.shape[1])] = temp
                            row_ind += temp.shape[0]
                    else:
                        row_ind += self.factor.col_coef[j].shape[1]  
            for j in self.row_l_list:
                if i == j:
                    insert_I(self,self.factor.col_coef[i].shape[1],row_ind,col_ind)
                    row_ind += self.factor.col_coef[i].shape[1]
                else:
                    if  j in self.factor.col_tree.child[i]:
                        if j%2 == 1 :
                            temp = self.factor.col_coef[i][:][0:self.factor.col_coef[j].shape[1]]
                            self.hyper[row_ind : (row_ind+temp.shape[0]),col_ind : (col_ind+temp.shape[1])] = temp
                            row_ind += temp.shape[0]
                        else:
                            temp = self.factor.col_coef[i][:][(self.factor.col_coef[i].shape[0]-self.factor.col_coef[j].shape[1]):self.factor.col_coef[i].shape[0]]
                            self.hyper[row_ind : (row_ind+temp.shape[0]),col_ind : (col_ind+temp.shape[1])] = temp
                            row_ind += temp.shape[0]
                    else:
                        row_ind += self.factor.col_coef[j].shape[1]             
            col_ind += self.factor.col_coef[i].shape[1]
        else:
            for j in self.row_nl_list:
                if i == j:
                    insert_I(self,self.factor.col_coef[i].shape[1],row_ind,col_ind)
                    row_ind += self.factor.col_coef[i].shape[1]
                else:
                    if  j in self.factor.col_tree.child[i]:
                        if (j%2 == 1) and (l_nl_child(self,i) == False):
                            insert_plus_I(self,self.factor.col_coef[i].shape[1],row_ind,col_ind)
                            row_ind += self.factor.col_coef[i].shape[1]
                        if (l_nl_child(self,i)):
                            insert_plus_I(self,self.factor.col_coef[j].shape[1],row_ind,col_ind)
                            row_ind += self.factor.col_coef[j].shape[1]
                            rem_l[0].append(j+1)
                            rem_l[1].append(self.factor.col_coef[j].shape[1])
                    else:
                        row_ind += self.factor.col_coef[j].shape[1]
            for j in self.row_l_list:
                if  j in self.factor.col_tree.child[i]:
                        if (j%2 == 1)  :
                            insert_plus_I(self,self.factor.col_coef[i].shape[1],row_ind,col_ind)
                            row_ind += self.factor.col_coef[i].shape[1]
                        if j in rem_l[0]:
                            insert_plus_I(self,self.factor.col_coef[j].shape[1],row_ind,col_ind+rem_l[1][rem_l[0].index(j)])
                            row_ind += self.factor.col_coef[j].shape[1]
                else:
                        row_ind += self.factor.col_coef[j].shape[1]
            col_ind += self.factor.col_coef[i].shape[1]





    
'''def sparse_C(self,row_ind_start,col_ind_start):
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
            col_ind +=  self.factor.row_coef[i].shape[1]'''
        
               
       
       
       
    
        
def sparse_A(self,row_ind_start,col_ind_start):
    rem_l = -1
    row_ind = row_ind_start
    col_ind = col_ind_start
    rem_l = [[],[]]
    for i in self.col_nl_list:
        col_ind = col_ind_start
        if self.factor.row_coef[i].shape[0] != 0:
            for j in  self.col_nl_list:
                if i == j:
                    insert_I(self,self.factor.row_coef[i].shape[1],row_ind,col_ind)
                    col_ind += self.factor.row_coef[i].shape[1]
                else:   
                    if  j in self.factor.row_tree.child[i]:
                        if j%2 == 1 :
                            temp = self.factor.row_coef[i][:][0:self.factor.row_coef[j].shape[1]].T
                            self.hyper[row_ind : (row_ind+temp.shape[0]),col_ind : (col_ind+temp.shape[1])] = temp
                            col_ind += temp.shape[1]
                        else:
                            
                            temp = self.factor.row_coef[i][:][(self.factor.row_coef[i].shape[0]-self.factor.row_coef[j].shape[1]):self.factor.row_coef[i].shape[0]].T
                            self.hyper[row_ind : (row_ind+temp.shape[0]),col_ind : (col_ind+temp.shape[1])] = temp
                            col_ind += temp.shape[1]
                    else:
                        col_ind +=  self.factor.row_coef[j].shape[1]
            for j in  self.col_l_list:
                if i == j:
                    insert_I(self,self.factor.row_coef[i].shape[1],row_ind,col_ind)
                    col_ind += self.factor.row_coef[i].shape[1]
                else:   
                    if  j in self.factor.row_tree.child[i]:
                         if j%2 == 1 :
                            temp = self.factor.row_coef[i][:][0:self.factor.row_coef[j].shape[1]].T
                            self.hyper[row_ind : (row_ind+temp.shape[0]),col_ind : (col_ind+temp.shape[1])] = temp
                            col_ind += temp.shape[1]
                         else:
                            temp = self.factor.row_coef[i][:][(self.factor.row_coef[i].shape[0]-self.factor.row_coef[j].shape[1]):self.factor.row_coef[i].shape[0]].T
                            self.hyper[row_ind : (row_ind+temp.shape[0]),col_ind : (col_ind+temp.shape[1])] = temp
                            col_ind += temp.shape[1]
                    else:
                        col_ind +=  self.factor.row_coef[j].shape[1]
            row_ind +=self.factor.row_coef[i].shape[1]
        else:
            for j in self.col_nl_list:
                if i == j:
                    insert_I(self,self.factor.row_coef[i].shape[1],row_ind,col_ind)
                    col_ind += self.factor.row_coef[i].shape[1]
                else:   
                    if  j in self.factor.row_tree.child[i]:
                        if j%2 == 1 and ( (l_nl_child(self,i) == False)):
                            insert_plus_I(self,self.factor.row_coef[i].shape[1],row_ind,col_ind)
                            col_ind += self.factor.row_coef[i].shape[1]
                            
                        if (l_nl_child(self,i)):
                            insert_plus_I(self,self.factor.row_coef[j].shape[1],row_ind,col_ind)
                            col_ind += self.factor.row_coef[j].shape[1]
                            rem_l[0].append(j+1)
                            rem_l[1].append(self.factor.row_coef[j].shape[1])
                        
                    else:
                        col_ind += self.factor.row_coef[j].shape[1]
            for j in self.col_l_list:
                if i == j:
                    insert_I(self,self.factor.row_coef[i].shape[1],row_ind,col_ind)
                    col_ind += self.factor.row_coef[i].shape[1]
                else:   
                    if  j in self.factor.row_tree.child[i]:
                        if j%2 == 1 :
                            insert_plus_I(self,self.factor.row_coef[i].shape[1],row_ind,col_ind)
                            col_ind += self.factor.row_coef[i].shape[1]
                        if j in rem_l[0]:
                            insert_plus_I(self,self.factor.row_coef[j].shape[1],row_ind+rem_l[1][rem_l[0].index(j)],col_ind)
                            col_ind += self.factor.row_coef[j].shape[1]
                    else:
                        col_ind += self.factor.row_coef[j].shape[1]
            
            row_ind +=self.factor.row_coef[i].shape[1]
            
                                
                   
def sparse_E(self,row_ind_start,col_ind_start):
    tr = self.factor.col_coef 
    row_ind = row_ind_start
    col_ind = col_ind_start
    for i in self.col_l_list:
        if self.factor.col_coef[i].shape[0] != 0:
            temp = tr[i]
            self.hyper[row_ind : (row_ind+temp.shape[0]),col_ind : (col_ind+temp.shape[1])] = temp
            row_ind += temp.shape[0]
        else:
            temp = np.identity(self.factor.col_coef[i].shape[1])
            self.hyper[row_ind : (row_ind+temp.shape[0]),col_ind : (col_ind+temp.shape[1])] = temp
            row_ind += temp.shape[0]    
        col_ind +=tr[i].shape[1]
                    
      
        
     
    
        
def sparse_D(self,row_ind_start,col_ind_start):
    row_ind = row_ind_start
    col_ind = col_ind_start
    for i in self.row_l_list:
        row_ind = row_ind_start
        for j in self.row_l_list:
            if self.factor.row_coef[i].shape[0] != 0:
                if i == j:
                    temp = self.factor.row_coef[j].T
                    self.hyper[row_ind : (row_ind+temp.shape[0]),col_ind : (col_ind+temp.shape[1])]=temp
                    row_ind += temp.shape[1]
                else:   
                    row_ind += self.factor.row_coef[j].shape[1]
            else:
                if i == j:
                    temp = np.identity(self.factor.col_coef[i].shape[1])
                    self.hyper[row_ind : (row_ind+temp.shape[0]),col_ind : (col_ind+temp.shape[1])]=temp
                    row_ind += temp.shape[1]
                else:   
                    row_ind += self.factor.row_coef[j].shape[1]


        col_ind += self.factor.row_coef[i].shape[0]
        if self.factor.row_coef[i].shape[0] == 0:
            col_ind += self.factor.row_coef[i].shape[1]
        
        
        
                                
    

'''def sparse_CL(self,row_ind_start,col_ind_start):
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
        
    '''
        
    
def insert_I_to_prec(self,size_I,row_ind,col_ind):
    for i in xrange(size_I):
        self.prec[row_ind+i,col_ind+i] = 1
        
def insert_I(self,size_I,row_ind,col_ind):
    for i in xrange(size_I):
        self.hyper[row_ind+i,col_ind+i] = -1  
        
         
def insert_plus_I(self,size_I,row_ind,col_ind):
    for i in xrange(size_I):
        self.hyper[row_ind+i,col_ind+i] = 1
        

    


def hyp_test(self,y,x):
    W = np.zeros((0,1))
    for i in self.row_nl_list:
        W =  np.vstack((W, self.factor.node_answer[i])) 
    for i in self.row_l_list:
        W =  np.vstack((W, self.factor.node_answer[i]))
        
        
        
    Q_l=np.zeros((0,1))
    Q_nl=np.zeros((0,1))
    col_size = self.factor.col_tree.level[-1]    
    for i in xrange(col_size):
        if len(self.factor.row_tree.child[i]) == 0:
            Q_l =  np.vstack((Q_l, self.factor.node_weight[i]))
        else:
            Q_nl =  np.vstack((Q_nl, self.factor.node_weight[i]))
    Q = Q_nl
    Q = np.vstack((Q,Q_l))
    
    
    
    x = x.reshape(x.shape[0],1)
    tree = self.factor.col_tree
    #transfer = self.factor.col_coef            
    size = tree.level[-1]
    #level_count = len(tree.level)-1
    #node_weight = [np.zeros((0, x.shape[1]), dtype = x.dtype) for i in xrange(size)]
    # Loop is query-dependant
    od = []
    tmp_ar = []
    for j in xrange(size):
            if tree.notransition[j]:
                continue
            if len(tree.child[j]) is 0:
                tmp = x[tree.index[j]]
                od.append(j)
                tmp_ar.append(tmp)
                
    '''for i in xrange(level_count-1):
        for j in xrange(tree.level[level_count-i-2], tree.level[level_count-i-1]):
                if tree.notransition[j]:
                    continue
                if len(tree.child[j]) is 0:
                    tmp = x[tree.index[j]]
                    od.append(j)
                    tmp_ar.append(tmp)
                else:
                    tmp = []
                    for k in tree.child[j]:
                        tmp.append(node_weight[k])
                    tmp = np.vstack(tmp)
                if transfer[j].shape[0] is 0:
                    node_weight[j] = tmp
                else:
                    node_weight[j] = transfer[j].T.dot(tmp)'''
    Q_hat = np.zeros((0,1)) 
    for i in xrange(col_size):
    	if i in od:
    		Q_hat = np.vstack((Q_hat,tmp_ar[od.index(i)]))
    		
    		
    		
    		
    y = y.reshape(y.shape[0],1)
    W_hat = copy.deepcopy(y)
    p = 0
    for i in self.row_l_list:
        if self.factor.row_coef[i].shape[0] == 0:
            for j in xrange(self.factor.row_coef[i].shape[1]):
                W_hat[p,0] =  y[self.factor.row_tree.index[i][j],0] 
                p+=1
        else:
            for j in xrange(self.factor.row_coef[i].shape[0]):
                W_hat[p,0] =  y[self.factor.row_tree.index[i][j],0] 
                p+=1
    
    
       
    
    giper_variables_col = Q_hat
    giper_variables_col = np.vstack((giper_variables_col,Q_nl))
    giper_variables_col = np.vstack((giper_variables_col,Q_l))
    giper_variables_col = np.vstack((giper_variables_col,W))
    
    self.r_g_a = W_hat
    self.y = W_hat
    self.x = Q_hat
    self.r_g_a = np.vstack((self.r_g_a, np.zeros((Q_l.shape[0]+Q_nl.shape[0]+W.shape[0],1))))
    
    
    giper_answer  = self.hyper.T.dot(giper_variables_col)
    g_a_s =  giper_answer[Q_l.shape[0]+Q_nl.shape[0]+W.shape[0]:,:]
    for i in xrange(giper_answer.shape[0]):
        if abs(giper_answer[i] -self.r_g_a[i]) > 0.0001:
               print  "i",i,giper_answer[i],self.r_g_a[i],giper_answer[i]-self.r_g_a[i]
               #import ipdb; ipdb.set_trace()
      
                  
    print "H*sol - rh = ",  sp.linalg.norm(giper_answer-self.r_g_a)
    
    #temp = self.hyper[:,5875:8552]
    #save_hyp(self)
    #import ipdb; ipdb.set_trace()
    #solving(self)
    #csr_hyp =  sp.sparse.csr_matrix(self.hyper.T)
    #aaa = sp.sparse.linalg.spsolve(csr_hyp,self.r_g_a)
    #for i in xrange(aaa.shape[0]):
        #if abs(aaa[i] - giper_variables_col[i])> 0.001:
            #print i, aaa[i],giper_variables_col[i]
    #import ipdb; ipdb.set_trace()
    
def info_hyp(self):
    print "Number of nonzeros",self.hyper.nnz/self.hyper.shape[0]
    #import pdb; pdb.set_trace()
    sp_fu = self.hyper.todense()
    print np.linalg.cond(sp_fu)
    inv = np.linalg.inv(sp_fu)
    inv[abs(inv) <= 1e-2 * 1300] = 0
    plt.pyplot.spy(inv.T,markersize=2,precision=0,color = 'green')
    plt.pyplot.show()
    #import pdb; pdb.set_trace()
 

def save_prec(self):
    import scipy.io
    pr =  sp.sparse.csc_matrix(self.for_prec.T)
    scipy.io.mmwrite('pr.mtx',pr)
    
def save_hyp(self):
    import scipy.io
    self.r_g_a.tofile('Prga')
    self.y.tofile('W')
    self.x.tofile('Q')
    csr_hyp =  sp.sparse.csr_matrix(self.hyper.T)
    scipy.io.mmwrite('Phyper.mtx',csr_hyp)
    #import scipy.io
    #E =  scipy.io.mmread('hyper.mtx')
    
def show_hyp(self):       
    plt.pyplot.spy(self.hyper.T,markersize=2,precision=0,color = 'green')
    plt.pyplot.show()       
       # import pdb; pdb.set_trace()  
