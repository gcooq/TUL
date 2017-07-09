# -*- coding:  UTF-8 -*-
'''
Created on Jan 23, 2017
Using the two Pooling struct
@author: gaoqiang
'''
#bidirectional LSTM
from __future__ import division
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn

import tensorflow as tf
import numpy as np
#from tensorflow.python.ops.constant_op import constant

import operator
import sys
import gc
import re
#listdir
from os import listdir

from fileinput import filename
import math
import matplotlib
import matplotlib.pyplot as plt
import os
#
learning_rate=0.00005 #0.00095
train_iters=30 #
dispaly_step=1000
batch_size=1 #
User_List=list() #
keep_prob=tf.placeholder(tf.float32)
#
lambda_loss_amount=0.001
n_input=250 #embedding size
n_hidden=300 #hidden size 
w_2=28
n_classes=201 #clss number 
n_steps=[3]
x=tf.placeholder("float",[batch_size,None,n_input]) #
istate_fw=tf.placeholder("float",[None,2*n_hidden])
istate_bw=tf.placeholder("float",[None,2*n_hidden])
y_out=tf.placeholder("float",[batch_size,n_classes])
#
weights={
         'out':tf.Variable(tf.random_normal([2*n_hidden,n_classes]))
}
biases={
        'out':tf.Variable(tf.random_normal([n_classes]))
}
table_Y={} #
table_X={}
Train_acc=[]
Test_acc=[]
Test_acc5=[]
Test_macro=[]
Test_10=[]
seq_length=tf.placeholder(tf.int32, [None])
def RNN(x,weights,biases,keep_prob): #define RNN
    x=tf.transpose(x,[1,0,2])
    n_splits=n_steps[0]
    fw_lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(n_hidden,forget_bias=1.0,state_is_tuple=True) #forward , state_is_tuple=True
    fw_lstm_cell=tf.nn.rnn_cell.DropoutWrapper(fw_lstm_cell,output_keep_prob=keep_prob) #use dropout
    bw_lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(n_hidden,forget_bias=1.0,state_is_tuple=True) #backward , state_is_tuple=True
    bw_lstm_cell=tf.nn.rnn_cell.DropoutWrapper(bw_lstm_cell,output_keep_prob=keep_prob) #use dropout

    #bidirectional LSTM
    cell_fw=tf.nn.rnn_cell.MultiRNNCell([fw_lstm_cell], state_is_tuple=True)
    cell_bw=tf.nn.rnn_cell.MultiRNNCell([bw_lstm_cell], state_is_tuple=True)
    #dynamic rnn 
    #seq_len=tf.fill([batch_size], constant(seq_len,dtype=tf.int64))
    (outputs,states)=tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,x,dtype=tf.float32,time_major=True,sequence_length=seq_length) #,dtype=tf.float32,time_major=True  ,initial_state_fw=istate_fw,initial_state_bw=istate_bw
    #(outputs, states)=tf.nn.dynamic_rnn(lstm_cell,x,time_major=True,dtype=tf.float32)
    #h_fc1=tf.nn.tanh((tf.matmul(outputs[-1], weights['out']) + biases['out']))
#     fw_output=tf.reshape(outputs[0], [-1,n_hidden])
#     fb_output=tf.reshape(outputs[1], [-1,n_hidden])
    fw_output,fb_output=outputs
    new_outputs=tf.concat(2, outputs)
    val=tf.matmul(new_outputs[-1],weights['out'])+biases['out']
    #val=tf.add(tf.add(tf.matmul(fw_output[-1],weights['out']),tf.matmul(fb_output[-1],weights['out'])), biases['out'])
    
    return val
    #return (tf.matmul(h_fc1, weights_2['out']) + biases_2['out']),outputs[-1]

pred=RNN(x,weights,biases,keep_prob)

cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,y_out))

optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) #
#
correct_pred=tf.equal(tf.arg_max(pred,1),tf.argmax(y_out,1)) #1
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
 
# 
init=tf.initialize_all_variables()

def getXs(): #
    fpointvec=open('data/gowalla_vector_new250d.dat','r') #
#     table_X={}  #
    item=0
    for line in fpointvec.readlines():
       lineArr=line.split()
       if(len(lineArr)<100 or lineArr[0]=='</s>'):
           continue
       item+=1 #
       X=list()
       for i in lineArr[1:]:
           X.append(float(i)) #
       table_X[int(lineArr[0])]=X
    print "point number item=",item 

 
    return table_X
def printXs():
    fuservec_out=open('out_data/gowalla-vector_out.dat','w')#
    out_table=getXs()
    for key in out_table.keys():
        temp=out_table[key]
        fuservec_out.write('%d '%key)
        for i in temp:
            fuservec_out.write('%f '%i)
        fuservec_out.write('\n')
    print 'point vector print end'
def getOneHot(i):
    x=[0]*n_classes
    x[i]=1
    return x
def get_index(userT):
    userT=list(set(userT))
    User_List=userT
    #print userT
    return User_List
def get_mask_index(value,User_List):
#     print User_List #weikong
    return User_List.index(value)
def get_true_index(index,User_List):
    return User_List[index] 
def readtraindata():

    test_T=list()
    test_UserT=list()
    test_lens=list()
    ftraindata=open('data/gowalla_scopus_1104.dat','r') #gowalla_scopus_1006.dat 
    tempT=list()  #
    pointT=list() # 
    userT=list()  #User ID
    seqlens=list() #
    item=0
    for line in ftraindata.readlines():
        lineArr=line.split()
        X=list()
        for i in lineArr:
            X.append(int(i))
        tempT.append(X)
        userT.append(X[0])
        pointT.append(X[1:])
        seqlens.append(len(X)-1) #
        item+=1
    #Test 98481
    Train_Size=20000
    pointT=pointT[:Train_Size]
    userT=userT[:Train_Size]
    seqlens=seqlens[:Train_Size]
    User_List=get_index(userT)
    print 'Count average length of trajectory'
    avg_len=0
    for i in range(len(pointT)):
        avg_len+=len(pointT[i])
    print '--------Average Length=',avg_len/(len(pointT))
    print User_List
    print "Index numbers",len(User_List)
    print "point T",pointT[Train_Size-1]
    flag=0
    count=0;
    temp_pointT=list()
    temp_userY=list()
    temp_seqlens=list()
    User=0 #
    rate=0.1 #10% for test
    for index in range(len(pointT)):
        if(userT[index]!=flag or index==(len(pointT)-1)):
            User+=1
            #split data 
            if(count>1): #
                #print "count",count," ",index
                test_T+=(pointT[int((index-math.ceil(count*rate))):index]) #
                test_UserT+=(userT[int((index-math.ceil(count*rate))):index])   #
                test_lens+=(seqlens[int((index-math.ceil(count*rate))):index])  #
                temp_pointT+=(pointT[int((index-count)):int((index-count*rate))])
                temp_userY+=(userT[int((index-count)):int((index-count*rate))])
                temp_seqlens+=(seqlens[int((index-count)):int((index-count*rate))])
            else:
                temp_pointT+=(pointT[int((index-count)):int((index))])
                temp_userY+=(userT[int((index-count)):int((index))])
                temp_seqlens+=(seqlens[int((index-count)):int((index))])
            count=1; #
            flag=userT[index] #
        else:
            count+=1
             
    pointT=temp_pointT
    userT=temp_userY
    seqlens=temp_seqlens
    print 'training Numbers=',item-1
    print 'div number=',Train_Size
    print 'Train Size=',len(userT),' Test Size=',len(test_UserT),"User numbers=",User
    print test_T[-1]
#     for i in range(len(pointT)):
#         print 'T train',userT[i]," ",pointT[i]
#     for j in range(len(test_T)):
#         print 'Test',test_UserT[j]," ",test_T[j]
     
    return tempT,pointT,userT,seqlens,test_T,test_UserT,test_lens,User_List #
def getPvector(i): #
    return table_X[i]
def getUvector(i):
    return table_Y[i]
def compare(sess,test_T,test_U,allAcc,User_List,iter,filename='out_data/TEST_resultVec.txt'):  # Test
    ftestw=open(filename,'a+')
    ftestw.write("iters:="+str(iter)+" ")
    ftestw.write("allAcc:"+str(allAcc)+" ")
     
    step=0
    acc=0
    AccTop5=0
    AccTop10=0
    #print 'test_U',test_U
    #print 'test_U Size',len(test_U)
    tempU=list(set(User_List))
    #print 'tempU',tempU
    Dic={}   
    #COU_LIST=[[0 for i in range(4)] for i in range(len(tempU))] # UserID,True,False,sum
    for i in range(len(tempU)):
        #print tempU[i]
        #COU_LIST[i][0]=tempU[i]
        Dic[i]=[0,0,0,0] #
    #print COU_LIST
    #print Dic
    while step<len(test_T): #
        #length=len(test_T[step])
        value=list()

        xsx_step=[map(lambda x:getPvector(x), [a for a in test_T[step]])] #
        #xsy_step=[getUvector(test_U[step])] #
        #xsy_step=[getOneHot(test_U[step])] #
        xsy_step=[getOneHot(get_mask_index(test_U[step],User_List))] #
        user_id=get_mask_index(test_U[step],User_List)
        #print user_id
        Dic.get(user_id)[2]+=1 #a+c 
        x_seqlens=[len(test_T[step])]
        nowVec=sess.run(pred,feed_dict={x:xsx_step,y_out:xsy_step,keep_prob:1.0,seq_length:x_seqlens})[0] #
        predictList=np.argpartition(a=-nowVec, kth=5)[:5]
        top10=np.argpartition(a=-nowVec, kth=10)[:10]
        top1=np.argpartition(a=-nowVec, kth=1)[:1]
        Dic.get(user_id)[1]+=1 #a+b 
        for i in range(len(top10)):
            value.append(get_true_index(top10[i],User_List))
        Test_10.append(value)
       # print test_U[step]
        for index in range(0,5):
            if(predictList[index]==get_mask_index(test_U[step],User_List)):
                AccTop5+=1
                break
               # continue
        #if(index<5):
            #AccTop5+=1
        predictPred=sess.run(correct_pred,feed_dict={x:xsx_step,y_out:xsy_step,keep_prob:1.0,seq_length:x_seqlens})
        if(predictPred==1):
            acc+=1
            Dic.get(user_id)[0]+=1 #a 
        step+=1
    #Count Macro-F1
    macro=0
    a=0
    for i in Dic.keys():
        #print i
        if(( Dic.get(i)[1]+ Dic.get(i)[2])>0):
            Dic.get(i)[3]=(2* Dic.get(i)[0])/( Dic.get(i)[1]+ Dic.get(i)[2])
            macro+= Dic.get(i)[3]
            a+= Dic.get(i)[0]
    macro=macro/len(Dic) 
    print 'Dic length',len(Dic)
    try:
        ftestw.write("OUT CONSOLE: ")
         
        ftestw.write("step="+str(step)+" ")
        ftestw.write(" length="+str(len(test_T))+" ")
        ftestw.write(" Accuracy1 numbers:"+str(acc)+" ")
        ftestw.write(" Accuracy5 numbers:"+str(AccTop5)+" ")
        ftestw.write(" Accuracy1:"+str(acc/step)+" ")
        ftestw.write(" Accuracy5:"+str(AccTop5/step)+" ")
        ftestw.write(" Macro-F1:"+str(macro))
        ftestw.write('\n')
    except Exception:
        print 'get error in count acc'
    ftestw.close()
    print "OUT_PUT accuraccy1=",acc
    print "OUT_PUT accuraccy5=",AccTop5
    print "Macro-F1",macro,'A=',a
    Test_acc.append(acc/step)
    Test_acc5.append(AccTop5/step)
    Test_macro.append(macro)
    #print Dic
    return 0
def draw_pic():
    font={'family':'Trajectory',
          'weight':'bold',
          'size':18
    }
    width=12
    height=12
    plt.figure(figsize=(width,height))
    train_axis=np.array(range(1,len(Train_acc)+1,1))
    plt.plot(train_axis,np.array(Train_acc),"b--",label="Train accuracies")
    test_axi=np.array(range(1,len(Test_acc)+1,1))
    plt.plot(train_axis, np.array(Test_acc), "g--", label="Test Top1 accuracies")
    plt.plot(train_axis, np.array(Test_acc5), "b-", label="Test Top5 accuracies")
    plt.plot(train_axis, np.array(Test_macro), "r-", label="Macro-F1")
    plt.title("Trajectory Classification")
    plt.legend(loc='upper right',shadow=True)
    plt.ylabel('Processing(Accuracy values)')
    plt.xlabel('Training iteration')
    plt.show()
def save():
    ftop=open("out_data/top10.dat","w")
 
    for i in range(len(Test_10)):
        for j in range(len(Test_10[i])):
            ftop.write(str(Test_10[i][j])+" ")
        ftop.write("\n")
    ftop.close()
 
def start(n_steps):
    saver=tf.train.Saver()#
    with tf.Session() as sess:
        gc.collect()
        #sess.run(init)
        saver.restore(sess, './out_data/temp_rnn.pkt') #
        tempT,pointT,userT,seqlens,test_T,test_UserT,test_lens,User_List=readtraindata() #初始化参数 Python参数返回一一对应的，不能简单输出一个
        print 'Point Number',len(test_T)+len(pointT)
        print "Index numbers",len(User_List)
        test=list(set(test_UserT))
        print 'User number test:', len(test)
        #print 'User test:', test
        if(os.path.exists('out_data/TEST_resultVec.txt')):os.remove('out_data/TEST_resultVec.txt')
        for i in range(train_iters): #
            step=0;acc=0;allAcc=0
            while step<len(pointT): #
                length=len(pointT[step]); #
                xsx_step=[map(lambda x:getPvector(x), [a for a in pointT[step]])] #
                #xsy_step=[getOneHot(userT[step])]
                xsy_step=[getOneHot(get_mask_index(userT[step],User_List))]
                #xsy_step=[getUvector(userT[step])] #
#                 print "UserT",userT[step]
#                 print "single tra",len(xsx_step),len(xsx_step[0]),len(xsx_step[0][0])
#                 print "xsy_step",xsy_step
                x_seqlens=[seqlens[step]] #
                sess.run(optimizer,feed_dict={x:xsx_step,y_out:xsy_step,keep_prob:0.5,seq_length:x_seqlens})
#                 print "ACC",sess.run(accuracy,feed_dict={x:xsx_step,y_out:xsy_step})
                if(sess.run(correct_pred,feed_dict={x:xsx_step,y_out:xsy_step,keep_prob:0.5,seq_length:x_seqlens})==1):
                    acc+=1
#                 now=sess.run(pred,feed_dict={x:xsx_step,y_out:xsy_step})[0]
#                 print "now",now,len(now)
                if(step%dispaly_step==0):
                    loss=sess.run(cost,feed_dict={x: xsx_step,y_out:xsy_step,keep_prob:0.5,seq_length:x_seqlens})
                    print 'step=',step,' cost=',loss
                    #acc+=sess.run(accuracy, feed_dict={x: xsx_step,y_classes:xsy_step})
                    print "Iter " + str(i) + ", MiniNowItem Loss= " + \
                          "{:.6f}".format(loss)\
                 + ", Training Accuracy= " + \
                      "{:.5f}".format(acc);allAcc+=acc;acc=0;
         
                step+=1
            print "Optimization No:",i,"Finished! allAcc=",allAcc
            print "START TEST ",i
            Train_acc.append(allAcc/len(pointT))
            saver.save(sess, './out_data/temp_rnn.pkt')
             
            compare(sess, test_T, test_UserT,allAcc, User_List,iter=i)#,'out_data/TEST_record'+str(i)+'vector.txt'
            #recordVector(sess,pointT,userT,'out_data/record'+str(i)+'vector.txt')
            print 'record over!'
        print "All Optimization Finished!" 
        fresultw=open('out_data/result_Weights.txt','w')
        fresultb=open('out_data/result_Biases.txt','w')
        rweights=sess.run(weights['out'])
        rbiases=sess.run(biases['out']) 
        for i in range(n_hidden): #
            for j in range(n_classes):
                fresultw.write('%f '% rweights[i][j])
            fresultw.write('\n')       
         
        for i in range(n_classes): #
            fresultb.write('%f '%rbiases[i])    
        fresultw.close()
        fresultb.close()
        #
        #recordVector(sess, pointT, userT)
        print 'record vector over!'
         
if __name__=="__main__":
    print 'getXs()'
    getXs()

    n_steps=[3]
    start(n_steps)
 
    print("Start save Top10 file...")
    save()
    draw_pic()
#     print getUvector(0)
#     print "TEST"
#     readtraindata()
 
 
 
            
