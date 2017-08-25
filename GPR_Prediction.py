from __future__ import print_function
import numpy as np
import tensorflow as tf
from collections import defaultdict
import sys
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
import time
import argparse
import os


def timer(start,end):
       hours, rem = divmod(end-start, 3600)
       minutes, seconds = divmod(rem, 60)
       print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


class Gaussian_Process_Regression(object):

	def __init__(self,num_test,dim_input,dim_output,age_mean,num_data):
       
		self.Y_train = tf.placeholder(tf.float64,shape=(num_data,dim_output))
		self.X_train = tf.placeholder(tf.float64,shape=(num_data,dim_input))
		self.X_test = tf.placeholder(tf.float64,shape=(num_test,dim_input))
		self.Y_test = tf.placeholder(tf.float64,shape=(num_test,dim_output))
		self.age_mean = age_mean

	def restore_model(self):      
		self.restored_sess = tf.Session()
		new_saver = tf.train.import_meta_graph('./logs/GPR_model.meta')
		new_saver.restore(self.restored_sess,tf.train.latest_checkpoint('./logs/'))  
		self.variance_output = tf.get_collection('variance_output')[0]
		self.variance_kernel = tf.get_collection('variance_kernel')[0]
		print('we print variance_output')
		print(self.restored_sess.run(self.variance_output))
		print('we print variance kernel')
		print(self.restored_sess.run(self.variance_kernel))
		self.Kuu= tf.get_collection('sim_matrix')[0]
		print('we restored the variables from the previous model')

	def chol_solve(self,L,X):

		return tf.matrix_triangular_solve(tf.transpose(L),tf.matrix_triangular_solve(L,X),lower=False)

	def chol_solve_reverse(self,L,X):
    
		return tf.transpose(tf.matrix_triangular_solve(tf.transpose(L),tf.matrix_triangular_solve(L,tf.transpose(X)),lower=False))

	def eye(self,N):
        
		return tf.diag(tf.ones(tf.stack([N,]),dtype=tf.float64))

	def condition(self,X):

		return X + self.eye(tf.shape(X)[0]) * 1e-6


	def RBF(self,X1,X2):

                return tf.matmul(X1 * tf.exp(self.variance_kernel),X2,transpose_b=True)



	def build_predict(self,X_new,full_cov=False):
  
		Kx = self.RBF(self.X_train, X_new)
		#Kuu = self.RBF(self.X_train,self.X_train)
		L = tf.cholesky(self.condition(self.Kuu))
		A = tf.matrix_triangular_solve(L, Kx, lower=True)
		V = tf.matrix_triangular_solve(L, self.Y_train)

		fmean = tf.matmul(A, V, transpose_a=True) + self.age_mean

		'''
		if full_cov:
			fvar = self.RBF(X_new,X_new) - tf.matmul(A, A, transpose_a=True) + self.variance_output * self.eye(X_new.shape[0])
		else:
			fvar = tf.diag(tf.diag_part(self.RBF(X_new,X_new) - tf.matmul(A, A, transpose_a=True))+self.variance_output*self.eye(X_new.shape[0]) )
		'''
		
		return fmean

	def session_TF(self,X_testing,X_training,Y_training):

		self.restore_model()
		predictions = self.build_predict(self.X_test)		
		predictions_now = self.restored_sess.run(predictions,feed_dict={self.X_test:X_testing,self.X_train:X_training,self.X_train:X_training,self.Y_train:Y_training})
		print predictions_now		

if __name__=='__main__':


	parser = argparse.ArgumentParser()
	parser.add_argument('--input_feature_path_testing',type=str,help='the absolute path of the file which contains your features dataset for new subjects')
	parser.add_argument('--input_feature_path_training',type=str,help='the absolute path of the file which contains the training features previously used for training the model')
	parser.add_argument('--age_training_path',type=str,help='the absolute path of the file which contains the chrnological age of the subjects used for training')
		
 	args = parser.parse_args()
 
 	X_total_training = np.genfromtxt(args.input_feature_path_training,dtype=np.float64)
 	Y_total_training = np.genfromtxt(args.age_training_path,dtype=np.float64)
 	Y_total_training = Y_total_training.reshape((X_total_training.shape[0],1))
	X_total_testing = np.genfromtxt(args.input_age_path,dtype=np.float64)
	np.random.seed(7)
	perm = np.random.permutation(2001)
	X_total_training = X_total_training[perm,:]
	Y_total_training = Y_total_training[perm,:]
	X_total_training = X_total_training[:1600,:]
	age_mean = np.mean(Y_total_training)

	obiect =  Gaussian_Process_Regression(num_test=X_testing.shape[0],dim_input=X_testing.shape[1],dim_output=Y_testing.shape[1],age_mean=age_mean,num_data=X_training.shape[0])
	obiect.session_TF(X_testing=X_total_testing,X_training=X_total_training,Y_training=Y_total_training)
	


	
