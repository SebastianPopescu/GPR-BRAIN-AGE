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
import subprocess
import nibabel as nib

app_dir='/apps/software/brain_age'
app_version='v1.0_18Jan2018'

def timer(start,end):
       hours, rem = divmod(end-start, 3600)
       minutes, seconds = divmod(rem, 60)
       print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

def data_factory(path_gm_data,path_wm_data,matter_involved):
	
	if matter_involved=='both':
        
		### this part is for white matter data
		### get the mask
		wm_mask_object = nib.load('%s/%s/banc_data/masks/wm_mask.nii' %(app_dir, app_version))
		wm_mask_data = wm_mask_object.get_data()

		wm_mask_data_reshaped = wm_mask_data.reshape(np.prod(wm_mask_data.shape),)
		wm_mask_data_full_bool = wm_mask_data_reshaped.astype(bool)

		comanda = ['ls', path_wm_data]
		str_output,plm = subprocess.Popen(comanda,stdout=subprocess.PIPE).communicate()
		
		lista_output = str_output.rsplit('\n')
		lista_output = lista_output[:-1]
		if len(lista_output) ==1:
			comanda_temporara = path_wm_data + '/' + str(lista_output[0])
			temporar_object = nib.load(comanda_temporara)
			temporar_data = temporar_object.get_data()

			temporar_data_reshaped = temporar_data.reshape(np.prod(temporar_data.shape),)
			imagini_wm = temporar_data_reshaped[wm_mask_data_full_bool]
			imagini_wm  =np.asarray(imagini_wm)
		else:

			lista_imagini_wm = []
			for line in lista_output:
				comanda_temporara = path_wm_data + '/' + str(line)
				temporar_object = nib.load(comanda_temporara)
				temporar_data = temporar_object.get_data()

				temporar_data_reshaped = temporar_data.reshape(np.prod(temporar_data.shape),)
				lista_imagini_wm.append(temporar_data_reshaped[wm_mask_data_full_bool])

			imagini_wm = np.stack(lista_imagini_wm)

                ### this part is for grey matter data
                ### get the mask
                gm_mask_object = nib.load('%s/%s/banc_data/masks/gm_mask.nii' %(app_dir, app_version))
                gm_mask_data = gm_mask_object.get_data()

                gm_mask_data_reshaped = gm_mask_data.reshape(np.prod(gm_mask_data.shape),)
                gm_mask_data_full_bool = gm_mask_data_reshaped.astype(bool)

                comanda = ['ls', path_gm_data]                                                                                                                                                       
                str_output,plm = subprocess.Popen(comanda,stdout=subprocess.PIPE).communicate()

                lista_output = str_output.rsplit('\n')
                lista_output = lista_output[:-1]
                if len(lista_output) ==1:
                        comanda_temporara = path_gm_data + '/' + str(lista_output[0])                                                                                                                
                        temporar_object = nib.load(comanda_temporara)
                        temporar_data = temporar_object.get_data()

                        temporar_data_reshaped = temporar_data.reshape(np.prod(temporar_data.shape),)                                                                                                
                        imagini_gm = temporar_data_reshaped[gm_mask_data_full_bool]                                                                                                            
			imagini_gm = np.asarray(imagini_gm)
                else:

                        lista_imagini_gm = []
                        for line in lista_output:
                                comanda_temporara = path_gm_data + '/' + str(line)
                                temporar_object = nib.load(comanda_temporara)
                                temporar_data = temporar_object.get_data()

                                temporar_data_reshaped = temporar_data.reshape(np.prod(temporar_data.shape),)
                                lista_imagini_gm.append(temporar_data_reshaped[gm_mask_data_full_bool])

                        imagini_gm = np.stack(lista_imagini_gm)
		imagini = np.concatenate((imagini_gm,imagini_wm),axis=1)
		print('here we print the dimensions of the testing data')
		print(imagini.shape)
		return imagini,lista_output

	elif matter_involved=='gm':

                ### this part is for grey matter data
                ### get the mask
                gm_mask_object = nib.load('%s/%s/banc_data/masks/gm_mask.nii' %(app_dir, app_version))
                gm_mask_data = gm_mask_object.get_data()

                gm_mask_data_reshaped = gm_mask_data.reshape(np.prod(gm_mask_data.shape),)
                gm_mask_data_full_bool = gm_mask_data_reshaped.astype(bool)

                comanda = ['ls', path_gm_data]
                str_output,plm = subprocess.Popen(comanda,stdout=subprocess.PIPE).communicate()

                lista_output = str_output.rsplit('\n')
                lista_output = lista_output[:-1]
                if len(lista_output) ==1:
                        comanda_temporara = path_gm_data + '/' + str(lista_output[0])
                        temporar_object = nib.load(comanda_temporara)
                        temporar_data = temporar_object.get_data()

                        temporar_data_reshaped = temporar_data.reshape(np.prod(temporar_data.shape),)
                        imagini_gm = temporar_data_reshaped[gm_mask_data_full_bool]
			imagini_gm = np.asarray(imagini_gm)
                else:

                        lista_imagini_gm = []
                        for line in lista_output:
                                comanda_temporara = path_gm_data + '/' + str(line)
                                temporar_object = nib.load(comanda_temporara)
                                temporar_data = temporar_object.get_data()

                                temporar_data_reshaped = temporar_data.reshape(np.prod(temporar_data.shape),)
                                lista_imagini_gm.append(temporar_data_reshaped[wm_mask_data_full_bool])

                        imagini_gm = np.stack(lista_imagini_gm)
		print('here we print the dimensions of the testing data')
		print(imagini_gm.shape)
		return imagini_gm,lista_output

	elif matter_involved=='wm':

                ### this part is for white matter data
                ### get the mask
                wm_mask_object = nib.load('%s/%s/banc_data/masks/wm_mask.nii' %(app_dir, app_version))
                wm_mask_data = wm_mask_object.get_data()

                wm_mask_data_reshaped = wm_mask_data.reshape(np.prod(wm_mask_data.shape),)
                wm_mask_data_full_bool = wm_mask_data_reshaped.astype(bool)

                comanda = ['ls', path_wm_data]
                str_output,plm = subprocess.Popen(comanda,stdout=subprocess.PIPE).communicate()

                lista_output = str_output.rsplit('\n')
                lista_output = lista_output[:-1]
                if len(lista_output) ==1:
                        comanda_temporara = path_wm_data + '/' + str(lista_output[0])
                        temporar_object = nib.load(comanda_temporara)
                        temporar_data = temporar_object.get_data()

                        temporar_data_reshaped = temporar_data.reshape(np.prod(temporar_data.shape),)
                        imagini_wm = temporar_data_reshaped[wm_mask_data_full_bool]
			imagini_wm = np.asarray(imagini_wm)
			imagini_wm = np.reshape(imagini_wm,(1,194831))

                else:

                        lista_imagini_wm = []
                        for line in lista_output:
                                comanda_temporara = path_wm_data + '/' + str(line)
                                temporar_object = nib.load(comanda_temporara)
                                temporar_data = temporar_object.get_data()

                                temporar_data_reshaped = temporar_data.reshape(np.prod(temporar_data.shape),)
                                lista_imagini_wm.append(temporar_data_reshaped[wm_mask_data_full_bool])

                        imagini_wm = np.stack(lista_imagini_wm)
		print('here we print the dimensions of the testing data')
		print(imagini_wm.shape)
		return imagini_wm,lista_output
	else:
		print('error in specification of brain tissues')
		sys.exit();

class Gaussian_Process_Regression(object):

	def __init__(self,num_test,dim_input,dim_output,age_mean,num_data,matter_involved,current_directory):
       
		self.Y_train = tf.placeholder(tf.float64,shape=(num_data,dim_output),name='Y_training')
		self.X_train = tf.placeholder(tf.float64,shape=(num_data,dim_input),name='X_training')
		self.X_test = tf.placeholder(tf.float64,shape=(num_test,dim_input),name='X_testing')
		self.Y_test = tf.placeholder(tf.float64,shape=(num_test,dim_output),name='Y_testing')
		self.age_mean = age_mean
		self.matter_involved = matter_involved
		self.current_directory = current_directory

	def restore_model(self):      
		self.restored_sess = tf.Session()
		if self.matter_involved == 'both':

			new_saver = tf.train.import_meta_graph('%s/%s/GPR-BRAIN-AGE/logs/both_model/GPR_model.meta' %(app_dir, app_version))
			new_saver.restore(self.restored_sess,tf.train.latest_checkpoint('%s/%s/GPR-BRAIN-AGE/logs/both_model/' %(app_dir, app_version)))  

		elif self.matter_involved == 'grey':

			new_saver = tf.train.import_meta_graph('%s/%s/GPR-BRAIN-AGE/logs/gm_model/GPR_model.meta' %(app_dir, app_version))
			new_saver.restore(self.restored_sess,tf.train.latest_checkpoint('%s/%s/GPR-BRAIN-AGE/logs/gm_model/' %(app_dir, app_version)))

		else:

			new_saver = tf.train.import_meta_graph('%s/%s/GPR-BRAIN-AGE/logs/wm_model/GPR_model.meta' %(app_dir, app_version))
			new_saver.restore(self.restored_sess,tf.train.latest_checkpoint('%s/%s/GPR-BRAIN-AGE/logs/wm_model/' %(app_dir, app_version)))

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
		
		if full_cov:
			fvar = self.RBF(X_new,X_new) - tf.matmul(A, A, transpose_a=True) + tf.exp(self.variance_output) * self.eye(X_new.shape[0])
		else:
			fvar = tf.diag_part(self.RBF(X_new,X_new) - tf.matmul(A, A, transpose_a=True)) + tf.exp(self.variance_output) 
		
		return fmean,fvar

	def session_TF(self,X_testing,X_training,Y_training,testing_nifti_names):

		self.restore_model()
		predictions = self.build_predict(self.X_test)		
		[predictions_now,predictions_now_variance] = self.restored_sess.run(predictions,feed_dict={self.X_test:X_testing,self.X_train:X_training,self.Y_train:Y_training})
		print(predictions_now.shape)
		print(predictions_now_variance.shape)

		#### create an output file of the type: name of nifti_file predicted_brain_age confidence_intervals

		text=''
		for i in range(X_testing.shape[0]):
			text+=str(testing_nifti_names[i])+' '+str(predictions_now[i][0])+' ['+str(predictions_now[i][0]-1.96*np.sqrt(predictions_now_variance[i]))+','+str(predictions_now[i][0]+1.96*np.sqrt(predictions_now_variance[i]))+']\n'
		with open(self.current_directory+'/predictions/predicted_brain_age.txt','w') as f:
			f.write(text)
		f.close()

if __name__=='__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--path_gm_data',type=str,help='the absolute path of the folder which contains the nifti files for grey matter data',default='./')
	parser.add_argument('--path_wm_data',type=str,help='the absolute path of the folder which contains the nifti files for white matter data',default='./')
	parser.add_argument('--matter_involved',type=str,help='whether to use white, grey or both types of brain tissue')		
	parser.add_argument('--current_directory',type=str,help='where you currently have the raw nifti files')
 	args = parser.parse_args()
 
 	Y_total_training = np.genfromtxt('%s/%s/training_data/Y_original.txt' %(app_dir, app_version),dtype=np.float64)
 	Y_total_training = Y_total_training.reshape((2001,1))

	if args.matter_involved=='both':
		X_total_training_gm = np.genfromtxt('%s/%s/training_data/X_original_gm.txt' %(app_dir, app_version),dtype=np.float64)
		X_total_training_wm = np.genfromtxt('%s/%s/training_data/X_original_wm.txt' %(app_dir, app_version),dtype=np.float64)
		X_total_training = np.concatenate((X_total_training_gm,X_total_training_wm), axis=1)

		X_total_testing,testing_nifti_names = data_factory(path_gm_data=args.path_gm_data,path_wm_data=args.path_wm_data,matter_involved='both')

	elif args.matter_involved=='gm':
		X_total_training = np.genfromtxt('%s/%s/training_data/X_original_gm.txt' %(app_dir, app_version),dtype=np.float64)
		X_total_testing,testing_nifti_names = data_factory(path_gm_data=args.path_gm_data,path_wm_data=args.path_wm_data,matter_involved='gm')
	
	elif args.matter_involved=='wm':
		X_total_training = np.genfromtxt('%s/%s/training_data/X_original_wm.txt' %(app_dir, app_version),dtype=np.float64)
		X_total_testing,testing_nifti_names = data_factory(path_gm_data=args.path_gm_data,path_wm_data=args.path_wm_data,matter_involved='wm')

	else:
		print('error in specification of brain tissues')
		sys.exit();

	np.random.seed(7)
	perm = np.random.permutation(2001)
	X_total_training = X_total_training[perm,:]
	Y_total_training = Y_total_training[perm,:]
	X_total_training = X_total_training[:1600,:]
	Y_total_training = Y_total_training[:1600,:]
	age_mean = np.mean(Y_total_training)
	Y_total_training = Y_total_training - age_mean

	obiect =  Gaussian_Process_Regression(num_test=X_total_testing.shape[0],dim_input=X_total_training.shape[1],dim_output=Y_total_training.shape[1],age_mean=age_mean,num_data=X_total_training.shape[0],matter_involved=args.matter_involved,current_directory=args.current_directory)
	obiect.session_TF(X_testing=X_total_testing,X_training=X_total_training,Y_training=Y_total_training,testing_nifti_names=testing_nifti_names)
