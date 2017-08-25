import numpy as np
import nibabel as nib
import sys
import subprocess
from subprocess import call
from collections import defaultdict
import os
import argparse



def data_factory(mask_path,data_info_path,folder_nifti_path):


	### get the mask
	wm_mask_object = nib.load(mask_path)
	wm_mask_data = wm_mask_object.get_data()

	wm_mask_data_reshaped = wm_mask_data.reshape(np.prod(wm_mask_data.shape),)
	wm_mask_data_full_bool = wm_mask_data_reshaped.astype(bool)


	lista_age = []
	with open(data_info_path) as f:
		for line in f.readlines():

			temporar = line.rsplit(',')[3]
			lista_age.append([np.float(temporar)])

	age = np.stack(lista_age)


	comanda = ['ls', folder_nifti_path]
	str_output,plm = subprocess.Popen(comanda,stdout=subprocess.PIPE).communicate()
	lista_imagini = []

	lista_output = str_output.rsplit('\n')
	lista_output = lista_output[:-1]


	for line in lista_output:
		comanda_temporara = folder_nifti_path + '/' + str(line)
		temporar_object = nib.load(comanda_temporara)
		temporar_data = temporar_object.get_data()


		temporar_data_reshaped = temporar_data.reshape(np.prod(temporar_data.shape),)
		lista_imagini.append(temporar_data_reshaped[wm_mask_data_full_bool])


	imagini = np.stack(lista_imagini)
	return imagini,age



if __name__ == '__main__':        

	parser= argparse.ArgumentParser()
	parser.add_argument('--mask_path_gm',type=str,help='the absolute path to the nifti file containing the gray matter mask you want to apply')
	parser.add_argument('--mask_path_wm',type=str,help='the absolute path to the nifti file containing the white matter mask you want to apply')	
	parser.add_argument('--data_info_path',type=str,help='the absolute path to the csv file containing meta data about yur dataset, such as BANC_2016.csv')
	parser.add_argument('--folder_nifti_path_gm',type=str,help='the absolute path to the folder which contains your nifti files for gray matter')
	parser.add_argument('--folder_nifti_path_wm',type=str,help='the absolute path to the folder which contains your nifti files for white matter')	
	parser.add_argument('--output_folder',type=str,help='the absolute path of the folder where you want to store the array files created by this script')
	parser.add_argument('--output_X_name',type=str,help='the name of the .txt file where we store the array containing the input features')
	parser.add_argument('--output_Y_name',type=str,help=' the name of the .txt file where we store the column-vector containing the chrnological age of subjects')
	parser.add_argument('--matters_involved',type=str,help='proide a string from the following list : gm ; wm ; both')

	args = parser.parse_args()

	if args.matters_involved == 'wm':

		imagini,age = data_factory(mask_path=args.mask_path_wm,data_info_path=args.data_info_path,folder_nifti_path=args.folder_nifti_path_wm)

	elif args.matters_involved =='gm':

		imagini,age = data_factory(mask_path=args.mask_path_gm,data_info_path=args.data_info_path,folder_nifti_path=args.folder_nifti_path_gm)
	else:
		imagini_wm,age = data_factory(mask_path=args.mask_path_wm,data_info_path=args.data_info_path,folder_nifti_path=args.folder_nifti_path_wm)
		imagini_gm,age = data_factory(mask_path=args.mask_path_gm,data_info_path=args.data_info_path,folder_nifti_path=args.folder_nifti_path_gm)
		imagini = np.concatenate((imagini_wm,imagini_gm),axis=1)


	np.savetxt(args.output_folder+'/'+args.output_X_name,imagini)
	np.savetxt(args.output_folder+'/'+args.output_Y_name,age)


