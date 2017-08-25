# GPR-BRAIN-AGE

### Update:

If you decide to train the GPR by yourself,after the training process finishes you can visualize in Tensorboard how the variables and metrics chance by doing the following in the terminal:

tensorboard --logdir=./tensorboard/



***Requirements***

install:
1. nibabel
2. tensorflow

1. Create "logs" folder
2. Store the log files in that folder

Current Version:

White Matter BANC_2016

Linear kernel, non-ARD

Results on 200 testing subjects:

***RMSE at testing time is :7.20819736674
***MAE at testing time is :5.64525260886


**** How to use scripts ***

1. use create_dataset.py
it has the following arguments:

	'--mask_path' help='the absolute path to the nifti file containing the mask you want to apply'
	
	'--data_info_path' help='the absolute path to the csv file containing meta data about yur dataset, such as BANC_2016.csv'
	
	'--folder_nifti_path' help='the absolute path to the folder which contains your nifti files'
	
	'--output_folder' help='the absolute path of the folder where you want to store the array files created by this script'
	
	'--output_X_name' help='the name of the .txt file where we store the array containing the input features'
	
	'--output_Y_name' help=' the name of the .txt file where we store the column-vector containing the chrnological age of subjects'

2. use GPR.py to train a new model if you want;otherwise go to step 3
it has the following arguments:

	'--input_feature_path' help='the absolute path of the file which contains your features dataset'
	
	'--input_age_path' help='the absolute path of the file which contains your chrnonological age columns-vector'
	
	'--num_iterations' default=1000 help='the number of iterations in the training process' 

3. use GPR_Prediction.py to get predictions for new subjects; obv use step 1 again to create the array folders for your externat dataset
it has the following arguments:

	'--input_feature_path' help='the absolute path of the file which contains your features dataset'
	
	'--input_age_path' help='the absolute path of the file which contains your chrnonological age columns-vector'
	





