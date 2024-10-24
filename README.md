This code is a 3D version of the Pix2pix HD model, made for the synthesis if sCT from MRI images.
By changing the options (base_option and train_option), you can : 
- define the name of your model
- define your data root and if needed, the path to your list of testing subjects
- change the number of channels for your input if you want to use dixon sequence
- decide if you want to create your testing set randomly or take the subjects from a given path (list the patients in a .txt file)
- choose your patch size and patch overlap

Once the model is trained, it can be evaluated using the test.py code. By giving the epoch you want to evaluate, it will create a folder containing the sCT created with the given model. 
It is possible to compute comparisons using the totalsegmentator tool, using the 3 functions in the folder 'Compute comparison': 
- first use 'Run_TotalSegmentator.py' to run the totalsegmentator tool on the sCT folder
- secondly use 'Run_VolumesComputation_CT_MRI_sCT.py' to create .txt files containing the volumes for each tissue on the 3 types of scans
- thirdly use 'Run_comparison_TotalSegmentator' to create a final report that compares the results of the segmentation for the sCT and the MRI (compared to the real CT)

While training, you can visualize the results by executing the command : 

tensorboard --logdir "checkpoints/name_of_your_model/Tensorboard/name_of_your_model"

It is possible to change the frequency for prints and validation in the options. 
