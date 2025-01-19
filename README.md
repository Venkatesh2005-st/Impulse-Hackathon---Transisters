\\\\\\\     Task4.1 


The folder task4.1 contains task1_final.py which is the python program used to create the required plots and calculate the metrics
output_plots contains class folders for each of the four classes with each class folder containing:
- png files of plots for each channel of one datapoint
- a csv file with the required metrics



\\\\\\\\    Task5.1

The folder task5.1 contains task5.1.py which is the python program used to create the required plots and calculate the metrics
output_plots contains class folders for each of the four classes with each class folder containing:
- fourier_transforms - contains the fourier transform of each channel of a single datapoint
- spectrograms - contains spectrograms for each channel of a single datapoint
- wavelet_decomposition - contains the four level wavelet decomposition for each channel of a single datapoint as well as a csv file with 			  the most similar wavelet level for each cahnnel


\\\\\\\\    Task6.1


- outputs contains the csv files used to train the baseline model
- test_data_csv contains csv files of test data
- validation_data_csv contains csv fils of validation data
- make_train_set.py uses the files in output to construct a single train_set_df.csv which is used for training the model
- metrics.py and fourier_metrics.py were used to prepare the data for training 
- model.py is used to train he model using sklearn.svm and gives an accuracy score
