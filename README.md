# COMP8755Project
This project contains *two* main parts, that is, ***Dropouts_prediction*** and ***Curriculum_Analysis*** in the seperate corresponding folders. There is also a README file in each folder.

## Dropout_Prediction
This sub-project is for predicting student dropout in 3 degree (ADM, ARQ, and CSI) at a Brazilian university. By using Genetic Algorithm (GA)+ Support Vector machine (SVM) for feature selection and Long short-term memory (LSTM) for Time Series (TS) data training and test.

### Dataset
`historicisFunal.csv` is the raw data for this project, it contains **32** features including student info such as *course*, *degree*, *grades*, *enrollment status* and so on.

`ADM.csv, ARQ.csv, and CSI.csv` are three datasets are seperated from raw data by degree

`xx_clean.csv, xx=ADM,ARQ, or CSI` is the clean dataset processed by `preprocessor.py`. Derive from the above three datasets.

`xx_final.csv, xx=ADM,ARQ, or CSI` is the final dataset selected by `feature_selector.py` and ready for training. Derive from above three datasets.


### Imported Libraries
The scripts of projects are all based in Python 3.8 and anaconda package manager. The external dependencies are as following:

* torch
* torch.nn
* torch.utils.data
* torch.autograd
* torch.utils.data
* matplotlib.pyplot 
* sklearn.svm
* sklearn.model_selection
* sklearn.preprocessing
* sklearn.impute
* sklearn.ensemble

***Note: Install all libraries by using `conda install` or `pip install`***

### Pre-processing
`preprocessing.py` implements the pre-processing. `subset_seperator(self)` is separating raw dataset by degree; `organise(self)` is for data cleaning, data normalization and Data validation. `impute_missing_value(self)` is for data imputation.

 




### Feature Selection

`feature_selector.py` implements the feature selection by using GA+SVM. `get_fitness(self,pop,path)` is for getting fitness of each individual by SVM; `select(self,pop, fitness), crossover(self,parent, pop), mutate(self,child), and evolution(self)` are the steps of GA.


### Training+Test
`xx_dataloader.py, xx=ADM,ARQ, or CSI` implements the training and test by LSTM.

### Running project

***Directly Run `xx_dataloader.py, xx=ADM,ARQ, or CSI`*** to get the result of preprocessing, feature selection and training and test.

## Curriculum_Analysis
This sub-project contains two parts, one is ***similarity measurement***, which aims to get the similarity between two courses with using *BERT*; another one is ***prerequisite identification***, which is about using *SemfD* to analysis semantic relation between two concepts and courses further.
###Dataset
The dataset this project used is from ANU Course & Program https://programsandcourses.anu.edu.au/, which is a website providing the course description and curriculum arrangement.

The `raw_data` stores the each course in txt format； the `raw_data_csv` store each course in csv format.  
### Imported Libraries
The scripts of projects are all based in Python 3.8 and anaconda package manager. The external dependencies are as following:
* bert_serving.client
* termcolor
* seaborn
* textrazor
* SPARQLWrapper

***Note: Install all libraries by using `conda install` or `pip install`***
### Similarity Measurement

This part focuses on the similarity measurement relying on ***BERT*** to conduct sentence embedding. Note that this work is based on the @blindreviewrepo on Github. https://github.com/blindreviewrepo/WebAndReligion#run-the-script-scriptsimilarity_fullpy-book_file_1txt-book_file_2txt-output_matrix_filecsv
#### Installment of BERT as Service and runing
The installment of BERT used in this project as the link above or following.
1. Install Bert as a Service by `pip install -U bert-serving-client`(instruction is available on https://github.com/hanxiao/bert-as-service)
2. Download large BERT model at https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip
3. Run the Bert as a Service pointing to the unzipped downloaded model using the following command: `bert-serving-start -model_dir /your_directory/wwm_uncased_L-24_H-1024_A-16 -num_worker=4 -port XXXX -max_seq_len NONE`

#### Sentence Embedding Generation and similarity acquirement

run `python similarity.py` to get the result of two sentences in two courses comparison. The result will be in `/result/similarity_measurement/full_similarity`

run `python text_similarity.py` to get the result of two courses comparison, which will be stored in `/result/similarity_measurement/full_similarity/similarity_full.csv`

run `python diagram.py` to visualize the result.

### Prerequisite Identification

This part focuses on the ***prerequisite of concepts and courses*** in further move.

1. Install the external libraries by using conda or pip command
2. `RefDSimple.py` is for using RefD to retrieve and get potential candidates from DBpedia https://www.dbpedia.org/ which is a main knowledge graph in Semantic Web.
3. `entity_extractor.py` is using TextRazor (https://www.textrazor.com/) to segment and extract entities from text.
4. `config.cfg` is the configuration file, you may need to change the proxy before using it.
5. run `entity_extractor.py` to get the result which will be in `/result/prerequisite_identification`
