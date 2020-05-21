#### Predicting 30 Readmission risk for ICU patients:  unsupervised clustering of patient sub-populations for multitask learning using electronic health records

In this study, I will use single task and multi task models to predict 30 day readmission risk using the MIMIC III dataset. This will be done by first clustering patients using a data driven unsupervised approach to cluster patients using sociodemographic, operational, and clinical factors captured in the last 48 hours of their stay.This will then be used to predict the patient's risk of readmission within 30 days of discharge from the ICU using a multi-task framework. This approach was first introduced by Suresh et.al., (2018)$^{13}$ to predict in hospital mortality in the ICU. To the best of my knowledge this is the first study that uses this two step approach to predict 30 day readmission risks of heterogenous patients in the ICU. Additionally, it is key to note that this study focuses on the implementation and tests the feasibility of this modeling approach,  which if sucessfull will then lead to possible follow-ups and improvements of this study in a subsequent post.

How to run the code in this repo:

1. Clone this repo

   ```
   git clone https://github.com/sparalic/30-day-Readmission.git
   ```

   

2. Create a conda env using the environment yaml file

   ```shell
   conda env create -f environment.yml -n <name of env>
   ```

3. cd into the repo folder and run the GMM file 

   ```shell
   python3 GMM_clustering.py
   ```

4. Run the modeling file after selecting the desired model type

   ```
   python3 modeling.py
   ```

   

