#  Utilizing Longitudinal Microbiome Taxonomic Profiles to Predict Food Allergy via Long Short-Term Memory Networks

Food allergy is usually difficult to diagnose in early life, and the inability to diagnose patients with atopic diseases at an early age may lead to severe complications. Numerous studies have suggested an association between the infant gut microbiome and development of allergy. Here, we investigated the capacity of Long Short-Term Memory (LSTM) networks to predict food allergies in early life (0-3 years) from subjects' longitudinal gut microbiome profiles using the DIABIMMUNE dataset. Also, We considered sparse autoencoder for extraction of potential latent representations.



# Dataset: 
In order to evaluate our proposed model, we used the longitudinal gut microbiome profiles from the DIABIMMUNE project (https://pubs.broadinstitute.org/diabimmune/three-country-cohort), a study that aimed to characterize host-microbe immune interactions contributing to autoimmunity and allergy.







# Execution

##  Prerequisites:
  * Python (v3.6.2)
  * Tensorflow (v1.6.0)
  * Libraries: numpy, pandas, scikit-learn, scipy
  

### To extract latent features using autoencoder:
```
python3.6 autoencoder_diabimmune.py -i <input_file.csv> -m <meta_file.csv> -o <output_prefix>
```


### To train LSTM on DIABIMMUNE dataset:
```
python3.6 lstm_diabimmune.py -i <input_file.csv> -m <meta_file.csv> -o <output_prefix>
```
