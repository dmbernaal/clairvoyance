# Clairvoyance Training
Data augmentation method for tabular-structured data. 

## About
Clairvoyance is an **experimental** approach to performing *data augmentation* on any tabular-structured dataset. Data augmentation is a common method used in training computer vision models - allowing for better regualization of the model. *tabular structured here means SQL, CSV formatted data*.

Here we take the same intuition of adding 'additional dimensions' to a raw dataset, with the purpose of generating more data. The hypothesis here is that with appropriately augmented data, the deep learning model will both generalize better and perform better than baseline. 

When dealing with tabular-structured data, the most common algorithms used are *XGboost, Tree-based* 'classical' machine learning algorithms. Typically, there is a **lack** of data in which case these models will almost always outperform a stand-alone Feed-Forward-Net. 

**Clairvoyance** challenges this practice by creating augmented data through carefully designed **Autoencoders**. 

## Method
We first will take the raw dataset and train a **sparse-autoencoder** to reconstruct the dataset. The goal here is the create a great enough reconstruction to form the new dataset. 

Once we come to an *appropriate* loss & further reconstruction of the raw data, we then take the *encoder* of the model to perform data augmentation. 

This *encoder* has been trained to perform feature extraction - creating a different sub-space of the raw dimensions in a lower (or) higher dimensional representation. This *latent* space is then used to create the **augmentation**. 

# Results
Below are some test results. Notebooks to each test can be found in this repository above.

### Breast Cancer 1 - ```99.3% Accuracy```
link: https://www.kaggle.com/uciml/breast-cancer-wisconsin-data

#### Approach:
For this approach we will use ```clairvoyance``` training method. This include using an Auto-Encoder and FFN Ensemble. The Auto-encoder will perform feature extraction and will serve as a *data augmentation* method on our raw-continuous data. 

The FFN will be a basic FFN - Ranger Optimization, SeLU activation, and linear training method. We will also use a Flattened Label Smoothing Cross Entropy.

#### Results: 
```99.3%``` Accuracy: **Clairvoyance Training**
```98%``` Acuracy: *Baseline Model*

## To Do:
1. Perform experiments in the following datasets.

* Breast Cancer 2 - TBD
  * link: https://www.kaggle.com/piotrgrabo/breastcancerproteomes#77_cancer_proteomes_CPTAC_itraq.csv

* Blindness Detection - TBD
  * https://www.kaggle.com/c/aptos2019-blindness-detection/data

* Diabetic Retinopathy Detection - TBD
  * https://www.kaggle.com/c/diabetic-retinopathy-detection/data

* Gene Expression Cancer RNA-Sequence - TBD
  * https://www.kaggle.com/murats/gene-expression-cancer-rnaseq

* Hopsital Triage and Patient History Data - TBD
  * https://www.kaggle.com/maalona/hospital-triage-and-patient-history-data

* Respiratory Sound Database - TBD
  * https://www.kaggle.com/vbookshelf/respiratory-sound-database#104_1b1_Ar_sc_Litt3200.txt

* Skin Cancer - TBD
  * https://github.com/GalAvineri/ISIC-Archive-Downloader

2. Perfrom an in-depth analysis of the technique - measuring model telemetry.
