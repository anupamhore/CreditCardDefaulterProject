<h1>Introduction</h1>
<hr>
<strong>Credit Card Default Detection System</strong> is a ML solution to predict the probability of credit default based on credit card owner’s characteristics and payment history.

<div></div>

<strong>Problem Statement</strong>
<hr>
<div>Financial threats are displaying a trend about the credit risk of commercial banks as the incredible improvement in the financial industry has arisen. In this way, one of the biggest threats faces by commercial banks is the risk prediction of credit clients. The objective of this project is to reduce such threats through predictive machine learning models and statistical insights which will allow the banks to take necessary businsess decisions to minimize the credit risk.</div>
<hr>
The data  for this project is taken from <a href="https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset" target='_blank'>Kaggle.com</a>. Original source of this data is from
<div>Lichman, M. (2013). UCI Machine Learning Repository <a href="http://archive.ics.uci.edu/ml" target="_blank">[http://archive.ics.uci.edu/ml].</a>  Irvine, CA: University of California, School of Information and Computer Science.</div>

<h1>Source</h1>
<hr>
This dataset contains information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients in Taiwan from April 2005 to September 2005.


<h1>Approach</h1>
<hr>
Classical Machine Learning tasks are performed in this project like Data Exploration, Data Cleaning, Feature Engineering, Model Building and Model Testing.

A Flask UI is presented as the front-end to use for prediction. Users can select any test csv files which has the same format as the one from Kaggle and can check the predictions for those dataset.

It is hosted in Heroku cloud at 

<a href="https://credit-card-defauldetection.herokuapp.com/" target="_blank">Credit Card Default Detection System</a>

<img src="https://www.dropbox.com/s/4bim4be7j7002zw/predictmodel.png?raw=1" style="width:600px;height:250px"/>

<h1>Installation</h1>
<hr>
Dependencies
 <ul>
    <li>Python(>=3.8)</li>
    <li>cassandra_driver==3.25.0</li>
    <li>Flask==2.0.1</li>
    <li>flask_cors==3.0.10</li>
    <li>imblearn==0.0</li>
    <li>kneed==0.7.0</li>
    <li>matplotlib==3.4.3</li>
    <li>numpy==1.21.2</li>
    <li>pandas==1.3.3</li>
    <li>scikit_learn==0.24.2</li>
    <li>scipy==1.7.1</li>
    <li>statsmodels==0.12.2</li>
    <li>Werkzeug==2.0.1</li>
    <li>xgboost==1.4.2</li>
</ul>

<h2>Please go through <a href="https://drive.google.com/file/d/1LCuQAlMBKlEIa9V1RtiFMSyYDiAzgh8D/view?usp=sharing" target="_blank">High Level Design</a> for more information</h2>


<h1>Contributors</h1>
<hr>
<ul>
<li>Anupam Hore</li>
</ul>
