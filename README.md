# Amazon_Sentiment_Analysis <img src="https://github.com/user-attachments/assets/74294622-c5ff-4800-866e-e57057c9cb28" alt="amazon-logo" width="80" style="vertical-align:middle; margin-left:10px;">

## Aim of the project:<br>
1) Enhance the ability to predict customer sentiment, which can be useful for businesses in improving their products and services.<br>
2) Identify key factors that contribute to positive and negative experiences.<br>
3) Develop practical skills in **natural language processing (NLP) and machine learning**, which are valuable in various data-driven fields.<br>

## About the dataset:
1) The Amazon reviews polarity dataset is taken from Kaggle, constructed by taking **review score 1 and 2 as negative, and 4 and 5 as positive. Samples of score 3 is ignored**.
2) Each class has 1,800,000 training samples and 200,000 testing samples. Further sampling is done to reduce the size of the training and testing data.
3) The CSVs contain polarity, title, text. These 3 columns in them, correspond to class index (1 or 2), review heading and review body.

## Data Cleaning Steps:
1) review_cleaning function is applied to make **text lowercase, remove links, punctuations and numbers**<br>
2) **Negative Stop words were taken special care of**, while removing the stop words as they carry meaning towards negative sentiment.<br>
3) Words were **lemmatized instead of stemmed** to withold the sematic meaning of words.<br>

## Visualizations from text

1) The polarity of the sentiments were **normally distributed** but it was not standard normal.
<img src="https://github.com/user-attachments/assets/24b1fa4d-8f11-4b03-8c1a-6b2c6dbe4398" alt="polarity distribution" width="750"><br>

2) Histogram of **total length** of Reviews and total **word count** were plotted

  <img src="https://github.com/user-attachments/assets/f5320734-bc38-4881-a94f-5ff7de785b2f" alt="total length of reviews" width="500"> <img src="https://github.com/user-attachments/assets/1f9f5fa3-31e0-43a7-9988-8772ddeced44" alt="total word count" width="500"><br>
  
3) Frequent 1,2 & 3 grams in both positive  & negative reviews:

<img src="https://github.com/user-attachments/assets/f5edf8df-5fc9-466c-9920-1d9736ddff2d" alt="image 1" style="width: 500px; height: 380px;">

<img src="https://github.com/user-attachments/assets/d4d05c88-13f3-4376-bfe4-c7088bad9147" alt="image 2" style="width: 500px; height: 380px;">
<br><br><br>

<img src="https://github.com/user-attachments/assets/48eb7ef1-2dc9-428e-98a4-69840441efed" alt="image 3" style="width: 500px; height: 380px;">

<img src="https://github.com/user-attachments/assets/d744e0e6-6793-4db1-b09a-05393b40e409" alt="image 4" style="width: 500px; height: 380px;">
<br><br><br>

<img src="https://github.com/user-attachments/assets/8a49c8fa-c3ec-47a0-9a52-0033ce9b267e" alt="image 5" style="width: 500px; height: 380px;">

<img src="https://github.com/user-attachments/assets/2773f464-05fd-456e-bfcd-868d07868f93" alt="image 6" style="width: 500px; height: 380px;">
<br><br>

4) **Word Clouds**

<table>
  <tr>
    <td style="text-align: center;">
      <h3>Negative Word Cloud</h3>
      <img src="https://github.com/user-attachments/assets/cc9b10c5-f3c3-4a5f-b4e4-6d7427e75392" alt="Negative Word Cloud" style="width: 550px; height: auto;">
    </td>
    <td style="text-align: center;">
      <h3>Positive Word Cloud</h3>
      <img src="https://github.com/user-attachments/assets/0cf25a44-1853-4f23-b3c2-604f25559ba7" alt="Positive Word Cloud" style="width: 550px; height: auto;">
    </td>
  </tr>
</table>

# Use of Bag of Words and Tfidf Vectorizer

Both BOW and Tfidf were used to convert texts into vectors using max_features 10,000 and ngram_range 1 to 3. These vectors were then utilised for applying Machine Learning models. Here are the performances of various ML Models on these vectors:<br>
### Logistic Regression:
<table>
  <tr>
    <td style="text-align: center;">
      <h3>Trained on BOW</h3>
      <img src="https://github.com/user-attachments/assets/1037f90d-eebb-4c5d-bbf3-d76156491284" alt="Logistic Regression" style="width: 550px; height: auto;">
    </td>
    <td style="text-align: center;">
      <h3>Trained On Tfidf</h3>
      <img src="https://github.com/user-attachments/assets/fc300e58-aa03-4cb7-ae92-3d2a60fddab9" alt="Logistic Regression" style="width: 550px; height: auto;">
    </td>
  </tr>
</table>


### Multinomial Naive Bayes
<table>
  <tr>
    <td style="text-align: center;">
      <h3>Trained on BOW</h3>
      <img src="https://github.com/user-attachments/assets/56564502-a7d8-4e22-8c97-c0023cd46d7d" alt="Multinomial Naive Bayes" style="width: 550px; height: auto;">
    </td>
    <td style="text-align: center;">
      <h3>Trained On Tfidf</h3>
      <img src="https://github.com/user-attachments/assets/4cc9cf77-f2e4-44f8-868b-383423042161" alt="Multinomial Naive Bayes" style="width: 550px; height: auto;">
    </td>
  </tr>
</table>

### Random Forest
<table>
  <tr>
    <td style="text-align: center;">
      <h3>Trained on BOW</h3>
      <img src="https://github.com/user-attachments/assets/f8b65817-53f2-40b5-8c84-80791cf28970" alt="Random Forest" style="width: 550px; height: auto;">
    </td>
    <td style="text-align: center;">
      <h3>Trained On Tfidf</h3>
      <img src="https://github.com/user-attachments/assets/e857235a-0e7a-4d57-931f-563b9d8924c8" alt="Random Forest" style="width: 550px; height: auto;">
    </td>
  </tr>
</table>

**Logistic Regression was the best performing model showcasing ROC_AUC score of 0.9016 on Tfidf Vectors**

# Training LSTM model

**Tokenization and Padding**<br>
Max features were kept as 10,000 to maintain uniformity. Sentences were tokenised first using Keras' tokenizer. Distribution of the tokenised sequences was checked:
<br>

<div style="text-align: center;">
  <img src="https://github.com/user-attachments/assets/ff7c0218-6566-4c2b-87ac-c6b8be50adad" alt="Tokenized Sequences Distribution" style="width: 600px; height: auto;">
</div>
<br>

Max length for padding was decided to be 128.

As for the **architecture of the model**, an embedding layer at the start, followed by two Bidirectional LSTM along with Batch Normalization, and then a Dropout layer followed by some Dense layers. 

<img src="https://github.com/user-attachments/assets/2a6c17c4-ccd3-4f70-bb7d-ee23d4961174" alt="image 1" style="width: 550px; height: 570px;">

<img src="https://github.com/user-attachments/assets/b2918745-b6e5-4244-a24e-b38ba3f2e4bc" alt="image 2" style="width: 350px; height: 675px;">


**Performance On test Data**
<table>
  <tr>
    <td style="text-align: center;">
      <h3>Classification Report</h3>
      <img src="https://github.com/user-attachments/assets/b3e9cd2b-698a-4f01-8469-d643a0c99983" alt="LSTM" style="width: 450px; height: auto;">
    </td>
    <td style="text-align: center;">
      <h3>Confusion Matrix</h3>
      <img src="https://github.com/user-attachments/assets/5a224f7c-a613-4c1d-bad3-fa5830478d8e" alt="LSTM" style="width: 450px; height: auto;">
    </td>
  </tr>
</table>
<br>

**LSTM Model performance comparison on Training and Test data**

<table>
  <tr>
    <td style="text-align: center;">
      <h3>Accuracy comparison</h3>
      <img src="https://github.com/user-attachments/assets/fa2eb848-5530-4299-9f0b-ab044c9cb210" alt="LSTM" style="width: 550px; height: auto;">
    </td>
    <td style="text-align: center;">
      <h3>Loss Comparison</h3>
      <img src="https://github.com/user-attachments/assets/25b43ba8-d9dc-4a70-9979-f5549f21c7f6" alt="LSTM" style="width: 550px; height: auto;">
    </td>
  </tr>
</table>
