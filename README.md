# Good-Or-Bad-Credit

For banks, accurately classifying loans as good or bad is a critical balancing act. Extending credit to creditworthy borrowers fuels economic growth, while mistakenly lending to risky borrowers can trigger losses and even financial instability. And this project is about better classification using different model to avoid the risk of NPA

# Data Source
--------------
UCIREPO - German Credit Data

Columns - ![image](https://github.com/shashank-2010/Good-Or-Bad-Credit/assets/153171192/9d91c05b-6b97-4fa6-bdc4-178201960022)

Data Cleaning
-------------
In machine learning, data cleaning, also known as data cleansing, data wrangling, or data preprocessing, is the crucial process of preparing raw data for analysis and modeling. It involves identifying and resolving issues with the data that could adversely affect the performance and accuracy of your machine learning models.
It involves-
1. Removing Null
   ![image](https://github.com/shashank-2010/Good-Or-Bad-Credit/assets/153171192/2ecf51ee-5c18-42a1-972f-64ecedb67c9d)

2. FIlling the null values either with mean, median or any other value
3. Removing Duplicates
   ![image](https://github.com/shashank-2010/Good-Or-Bad-Credit/assets/153171192/325f1221-08b6-4f5f-871d-0f6e29d491d1)

4. Removing unnecessary columns like index columns
7. Identifying and removing outliers

Data Analysis and Feature Engineering
-------------------------------------
1. Check for the imbalance in the data, as an imbalanced dataset may affect the accuracy of the model.
   ![image](https://github.com/shashank-2010/Good-Or-Bad-Credit/assets/153171192/5dcc5c80-cadf-45d4-861b-9040c2966b11)

2. Encoding of the columns - As machine understands the numerical data for training testing and predicting. This model has used the one hot encoding process.
   ![image](https://github.com/shashank-2010/Good-Or-Bad-Credit/assets/153171192/8735f077-a89e-43db-ab30-096c6d9be0ee)

Model Creation and fitting of data
----------------------------------
1. No one model is the best model thus there is a need to build different models and then check of the accuracy. This project has used models like
   - Logistic Regression
     ![image](https://github.com/shashank-2010/Good-Or-Bad-Credit/assets/153171192/fb5c6118-cb73-49be-ac6a-b876a5836d11)

   -Decision Tree
   ![image](https://github.com/shashank-2010/Good-Or-Bad-Credit/assets/153171192/30f01789-7091-47e6-86b7-f75fb7897d5c)

   - KNN
   - RandomForestClassifier
   - Gradient Boost
  
2. The training and testing of the data is done using the sklearn.model_selection.train_test_split() function.
   ![image](https://github.com/shashank-2010/Good-Or-Bad-Credit/assets/153171192/17ddc662-e518-4501-aa96-754fdf5503e8)

Checking for the accuracy
-------------------------




