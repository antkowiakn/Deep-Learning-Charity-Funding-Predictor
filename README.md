# Deep Learning: Charity Funding Predictor  
  
## Background  
  
The non-profit foundation Alphabet Soup wants to create an algorithm to predict whether or not applicants for funding will be successful. With knowledge of machine learning and neural networks, using the features in the dataset; create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup.

The CSV contains more than 34,000 organizations that have received funding from Alphabet Soup over the years. The dataset has a number of columns that capture metadata about each organization as follows:  
  
* **EIN** and **NAME**—Identification columns  
* **APPLICATION_TYPE**—Alphabet Soup application type  
* **AFFILIATION**—Affiliated sector of industry  
* **CLASSIFICATION**—Government organization classification  
* **USE_CASE**—Use case for funding  
* **ORGANIZATION**—Organization type  
* **STATUS**—Active status  
* **INCOME_AMT**—Income classification  
* **SPECIAL_CONSIDERATIONS**—Special consideration for application  
* **ASK_AMT**—Funding amount requested  
* **IS_SUCCESSFUL**—Was the money used effectively  
  
  
### Preprocess the data  
  
Using Pandas and the Scikit-Learn’s `StandardScaler()`, preprocess the dataset in order to compile, train, and evaluate the neural network model later in the process.  
1. Read in the charity_data.csv to a Pandas DataFrame, identify the following in the dataset:  
  * What variable(s) are considered the target(s)?  
  * What variable(s) are considered the feature(s)?  
2. Drop the `EIN` and `NAME` columns.  
3. Determine the number of unique values for each column.  
4. For those columns that have more than 10 unique values, determine the number of data points for each unique value.  
6. Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, `Other`, and then check if the binning was successful.  
7. Use `pd.get_dummies()` to encode categorical variables.  
  
### Compile, Train, and Evaluate the Model  

Using knowledge of TensorFlow, design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup–funded organization will be successful based on the features in the dataset. Once completed compile, train, and evaluate the binary classification model to calculate the model’s loss and accuracy.  
  

1. Create a neural network model by assigning the number of input features and nodes for each layer using Tensorflow Keras.  
2. Create the first hidden layer and choose an appropriate activation function.  
3. If necessary, add a second hidden layer with an appropriate activation function.  
4. Create an output layer with an appropriate activation function.  
5. Check the structure of the model.  
6. Compile and train the model.  
7. Create a callback that saves the model's weights every 5 epochs.  
8. Evaluate the model using the test data to determine the loss and accuracy.  
9. Save and export results to an HDF5 file, and name it `AlphabetSoupCharity.h5`.  
  
### Optimize the Model  
  
Optimize the model in order to achieve a target predictive accuracy higher than 75%. If the model can't achieve an accuracy higher than 75%, make at least three attempts to do so.  
  
Optimize the model in order to achieve a target predictive accuracy higher than 75% by using any or all of the following:  
  
* Adjusting the input data to ensure that there are no variables or outliers that are causing confusion in the model, such as:  
  * Dropping more or fewer columns.  
  * Creating more bins for rare occurrences in columns.  
  * Increasing or decreasing the number of values for each bin.  
* Adding more neurons to a hidden layer.  
* Adding more hidden layers.  
* Using different activation functions for the hidden layers.  
* Adding or reducing the number of epochs to the training regimen.  
  
1. Create a new Jupyter Notebook file and name it `AlphabetSoupCharity_Optimzation.ipynb`.  
2. Import your dependencies, and read in the `charity_data.csv` to a Pandas DataFrame.  
3. Preprocess the dataset like you did in Step 1, taking into account any modifications to optimize the model.  
4. Design a neural network model, taking into account any modifications that will optimize the model to achieve higher than 75% accuracy.  
5. Save and export your results to an HDF5 file, and name it `AlphabetSoupCharity_Optimization.h5`.  

### Report on the Neural Network Model  

Write a report on the performance of the deep learning model created for AlphabetSoup.  
  
The report should contain the following:  
  
1. **Overview** of the analysis: Explain the purpose of the analysis.  
  
2. **Results**: Using bulleted lists and images to support the answers, address the following questions.  
  
  * Data Preprocessing  
    * What variable(s) are considered the target(s) for the model?  
    * What variable(s) are considered to be the features for the model?  
    * What variable(s) are neither targets nor features, and should be removed from the input data?  
  * Compiling, Training, and Evaluating the Model  
    * How many neurons, layers, and activation functions were selected for the neural network model, and why?  
    * Were you able to achieve the target model performance?  
    * What steps did were taken to try and increase model performance?  
  
3. **Summary**: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and explain your recommendation.  

