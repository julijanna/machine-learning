# Machine Learning Engineer Nanodegree
## Capstone Project
Malgorzata Kot
29.07.2018

## I. Definition

### Project Overview
Machine Learning can support diverse discplines and one of them is marketing. For example, according to research using machine learning models for customer classification results in higher conversion rates<sup>1</sup>, better potential customer identification<sup>2</sup> or can be used for quick sentiment classification <sup>3</sup>, which was previosuly done manually. One of the problems, marketing is facing every day, is predicting, who of the potential customers will buy a product. It helps in efficient allocation of ressources as well as getting to know the customer base. For companies with large numbers of customers with heterogenous characteristics, supervised machine learning can be used for predicting potential customer interest in buying a product.

In this project, potential customers will be classified based on bank marketing dataset from UCI Machine Learning Repository. ([link](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)) The goal is to predict the variable "y", which explains if customer has subscribed a deposit or not, which is equivalent to: did he buy a product or not.

### Problem Statement
I will classify potential customers of a bank based on an information if they bought a product or not. The classifier, after getting data from a potential customer, should return a 1 or 0, 1 meaning customer will buy the product and 0 he/she will not buy. Additionally, there will be probabilty returned of how likely it is, that customer will subscribe to the deposit.

### Metrics
Main metric will be F-2 Score (derived from F-Beta score), but additionally it will be measured with accuracy and confusion matrix. Ideally all of them should be higher than in the base model.

Precision measures how many true prediction were made out all values predictied true (true positives and false positives). Recall measures how many true positives were out of all true values in dataset. F-Beta score combines both precision and recall in order to find the balance between them. Because this is marketing data, where we look for potential opportunities, identifying all potential customers is more improtant than making sure, that they are no false identifications. That is why I will use F-2 score, where beta = 2, what give more credit to recall than precision.

Accuracy is the amount of correctly predicted values divided by all predictions and multiplied by 100. Accuracy is not the most useful measure for this problem. If the assumption in benchmark model is, that all contacted clients are willing to buy, the model would get a low accuracy (11%). But if the assumption would be, that none of called people would by a product, accuracy will be very high (89%). Where second predictions has a very high accuracy, it makes completely no sense, as we look for people who are willing to buy a product. First prediction makes more sense, but the number is very low and easy to improve. So if final model would have 15% accuracy it would be already 30% better thank benchmark but still bad. The problem is, that it is very hard to say where bad accuracy ends and good accuracy starts.

Additionally, I will create confusion matrix in order to see how many false/true positives and negatives are predicted.

## II. Analysis

This project will use Bank Marketing Dataset from Machine Learning Repository ([link](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)). This dataset represent marketing activities from a bank.

The dataset contains of 20 input variables and out output variable. Variables' descriptions were taken from Machine Learning Repository Website, linked above.


######Input variables

Bank client data:
1 - age (numeric)
2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
5 - default: has credit in default? (categorical: 'no','yes','unknown')
6 - housing: has housing loan? (categorical: 'no','yes','unknown')
7 - loan: has personal loan? (categorical: 'no','yes','unknown')

Related with the last contact of the current campaign:
8 - contact: contact communication type (categorical: 'cellular','telephone')
9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.

Other attributes:
12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
14 - previous: number of contacts performed before this campaign and for this client (numeric)
15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')

Social and economic context attributes
16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
17 - cons.price.idx: consumer price index - monthly indicator (numeric)
18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)
19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
20 - nr.employed: number of employees - quarterly indicator (numeric)

######Output variable

Output variable:
21 - y - has the client subscribed a term deposit? (binary: 'yes','no').

Output variable gives an information if the person bought a product (subscribed a term deposit in this dataset). Input variables are actually a description of a person in data, description of campaign, contact time and additional attributes. These attributes are most likely the data which most businesses own and use when advertising their products.

### Data Exploration

Dataset contains 21 variables, of which 1 is output variable and 41188 observations.

![](/Users/mkot/Documents/Edu/Machine Learning Nanodegree/fork/machine-learning/projects/capstone/images/1 dataset shape.png)
Fig. 1: Dataset shape

Out of 20 input variables in the dataset half is numerical and another one is categorical, stored as objects (strings).

![](/Users/mkot/Documents/Edu/Machine Learning Nanodegree/fork/machine-learning/projects/capstone/images/2 dataset types.png)
Fig. 2: Data types in dataset

![](/Users/mkot/Documents/Edu/Machine Learning Nanodegree/fork/machine-learning/projects/capstone/images/3 dataset head.png)
Fig. 3: Datset sample

Out of numerical variables, only cons.conf.idx and cons.price.idx look like they would not be skewed. There are also big scale differences between them. Eg. 'emp.var.rate' varies from -3.4 to 1.4 and duration from 0 to 4918.

Most of the categorical variables habe 2 to 4 options, apart of job and education as well as date related variables. Some of categorical variables seem to have not categorized data, eg. 'poutcome' features has 'nonexistent' as the most often input.

![](/Users/mkot/Documents/Edu/Machine Learning Nanodegree/fork/machine-learning/projects/capstone/images/4 dataset describe.png)
Fig. 4: Dataset variables description

### Exploratory Visualization
Dataset contains numerical and categorical variables. Out of numerical variables, none has normal distributions. Most distributions are positively skewed (age, duration, nr.employed), but some of theme have discrete-like distribution (cons.price.idx, cons.conf.idx)

![](/Users/mkot/Documents/Edu/Machine Learning Nanodegree/fork/machine-learning/projects/capstone/images/5 numerical variables.png)
Fig. 5: Distributions of numerical variables

In categorical variables, there are some which are balanced as contact or housing, with housing having very little "unknown" share. There are a few impalanced classes (loan, default) also with very small "unknown" share. Poutcome is imbalanced having "unknown" as the most frequent feature. Marital, Job and Education are only classes with more than 2 labels (when not counting "unknown" as label).

![](/Users/mkot/Documents/Edu/Machine Learning Nanodegree/fork/machine-learning/projects/capstone/images/6 categorical variables.png)
Fig. 6: Distributions of categorical variables

Output variable is an imbalanced two label class with only 11% share of "no" and 89% "yes".

![](/Users/mkot/Documents/Edu/Machine Learning Nanodegree/fork/machine-learning/projects/capstone/images/7 output variable distribution.png)
Fig. 7: Distribution of output variable


### Algorithms and Techniques

I will choose the best performing classifier out of following:

##### 1. Logistic Regression

Logistic regression is based on a probability of belonging to one or other class. Probability is counted on basis of fit to logistic regression curve (sigmoid function).

![](/Users/mkot/Documents/Edu/Machine Learning Nanodegree/fork/machine-learning/projects/capstone/images/8 sigmoid function.png)
Fig. 8<sup>4</sup>: Sigmoid function formula

Once data is fitted to the formula, having features coefficients and intercept, every data point can be fitted to the probability function.

![](/Users/mkot/Documents/Edu/Machine Learning Nanodegree/fork/machine-learning/projects/capstone/images/9 probability function with one input variable.png)
Fig. 9<sup>4</sup>: Probability function (logistic function) with one dependent variable x

##### 2. Support Vector Machines

Support vector machines construct a hyperplane or set of hyperplanes which divides points in space into two classes. If data is linearly separable, superplane is chosen as line between two closest points from different classes. When data is not linearly separable, so called "kernel trick" can be used in order to make data linearly separable. That means mapping to more-dimensional space.

##### 3. Decision Tree

Decision trees choose sort dependent features based on their importance (having most information about the data) and build from the a tree according to sorting order. It means, that they while working top-down every time choose variable which gives the best split of output variable.

##### 4. Random Forest

Random forest uses decision trees as preliminary for choosing optimal solution. It divides dataset into samples and for every sample builds a decision tree. Additionally, a random subset of features for every sample will be selected. The chosen tree is this, which is built most often.

##### 5. Gaussian NB

Naive bayes apply on Bayes' theorem with assumption of independence of features.  It calculates probability of hypothesis h givenn data d. The model calculates class probabilities (how many % of data belongs to which class) and conditional probabilities, which are probabilities of every feature value for every class value. So in marketing dataset, probability of all 20 input features when given result subscribed or not subscribed.

### Benchmark
In order to create a benchmark I assumed, that every contacted client would buy a product. This approach achieved 0.11 accuracy and 0.39 F-2 score.


## III. Methodology

### Data Preprocessing

#### Missing data

There is no missing data in this dataset:

![](/Users/mkot/Documents/Edu/Machine Learning Nanodegree/fork/machine-learning/projects/capstone/images/10 missing data.png)
Fig. 10: Missing data

But many features have "unknown" which can be considered as missing data. Because I will scale all features I want unknown to be a constant between "yes" and "no". I will change all "unknown" or "nonexistent" to 0 and accordingly, "yes" or "success" to 1 and "no" or "failure" to -1.

![](/Users/mkot/Documents/Edu/Machine Learning Nanodegree/fork/machine-learning/projects/capstone/images/11 Replacing unknown and nonexistent.png)
Fig. 11: Replacing "unknonwn" and "nonexistent" with 0

Additionally, I will insert dummies for categorical variables.

![](/Users/mkot/Documents/Edu/Machine Learning Nanodegree/fork/machine-learning/projects/capstone/images/12 inserting dummies.png)
Fig. 12: Inserting dummies

#### Skewed distributions
As many distributions are skewed, I will transform then using log transformation, log(1+x).

![](/Users/mkot/Documents/Edu/Machine Learning Nanodegree/fork/machine-learning/projects/capstone/images/13 transformed age.png)
Fig. 13: Transformed age feature

![](/Users/mkot/Documents/Edu/Machine Learning Nanodegree/fork/machine-learning/projects/capstone/images/14 transformed duration.png)
Fig. 14: Transformed duration feature

![](/Users/mkot/Documents/Edu/Machine Learning Nanodegree/fork/machine-learning/projects/capstone/images/15 transformed campaign.png)
Fig. 15: Transformed campaign feature

### Implementation

Before data was divided into training and test set, it had to be randomly under sampled in order to achieve higher performance.

#### Data preprocessing

All data was preprocessed, as described earlier before starting any model evaluation or fitting. Then, before splitting data info test and training data, it had to be under-sampled in order to minimize classes imbalance.

#### Random Under-Sampling

Because this dataset is very imbalanced ("yes" output class is only 11% of whole dataset) and also pretty large (41k observations), I will perform under-sampling in order to train and test on more balanced dataset. That means, that I will randomly choose 30% of data with "no" output and combine in with all data with "yes" output before split dataset into training and testing. It causes dataset to be smaller, only with 15k observations, but it also more balanced, with 30% share of "yes" class.

![](/Users/mkot/Documents/Edu/Machine Learning Nanodegree/fork/machine-learning/projects/capstone/images/16 dataset shape after under sampling.png)
Fig. 16: Dataset shape after under-sampling

![](/Users/mkot/Documents/Edu/Machine Learning Nanodegree/fork/machine-learning/projects/capstone/images/17 classes share after under sampling.png)
Fig. 17: Classes share after under-sampling

Random under-sampling was done 10 times every time with another split. This gives the possibility to test the model robustness on different data splits and decide which model performs the best.

#### Model fitting

For every of 10 different under-samplings, there are 10 random training and test splits done. On every of these 100 different splits, each of 5 models is trained and measured. Model evaluation will be done for average and distribution of all 100 performances.

#### Choosing best model

Out of all models, SVMs have performed the best in terms of F2-Score (fbeta) score, accuracy and recall. Precision was no the best of all models, but as written above, precision is not the main measure in this project. SVM F-2 Score was also significantly bigger in t-test than second biggest score, which came from ligstic regression.

![](/Users/mkot/Documents/Edu/Machine Learning Nanodegree/fork/machine-learning/projects/capstone/images/18 FBeta score box plot all.png)
Fig. 18: FBeta score of all algorithms incl. benchmark

![](/Users/mkot/Documents/Edu/Machine Learning Nanodegree/fork/machine-learning/projects/capstone/images/19 FBeta score mean all.png)
Fig. 19: FBeta score mean of all algorithms incl. benchmark

### Refinement

In support vector machines, two main hyperparameters can be tuned: C and kernel. Kernel means how the data is splitted in the space - if plane should be straight line, polynomial or RBF, which plotted on a 2 dimensional spaces look something like circle.

![](/Users/mkot/Documents/Edu/Machine Learning Nanodegree/fork/machine-learning/projects/capstone/images/20 SVM kernel types.png)
Fig. 20<sup>5</sup>: SVM kernel types

C parameter tells SVM how much sensitive it should be for missclasifying points. The bigger C is, the more points should be classified correctly. Keeping that in mind, bigger C runs into risk of overfitting the data.

![](/Users/mkot/Documents/Edu/Machine Learning Nanodegree/fork/machine-learning/projects/capstone/images/21 SVM C parameter.png)
Fig. 21<sup>6</sup>: SVM C parameter

Additionally, for kernel function, there is a gamma parameter, which indicates how much important the variance is. That means, for small gamma, two points can be further away to be considered as similar, but for large gamma, points have to be close to each other to be considered similar.

I did hyperparameter tuning using GridSearchCV with 2 shuffle splits and 3 folds for each split. I used this example for my implementation ([link](http://scikit-learn.org/stable/auto_examples/svm/plot_svm_scale_c.html#sphx-glr-auto-examples-svm-plot-svm-scale-c-py)). Because recall is most important factor for me, which is also used for counting f2-score, I optimized GridSearchCV on recall. The result is, that the best hyperparameters are:

![](/Users/mkot/Documents/Edu/Machine Learning Nanodegree/fork/machine-learning/projects/capstone/images/22 SVM Parameters.png)
Fig. 22: Optimized SVM parameters

Optimized model has 0.001 better F2-Score than not optimized - from 0.806 to 0.807.

## IV. Results

### Model Evaluation and Validation
Chosen model was support vector machines with hyperparameters described in refinement section. The goal of the model was to achieve higher F2-score than benchmark. SMVs performed best out of 5 tested models and additionaly their performance could be slightly improved through hyperparameter tuning. Because model training was run 100 times, final results will be evaluated based on mean and average performances.

In order to achieve robustness, model was run 100 times with different splits and different under-samplings. This caused small fluctuation, with minimum F2-score of 0.79 and maximum 0.83. However, it was still the best out of all tested models.

### Justification
SVM achieved 0.807 F2-score which is 208% more than benchmark. It also achieved higher accuracy and precision, but smaller recall. Reason for achieving smaller recall is that in the benchmark we assumed that all people which are contacted will subscribe to a product, which causes that all positives are predicted correctly. In layman terms, it is more the reason that benchmark had very high recall than the SVMs are performing badly. Minimal F2-Score of SVMs before tuning was on the same level as best score from second classifier - logistic regression.

![](/Users/mkot/Documents/Edu/Machine Learning Nanodegree/fork/machine-learning/projects/capstone/images/23 final results box plot.png)
Fig. 23: Final results model vs. benchmark

![](/Users/mkot/Documents/Edu/Machine Learning Nanodegree/fork/machine-learning/projects/capstone/images/24 final results mean.png)
Fig. 24: Final results model (means) vs. benchmark

SVM were significantly better than benchmark according also to two sample mean t-test. In figure below, sample one is F2-Score of SVMs and sample 2 is benchmark F2-Score.

![](/Users/mkot/Documents/Edu/Machine Learning Nanodegree/fork/machine-learning/projects/capstone/images/25 scores t test.png)
Fig. 24<sup>7</sup>: T-Test result for SVMs score vs. benchmark


## V. Conclusion

### Free-Form Visualization
This project would be used in marketing department in order to decide which potential customers to contact. Because sales people are contacting customers through telephone, it takes time and costs to contact every customer which will not subscribe to a deposit. Proposed algorithm has 0.82 recall, which means that if this algorithm would be used to choose which potential customers should be contacted, only 82% of customers who bought the product would buy it, because other 18% of them would not be contacted at all. It can mean, that it would decrease sales, but there is another side of it - because less potential people would be contacted, it could save a lot of money and time for the company.

![](/Users/mkot/Documents/Edu/Machine Learning Nanodegree/fork/machine-learning/projects/capstone/images/26 final visualization.png)
Fig. 26: T-Test result for SVMs score vs. benchmark

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?

<sup>1</sup> Pål Sundsøy, Johannes Bjelland, Asif M Iqbal, Alex Sandy Pentland, Yves-Alexandre de Montjoye. _Big Data-Driven Marketing: How machine learning outperforms marketers’ gut-feeling._ In _Social Computing, Behavioral-Cultural Modeling & Prediction Lecture Notes_ in _Computer Science_ Volume 8393. 2014, pp. 367-374.
<sup>2</sup> Dirk Thorleuchter, Dirk Van den Poel, Anita Prinzie. _Analyzing existing customers' websites to improve the customer acquisition process as well as the profitability prediction in B-to-B marketing._ In _Expert Systems with Applications_ Volume 39(3). February 2012, pp. 2597-2605.
<sup>3</sup> Bo Pang, Lillian Lee, Shivakumar Vaithyanathan. _Thumbs up? Sentiment Classification using Machine Learning Techniques_ In _Proceedings of EMNLP_. 2002, pp. 79–86.
<sup>4</sup> https://en.wikipedia.org/wiki/Logistic_regression
<sup>5</sup> http://scikit-learn.org/stable/auto_examples/exercises/plot_iris_exercise.html#sphx-glr-auto-examples-exercises-plot-iris-exercise-py
<sup>6</sup> https://stats.stackexchange.com/questions/31066/what-is-the-influence-of-c-in-svms-with-linear-kernel
<sup>7</sup> http://www.evanmiller.org/ab-testing/t-test.html