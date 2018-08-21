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

### Refinement

In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_


## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

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