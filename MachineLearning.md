# Machine Learning
Sean Davis  
July 7, 2017  

# Preliminaries

## Install required libraries.




```r
library(BiocInstaller)
biocLite(c("mlbench", "adabag", "e1071", "randomForest", "party", 
    "mboost", "rpart.plot", "formatR"))
```


```r
require(c("mlbench", "adabag", "e1071", "randomForest", "party", 
    "mboost", "rpart.plot", "formatR"))
```

### Some links of interest

- [caret](https://https://cran.r-project.org/package=caret), [party](https://cran.r-project.org/package=party), [randomForest](https://cran.r-project.org/package=randomForest), [mlbench](https://cran.r-project.org/package=mlbench), [mlr](https://cran.r-project.org/package=mlr)
- [Max Kuhn's old machine learning tutorial](https://www.r-project.org/conferences/useR-2013/Tutorials/kuhn/user_caret_2up.pdf)

# Overview

## What is machine learning?

Machine learning is a broad set of fields related to computers learning from "experience" (data). 

- Focusing on *predictive modeling* with a goal of *producing the most accurate estimates of some quantity or the most likely output of an event*. 
- These models are sometimes based on similar models for inference (testing against a null hypothesis, such as linear regression), but in many cases, predictive models are not well-suited for inference (think k-nearest-neighbor, for example). 

## The formula interface


```r
outcome ~ var1 + var2 + ...
```

The variable `outcome` is predicted by `var1, var2, ...`


```r
some_model_function(price ~ numBedrooms + numBaths + acres, data = housingData)
```

Conveniences of the formula interface:

- Transformations such as `log10(acres)` can be specified inline. 
- Factors are converted into dummy variables automatically.

## The Non-formula interface

- The non-formula interface specifies the predictors as a matrix or data frame. 
- The outcome data are then passed into the model as a vector.


```r
some_model_function(x = housePredictors, y = price)
```

Many R functions offer both a formula and a non-formula interface, but not all.

## General workflow for machine learning in R

1. Fit the model to a set of training data
    
    ```r
    fit <- knn(trainingData, outcome, k = 5)
    ```
2. Assess the properties of the model using `print`, `plot`, `summary` or other methods
3. Predict outcomes for samples using the predict method:
    
    ```r
    predict(fit, newSamples).
    ```

# Exercise 1
Playing with regression

## Is `mpg` a function of `wt`? {.smaller}

The formula interface in action:


```r
data(mtcars)
fit = lm(mpg ~ wt, data = mtcars)
summary(fit)
```

```
## 
## Call:
## lm(formula = mpg ~ wt, data = mtcars)
## 
## Residuals:
##     Min      1Q  Median      3Q     Max 
## -4.5432 -2.3647 -0.1252  1.4096  6.8727 
## 
## Coefficients:
##             Estimate Std. Error t value Pr(>|t|)    
## (Intercept)  37.2851     1.8776  19.858  < 2e-16 ***
## wt           -5.3445     0.5591  -9.559 1.29e-10 ***
## ---
## Signif. codes:  
## 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 3.046 on 30 degrees of freedom
## Multiple R-squared:  0.7528,	Adjusted R-squared:  0.7446 
## F-statistic: 91.38 on 1 and 30 DF,  p-value: 1.294e-10
```

And make a plot.


```r
plot(mpg ~ wt, data = mtcars)
abline(fit)
```

## Is `mpg` a function of `wt`?

![](MachineLearning_files/figure-html/unnamed-chunk-11-1.png)<!-- -->

## Use `wt` to predict `mpg`

And predict the original data based on the fitted model.


```r
pred_mpg = predict(fit, mtcars)
summary(pred_mpg)
```

```
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
##   8.297  17.992  19.515  20.091  23.490  29.199
```

And look at the predicted values:


```r
plot(mpg ~ wt, data = mtcars)
abline(fit)
points(y = pred_mpg, x = mtcars$wt, col = "red")
```

## Use `wt` to predict `mpg`

![](MachineLearning_files/figure-html/unnamed-chunk-14-1.png)<!-- -->

## Quantifying "goodness-of-fit"

![](MachineLearning_files/figure-html/unnamed-chunk-15-1.png)<!-- -->

## Quantifying "goodness-of-fit"

- Residual Sum of Squares
    $$ RSS = \sum_{N} (y_i - f(x_i))^{2} $$

## Quantifying "goodness-of-fit"


```r
rss = sum((mtcars$mpg - predict(fit, mtcars))^2)
rss
```

```
## [1] 278.3219
```

```r
anova(fit)
```

```
## Analysis of Variance Table
## 
## Response: mpg
##           Df Sum Sq Mean Sq F value    Pr(>F)    
## wt         1 847.73  847.73  91.375 1.294e-10 ***
## Residuals 30 278.32    9.28                      
## ---
## Signif. codes:  
## 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

## Training versus testing

- What did we do wrong in quantifying our "goodness-of-fit"?

# Splitting data and model performance evaluation


## Common steps during training

- estimating model parameters (i.e. training models)
- determining the values of tuning parameters that cannot be directly calculated from the data
- calculating the performance of the final model that will generalize to new data

## Spending the data to find an optimal model? 

- Split data into training and test data sets
- *Training Set*: these data are used to estimate model parameters and
to pick the values of the complexity parameter(s) for the model.
- *Test Set (aka validation set)*: these data can be used to get an independent assessment of model accuracy. The test data should never be used in any aspect of model training.

## Tradeoffs in spending data

The more data we spend, the better estimates we’ll get (provided the data is accurate). Given a fixed amount of dat:

- Too much spent in training won’t allow us to get a good assessment of predictive performance. We may find a model that fits the training data very well, but is not generalizable (over–fitting)
- Too much spent in testing won’t allow us to get a good assessment of
model parameters

Statistically, the best course of action would be to use all the data for
model building and use statistical methods to get good estimates of error, but from a non–statistical perspective, many consumers of of these models
emphasize the need for an untouched set of samples the evaluate
performance.

## Example using `mtcars`

Using 50% of the data for training and 50% for testing is a place to start.


```r
set.seed(1)
trainIdx = sample(1:nrow(mtcars), 16)
trainDat = mtcars[trainIdx, ]
testDat = mtcars[-trainIdx, ]
```

## Train the model using the training data


```r
fit = lm(mpg ~ wt, data = trainDat)
anova(fit)
```

```
## Analysis of Variance Table
## 
## Response: mpg
##           Df Sum Sq Mean Sq F value    Pr(>F)    
## wt         1 433.81  433.81  39.848 1.915e-05 ***
## Residuals 14 152.41   10.89                      
## ---
## Signif. codes:  
## 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

## Train the model using the training data

![](MachineLearning_files/figure-html/unnamed-chunk-19-1.png)<!-- -->

## Test our model using the testing data


```r
pred_mpg = predict(fit, testDat)
rss = sum((testDat$mpg - pred_mpg)^2)
rss
```

```
## [1] 194.9261
```

```r
anova(fit)
```

```
## Analysis of Variance Table
## 
## Response: mpg
##           Df Sum Sq Mean Sq F value    Pr(>F)    
## wt         1 433.81  433.81  39.848 1.915e-05 ***
## Residuals 14 152.41   10.89                      
## ---
## Signif. codes:  
## 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

## Test our model using the testing data

![](MachineLearning_files/figure-html/unnamed-chunk-21-1.png)<!-- -->

# Example 2
A classification example

## Classification Trees

As a simple dataset to try with machine learning, we are going to predict the species of 
`iris` based on four measurements.


```r
data(iris)
View(iris)
pairs(iris[, 1:4], col = iris$Species)
```

## Iris Data

![](MachineLearning_files/figure-html/unnamed-chunk-23-1.png)<!-- -->


## Another slide

We can start with a simple learner, a [classification tree](https://en.wikipedia.org/wiki/Decision_tree_learning). This learner requires:

- A known class for each observation
- A set of "features" that will serve a potential predictors

1. Start with whole dataset.
2. Choose features one-at-a-time and look for a value of each variable that ends up with the most homogeneous two groups after splitting on that variable/value.
3. For each resulting group, repeat step 2 until all remaining groups have only one class in them.
4. Optionally, "prune" the tree to keep only splits that are "statistically significant".

## Learning the model

The `party` package includes a function, `ctree` to "learn" a tree from data.


```r
library(party)
x = ctree(Species ~ ., data = iris)
plot(x)
```

## Learning the model

![](MachineLearning_files/figure-html/unnamed-chunk-25-1.png)<!-- -->



## Checking the model


```r
library(caret)
library(e1071)
prediction = predict(x, iris)
table(prediction)
```

```
## prediction
##     setosa versicolor  virginica 
##         50         54         46
```

```r
confusionMatrix(iris$Species, prediction)
```

```
## Confusion Matrix and Statistics
## 
##             Reference
## Prediction   setosa versicolor virginica
##   setosa         50          0         0
##   versicolor      0         49         1
##   virginica       0          5        45
## 
## Overall Statistics
##                                          
##                Accuracy : 0.96           
##                  95% CI : (0.915, 0.9852)
##     No Information Rate : 0.36           
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.94           
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: setosa Class: versicolor
## Sensitivity                 1.0000            0.9074
## Specificity                 1.0000            0.9896
## Pos Pred Value              1.0000            0.9800
## Neg Pred Value              1.0000            0.9500
## Prevalence                  0.3333            0.3600
## Detection Rate              0.3333            0.3267
## Detection Prevalence        0.3333            0.3333
## Balanced Accuracy           1.0000            0.9485
##                      Class: virginica
## Sensitivity                    0.9783
## Specificity                    0.9519
## Pos Pred Value                 0.9000
## Neg Pred Value                 0.9900
## Prevalence                     0.3067
## Detection Rate                 0.3000
## Detection Prevalence           0.3333
## Balanced Accuracy              0.9651
```

## Data splitting, take 2

What is the problem with what we just did to determine our prediction accuracy?  
To deal with this problem, we can split the dataset into a "training" set and then check
our prediction on the other piece of the data, the "test" set.


```r
set.seed(42)
trainIdx = sample(c(TRUE, FALSE), size = nrow(iris), prob = c(0.2, 
    0.8), replace = TRUE)
irisTrain = iris[trainIdx, ]
irisTest = iris[!trainIdx, ]
nrow(irisTrain)
```

```
## [1] 35
```

```r
nrow(irisTest)
```

```
## [1] 115
```

## "train" our tree on the "training" set.


```r
trainTree = ctree(Species ~ ., data = irisTrain)
plot(trainTree)
```

![](MachineLearning_files/figure-html/unnamed-chunk-28-1.png)<!-- -->

## "train" our tree on the "training" set.

![](MachineLearning_files/figure-html/unnamed-chunk-29-1.png)<!-- -->

## Test our predictions on the "training" data

And how does our `trainTree` do at predicting the original classes in the "training" data?


```r
trainPred = predict(trainTree, irisTrain)
confusionMatrix(irisTrain$Species, trainPred)
```

##  Test our predictions on the "training" data


```
## Confusion Matrix and Statistics
## 
##             Reference
## Prediction   setosa versicolor virginica
##   setosa         18          0         0
##   versicolor      0          0         5
##   virginica       0          0        12
## 
## Overall Statistics
##                                           
##                Accuracy : 0.8571          
##                  95% CI : (0.6974, 0.9519)
##     No Information Rate : 0.5143          
##     P-Value [Acc > NIR] : 2.275e-05       
##                                           
##                   Kappa : 0.7489          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: setosa Class: versicolor
## Sensitivity                 1.0000                NA
## Specificity                 1.0000            0.8571
## Pos Pred Value              1.0000                NA
## Neg Pred Value              1.0000                NA
## Prevalence                  0.5143            0.0000
## Detection Rate              0.5143            0.0000
## Detection Prevalence        0.5143            0.1429
## Balanced Accuracy           1.0000                NA
##                      Class: virginica
## Sensitivity                    0.7059
## Specificity                    1.0000
## Pos Pred Value                 1.0000
## Neg Pred Value                 0.7826
## Prevalence                     0.4857
## Detection Rate                 0.3429
## Detection Prevalence           0.3429
## Balanced Accuracy              0.8529
```

## Test our predictions on the "testing" data

How is our prediction performance now on the "test" data?


```r
testPred = predict(trainTree, irisTest)
confusionMatrix(irisTest$Species, testPred)
```

## Test our predictions on the "testing" data


```
## Confusion Matrix and Statistics
## 
##             Reference
## Prediction   setosa versicolor virginica
##   setosa         30          0         2
##   versicolor      0          0        45
##   virginica       0          0        38
## 
## Overall Statistics
##                                           
##                Accuracy : 0.5913          
##                  95% CI : (0.4957, 0.6821)
##     No Information Rate : 0.7391          
##     P-Value [Acc > NIR] : 0.9998          
##                                           
##                   Kappa : 0.4018          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: setosa Class: versicolor
## Sensitivity                 1.0000                NA
## Specificity                 0.9765            0.6087
## Pos Pred Value              0.9375                NA
## Neg Pred Value              1.0000                NA
## Prevalence                  0.2609            0.0000
## Detection Rate              0.2609            0.0000
## Detection Prevalence        0.2783            0.3913
## Balanced Accuracy           0.9882                NA
##                      Class: virginica
## Sensitivity                    0.4471
## Specificity                    1.0000
## Pos Pred Value                 1.0000
## Neg Pred Value                 0.3896
## Prevalence                     0.7391
## Detection Rate                 0.3304
## Detection Prevalence           0.3304
## Balanced Accuracy              0.7235
```

# Example 3
k-nearest neighbor and cross-validation

## k-nearest-neighbor

Now, let's make this harder. We will now look at a dataset that is designed to "foil" classifiers. 


```r
library(mlbench)
set.seed(1)
spiral = mlbench.spirals(1000, sd = 0.1)
spiral = data.frame(x = spiral$x[, 1], y = spiral$x[, 2], class = factor(spiral$classes))
library(ggplot2)
ggplot(spiral, aes(x, y, color = class)) + geom_point()
```

## k-nearest-neighbor

![](MachineLearning_files/figure-html/unnamed-chunk-35-1.png)<!-- -->

## Without splitting data


```r
library(caret)
fit = knn3(class ~ ., data = spiral)
confusionMatrix(predict(fit, spiral, type = "class"), spiral$class)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   1   2
##          1 466  30
##          2  34 470
##                                          
##                Accuracy : 0.936          
##                  95% CI : (0.919, 0.9504)
##     No Information Rate : 0.5            
##     P-Value [Acc > NIR] : <2e-16         
##                                          
##                   Kappa : 0.872          
##  Mcnemar's Test P-Value : 0.7077         
##                                          
##             Sensitivity : 0.9320         
##             Specificity : 0.9400         
##          Pos Pred Value : 0.9395         
##          Neg Pred Value : 0.9325         
##              Prevalence : 0.5000         
##          Detection Rate : 0.4660         
##    Detection Prevalence : 0.4960         
##       Balanced Accuracy : 0.9360         
##                                          
##        'Positive' Class : 1              
## 
```

## Cross-validation

setup


```r
library(caret)
indxTrain <- createDataPartition(y = spiral$class, p = 0.75, 
    list = FALSE)
training <- spiral[indxTrain, ]
testing <- spiral[-indxTrain, ]
ctrl <- trainControl(method = "repeatedcv", repeats = 3)
knnFit <- train(class ~ ., data = training, method = "knn", trControl = ctrl, 
    tuneLength = 10)
```

## Cross-validation {.smaller}


```
## k-Nearest Neighbors 
## 
## 750 samples
##   2 predictor
##   2 classes: '1', '2' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold, repeated 3 times) 
## Summary of sample sizes: 674, 675, 674, 675, 675, 675, ... 
## Resampling results across tuning parameters:
## 
##   k   Accuracy   Kappa    
##    5  0.9261745  0.8523488
##    7  0.9284086  0.8568218
##    9  0.9297778  0.8595636
##   11  0.9266724  0.8533571
##   13  0.9253507  0.8507109
##   15  0.9253507  0.8507083
##   17  0.9257716  0.8515490
##   19  0.9235729  0.8471465
##   21  0.9240055  0.8480178
##   23  0.9262339  0.8524781
## 
## Accuracy was used to select the optimal model using 
##  the largest value.
## The final value used for the model was k = 9.
```

# Exercise 4
Ensembles of learners

## What is an ensemble of learners?

In some cases, a machine learning algorithm can have limited predictive power, but using multiple "instances" of such *weak learners* in combination can produce a good result.

It is probably obvious that a classification tree approach might be problematic for a dataset like the `spiral` dataset. In this example, we are going to use "boosting" to combine many trees, each with minimal prediction capabilities, into an "ensemble" learner with reasonable good prediction capabilities.

## Using trees to predict on the `spiral` dataset


```r
library(formatR)
library(party)
trainIdx = sample(c(TRUE, FALSE), nrow(spiral), replace = TRUE, 
    prob = c(0.5, 0.5))
spiralTrain = spiral[trainIdx, ]
trainTree = ctree(class ~ ., spiralTrain)
```

## Using trees to predict on the `spiral` dataset


![](MachineLearning_files/figure-html/unnamed-chunk-40-1.png)<!-- -->

## Using trees to predict on the `spiral` dataset {.smaller}

Training Data


```r
prediction = predict(trainTree, spiralTrain)
confusionMatrix(spiralTrain$class, prediction)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   1   2
##          1 246   2
##          2 177  63
##                                           
##                Accuracy : 0.6332          
##                  95% CI : (0.5887, 0.6761)
##     No Information Rate : 0.8668          
##     P-Value [Acc > NIR] : 1               
##                                           
##                   Kappa : 0.2575          
##  Mcnemar's Test P-Value : <2e-16          
##                                           
##             Sensitivity : 0.5816          
##             Specificity : 0.9692          
##          Pos Pred Value : 0.9919          
##          Neg Pred Value : 0.2625          
##              Prevalence : 0.8668          
##          Detection Rate : 0.5041          
##    Detection Prevalence : 0.5082          
##       Balanced Accuracy : 0.7754          
##                                           
##        'Positive' Class : 1               
## 
```

## Using trees to predict on the `spiral` dataset {.smaller}

Testing data


```r
spiralTest = spiral[!trainIdx, ]
prediction = predict(trainTree, spiralTest)
confusionMatrix(spiralTest$class, prediction)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   1   2
##          1 247   5
##          2 209  51
##                                          
##                Accuracy : 0.582          
##                  95% CI : (0.538, 0.6251)
##     No Information Rate : 0.8906         
##     P-Value [Acc > NIR] : 1              
##                                          
##                   Kappa : 0.1741         
##  Mcnemar's Test P-Value : <2e-16         
##                                          
##             Sensitivity : 0.5417         
##             Specificity : 0.9107         
##          Pos Pred Value : 0.9802         
##          Neg Pred Value : 0.1962         
##              Prevalence : 0.8906         
##          Detection Rate : 0.4824         
##    Detection Prevalence : 0.4922         
##       Balanced Accuracy : 0.7262         
##                                          
##        'Positive' Class : 1              
## 
```

## Using trees to predict on the `spiral` dataset 

Many trees have similar prediction capability, but each is really bad.  This is a 
characteristic of a "weak learner".  Here, we see that in action by performing a bootstrap
sampling (resample with replacement), train, plot, and check prediction accuracy.

## Using trees to predict on the `spiral` dataset

Must be run "locally" to see effect.


```r
plotBootSample = function(spiral) {
    trainIdx = sample(1:nrow(spiral), replace = TRUE)
    spiralTrain = spiral[trainIdx, ]
    trainTree = ctree(class ~ ., spiralTrain)
    plot(trainTree)
    spiralTest = spiral[-trainIdx, ]
    prediction = predict(trainTree, spiralTest)
    print(confusionMatrix(spiralTest$class, prediction)$overall["Accuracy"])
}
```


```r
# press 'ESC' or 'ctrl-c' to stop
while (TRUE) {
    par(ask = TRUE)
    plotBootSample(spiral)
}
```

## Boosting

We can "combine" a bunch of "weak learners", giving more "weight" to hard-to-classify observations as we build each new classifier.  In this case, we will be using the same classification tree approach again.


```r
library(adabag)
trainIdx = sample(c(TRUE, FALSE), nrow(spiral), replace = TRUE, 
    prob = c(0.5, 0.5))
spiralTrain = spiral[trainIdx, ]
boostTree = boosting(class ~ x + y, data = spiralTrain, control = rpart.control(maxdepth = 2))
prediction = predict(boostTree, spiralTrain)
```

## Boosting results {.smaller}


```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   1   2
##          1 222  26
##          2  23 227
##                                          
##                Accuracy : 0.9016         
##                  95% CI : (0.872, 0.9263)
##     No Information Rate : 0.508          
##     P-Value [Acc > NIR] : <2e-16         
##                                          
##                   Kappa : 0.8032         
##  Mcnemar's Test P-Value : 0.7751         
##                                          
##             Sensitivity : 0.9061         
##             Specificity : 0.8972         
##          Pos Pred Value : 0.8952         
##          Neg Pred Value : 0.9080         
##              Prevalence : 0.4920         
##          Detection Rate : 0.4458         
##    Detection Prevalence : 0.4980         
##       Balanced Accuracy : 0.9017         
##                                          
##        'Positive' Class : 1              
## 
```

## A few trees from our ensemble

![](MachineLearning_files/figure-html/unnamed-chunk-47-1.png)<!-- -->

## Boosted trees on test data


```r
spiralTest = spiral[!trainIdx, ]
prediction = predict(boostTree, spiralTest)
confusionMatrix(spiralTest$class, prediction$class)
```

## Boosted trees on test data


```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   1   2
##          1 217  35
##          2  33 217
##                                           
##                Accuracy : 0.8645          
##                  95% CI : (0.8315, 0.8933)
##     No Information Rate : 0.502           
##     P-Value [Acc > NIR] : <2e-16          
##                                           
##                   Kappa : 0.7291          
##  Mcnemar's Test P-Value : 0.9035          
##                                           
##             Sensitivity : 0.8680          
##             Specificity : 0.8611          
##          Pos Pred Value : 0.8611          
##          Neg Pred Value : 0.8680          
##              Prevalence : 0.4980          
##          Detection Rate : 0.4323          
##    Detection Prevalence : 0.5020          
##       Balanced Accuracy : 0.8646          
##                                           
##        'Positive' Class : 1               
## 
```


# Exercise 5
Random forests--ensembles to the max

## Random Forests


```r
library(randomForest)
res = randomForest(Species ~ ., data = iris)
res
```

```
## 
## Call:
##  randomForest(formula = Species ~ ., data = iris) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 4%
## Confusion matrix:
##            setosa versicolor virginica class.error
## setosa         50          0         0        0.00
## versicolor      0         47         3        0.06
## virginica       0          3        47        0.06
```

# sessionInfo

## sessionInfo


```
## R Under development (unstable) (2016-10-26 r71594)
## Platform: x86_64-apple-darwin13.4.0 (64-bit)
## Running under: macOS Sierra 10.12.4
## 
## locale:
## [1] en_US.UTF-8/en_US.UTF-8/en_US.UTF-8/C/en_US.UTF-8/en_US.UTF-8
## 
## attached base packages:
## [1] grid      stats4    stats     graphics  grDevices
## [6] utils     datasets  methods   base     
## 
## other attached packages:
##  [1] e1071_1.6-8         formatR_1.4        
##  [3] randomForest_4.6-12 rpart.plot_2.1.2   
##  [5] adabag_4.1          rpart_4.1-11       
##  [7] mlbench_2.1-1       caret_6.0-76       
##  [9] ggplot2_2.2.1.9000  lattice_0.20-35    
## [11] party_1.2-3         strucchange_1.5-1  
## [13] sandwich_2.3-4      zoo_1.8-0          
## [15] modeltools_0.2-21   mvtnorm_1.0-6      
## [17] knitr_1.16         
## 
## loaded via a namespace (and not attached):
##  [1] coin_1.2-0         reshape2_1.4.2    
##  [3] splines_3.4.0      colorspace_1.3-2  
##  [5] htmltools_0.3.5    yaml_2.1.14       
##  [7] mgcv_1.8-17        survival_2.41-3   
##  [9] rlang_0.1.1        ModelMetrics_1.1.0
## [11] nloptr_1.0.4       multcomp_1.4-6    
## [13] foreach_1.4.3      plyr_1.8.4        
## [15] stringr_1.2.0      MatrixModels_0.4-1
## [17] munsell_0.4.3      gtable_0.2.0      
## [19] codetools_0.2-15   evaluate_0.10     
## [21] labeling_0.3       SparseM_1.77      
## [23] class_7.3-14       quantreg_5.33     
## [25] pbkrtest_0.4-7     parallel_3.4.0    
## [27] TH.data_1.0-8      Rcpp_0.12.11      
## [29] scales_0.4.1       backports_1.0.5   
## [31] lme4_1.1-13        digest_0.6.12     
## [33] stringi_1.1.5      rprojroot_1.2     
## [35] tools_3.4.0        magrittr_1.5      
## [37] lazyeval_0.2.0     tibble_1.3.3      
## [39] car_2.1-4          MASS_7.3-47       
## [41] Matrix_1.2-8       minqa_1.2.4       
## [43] rmarkdown_1.4      iterators_1.0.8   
## [45] compiler_3.4.0     nnet_7.3-12       
## [47] nlme_3.1-131
```
