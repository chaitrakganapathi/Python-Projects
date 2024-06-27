#!/usr/bin/env python
# coding: utf-8

# # Step 1 - Import the required modules and libraries

# In[1]:


# Add numpy library to allow more effective calculations
import numpy as np

# Add pandas library to support data frames
import pandas as pd

# Add seaborn, matplotlib library to support data visualization
import seaborn as seaPlt
import matplotlib.pyplot as matPlt

from sklearn.feature_selection import chi2, SelectKBest
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import GridSearchCV
from imblearn.combine import SMOTEENN
from sklearn.model_selection import cross_val_score

import math
import collections
from collections import Counter

import pickle

# Hide the unwanted warnings
import warnings
warnings.filterwarnings('ignore')


# # Step 2 - Load the PCOS Dataset
# **Dataset Details** - The 'PCOS Dataset' from Kaggle will be analyzed by using our implemented version of Random Forest Classifier. This dataset has information of 541 different patients. The number of follicles in the left and right ovaries, cycle regularity, weight growth, acne problems, etc. are only a few of the 44 features included in the dataset.
# The target feature is ‘PCOS’, which tells if the patient has PCOS or not (Yes/No).
# 
# https://www.kaggle.com/datasets/shreyasvedpathak/pcos-dataset?select=PCOS_data.csv

# In[2]:


# Define the column names
columnNames = ['serial_number', 'patient_file_number', 'is_pcos_present', 'age',
              'weight', 'height', 'bmi', 'blood_group', 'pulse_rate', 'rr_breaths_per_min',
              'hemoglobin', 'is_cycle_regular', 'cycle_length', 'marriage_yrs', 'is_pregnant',
              'number_of_abortions', 'I_beta_hcg', 'II_beta_hcg', 'fsh', 'lh', 'fsh_lh_ratio',
              'hip_inch', 'waist_inch', 'waist_hip_ratio', 'tsh', 'amh', 'prl', 'vit_d3', 'prg',
              'rbs', 'is_weight_gained', 'is_hair_growth', 'is_skin_darkening', 'is_hair_loss',
              'pimples_present', 'eat_fast_food', 'is_regular_exerciser', 'systolic_bp',
              'diastolic_bp', 'left_ovary_follicle_number', 'right_ovary_follicle_number',
              'avg_left_follicle_size', 'avg_right_follicle_size', 'endometrium', 'unnamed']

# Load the PCOS dataset as a panda data frame
pcosDataset = pd.read_csv('PCOS_data.csv')
pcosDataset.columns = columnNames

# Print the number of rows and columns
print("There are", pcosDataset.shape[0], "rows and", pcosDataset.shape[1], "columns in the loaded dataset.")

# Verify that the dataset is properly loaded
pcosDataset.head()


# In[3]:


pcosDataDF = pcosDataset.copy(deep=True)


# ## 2.1 - Explore the PCOS Dataset

# ### a) Examine the dataset for null values and learn about the datatypes

# In[4]:


pcosDataDF.info()


# **The dataset has 3 columns with null values. Furthermore, three columns contain data of the object type, which may need to be transformed into numerical format because ML algorithms only work with numerical data.**  

# ### b) Find the total number of null values present in the dataset

# In[5]:


pcosDataDF.isnull().sum()


# **'marriage_yrs' has 1 null value, 'eat_fast_food' has 1 null value as well, and 'unnamed' has 539 null values in total.**

# ## 2.2 - Clean the PCOS Dataset

# ### a) Drop the unwanted columns

# In[6]:


# Drop the column 'unnamed' containg almost all null and uninterpretable values
pcosDataDF = pcosDataDF.drop(columns='unnamed')
# Also, drop unnecessary columns 'serial_number' and 'patient_file_number', because these coluns have no impact on PCOS prediction
pcosDataDF = pcosDataDF.drop(columns=['serial_number','patient_file_number'])


# In[7]:


pcosDataDF.head()


# ### b) Handle missing values

# In[8]:


# Filled the ‘NAN’ values with mean value of the respective feature
pcosDataDF['marriage_yrs'].fillna(value = round(pcosDataDF['marriage_yrs'].mean(), 1), inplace=True)
pcosDataDF['eat_fast_food'].fillna(value = round(pcosDataDF['eat_fast_food'].mean(), 0), inplace=True)
pcosDataDF.isnull().sum()


# ### c) Replace irrelevant values

# In[9]:


# The information in the column "is_cycle_regular" indicates 
# whether a woman has a regular or irregular period. 
# Regular is denoted by 2, while irregular is denoted by 4. 
# The column contains one incorrect value, which is '5'. 
# As a result, we are changing it to 4.
pcosDataDF['is_cycle_regular'].astype('category').value_counts()


# In[10]:


pcosDataDF["is_cycle_regular"].replace({5: 4}, inplace=True)


# In[11]:


# One of the readings for systolic blood pressure is 12 and for,
# diastolic blood pressure is 8,
# which is impossible. Hence, changing 12 to 120 and 8 to 80.
pcosDataDF["systolic_bp"].replace({12: 120}, inplace=True)
pcosDataDF["diastolic_bp"].replace({8: 80}, inplace=True)


# ### d) Convert object data types into numeric data types ('II    beta-HCG(mIU/mL)', 'AMH(ng/mL)')

# In[12]:


# The columns "II_beta_hcg" & "amh" are an object data type 
# since they contain some irrelevant values.

# Fixing the column 'II_beta_hcg' values
pcosDataDF["II_beta_hcg"].replace({"1.99.": 1.99}, inplace=True)
pcosDataDF["II_beta_hcg"] = pcosDataDF["II_beta_hcg"].astype(float)


# In[13]:


# Fixing the column 'amh' values
pcosDataDF[pcosDataDF["amh"]== "a"].T


# In[14]:


# Only one row in the dataset has an invalid valid as 'a' for the 'AMH' column. 
# Hence, we are just dropping that record.
pcosDataDF.drop(pcosDataDF.loc[pcosDataDF["amh"] == "a"].index, inplace = True);


# In[15]:


# Due to the fact that we have already dealt with the string values, 
# we can convert 'AMH' column to the float datatype.
pcosDataDF["amh"] = pcosDataDF["amh"].astype(float)


# In[16]:


pcosDataDF.info()


# ## 2.3 - Visualization of dataset features

# In[17]:


pcodCount = pcosDataDF['is_pcos_present'].value_counts()
print("The dataset contains", pcodCount[0], "patients without PCOD and", pcodCount[1], "patients with PCOD.")
matPlt.figure(figsize=(3, 3))
seaPlt.countplot( x='is_pcos_present', data = pcosDataDF);


# In[18]:


matPlt.figure(figsize=(3,3))
pieChart = pcosDataDF['is_pcos_present'].value_counts()
explode = (0.05, 0)
colors = ['blue', 'lawngreen']
labels = ['benign', 'malignant']
seaPlt.set(font_scale = 1.2)
matPlt.pie(pieChart, labels = ('PCOS(N)', 'PCOS(Y)'), autopct = "%.2f%%", explode = explode, colors = colors)


# In[19]:


matPlt.figure(figsize=(5, 5))
seaPlt.countplot(x ='cycle_length',data=pcosDataDF)


# In[20]:


matPlt.figure(figsize=(4, 4))
seaPlt.countplot(x='systolic_bp', data=pcosDataDF);


# In[21]:


# After cleaning the dataset, 
# print the number of rows and columns that are present.
print("After cleaning the dataset, there are", pcosDataDF.shape[0], "rows and", pcosDataDF.shape[1], "columns in the dataset.")


# In[22]:


pcosDataDF.head()


# ## 2.4 - Prepare the data required to build the model

# In[23]:


# Create xFeatures and yTarget data
xFeatures = pcosDataDF.iloc[:, 1:42]

# PCOS (Y/N)
yTarget = pd.DataFrame(pcosDataDF.iloc[:, 0])


# # Step 3 - Implement the Decision Tree Classifier

# ## 3.1 - Define class TreeNode
# A class that represents a node in a decision tree

# In[24]:


class TreeNode:
    # Constructor for initializing instance variables
    def __init__(self, featurePointer=None, splitValue=None, leftSubTree=None, rightSubTree=None, informationGain=None, 
                 leafNodeValue=None):
        
        # Attributes of a decision node
        # featurePointer - the chosen feature for splitting the data into two halves
        # splitValue - the chosen feature value for splitting the data into two halves
        # leftSubTree - the left sub-tree
        # rightSubTree - the right sub-tree
        # informationGain - the information gained through data splitting
        self.featurePointer = featurePointer
        self.splitValue = splitValue
        self.leftSubTree = leftSubTree
        self.rightSubTree = rightSubTree
        self.informationGain = informationGain
        
        # Attributes of a leaf node
        # leafNodeValue - the value of the leaf node (the type of class)
        self.leafNodeValue = leafNodeValue


# ## 3.2 - Define class DecisionTree
# A class that represents a decision tree

# In[25]:


class DecisionTree:
    # Constructor for initializing instance variables
    def __init__(self, minimumSamplesSplit = 2, maximumTreeDepth = 2, maxFeatures = None, parent = "RandomForest"):
        
        # minimumSamplesSplit - is the minimum number of samples required in the dataset for it to be splitted
        # maximumTreeDepth - tells the tree how deep it should grow
        # maxFeatures - The number of features to consider when looking for the best split
        # parent - Parent's name (RandomForest/ExtraTree)
        # treeRoot - root node of the tree
        self.minimumSamplesSplit = minimumSamplesSplit
        self.maximumTreeDepth = maximumTreeDepth
        self.maxFeatures = maxFeatures 
        self.parent = parent
        self.treeRoot = None
        
        # A dictionary for storing the feature names 
        # and information gain of decision tree nodes
        self.featureImportances = {}
        

    # Summary - A function that uses recursion to construct the decision tree
    # 1) Separate the features columns from the target column in the loaded dataset.
    # 2) Determine the total number of features and samples in the dataset.
    # 3) Calculate the optimal split if the number of samples in the dataset is larger than or equal to 
    # the'minimumSamplesSplit' and the current tree depth is less than or equal to the'maximumTreeDepth'.
    # 3.1) If the information gain from the split is larger than zero, employ the same function 'constructDecisionTree' 
    # to build the left and right sub trees.
    # 3.2) Create and return an instance of class TreeNode
    # 4) Assign the leaf node value to the class with the most occurrences in the dataset.
    # 5) Create and return an instance of class TreeNode
    # Param 1 - self - Used to access methods of the class and instance variables
    # Param 2 - loadedDataset - the loaded dataset
    # Param 3 - currentTreeDepth - the current depth of the tree
    # Returns - An instance of class TreeNode
    def constructDecisionTree(self, loadedDataset, currentTreeDepth=0):
        
        # Declarations
        xFeatures, yTarget = loadedDataset[:,:-1], loadedDataset[:,-1]
        numberOfSamples, numberOfFeatures = np.shape(xFeatures)
        
        if numberOfSamples >= self.minimumSamplesSplit and currentTreeDepth <= self.maximumTreeDepth:
            # 3 - Calculate the best split
            bestSplit = self.gainBestSplit(numberOfSamples, numberOfFeatures, loadedDataset)
            
            # 3.1 - If the information gain from the split is larger than zero, employ the same function 'constructDecisionTree'to build the left and right sub trees 
            if bestSplit["informationGain"] > 0:
                leftSubTree = self.constructDecisionTree(bestSplit["leftDataset"], currentTreeDepth + 1)
                rightSubTree = self.constructDecisionTree(bestSplit["rightDataset"], currentTreeDepth + 1)
                
                # 3.2 - Create and return an instance of class TreeNode
                return TreeNode(bestSplit["featurePointer"], bestSplit["splitValue"], 
                              leftSubTree, rightSubTree, bestSplit["informationGain"])
            
        # 4 - Assign the leaf node value to the class with the most occurrences in the dataset.
        yTarget = list(yTarget)
        leafValue = max(yTarget, key=yTarget.count)
        
        # 5 - Create and return an instance of class TreeNode
        return TreeNode(leafNodeValue=leafValue)
                
    # Summary - A function that determines the dataset's optimal split
    # 1) Iterate through the randomly selected features
    # 1.1) If parent is 'RandomForest', then for every feature, extract its unique values
    # 1.1) If parent is 'ExtraTree', then for every feature, 
    # draw a random cut-point that uniformly lies between its min and max value
    # 1.2) Iterate through each unique value(RandomForest) 
    # or just the single value random cut-point(ExtraTree)
    # 1.2.1) Split the dataset into two halves(left child, right child) by 'thresholdValue'
    # 1.2.2) Calculate the information gain of the split
    # 1.2.3) If the 'currentInformationGain' information gain is greater than 'maximumInformationGain':
    # 1.2.3.1) Update the 'bestSplit' dictionary and assign the 'currentInformationGain' to 'maximumInformationGain'
    # 2) Return the dictionary 'bestSplit'
    # Param 1 - self - Used to access methods of the class and instance variables
    # Param 2 - numberOfSamples - number of samples currently present in the dataset
    # Param 3 - numberOfFeatures - number of features in the dataset
    # Param 4 - dataset
    # Returns - Dictionary 'bestSplit'
    def gainBestSplit(self, numberOfSamples, numberOfFeatures, dataset):
        
        # Declarations
        # bestSplit - A dictionary-type variable to hold data about the data split
        bestSplit = {}
        maximumInformationGain = -float("inf")
        featureIdxs = np.random.choice(numberOfFeatures, self.maxFeatures, replace = False)
        
        # 1) Iterate through the randomly selected features
        for featurePointer in featureIdxs:
            featureValues = dataset[:, featurePointer]
            
            # 1.1) If parent is 'RandomForest', then for every feature, extract its unique values
            potentialThresholds = np.unique(featureValues)
            
            # 1.1) If parent is 'ExtraTree', then for every feature, 
            # draw a random cut-point that uniformly lies between its min and max value
            if self.parent is "ExtraTree":
                minValue = np.min(potentialThresholds)
                maxValue = np.max(potentialThresholds)
                randomCutPoint = np.random.uniform(minValue, maxValue)
                potentialThresholds = [randomCutPoint]
            
            # 1.2) Iterate through each unique value
            for thresholdValue in potentialThresholds:
                # 1.2.1) Split the dataset into two halves(left child, right child) by 'thresholdValue'
                leftSubTreeData, rightSubTreeData = self.splitData(featurePointer, thresholdValue, dataset)
                
                # 1.2.2) Calculate the information gain of this split
                if len(leftSubTreeData) > 0 and len(rightSubTreeData) > 0:
                    parentTargetData, leftTargetData, rightTargetData = dataset[:,-1], leftSubTreeData[:,-1], rightSubTreeData[:,-1]
                    currentInformationGain = self.calculateInformationGain(parentTargetData, leftTargetData, rightTargetData)
                else:
                    currentInformationGain = 0
                    
                if currentInformationGain > maximumInformationGain:
                    bestSplit["featurePointer"] = featurePointer
                    bestSplit["splitValue"] = thresholdValue
                    bestSplit["leftDataset"] = leftSubTreeData
                    bestSplit["rightDataset"] = rightSubTreeData
                    bestSplit["informationGain"] = currentInformationGain
                    maximumInformationGain = currentInformationGain
                        
        return bestSplit
                
    # Summary - A function that splits the dataset in half by feature and its split value
    # Param 1 - self - Used to access methods of the class and instance variables
    # Param 2 - featurePointer - the chosen feature for splitting the data into two halves
    # Param 3 - splitValue - the chosen feature value for splitting the data into two halves
    # Param 4 - dataset
    # Returns - left and right sub tree data
    def splitData(self, featurePointer, splitValue, dataset):
        leftSubTreeData = np.array([dataRow for dataRow in dataset if dataRow[featurePointer] <= splitValue])
        rightSubTreeData = np.array([dataRow for dataRow in dataset if dataRow[featurePointer] > splitValue])
        return leftSubTreeData, rightSubTreeData
    
    # Summary - A function that calculates the information gain
    # Information Gain(IG) is calculated as follows:
    # IG = Entropy_parent - Entropy_child
    # The entropy of the parent node is 'Entropy_parent', 
    # while the average entropy of the child nodes is 'Entropy_child'.
    # Param 1 - self - Used to access methods of the class and instance variables
    # Param 2 - parentTargetData - parent node target data
    # Param 3 - leftTreeTargetData - left child node target data
    # Param 4 - rightTreeTargetData - right child node target data
    # Returns - information gain
    def calculateInformationGain(self, parentTargetData, leftTreeTargetData, rightTreeTargetData):
        weightOfLeftChild = len(leftTreeTargetData) / len(parentTargetData)
        weightOfRightChild = len(rightTreeTargetData) / len(parentTargetData)
        
        # Calculate the entropy of parent node
        parentEntropy = self.calculateEntropy(parentTargetData)
        
        # Calculate the entropy of the children node
        lChildEntropy = weightOfLeftChild * self.calculateEntropy(leftTreeTargetData)
        rChildEntropy = weightOfRightChild * self.calculateEntropy(rightTreeTargetData)

        # Calculate the information gain
        infoGain = parentEntropy - (lChildEntropy + rChildEntropy)
        
        return infoGain
        
    # Summary - A function that calculates the entropy of a node
    # The entropy formula(E) for the 'Wine Recognition Dataset' will be as follows:
    # E = -(P(class_0)log2(P(class_0)) + P(class_1)log2(P(class_1)) + P(class_2)log2(P(class_2)))
    # where P is the probability of the target class
    # Param 1 - self - Used to access methods of the class and instance variables
    # Param 2 - targetData - the target data of the node
    # Returns - the entropy of node
    def calculateEntropy(self, targetData):
        # Declarations
        nodeEntropy = 0
        
        # Obtain unique class labels, then compute the entropy
        classLabels = np.unique(targetData)
        for clsLbl in classLabels:
            classProbability = len(targetData[targetData == clsLbl]) / len(targetData)
            nodeEntropy += -classProbability * np.log2(classProbability)
            
        return nodeEntropy
    
    # Summary - A function that uses the training data to train the decision tree classifier that has been constructed
    # Param 1 - self - Used to access methods of the class and instance variables
    # Param 2 - xTrainData - features training data
    # Param 3 - yTrainData - target training data
    def fitModel(self, xTrainData, yTrainData):
        self.maxFeatures = xTrainData.shape[1] if not self.maxFeatures else min(self.maxFeatures, xTrainData.shape[1])
        trainingDataset = np.concatenate((xTrainData, yTrainData), axis = 1)
        self.treeRoot = self.constructDecisionTree(trainingDataset)
        
    # Summary - A function that predicts the class of an instance test data
    # Param 1 - self - Used to access methods of the class and instance variables
    # Param 2 - xTestData - testing data which includes only features data
    # Returns - the predictions clsPreditions
    def predictInstanceClass(self, xTestData):
        clsPredictions = [self.makeClsPrediction(xRow, self.treeRoot) for xRow in xTestData]
        return clsPredictions
    
    # Summary - A recursive function that traverses the built-in decision tree to anticipate the class of test data
    # Param 1 - self - Used to access methods of the class and instance variables
    # Param 2 - testRow - test data row which includes only features data
    # Param 3 - decTree - the constructed decision tree model
    # Returns - the prediction for the test data row
    def makeClsPrediction(self, testRow, decTree):
        if decTree.leafNodeValue != None: 
            return decTree.leafNodeValue
        
        # Get the feature value of the feature index
        featureValue = testRow[decTree.featurePointer]
        
        # Travel over the left or right subtree based on the featureValue
        if featureValue <= decTree.splitValue:
            return self.makeClsPrediction(testRow, decTree.leftSubTree)
        else:
            return self.makeClsPrediction(testRow, decTree.rightSubTree) 
        
    # Summary - A recursive function that traverses the TreeNode and displays the corresponding information of each node and its subtrees
    # Param 1 - self - Used to access methods of the class and instance variables
    # Param 2 - independentFeatures - the training features
    # Param 2 - treeNode - The constructed decision tree model
    # Param 3 - space - The constant for displaying space         
    def displayTree(self, independentFeatures, treeNode=None, space=" "):
        if not treeNode:
            treeNode = self.treeRoot

        if treeNode.leafNodeValue is not None:
            print(treeNode.leafNodeValue)

        else:
            print("feature_"+ str(treeNode.featurePointer) + "_" + str(independentFeatures.columns[treeNode.featurePointer]), 
                  "<=", treeNode.splitValue, "?", "InfoGain:", treeNode.informationGain)
            print("%sleft:" % (space), end = "")
            self.displayTree(treeNode.leftSubTree, space + space)
            print("%sright:" % (space), end = "")
            self.displayTree(treeNode.rightSubTree, space + space)
            
    # Summary - A recursive function that calculates the feature importances.
    # The more information gained, the more significant the feature is to the model
    # Param 1 - self - Used to access methods of the class and instance variables
    # Param 2 - independentFeatures - the training features
    # Param 2 - treeNode - The constructed decision tree model  
    def getFeatureImportances(self, independentFeatures, treeNode=None):
        if not treeNode:
            treeNode = self.treeRoot
            
        if treeNode.leafNodeValue is not None:
            return
        else:
            featureName = str(independentFeatures.columns[treeNode.featurePointer])
            infoGain = treeNode.informationGain
            
            if self.featureImportances is not None and featureName in self.featureImportances:
                self.featureImportances[featureName] = self.featureImportances[featureName] + infoGain
            else:
                self.featureImportances[featureName] = infoGain
            
            self.getFeatureImportances(independentFeatures, treeNode.leftSubTree)
            self.getFeatureImportances(independentFeatures, treeNode.rightSubTree)


# # Step 4 - Implement the Random Forest Classifier

# ## 4.1 - Define class RandomForest
# A class that represents a random forest

# In[26]:


class RandomForest:
    # Constructor for initializing instance variables
    def __init__(self, numOfTrees = 25, maxFeatures = None, 
                 minimumSamplesSplit = 3, maximumTreeDepth = 3):
        # numOfTrees - Number of trees to be used for random forest building
        # maxFeatures - The number of features to consider when looking for the best split
        # minimumSamplesSplit - is the minimum number of samples required in the dataset for it to be splitted
        # maximumTreeDepth - tells the tree how deep it should grow
        # decisionTrees - An array storing individually trained decision trees
        self.numOfTrees = numOfTrees
        self.maxFeatures = maxFeatures
        self.minimumSamplesSplit = minimumSamplesSplit
        self.maximumTreeDepth = maximumTreeDepth
        self.decisionTrees = []
        
    @staticmethod
    def _sample(xFeatures, yTarget):
        numberOfRows = xFeatures.shape[0]
        
        # Create samples with replacement
        index = np.random.choice(a = numberOfRows, 
                                   size = numberOfRows, 
                                   replace = True)
        
        return xFeatures[index], yTarget[index]
    
    
    def fit(self, xFeatures, yTarget):
        xFeatures = xFeatures.values
        yTarget = yTarget.values
        
        # Reset the values of the class variables
        if len(self.decisionTrees) > 0:
            self.decisionTrees = []
            
        # If the maxFeatures is not given, then consider 
        # sqaure root value of all the features
        numberOfFeatures = xFeatures.shape[1]
        if self.maxFeatures == None:
            self.maxFeatures = int(math.sqrt(numberOfFeatures))
            
        # Construct each decision tree of the random forest
        countOfBuiltTrees = 0
        while countOfBuiltTrees < self.numOfTrees:
            try:
                # Create an instance of DecisionTree
                decisionTreeClassifier = DecisionTree(minimumSamplesSplit = self.minimumSamplesSplit, 
                                                      maximumTreeDepth = self.maximumTreeDepth,
                                                      maxFeatures = self.maxFeatures)
                
                # Obtain bootstrapped dataset
                sampledXFeatures, sampledYTarget = self._sample(xFeatures, yTarget)
                
                # Train the decision tree on the training data
                decisionTreeClassifier.fitModel(sampledXFeatures, sampledYTarget)
                
                # Save the classifier
                self.decisionTrees.append(decisionTreeClassifier)
            
                countOfBuiltTrees += 1
                
            except Exception as e:
                print("An error has occurred while building decision trees of a random forest:", e)
                countOfBuiltTrees += 1
                continue
                
    def predict(self, xTestData):
        xTestData = xTestData.values
        
        # Make predictions with every decision tree in the forest
        yPredictions = []
        
        for tree in self.decisionTrees:
            yPredictions.append(tree.predictInstanceClass(xTestData))
        
        # Reshape so we can find the most common value
        yPredictions = np.swapaxes(a = yPredictions, 
                                   axis1 = 0, 
                                   axis2 = 1)
        
        # Use majority voting for the final prediction
        finalPredictions = []
        for preds in yPredictions:
            counter = Counter(preds)
            finalPredictions.append(counter.most_common(1)[0][0])
        return finalPredictions


# # Step 5 - Implement the Extra Tree Classifier

# In[27]:


class ExtraTreeClassifier:
    # Constructor for initializing instance variables
    def __init__(self, numOfTrees = 25, maxFeatures = None, 
                 minimumSamplesSplit = 3, maximumTreeDepth = 3):
        # numOfTrees - Number of trees to be used for random forest building
        # maxFeatures - The number of features to consider when looking for the best split:
        # minimumSamplesSplit - is the minimum number of samples required in the dataset for it to be splitted
        # maximumTreeDepth - tells the tree how deep it should grow
        self.numOfTrees = numOfTrees
        self.maxFeatures = maxFeatures
        self.minimumSamplesSplit = minimumSamplesSplit
        self.maximumTreeDepth = maximumTreeDepth
        
        # A dictionary for storing the feature names 
        # and information gain of decision tree nodes
        # of all decision trees
        self.treeImportances = {}
        
    def fit(self, xFeatures, yTarget):
        
        xFeaturesDF = xFeatures
        xFeatures = xFeatures.values
        yTarget = yTarget.values
        
        # If the maxFeatures is not given, then consider 
        # sqaure root value of all the features.
        numberOfFeatures = xFeatures.shape[1]
        if self.maxFeatures == None:
            self.maxFeatures = int(math.sqrt(numberOfFeatures))
            
        # Construct each decision tree of the extra tree classifier
        countOfBuiltTrees = 0
        while countOfBuiltTrees < self.numOfTrees:
            try:
                # Create an instance of DecisionTree
                decisionTreeClassifier = DecisionTree(minimumSamplesSplit = self.minimumSamplesSplit, 
                                                      maximumTreeDepth = self.maximumTreeDepth,
                                                      maxFeatures = self.maxFeatures, 
                                                      parent = "ExtraTree")

                # Train the decision tree on the training data
                decisionTreeClassifier.fitModel(xFeatures, yTarget)

                # Get the feature importances for this decision tree
                # And store the data in self.treeImportances
                decisionTreeClassifier.getFeatureImportances(xFeaturesDF)
                featuresImps = decisionTreeClassifier.featureImportances
                for featureKey, featureValue in featuresImps.items():
                    if self.treeImportances is not None and featureKey in self.treeImportances:
                        self.treeImportances[featureKey] = self.treeImportances[featureKey] + featureValue
                    else:
                        self.treeImportances[featureKey] = featureValue

                countOfBuiltTrees += 1
                
            except Exception as e:
                print("An error has occurred while building decision trees of an extra tree classifier.:", e)
                countOfBuiltTrees += 1
                continue
                
    def getFeatureImportances(self):
        return self.treeImportances


# # Step 6 - Define the constants

# In[28]:


numberOfFeaturesUsedForAnalysis = 20
extraTreeFeatureExtractionRandomState = 99
crossFolds = 5


# # Step 7 - Applying Feature Selection Techniques
# **The following feature selection strategies were employed in the research paper.**
# - Extra Tree Classifier
# - Chi-Square Method
# - Correlation Matrix

# ## 7.1 - Extra Tree Classifier

# ### a - Use Sklearn's Extra Tree Classifier

# In[29]:


# Use Sklearn's Extra Tree Classifier to extract the top 20 features
sklearnExtraTree = ExtraTreesClassifier(criterion ='entropy', 
                                         random_state = extraTreeFeatureExtractionRandomState, 
                                         max_features = 'sqrt', 
                                         n_estimators = 40,
                                         min_samples_split = 3,
                                         max_depth = 3)

# Training the model
sklearnExtraTree.fit(xFeatures, yTarget)
  
# Computing the importance of each feature,
# and then normalizing the individual importances
sklearnNormalizedFeatureImportances = np.std([tree.feature_importances_ for tree in 
                                        sklearnExtraTree.estimators_],
                                        axis = 0)

skLearnFeatureImpSeries = pd.Series(sklearnNormalizedFeatureImportances, 
                                    index = xFeatures.columns)

skLearnFeatureImpSeries.sort_values(ascending = False , inplace = True)

# Print the top 'numberOfFeaturesUsedForAnalysis' features,
# obtained through sklearn's extra tree classifier
print(skLearnFeatureImpSeries[0:numberOfFeaturesUsedForAnalysis])


# In[30]:


skLearnFeatureImpDF = pd.DataFrame({'Features':skLearnFeatureImpSeries.index, 
              'Importances':skLearnFeatureImpSeries.values})
skLearnFeatureImpDF.sort_values('Importances', inplace=True)

matPlt.figure(figsize = (50,50))

skLearnTopFeatureImpDF = skLearnFeatureImpDF.tail(numberOfFeaturesUsedForAnalysis)
skLearnTopFeatureImpDF.sort_values('Importances', 
                                   ascending = False).plot(kind = 'barh', 
                                                           y = 'Importances', 
                                                           x = 'Features', 
                                                           color = 'indigo',
                                                           title = 'Sklearn Extra Tree Classifier for Feature Selection')


# **The importance of features might have different values than the ones that are presented in the paper, because of the random nature of feature samples.**

# ### b - Use our implemented Extra Tree Classifier

# In[31]:


#np.random.seed(12)

# Create an instance of ExtraTreeClassifier
extraTrees = ExtraTreeClassifier(numOfTrees = 40, 
                                 minimumSamplesSplit = 3, 
                                 maximumTreeDepth = 3)

# Training the model
extraTrees.fit(xFeatures, yTarget)

# Get the feature importances
# To perform feature selection, 
# each feature is ordered in descending order according to the 
# total information gained of each feature 
# and the user selects the top k features according to his/her choice.
impFeatures = extraTrees.getFeatureImportances()
sortedImpFeatures = sorted(impFeatures.items(), 
                           key = lambda x: x[1], reverse=True)
sortedTopTwentyFeatures = sortedImpFeatures[0:20]

featureNames = []
featureImps = []
for featureName, infoGain in sortedTopTwentyFeatures:
    featureNames.append(featureName)
    featureImps.append(infoGain)
    
sortedTopTwentyFeatures


# In[32]:


featureImpDF = pd.DataFrame({'Features': featureNames, 
              'Importances': featureImps})
featureImpDF.sort_values('Importances', inplace=True)
featureImpDF.sort_values('Importances', 
                         ascending = False).plot(kind = 'barh', 
                                                 y = 'Importances', 
                                                 x = 'Features', 
                                                 color = 'indigo',
                                                 title = 'Our Extra Tree Classifier for Feature Selection')


# ### c - Extract top 5 measureable features

# In[33]:


# Top 5 home measureable features obtained through our and sklearn's extra tree classifier
extraTreeTop5FeaturesDF = xFeatures[['is_skin_darkening', 'is_hair_growth',
                                     'is_weight_gained', 'is_cycle_regular', 
                                     'eat_fast_food']].copy(deep=True)


# ## 7.2 - Sklearn's Chi-Square Method

# In[34]:


# Perform chi-square test on the cleaned dataset using SKLearn's chi2 module
chiSquareTest = chi2(xFeatures, yTarget)
print("The chi square values are:", chiSquareTest[0])
print("The p-values are:", chiSquareTest[1])


# In[35]:


# Features with low p-values are considered to be more strongly 
# associated with the target variable and are selected 
# for further analysis or modeling.
pValues = pd.Series(chiSquareTest[1], index = xFeatures.columns)
pValues.sort_values(ascending = True , inplace = True)

# Display the top 'numberOfFeaturesUsedForAnalysis'
pValues[0:numberOfFeaturesUsedForAnalysis]


# In[36]:


# Top 5 home measureable features obtained through sklearn's chi square test
chiSquareTop5FeaturesDF = xFeatures[['is_hair_growth', 
                                     'is_skin_darkening', 
                                     'is_weight_gained', 
                                     'weight', 
                                     'eat_fast_food']].copy(deep=True)


# ## 7.3 - Correlation Matrix
# Correlation states how the features are related to each other or the target variable.

# In[37]:


# To see correlations between variables, use a heatmap.
correlatedData = pcosDataDF.corr()
matPlt.figure(figsize=(50,30))
seaPlt.heatmap(correlatedData, cmap="YlGnBu", annot=True)
matPlt.show()


# In[38]:


corrMatrixImpFeaturesSeries = correlatedData['is_pcos_present'].copy()
corrMatrixImpFeaturesSeries.sort_values(ascending = False , inplace = True, key=abs)
corrMatrixImpFeaturesSeries
corrMatrixImpFeaturesSeries[1:(numberOfFeaturesUsedForAnalysis + 3)]


# In[39]:


# Top 5 home measureable features obtained through correlation matrix
corrMatrixTop5FeaturesDF = xFeatures[['is_skin_darkening', 'is_hair_growth', 
                                            'is_weight_gained', 'is_cycle_regular',
                                            'eat_fast_food']].copy(deep=True)


# # Step 8 - Separate the data into training and testing sets

# In[40]:


def getTrainTestDataset(featuresData, targetData, dataSplitRandomState):
    # The dataset is divided using the 'train_test_split()' method of the'sklearn.model_selection' module
    # 80% of the data is set aside for training and 20% for testing
    x_train, x_test, y_train, y_test = train_test_split(featuresData, targetData, 
                                                        train_size = 0.8, test_size = 0.2, 
                                                        random_state = dataSplitRandomState)
    
    return x_train, x_test, y_train, y_test


# # Step 9 - Scale the data

# In[41]:


# Scale the independent features
def scaleData(xTrain, columns, xTest = None):
    
    scaler = MinMaxScaler().fit(xTrain)
    
    xTrain = scaler.transform(xTrain)
    xTrain = pd.DataFrame(xTrain)
    xTrain.columns = columns
    
    if xTest is not None:
        xTest = scaler.transform(xTest)
        xTest = pd.DataFrame(xTest)
        xTest.columns = columns
    else:
        return xTrain
    
    return xTrain, xTest


# # Step 10 - Balance the dataset

# In[42]:


# For balancing the dataset, we have used the,
# 'Random Oversampling technique', that randomly duplicate examples
# in the minority class
def balanceDataset(featuresData, targetData, 
                   dataBalanceTechinique = 'RANDOM_OVER',
                   balanceDataRandomState = 43):
    
    resample = RandomOverSampler(random_state = balanceDataRandomState,
                                sampling_strategy = 'minority')
    
    if dataBalanceTechinique is 'SMOTEENN':
        resample = SMOTEENN(random_state = balanceDataRandomState, 
                            sampling_strategy = 'minority')
       
    X, y = resample.fit_resample(featuresData, targetData)
    return X, y


# # Step 11 - Use Random Forest Classifier to analyze the top 5 home-measurable features found using Sklearn's Chi Square Test (Unbalanced Dataset)

# In[43]:


# Print the number of rows and columns
print('---Unbalanced Dataset---')
print("There are", chiSquareTop5FeaturesDF.shape[0], 
      "rows and", 
      chiSquareTop5FeaturesDF.shape[1], 
      "columns in the loaded subset of the dataset.")

pcodCount = yTarget['is_pcos_present'].value_counts()
print("The dataset contains", pcodCount[0], "patients without PCOD and", pcodCount[1], "patients with PCOD.")

matPlt.figure(figsize=(3,3))
plotDataBal = yTarget.value_counts().plot.pie(autopct='%.2f', 
                                    colors = ['blue', 'lawngreen'],
                                   labels = ('0', '1'))

plotDataBal.set_title("Unbalanced Dataset")


# In[44]:


# Calculate the best params using Sklearn's cross-fold(5) validation and GridSearchCSV

# Scale the features
XUnbalancedScaledChi = scaleData(chiSquareTop5FeaturesDF, 
                       chiSquareTop5FeaturesDF.columns)

oldParamGrid = { 
    'n_estimators': [20, 40, 60, 80, 100, 125, 150, 175, 200, 300, 400, 500],
    'max_features': ['sqrt'],
    'max_depth' : [3,4,5,6,7,8],
    'criterion' :['entropy'],
    'random_state' : [12, 20, 43, 56, 99, 100],
    'min_samples_split':[2,3]
}

grid = { 
    'n_estimators': [40, 80, 100],
    'max_features': ['sqrt'],
    'max_depth' : [4,5,6],
    'criterion' :['entropy'],
    'random_state' : [12, 20, 100],
    'min_samples_split':[2]
}


# Apply Grid Search CSV and 5-fold cross validation
gridSearchCrossFoldChi = GridSearchCV(estimator = RandomForestClassifier(), 
                                      param_grid = grid, 
                                      cv = crossFolds, 
                                      scoring = 'accuracy',
                                      n_jobs = -1)

gridSearchCrossFoldChi.fit(XUnbalancedScaledChi, yTarget)

# Print the scores
gridSearchCrossFoldChiUnbalancedBestParams = gridSearchCrossFoldChi.best_params_

print(gridSearchCrossFoldChi.best_params_)
print(gridSearchCrossFoldChi.best_score_)


# In[45]:


# Split the features into training and testing
xTrainUnChi, xTestUnChi, yTrainUnChi, yTestUnChi = getTrainTestDataset(chiSquareTop5FeaturesDF, 
                                                               yTarget, 
                                                               gridSearchCrossFoldChiUnbalancedBestParams['random_state'])

# Scale the independent features
xTrainChiUnScaled, xTestChiUnScaled = scaleData(xTrainUnChi, xTrainUnChi.columns, xTestUnChi)

# Plot the class distribution
matPlt.figure(figsize=(3,3))
plotDataBal = yTrainUnChi.value_counts().plot.pie(autopct='%.2f', 
                                    colors = ['blue', 'lawngreen'],
                                   labels = ('0', '1'))

plotDataBal.set_title("Unbalanced Training Set")

def plotClassDistribution(targetData, title):
    # Plot the class distribution
    matPlt.figure(figsize=(3,3))
    plotDataBal = targetData.value_counts().plot.pie(autopct='%.2f', 
                                    colors = ['blue', 'lawngreen'],
                                   labels = ('0', '1'))
    plotDataBal.set_title(title)

def getTrainTestScaledData(features, target, 
                    dataSplitRandomState, 
                    balanceTrainingData = False,
                    balStrategy = 'RANDOM_OVER',
                    dataBalanceRandomState = 99):
    # Split the dataset
    xTrain, xTest, yTrain, yTest = getTrainTestDataset(features, 
                                                               target, 
                                                               dataSplitRandomState)
    if balanceTrainingData == True:
        xTrain, yTrain = balanceDataset(xTrain, yTrain, 
                           dataBalanceTechinique = balStrategy,
                           balanceDataRandomState = dataBalanceRandomState)
        
    # Scale the features
    xTrainScaled, xTestScaled = scaleData(xTrain, xTrain.columns, xTest)
    
    return xTrainScaled, xTestScaled, yTrain, yTest


# In[46]:


# Train sklearn random forest classifier on the training dataset
# Create an instance of sklearn's RandomForestClassifier
sklearnRandomForestUnBalanced = RandomForestClassifier(n_estimators = gridSearchCrossFoldChiUnbalancedBestParams['n_estimators'], 
                                 criterion="entropy", 
                                 random_state = gridSearchCrossFoldChiUnbalancedBestParams['random_state'], 
                                 max_depth = gridSearchCrossFoldChiUnbalancedBestParams['max_depth'],
                                 min_samples_split = gridSearchCrossFoldChiUnbalancedBestParams['min_samples_split'],
                                 max_features = 'sqrt' )

# Train the model
sklearnRandomForestUnBalanced.fit(xTrainChiUnScaled, yTrainUnChi)

# Make predictions
yPredictionsChiUnBal = sklearnRandomForestUnBalanced.predict(xTestChiUnScaled)

# Print the scores
print('--Sklearn Random Forest Classifier--')
print(accuracy_score(yTestUnChi, yPredictionsChiUnBal))
print(confusion_matrix(yTestUnChi, yPredictionsChiUnBal))
print(classification_report(yTestUnChi, yPredictionsChiUnBal))

# Train our implementation of random forest classifier on the training dataset
# Create an instance of our Random Forest classifier
customRandomForestChiUnBal = RandomForest(numOfTrees = gridSearchCrossFoldChiUnbalancedBestParams['n_estimators'], 
                                   maximumTreeDepth = gridSearchCrossFoldChiUnbalancedBestParams['max_depth'],
                                   minimumSamplesSplit = gridSearchCrossFoldChiUnbalancedBestParams['min_samples_split'])

# Train the model
customRandomForestChiUnBal.fit(xTrainChiUnScaled, yTrainUnChi)

# Make predictions
yPredictionsChiUnBal = customRandomForestChiUnBal.predict(xTestChiUnScaled)

# Print the scores
print('\n--Our Random Forest Classifier--')
print(accuracy_score(yTestUnChi, yPredictionsChiUnBal))
print(confusion_matrix(yTestUnChi, yPredictionsChiUnBal))
print(classification_report(yTestUnChi, yPredictionsChiUnBal))


# # Step 12 - Use Random Forest Classifier to analyze the top 5 home-measurable features found using Sklearn's Chi Square Test (Balanced Training Dataset)

# In[47]:


# Split the features into training and testing
xTrainUnChi, xTestUnChi, yTrainUnChi, yTestUnChi = getTrainTestScaledData(chiSquareTop5FeaturesDF, yTarget, 
                    gridSearchCrossFoldChiUnbalancedBestParams['random_state'], 
                    balanceTrainingData = True,
                    balStrategy = 'RANDOM_OVER',
                    dataBalanceRandomState = gridSearchCrossFoldChiUnbalancedBestParams['random_state'])


plotClassDistribution(yTrainUnChi, "RANDOM_OVER - Balanced Training Set")

# Train sklearn random forest classifier on the training dataset
# Create an instance of sklearn's RandomForestClassifier
sklearnRandomForestUnBalanced = RandomForestClassifier(n_estimators = gridSearchCrossFoldChiUnbalancedBestParams['n_estimators'], 
                                 criterion="entropy", 
                                 random_state = gridSearchCrossFoldChiUnbalancedBestParams['random_state'], 
                                 max_depth = gridSearchCrossFoldChiUnbalancedBestParams['max_depth'],
                                 min_samples_split = gridSearchCrossFoldChiUnbalancedBestParams['min_samples_split'],
                                 max_features = 'sqrt' )

# Train the model
sklearnRandomForestUnBalanced.fit(xTrainUnChi, yTrainUnChi)

# Make predictions
yPredictionsChiUnBal = sklearnRandomForestUnBalanced.predict(xTestUnChi)

# Print the scores
print('--Sklearn Random Forest Classifier--')
print(accuracy_score(yTestUnChi, yPredictionsChiUnBal))
print(confusion_matrix(yTestUnChi, yPredictionsChiUnBal))
print(classification_report(yTestUnChi, yPredictionsChiUnBal))

# Train our implementation of random forest classifier on the training dataset
# Create an instance of our Random Forest classifier
customRandomForestChiUnBal = RandomForest(numOfTrees = gridSearchCrossFoldChiUnbalancedBestParams['n_estimators'], 
                                   maximumTreeDepth = gridSearchCrossFoldChiUnbalancedBestParams['max_depth'],
                                   minimumSamplesSplit = gridSearchCrossFoldChiUnbalancedBestParams['min_samples_split'])

# Train the model
customRandomForestChiUnBal.fit(xTrainUnChi, yTrainUnChi)

# Make predictions
yPredictionsChiUnBal = customRandomForestChiUnBal.predict(xTestUnChi)

# Print the scores
print('\n--Our Random Forest Classifier--')
print(accuracy_score(yTestUnChi, yPredictionsChiUnBal))
print(confusion_matrix(yTestUnChi, yPredictionsChiUnBal))
print(classification_report(yTestUnChi, yPredictionsChiUnBal))

##############################################################################
# Split the features into training and testing
xTrainUnChi, xTestUnChi, yTrainUnChi, yTestUnChi = getTrainTestScaledData(chiSquareTop5FeaturesDF, yTarget, 
                    gridSearchCrossFoldChiUnbalancedBestParams['random_state'], 
                    balanceTrainingData = True,
                    balStrategy = 'SMOTEENN',
                    dataBalanceRandomState = gridSearchCrossFoldChiUnbalancedBestParams['random_state'])


plotClassDistribution(yTrainUnChi, "SMOTEENN - Balanced Training Set")

# Train sklearn random forest classifier on the training dataset
# Create an instance of sklearn's RandomForestClassifier
sklearnRandomForestUnBalanced = RandomForestClassifier(n_estimators = gridSearchCrossFoldChiUnbalancedBestParams['n_estimators'], 
                                 criterion="entropy", 
                                 random_state = gridSearchCrossFoldChiUnbalancedBestParams['random_state'], 
                                 max_depth = gridSearchCrossFoldChiUnbalancedBestParams['max_depth'],
                                 min_samples_split = gridSearchCrossFoldChiUnbalancedBestParams['min_samples_split'],
                                 max_features = 'sqrt' )

# Train the model
sklearnRandomForestUnBalanced.fit(xTrainUnChi, yTrainUnChi)

# Make predictions
yPredictionsChiUnBal = sklearnRandomForestUnBalanced.predict(xTestUnChi)

# Print the scores
print('--Sklearn Random Forest Classifier--')
print(accuracy_score(yTestUnChi, yPredictionsChiUnBal))
print(confusion_matrix(yTestUnChi, yPredictionsChiUnBal))
print(classification_report(yTestUnChi, yPredictionsChiUnBal))

# Train our implementation of random forest classifier on the training dataset
# Create an instance of our Random Forest classifier
customRandomForestChiUnBal = RandomForest(numOfTrees = gridSearchCrossFoldChiUnbalancedBestParams['n_estimators'], 
                                   maximumTreeDepth = gridSearchCrossFoldChiUnbalancedBestParams['max_depth'],
                                   minimumSamplesSplit = gridSearchCrossFoldChiUnbalancedBestParams['min_samples_split'])

# Train the model
customRandomForestChiUnBal.fit(xTrainUnChi, yTrainUnChi)

# Make predictions
yPredictionsChiUnBal = customRandomForestChiUnBal.predict(xTestUnChi)

# Print the scores
print('\n--Our Random Forest Classifier--')
print(accuracy_score(yTestUnChi, yPredictionsChiUnBal))
print(confusion_matrix(yTestUnChi, yPredictionsChiUnBal))
print(classification_report(yTestUnChi, yPredictionsChiUnBal))


# # Step 13 - Use Random Forest Classifier to analyze the top 5 home-measurable features found using using Extra Tree Classifier & Correlation Matrix (Unbalanced Dataset)

# In[48]:


# Scale the features
XScaledCorrExtraUnBal = scaleData(corrMatrixTop5FeaturesDF, 
                       corrMatrixTop5FeaturesDF.columns)

olderParamGrid = { 
    'n_estimators': [20, 40, 60, 80, 100, 125, 150, 175, 200, 300, 400, 500],
    'max_features': ['sqrt'],
    'max_depth' : [3,4,5,6,7,8],
    'criterion' :['entropy'],
    'random_state' : [12, 20, 43, 56, 99, 100],
    'min_samples_split':[2,3]
}

grid = { 
    'n_estimators': [100, 125, 150],
    'max_features': ['sqrt'],
    'max_depth' : [4,5,6],
    'criterion' :['entropy'],
    'random_state' : [56, 99, 100],
    'min_samples_split':[2]
}

# Apply Grid Search CSV and 5-fold cross validation
gridSearchCrossFoldCorrExtraUnBal = GridSearchCV(estimator = RandomForestClassifier(), 
                                      param_grid = grid, 
                                      cv = crossFolds, 
                                      scoring = 'accuracy',
                                      n_jobs = -1)

gridSearchCrossFoldCorrExtraUnBal.fit(XScaledCorrExtraUnBal, yTarget)

# Print the scores
gridSearchCrossFoldCorrExtraUnBalBestParams = gridSearchCrossFoldCorrExtraUnBal.best_params_

print(gridSearchCrossFoldCorrExtraUnBal.best_params_)
print(gridSearchCrossFoldCorrExtraUnBal.best_score_)


# In[49]:


# Split the features into training and testing
xTrainUnCorr, xTestUnCorr, yTrainUnCorr, yTestUnCorr = getTrainTestDataset(corrMatrixTop5FeaturesDF, 
                                                               yTarget, 
                                                               gridSearchCrossFoldCorrExtraUnBalBestParams['random_state'])

# Scale the independent features
xTrainCorrUnScaled, xTestCorrUnScaled = scaleData(xTrainUnCorr, xTrainUnCorr.columns, xTestUnCorr)

# Plot the class distribution
matPlt.figure(figsize=(3,3))
plotDataBal = yTrainUnCorr.value_counts().plot.pie(autopct='%.2f', 
                                    colors = ['blue', 'lawngreen'],
                                   labels = ('0', '1'))

plotDataBal.set_title("Unbalanced Training Set")


# In[50]:


# Train sklearn random forest classifier on the training dataset
# Create an instance of sklearn's RandomForestClassifier
sklearnRandomForestUnBalanced = RandomForestClassifier(n_estimators = gridSearchCrossFoldCorrExtraUnBalBestParams['n_estimators'], 
                                 criterion="entropy", 
                                 random_state = gridSearchCrossFoldCorrExtraUnBalBestParams['random_state'], 
                                 max_depth = gridSearchCrossFoldCorrExtraUnBalBestParams['max_depth'],
                                 min_samples_split = gridSearchCrossFoldCorrExtraUnBalBestParams['min_samples_split'],
                                 max_features = 'sqrt' )

# Train the model
sklearnRandomForestUnBalanced.fit(xTrainCorrUnScaled, yTrainUnCorr)

# Make predictions
yPredictionsCorrUnBal = sklearnRandomForestUnBalanced.predict(xTestCorrUnScaled)

# Print the scores
print('--Sklearn Random Forest Classifier--')
print(accuracy_score(yTestUnCorr, yPredictionsCorrUnBal))
print(confusion_matrix(yTestUnCorr, yPredictionsCorrUnBal))
print(classification_report(yTestUnCorr, yPredictionsCorrUnBal))

# Train our implementation of random forest classifier on the training dataset
# Create an instance of our Random Forest classifier
customRandomForestCorrUnBal = RandomForest(numOfTrees = gridSearchCrossFoldCorrExtraUnBalBestParams['n_estimators'], 
                                   maximumTreeDepth = gridSearchCrossFoldCorrExtraUnBalBestParams['max_depth'],
                                   minimumSamplesSplit = gridSearchCrossFoldCorrExtraUnBalBestParams['min_samples_split'])

# Train the model
customRandomForestCorrUnBal.fit(xTrainCorrUnScaled, yTrainUnCorr)

# Make predictions
yPredictionsCorrUnBal = customRandomForestCorrUnBal.predict(xTestCorrUnScaled)

# Print the scores
print('\n--Our Random Forest Classifier--')
print(accuracy_score(yTestUnCorr, yPredictionsCorrUnBal))
print(confusion_matrix(yTestUnCorr, yPredictionsCorrUnBal))
print(classification_report(yTestUnCorr, yPredictionsCorrUnBal))


# # Step 12 - Use Random Forest Classifier to analyze the top 5 home-measurable features found using  using Extra Tree Classifier & Correlation Matrix  (Balanced Training Dataset)

# In[51]:


# Split the features into training and testing
xTrainUnCorr, xTestUnCorr, yTrainUnCorr, yTestUnCorr = getTrainTestScaledData(corrMatrixTop5FeaturesDF, yTarget, 
                    gridSearchCrossFoldCorrExtraUnBalBestParams['random_state'], 
                    balanceTrainingData = True,
                    balStrategy = 'RANDOM_OVER',
                    dataBalanceRandomState = gridSearchCrossFoldCorrExtraUnBalBestParams['random_state'])

plotClassDistribution(yTrainUnCorr, "RANDOM_OVER - Balanced Training Set")

# Train sklearn random forest classifier on the training dataset
# Create an instance of sklearn's RandomForestClassifier
sklearnRandomForestUnBalanced = RandomForestClassifier(n_estimators = gridSearchCrossFoldCorrExtraUnBalBestParams['n_estimators'], 
                                 criterion="entropy", 
                                 random_state = gridSearchCrossFoldCorrExtraUnBalBestParams['random_state'], 
                                 max_depth = gridSearchCrossFoldCorrExtraUnBalBestParams['max_depth'],
                                 min_samples_split = gridSearchCrossFoldCorrExtraUnBalBestParams['min_samples_split'],
                                 max_features = 'sqrt' )

# Train the model
sklearnRandomForestUnBalanced.fit(xTrainUnCorr, yTrainUnCorr)

# Make predictions
yPredictionsCorrUnBal = sklearnRandomForestUnBalanced.predict(xTestUnCorr)

# Print the scores
print('--Sklearn Random Forest Classifier--')
print(accuracy_score(yTestUnCorr, yPredictionsCorrUnBal))
print(confusion_matrix(yTestUnCorr, yPredictionsCorrUnBal))
print(classification_report(yTestUnCorr, yPredictionsCorrUnBal))

# Train our implementation of random forest classifier on the training dataset
# Create an instance of our Random Forest classifier
customRandomForestCorrUnBal = RandomForest(numOfTrees = gridSearchCrossFoldCorrExtraUnBalBestParams['n_estimators'], 
                                   maximumTreeDepth = gridSearchCrossFoldCorrExtraUnBalBestParams['max_depth'],
                                   minimumSamplesSplit = gridSearchCrossFoldCorrExtraUnBalBestParams['min_samples_split'])

# Train the model
customRandomForestCorrUnBal.fit(xTrainUnCorr, yTrainUnCorr)

# Make predictions
yPredictionsCorrUnBal = customRandomForestCorrUnBal.predict(xTestUnCorr)

# Print the scores
print('\n--Our Random Forest Classifier--')
print(accuracy_score(yTestUnCorr, yPredictionsCorrUnBal))
print(confusion_matrix(yTestUnCorr, yPredictionsCorrUnBal))
print(classification_report(yTestUnCorr, yPredictionsCorrUnBal))

##############################################################################

# Split the features into training and testing
xTrainUnCorr, xTestUnCorr, yTrainUnCorr, yTestUnCorr = getTrainTestScaledData(corrMatrixTop5FeaturesDF, yTarget, 
                    gridSearchCrossFoldCorrExtraUnBalBestParams['random_state'], 
                    balanceTrainingData = True,
                    balStrategy = 'SMOTEENN',
                    dataBalanceRandomState = gridSearchCrossFoldCorrExtraUnBalBestParams['random_state'])

plotClassDistribution(yTrainUnCorr, "SMOTEENN - Balanced Training Set")

# Train sklearn random forest classifier on the training dataset
# Create an instance of sklearn's RandomForestClassifier
sklearnRandomForestUnBalanced = RandomForestClassifier(n_estimators = gridSearchCrossFoldCorrExtraUnBalBestParams['n_estimators'], 
                                 criterion="entropy", 
                                 random_state = gridSearchCrossFoldCorrExtraUnBalBestParams['random_state'], 
                                 max_depth = gridSearchCrossFoldCorrExtraUnBalBestParams['max_depth'],
                                 min_samples_split = gridSearchCrossFoldCorrExtraUnBalBestParams['min_samples_split'],
                                 max_features = 'sqrt' )

# Train the model
sklearnRandomForestUnBalanced.fit(xTrainUnCorr, yTrainUnCorr)

# Make predictions
yPredictionsCorrUnBal = sklearnRandomForestUnBalanced.predict(xTestUnCorr)

# Print the scores
print('--Sklearn Random Forest Classifier--')
print(accuracy_score(yTestUnCorr, yPredictionsCorrUnBal))
print(confusion_matrix(yTestUnCorr, yPredictionsCorrUnBal))
print(classification_report(yTestUnCorr, yPredictionsCorrUnBal))

# Train our implementation of random forest classifier on the training dataset
# Create an instance of our Random Forest classifier
customRandomForestCorrUnBal = RandomForest(numOfTrees = gridSearchCrossFoldCorrExtraUnBalBestParams['n_estimators'], 
                                   maximumTreeDepth = gridSearchCrossFoldCorrExtraUnBalBestParams['max_depth'],
                                   minimumSamplesSplit = gridSearchCrossFoldCorrExtraUnBalBestParams['min_samples_split'])

# Train the model
customRandomForestCorrUnBal.fit(xTrainUnCorr, yTrainUnCorr)

# Make predictions
yPredictionsCorrUnBal = customRandomForestCorrUnBal.predict(xTestUnCorr)

# Print the scores
print('\n--Our Random Forest Classifier--')
print(accuracy_score(yTestUnCorr, yPredictionsCorrUnBal))
print(confusion_matrix(yTestUnCorr, yPredictionsCorrUnBal))
print(classification_report(yTestUnCorr, yPredictionsCorrUnBal))


# # Step 13 - Use Random Forest Classifier & Random Over Sampling Technique to analyze the top 5 home-measurable features found using Sklearn's Chi Square Test

# ## Step 13.1 - Balance the dataset

# In[52]:


# Balance the dataset
balancedXChi, balancedYChi = balanceDataset(chiSquareTop5FeaturesDF, yTarget, 
                                            dataBalanceTechinique = 'RANDOM_OVER',
                                            balanceDataRandomState = 99)

print('\n---Balanced Dataset---')
print("After balancing the dataset by using the 'RANDOM_OVER' sampling technique",
      "there are", balancedXChi.shape[0], 
      "rows and", 
      balancedXChi.shape[1], 
      "columns in the loaded subset of the dataset.")

pcodCount = balancedYChi['is_pcos_present'].value_counts()
print("The dataset contains", pcodCount[0], "patients without PCOD and", pcodCount[1], "patients with PCOD.")

matPlt.figure(figsize=(3,3))
plotDataBal = balancedYChi.value_counts().plot.pie(autopct='%.2f', 
                                    colors = ['blue', 'lawngreen'],
                                   labels = ('0', '1'))

plotDataBal.set_title("RANDOM OVER Sampling - Balanced Dataset")


# ## Step 13.2 - Apply Grid Search & Cross Fold Validation to dataset

# ### a - Evaluating sklearn Random Forest by applying grid search & cross fold validation

# In[53]:


# Scale the features
balancedXScaledChi = scaleData(balancedXChi, balancedXChi.columns)

# Old grid params (Takes too much time to execute)
# Best Params Obtained: 
# {'criterion': 'entropy', 
# 'max_depth': 7, 
# 'max_features': 'sqrt', 
# 'min_samples_split': 2, 
# 'n_estimators': 100, 
# 'random_state': 99}

oldParamGrid = { 
    'n_estimators': [20, 40, 60, 80, 100, 125, 150, 175, 200, 300, 400, 500],
    'max_features': ['sqrt'],
    'max_depth' : [3,4,5,6,7,8],
    'criterion' :['entropy'],
    'random_state' : [12, 20, 43, 56, 99, 100],
    'min_samples_split':[2,3]
}

grid = { 
    'n_estimators': [40, 80, 100, 125],
    'max_features': ['sqrt'],
    'max_depth' : [5,6,7,8],
    'criterion' :['entropy'],
    'random_state' : [99],
    'min_samples_split':[2,3]
}

# Apply Grid Search CSV and 5-fold cross validation
gridSearchCrossFoldChi = GridSearchCV(estimator = RandomForestClassifier(), 
                                      param_grid = grid, 
                                      cv = crossFolds, 
                                      scoring = 'accuracy',
                                      n_jobs = -1)

gridSearchCrossFoldChi.fit(balancedXScaledChi, balancedYChi)

# Print the scores
gridSearchCrossFoldChiBestParams = gridSearchCrossFoldChi.best_params_

print(gridSearchCrossFoldChi.best_params_)
print(gridSearchCrossFoldChi.best_score_)


# ## Step 13.3 - Split the dataset into training & testing datasets and scale them

# In[54]:


# Split the features into training and testing
xTrainChi, xTestChi, yTrainChi, yTestChi = getTrainTestDataset(balancedXChi, 
                                                               balancedYChi, 
                                                               gridSearchCrossFoldChiBestParams['random_state'])

# Scale the independent features
xTrainChiScaled, xTestChiScaled = scaleData(xTrainChi, xTrainChi.columns, xTestChi)

matPlt.figure(figsize=(3,3))
plotDataBal = yTrainChi.value_counts().plot.pie(autopct='%.2f', 
                                    colors = ['blue', 'lawngreen'],
                                   labels = ('0', '1'))

plotDataBal.set_title("RANDOM OVER SAMPLER - Training Set")


# ## Step 13.4 - Apply Sklearn's Random Forest Classifier on dataset

# In[55]:


# Create an instance of sklearn's RandomForestClassifier
sklearnRandomForestChiBalRO = RandomForestClassifier(n_estimators = gridSearchCrossFoldChiBestParams['n_estimators'], 
                                 criterion="entropy", 
                                 random_state = gridSearchCrossFoldChiBestParams['random_state'], 
                                 max_depth = gridSearchCrossFoldChiBestParams['max_depth'],
                                 min_samples_split = gridSearchCrossFoldChiBestParams['min_samples_split'],
                                 max_features = 'sqrt' )

# Train the model
sklearnRandomForestChiBalRO.fit(xTrainChiScaled, yTrainChi)

# Make predictions
yPredictionsChi = sklearnRandomForestChiBalRO.predict(xTestChiScaled)

# Print the scores
print(accuracy_score(yTestChi, yPredictionsChi))
print(confusion_matrix(yTestChi, yPredictionsChi))
print(classification_report(yTestChi, yPredictionsChi))


# ## Step 13.5 - Apply our implemented Random Forest Classifier on dataset

# In[56]:


# Create an instance of our Random Forest classifier
customRandomForestChiBalRO = RandomForest(numOfTrees = gridSearchCrossFoldChiBestParams['n_estimators'], 
                                   maximumTreeDepth = gridSearchCrossFoldChiBestParams['max_depth'],
                                   minimumSamplesSplit = gridSearchCrossFoldChiBestParams['min_samples_split'])

# Train the model
customRandomForestChiBalRO.fit(xTrainChiScaled, yTrainChi)

# Make predictions
yPredictionsChi = customRandomForestChiBalRO.predict(xTestChiScaled)

# Print the scores
print(accuracy_score(yTestChi, yPredictionsChi))
print(confusion_matrix(yTestChi, yPredictionsChi))
print(classification_report(yTestChi, yPredictionsChi))


# # Step 14 - Use Random Forest Classifier & SMOTEENN Sampling Technique to analyze the top 5 home-measurable features found using Sklearn's Chi Square Test

# ## Step 14.1 - Balance the dataset

# In[57]:


# Balance the dataset
balancedXChiSmote, balancedYChiSmote = balanceDataset(chiSquareTop5FeaturesDF, yTarget, 
                                            dataBalanceTechinique = 'SMOTEENN',
                                            balanceDataRandomState = 99)

print('\n---Balanced Dataset---')
print("After balancing the dataset by using the 'SMOTEENN' sampling technique",
      "there are", balancedXChiSmote.shape[0], 
      "rows and", 
      balancedXChiSmote.shape[1], 
      "columns in the loaded subset of the dataset.")

pcodCount = balancedYChiSmote['is_pcos_present'].value_counts()
print("The dataset contains", pcodCount[0], "patients without PCOD and", pcodCount[1], "patients with PCOD.")

matPlt.figure(figsize=(3,3))
plotDataBal = balancedYChiSmote.value_counts().plot.pie(autopct='%.2f', 
                                    colors = ['blue', 'lawngreen'],
                                   labels = ('0', '1'))

plotDataBal.set_title("SMOTEENN Sampling - Balanced Dataset")


# ## Step 14.2 - Apply Grid Search & Cross Fold Validation to dataset

# ### a - Evaluating sklearn Random Forest by applying grid search & cross fold validation

# In[58]:


# Scale the features
balancedXScaledChiSmote = scaleData(balancedXChiSmote, balancedXChiSmote.columns)

# Old grid params (Takes too much time to execute)
# Best Params Obtained: 
# {'criterion': 'entropy', 
# 'max_depth': 8, 
# 'max_features': 'sqrt', 
# 'min_samples_split': 2, 
# 'n_estimators': 40, 
# 'random_state': 100}
oldParamGrid = { 
    'n_estimators': [20, 40, 60, 80, 100, 125, 150, 175, 200, 300, 400, 500],
    'max_features': ['sqrt'],
    'max_depth' : [3,4,5,6,7,8],
    'criterion' :['entropy'],
    'random_state' : [12, 20, 43, 56, 99, 100],
    'min_samples_split':[2,3]
}

grid = { 
    'n_estimators': [20, 40, 60, 80, 100],
    'max_features': ['sqrt'],
    'max_depth' : [6,7,8],
    'criterion' :['entropy'],
    'random_state' : [56, 99, 100],
    'min_samples_split':[2,3]
}


# Apply Grid Search CSV and 5-fold cross validation
gridSearchCrossFoldChi = GridSearchCV(estimator = RandomForestClassifier(), 
                                      param_grid = grid, 
                                      cv = crossFolds, 
                                      scoring = 'accuracy',
                                      n_jobs = -1)

gridSearchCrossFoldChi.fit(balancedXScaledChiSmote, balancedYChiSmote)

# Print the scores
gridSearchCrossFoldChiBestParamsSmote = gridSearchCrossFoldChi.best_params_
print(gridSearchCrossFoldChi.best_params_)
print(gridSearchCrossFoldChi.best_score_)


# ## Step 14.3 - Split the dataset into training & testing datasets and scale them

# In[59]:


# Split the features into training and testing
xTrainChi, xTestChi, yTrainChi, yTestChi = getTrainTestDataset(balancedXScaledChiSmote, 
                                                               balancedYChiSmote, 
                                                               gridSearchCrossFoldChiBestParamsSmote['random_state'])

# Scale the independent features
xTrainChiScaledSmote, xTestChiScaledSmote = scaleData(xTrainChi, xTrainChi.columns, xTestChi)

matPlt.figure(figsize=(3,3))
plotDataBal = yTrainChi.value_counts().plot.pie(autopct='%.2f', 
                                    colors = ['blue', 'lawngreen'],
                                   labels = ('0', '1'))

plotDataBal.set_title("SMOTEENN - Training Set")


# ## Step 14.4 - Apply Sklearn's Random Forest Classifier on dataset

# In[60]:


# Create an instance of sklearn's RandomForestClassifier
sklearnRandomForestChiBalSmote = RandomForestClassifier(n_estimators = gridSearchCrossFoldChiBestParamsSmote['n_estimators'], 
                                 criterion="entropy", 
                                 random_state = gridSearchCrossFoldChiBestParamsSmote['random_state'], 
                                 max_depth = gridSearchCrossFoldChiBestParamsSmote['max_depth'],
                                 min_samples_split = gridSearchCrossFoldChiBestParamsSmote['min_samples_split'],
                                 max_features = 'sqrt' )

# Train the model
sklearnRandomForestChiBalSmote.fit(xTrainChiScaledSmote, yTrainChi)

# Make predictions
yPredictionsChiSmote = sklearnRandomForestChiBalSmote.predict(xTestChiScaledSmote)

# Print the scores
print(accuracy_score(yTestChi, yPredictionsChiSmote))
print(confusion_matrix(yTestChi, yPredictionsChiSmote))
print(classification_report(yTestChi, yPredictionsChiSmote))


# ## Step 14.5 - Apply our implemented Random Forest Classifier on dataset

# In[61]:


# Create an instance of our Random Forest classifier
customRandomForestChiBalSmote = RandomForest(numOfTrees = gridSearchCrossFoldChiBestParamsSmote['n_estimators'], 
                                   maximumTreeDepth = gridSearchCrossFoldChiBestParamsSmote['max_depth'],
                                   minimumSamplesSplit = gridSearchCrossFoldChiBestParamsSmote['min_samples_split'])

# Train the model
customRandomForestChiBalSmote.fit(xTrainChiScaledSmote, yTrainChi)

# Make predictions
yPredictionsChiSmote = customRandomForestChiBalSmote.predict(xTestChiScaledSmote)

# Print the scores
print(accuracy_score(yTestChi, yPredictionsChiSmote))
print(confusion_matrix(yTestChi, yPredictionsChiSmote))
print(classification_report(yTestChi, yPredictionsChiSmote))


# # Step 15 - Use Random Forest Classifier & & Random Over Sampling Technique to analyze the top 5 home-measurable features found using Extra Tree Classifier & Correlation Matrix

# In[62]:


# Print the number of rows and columns
print('---Unbalanced Dataset---')
print("There are", corrMatrixTop5FeaturesDF.shape[0], 
      "rows and", 
      corrMatrixTop5FeaturesDF.shape[1], 
      "columns in the loaded subset of the dataset.")

pcodCount = yTarget['is_pcos_present'].value_counts()
print("The dataset contains", pcodCount[0], "patients without PCOD and", pcodCount[1], "patients with PCOD.")

matPlt.figure(figsize=(3,3))
plotDataBal = yTarget.value_counts().plot.pie(autopct='%.2f', 
                                    colors = ['blue', 'lawngreen'],
                                   labels = ('0', '1'))

plotDataBal.set_title("Unbalanced Dataset")


# ## Step 15.1 - Balance the dataset

# In[63]:


# Balance the dataset
balancedXExtraCorr, balancedYExtraCorr = balanceDataset(corrMatrixTop5FeaturesDF, yTarget, 
                                            dataBalanceTechinique = 'RANDOM_OVER',
                                            balanceDataRandomState = 99)

print('\n---Balanced Dataset---')
print("After balancing the dataset by using the 'RANDOM_OVER' sampling technique",
      "there are", balancedXExtraCorr.shape[0], 
      "rows and", 
      balancedXExtraCorr.shape[1], 
      "columns in the loaded subset of the dataset.")

pcodCount = balancedYExtraCorr['is_pcos_present'].value_counts()
print("The dataset contains", pcodCount[0], "patients without PCOD and", pcodCount[1], "patients with PCOD.")

matPlt.figure(figsize=(3,3))
plotDataBal = balancedYExtraCorr.value_counts().plot.pie(autopct='%.2f', 
                                    colors = ['blue', 'lawngreen'],
                                   labels = ('0', '1'))

plotDataBal.set_title("RANDOM OVER Sampling - Balanced Dataset")


# ## Step 15.2 - Apply Grid Search & Cross Fold Validation to dataset

# ### a - Evaluating sklearn Random Forest by applying grid search & cross fold validation

# In[64]:


# Scale the features
balancedXExtraCorrScaled = scaleData(balancedXExtraCorr, balancedXExtraCorr.columns)

# Old grid params (Takes too much time to execute)
# Best Params Obtained: 
# {'criterion': 'entropy', 
# 'max_depth': 4, 
# 'max_features': 'sqrt', 
# 'min_samples_split': 2, 
# 'n_estimators': 20, 
# 'random_state': 20}
oldParamGrid = { 
    'n_estimators': [20, 40, 60, 80, 100, 125, 150, 175, 200, 300, 400, 500],
    'max_features': ['sqrt'],
    'max_depth' : [3,4,5,6,7,8],
    'criterion' :['entropy'],
    'random_state' : [12, 20, 43, 56, 99, 100],
    'min_samples_split':[2,3]
}

grid = { 
    'n_estimators': [20, 40, 60, 80, 100],
    'max_features': ['sqrt'],
    'max_depth' : [3,4,5],
    'criterion' :['entropy'],
    'random_state' : [12, 20, 43],
    'min_samples_split':[2,3]
}

# Apply Grid Search CSV and 5-fold cross validation
gridSearchCrossFoldExtraCorr = GridSearchCV(estimator = RandomForestClassifier(), 
                                      param_grid = grid, 
                                      cv = crossFolds, 
                                      scoring = 'accuracy',
                                      n_jobs = -1)

gridSearchCrossFoldExtraCorr.fit(balancedXExtraCorrScaled, balancedYExtraCorr)

# Print the scores
gridSearchCrossFoldExtraCorrBestParams = gridSearchCrossFoldExtraCorr.best_params_

print(gridSearchCrossFoldExtraCorr.best_params_)
print(gridSearchCrossFoldExtraCorr.best_score_)


# ## Step 15.3 - Split the dataset into training & testing datasets and scale them

# In[65]:


# Split the features into training and testing
xTrainExtraCorr, xTestExtraCorr, yTrainExtraCorr, yTestExtraCorr = getTrainTestDataset(balancedXExtraCorr, 
                                                               balancedYExtraCorr, 
                                                               99)

# Scale the independent features
xTrainExtraCorrScaled, xTestExtraCorrScaled = scaleData(xTrainExtraCorr, 
                                                        xTrainExtraCorr.columns, 
                                                        xTestExtraCorr)

matPlt.figure(figsize=(3,3))
plotDataBal = yTrainExtraCorr.value_counts().plot.pie(autopct='%.2f', 
                                    colors = ['blue', 'lawngreen'],
                                   labels = ('0', '1'))

plotDataBal.set_title("RANDOM OVER SAMPLER - Training Set")


# ## Step 15.4 - Apply Sklearn's Random Forest Classifier on dataset

# In[66]:


# Create an instance of sklearn's RandomForestClassifier
sklearnRandomForestCorrBalRO = RandomForestClassifier(n_estimators = gridSearchCrossFoldExtraCorrBestParams['n_estimators'], 
                                 criterion="entropy", 
                                 random_state = gridSearchCrossFoldExtraCorrBestParams['random_state'], 
                                 max_depth = gridSearchCrossFoldExtraCorrBestParams['max_depth'],
                                 min_samples_split = gridSearchCrossFoldExtraCorrBestParams['min_samples_split'],
                                 max_features = 'sqrt' )

# Train the model
sklearnRandomForestCorrBalRO.fit(xTrainExtraCorrScaled, yTrainExtraCorr)

# Make predictions
yPredictionsExtraCorr = sklearnRandomForestCorrBalRO.predict(xTestExtraCorr)

# Print the scores
print(accuracy_score(yTestExtraCorr, yPredictionsExtraCorr))
print(confusion_matrix(yTestExtraCorr, yPredictionsExtraCorr))
print(classification_report(yTestExtraCorr, yPredictionsExtraCorr))


# ## Step 15.5 - Apply our implemented Random Forest Classifier on dataset

# In[67]:


# Create an instance of our Random Forest classifier
customRandomForestCorrBalRO = RandomForest(numOfTrees = gridSearchCrossFoldChiBestParams['n_estimators'], 
                                   maximumTreeDepth = gridSearchCrossFoldChiBestParams['max_depth'],
                                   minimumSamplesSplit = gridSearchCrossFoldChiBestParams['min_samples_split'])

# Train the model
customRandomForestCorrBalRO.fit(xTrainExtraCorrScaled, yTrainExtraCorr)

# Make predictions
yPredictionsExtraCorr = customRandomForestCorrBalRO.predict(xTestExtraCorr)

# Print the scores
print(accuracy_score(yTestExtraCorr, yPredictionsExtraCorr))
print(confusion_matrix(yTestExtraCorr, yPredictionsExtraCorr))
print(classification_report(yTestExtraCorr, yPredictionsExtraCorr))


# # Step 16 - Use Random Forest Classifier & SMOTEENN Sampling Technique to analyze the top 5 home-measurable features found using using Extra Tree Classifier & Correlation Matrix

# ## Step 16.1 - Balance the dataset

# In[68]:


# Balance the dataset
balancedXExtraCorrSmote, balancedYExtraCorrSmote = balanceDataset(corrMatrixTop5FeaturesDF, yTarget, 
                                            dataBalanceTechinique = 'SMOTEENN',
                                            balanceDataRandomState = 99)

print('\n---Balanced Dataset---')
print("After balancing the dataset by using the 'SMOTEENN' sampling technique",
      "there are", balancedXExtraCorrSmote.shape[0], 
      "rows and", 
      balancedXExtraCorrSmote.shape[1], 
      "columns in the loaded subset of the dataset.")

pcodCount = balancedYExtraCorrSmote['is_pcos_present'].value_counts()
print("The dataset contains", pcodCount[0], "patients without PCOD and", pcodCount[1], "patients with PCOD.")

matPlt.figure(figsize=(3,3))
plotDataBal = balancedYExtraCorrSmote.value_counts().plot.pie(autopct='%.2f', 
                                    colors = ['blue', 'lawngreen'],
                                   labels = ('0', '1'))

plotDataBal.set_title("SMOTEENN - Balanced Dataset")


# ## Step 16.2 - Apply Grid Search & Cross Fold Validation to dataset

# ### a - Evaluating sklearn Random Forest by applying grid search & cross fold validation

# In[69]:


# Scale the features
balancedXExtraCorrSmoteScaled = scaleData(balancedXExtraCorrSmote, 
                                          balancedXExtraCorrSmote.columns)

# Old grid params (Takes too much time to execute)
# Best Params Obtained: 
# {'criterion': 'entropy', 
# 'max_depth': 3, 
# 'max_features': 'sqrt', 
# 'min_samples_split': 2, 
# 'n_estimators': 80, 
# 'random_state': 12}
oldParamGrid = { 
    'n_estimators': [20, 40, 60, 80, 100, 125, 150, 175, 200, 300, 400, 500],
    'max_features': ['sqrt'],
    'max_depth' : [3,4,5,6,7,8],
    'criterion' :['entropy'],
    'random_state' : [12, 20, 43, 56, 99, 100],
    'min_samples_split':[2,3]
}

grid = { 
    'n_estimators': [40, 80, 100, 125],
    'max_features': ['sqrt'],
    'max_depth' : [3,4,5],
    'criterion' :['entropy'],
    'random_state' : [12, 20, 43],
    'min_samples_split':[2,3]
}

# Apply Grid Search CSV and 5-fold cross validation
gridSearchCrossFoldExtraCorrSmote = GridSearchCV(estimator = RandomForestClassifier(), 
                                      param_grid = grid, 
                                      cv = crossFolds, 
                                      scoring = 'accuracy',
                                      n_jobs = -1)

gridSearchCrossFoldExtraCorrSmote.fit(balancedXExtraCorrSmoteScaled, 
                                 balancedYExtraCorrSmote)

# Print the scores
gridSearchCrossFoldExtraCorrSmoteBestParams = gridSearchCrossFoldExtraCorrSmote.best_params_

print(gridSearchCrossFoldExtraCorrSmote.best_params_)
print(gridSearchCrossFoldExtraCorrSmote.best_score_)


# ## Step 16.3 - Split the dataset into training & testing datasets and scale them

# In[70]:


# Split the features into training and testing
xTrainExtraCorrSmote, xTestExtraCorrSmote, yTrainExtraCorrSmote, yTestExtraCorrSmote = getTrainTestDataset(balancedXExtraCorrSmote, 
                                                               balancedYExtraCorrSmote, 
                                                               gridSearchCrossFoldExtraCorrBestParams['random_state'])

# Scale the independent features
xTrainExtraCorrSmoteScaled, xTestExtraCorrSmoteScaled = scaleData(xTrainExtraCorrSmote, 
                                                        xTrainExtraCorrSmote.columns, 
                                                        xTestExtraCorrSmote)

matPlt.figure(figsize=(3,3))
plotDataBal = yTrainExtraCorrSmote.value_counts().plot.pie(autopct='%.2f', 
                                    colors = ['blue', 'lawngreen'],
                                   labels = ('0', '1'))

plotDataBal.set_title("SMOTEENN - Training Set")


# ## Step 16.4 - Apply Sklearn's Random Forest Classifier on dataset

# In[71]:


# Create an instance of sklearn's RandomForestClassifier
sklearnRandomForestCorrBalSmote = RandomForestClassifier(n_estimators = gridSearchCrossFoldExtraCorrBestParams['n_estimators'], 
                                 criterion="entropy", 
                                 random_state = gridSearchCrossFoldExtraCorrBestParams['random_state'], 
                                 max_depth = gridSearchCrossFoldExtraCorrBestParams['max_depth'],
                                 min_samples_split = gridSearchCrossFoldExtraCorrBestParams['min_samples_split'],
                                 max_features = 'sqrt' )

# Train the model
sklearnRandomForestCorrBalSmote.fit(xTrainExtraCorrSmoteScaled, 
                        yTrainExtraCorrSmote)

# Make predictions
yPredictionsExtraCorrSmote = sklearnRandomForestCorrBalSmote.predict(xTestExtraCorrSmote)

# Print the scores
print(accuracy_score(yTestExtraCorrSmote, yPredictionsExtraCorrSmote))
print(confusion_matrix(yTestExtraCorrSmote, yPredictionsExtraCorrSmote))
print(classification_report(yTestExtraCorrSmote, yPredictionsExtraCorrSmote))


# ## Step 16.5 - Apply our implemented Random Forest Classifier on dataset

# In[72]:


# Create an instance of our Random Forest classifier
customRandomForestCorrBalSmote = RandomForest(numOfTrees = gridSearchCrossFoldExtraCorrBestParams['n_estimators'], 
                                   maximumTreeDepth = gridSearchCrossFoldExtraCorrBestParams['max_depth'],
                                   minimumSamplesSplit = gridSearchCrossFoldExtraCorrBestParams['min_samples_split'])

# Train the model
customRandomForestCorrBalSmote.fit(xTrainExtraCorrSmoteScaled, 
                       yTrainExtraCorrSmote)

# Make predictions
yPredictionsExtraCorrSmote = customRandomForestCorrBalSmote.predict(xTestExtraCorrSmote)

#Create pickle file to use in webapp
pickle.dump(customRandomForestCorrBalSmote,open('pickleFinal.pkl','wb'))

# Print the scores
print(accuracy_score(yTestExtraCorrSmote, yPredictionsExtraCorrSmote))
print(confusion_matrix(yTestExtraCorrSmote, yPredictionsExtraCorrSmote))
print(classification_report(yTestExtraCorrSmote, yPredictionsExtraCorrSmote))

