# -*- coding: utf-8 -*-
# Name: Ding Wang
# Email: diw005@ucsd.edu
# PID: A53089251
from pyspark import SparkContext
sc = SparkContext()
# coding: utf-8
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint

from string import split,strip

from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
from pyspark.mllib.util import MLUtils

# Read the file into an RDD
# If doing this on a real cluster, you need the file to be available on all nodes, ideally in HDFS.
path='/covtype/covtype.data'
inputRDD=sc.textFile(path)

Label=2.0
Data=inputRDD.map(lambda line: [float(x) for x in line.split(',')])    .map(lambda V:LabeledPoint(1.0 if V[-1] == Label else 0.0, V[:-1]))

(trainingData,testData)=Data.randomSplit([0.7,0.3], seed=255)
trainingData.cache()
testData.cache()

errors={}
depth = 15
model=GradientBoostedTrees.trainClassifier(trainingData, {}, numIterations=10, maxDepth=depth)
errors[depth]={}
dataSets={'train':trainingData,'test':testData}
for name in dataSets.keys():  # Calculate errors on train and test sets
    data=dataSets[name]
    Predicted=model.predict(data.map(lambda x: x.features))
    LabelsAndPredictions=data.map(lambda x: x.label).zip(Predicted)
    Err = LabelsAndPredictions.filter(lambda (v,p):v != p).count()/float(data.count())
    errors[depth][name]=Err
print depth,errors[depth]
