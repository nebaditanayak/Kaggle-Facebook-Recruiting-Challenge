
# coding: utf-8

# In[114]:


#Make MongoDB connection
from pymongo import MongoClient
client = MongoClient('compute-0-11', 27017)

db = client.FacebookChallenge_akar1
collection1 = db.fb_hw
collection2  = db.fb_hw_test
contentsTrain = collection1.find().limit(1000)
contentsTest = collection2.find().limit(1000)


# In[115]:


#Create spark connector
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql import SQLContext
from pyspark import SparkConf, SparkContext

spark = SparkSession.builder         .appName("Facebook_nebaditn")         .getOrCreate()

sc = spark.sparkContext
sqlContext = SQLContext(sc)


# In[116]:



from bson import json_util, ObjectId
import json
rddSan1 = json.loads(json_util.dumps(contentsTrain))
rddSan2 = json.loads(json_util.dumps(contentsTest))


# In[117]:


#Generate RDD
rddTrain = sc.parallelize(rddSan1)
rddTest = sc.parallelize(rddSan2)


# In[118]:


#Create schema to generate train and test dataframe
schemaTrain = StructType([StructField("Body", StringType(), True), 
                     StructField("Id", IntegerType(), True), 
                     StructField("Tags", StringType(), True),
                     StructField("Title", StringType(), True),
                     StructField("_id", StringType(), True)])
schemaTest = StructType([StructField("Body", StringType(), True), 
                     StructField("Id", IntegerType(), True), 
                     StructField("Title", StringType(), True),
                     StructField("_id", StringType(), True)])

train = sqlContext.createDataFrame(rddTrain, schema=schemaTrain)
test = sqlContext.createDataFrame(rddTest, schema=schemaTest)


# In[119]:


#Print and check the data
train.show()
test.show()


# In[120]:


# Remove HTML Tags from the Body text
#Here I have used a library beautiful soup to remove the HTML tags from the body section in train and test
from bs4 import BeautifulSoup
from pyspark.sql.functions import udf
from pyspark.sql.types import *

train = train.rdd.map(lambda x: (x[0], x[1], x[2], x[3], x[4], BeautifulSoup(x[0]).text))
train = train.toDF()
test = test.rdd.map(lambda x: (x[0], x[1], x[2], x[3], BeautifulSoup(x[0]).text))
test =  test.toDF()


# In[121]:


#Check the new dataframe 
train.show()
test.show()


# In[122]:


#Filter columns from the data frame
workingTrain = train.selectExpr("_2 as ID", "_6 as Body", "_3 as Tags", "_4 as Title")
workingTest = test.selectExpr("_2 as ID", "_5 as Body", "_3 as Title")


# In[123]:


#Again check the data frame
workingTrain.show()
workingTest.show()


# In[125]:


#Select particular columns
workingTrain = workingTrain.selectExpr("ID","Body", "Title", "Tags")
workingTest = workingTest.selectExpr("ID", "Title", "Body")


# In[126]:


#UDF to concatenate two columns
import pyspark.sql.functions as F
from pyspark.sql.types import *
concat_udf = F.udf(lambda cols: "".join([x if x is not None else "*" for x in cols]), StringType())


# In[127]:


#Merge the title and the body tokenized column
   
workingTrain = workingTrain.withColumn("features", concat_udf(F.array("Title", "Body")))


# In[128]:


workingTrain.show()


# In[129]:


#Remove the stop words from the body
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType
#Tokenize the body column
tokenizer = Tokenizer(inputCol = "features" , outputCol = "tokenizedfeatures")
workingTrain = tokenizer.transform(workingTrain) # For the train data


# In[130]:


workingTrain.show()


# In[131]:


#Tokenize the Tags
tagTokenizer = Tokenizer(inputCol="Tags", outputCol="tokenizedTags")
workingTrain = tagTokenizer.transform(workingTrain)
#workingTrain.show()


# In[132]:


#Remove stop words from the Tokenized title
stopWordRemover = StopWordsRemover(inputCol = "tokenizedfeatures", outputCol = "filteredFeatures")
workingTrain = stopWordRemover.transform(workingTrain) #For Train data


# In[133]:


workingTrain = workingTrain.selectExpr("ID","filteredFeatures as features","tokenizedTags")


# In[134]:


#In thi step we are going to set count vectorizer the features
from pyspark.ml.feature import CountVectorizer
cvFeatures = CountVectorizer(inputCol = "features", outputCol="vectorizedFeatures" )
cvModel = cvFeatures.fit(workingTrain)
workingTrain = cvModel.transform(workingTrain)


# In[135]:


cvTag = CountVectorizer(inputCol = "tokenizedTags", outputCol="vectorizedTags" )
cvModelTag = cvTag.fit(workingTrain)
workingTrain = cvModelTag.transform(workingTrain)


# In[137]:


workingTrain = workingTrain.selectExpr("ID", "vectorizedFeatures", "vectorizedTags" )


# In[138]:


from pyspark.sql import SQLContext, Row
from pyspark.ml.feature import CountVectorizer
from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.mllib.linalg import Vector, Vectors

workingTrain2 = workingTrain.selectExpr("ID","vectorizedFeatures","vectorizedTags")
#workingTrain2.show()
corpus = workingTrain2.select("ID", "vectorizedFeatures").rdd.map(lambda (x,y): [x,Vectors.fromML(y)]).cache()


# In[139]:


# Cluster the documents into three topics using LDA
ldaModel1 = LDA.train(corpus, k=10,maxIterations=100,optimizer='online')


# In[140]:


topics = ldaModel1.topicsMatrix()
vocabArray = cvModel.vocabulary


# In[141]:


wordNumbers = 20  # number of words per topic


# In[142]:


topicIndices = sc.parallelize(ldaModel1.describeTopics(maxTermsPerTopic = wordNumbers))


# In[143]:


#Ref: https://stackoverflow.com/questions/42051184/latent-dirichlet-allocation-lda-in-spark

def topic_render(topic):  # specify vector id of words to actual words
    terms = topic[0]
    result = []
    for i in range(wordNumbers):
        term = vocabArray[terms[i]]
        result.append(term)
    return result


# In[144]:


topics_final = topicIndices.map(lambda topic: topic_render(topic)).collect()


# In[145]:


for topic in range(len(topics_final)):
    print ("Topic" + str(topic) + ":")
    for term in topics_final[topic]:
        print (term)
    print ('\n')


# In[154]:


#Since the LDA model mllib doesnt have the transform function, I used the function in ml library to get the probabilities of each post falling into a particular cluster.
from pyspark.ml.clustering import LDA
# Trains a LDA model.

workingTrain3 = workingTrain2.selectExpr("ID", "vectorizedFeatures as features")
lda = LDA(k=10, maxIter=100)
ldamodel = lda.fit(workingTrain3)


# In[156]:


# Shows the result
transformed = ldamodel.transform(workingTrain3)
transformed.show()


# In[157]:


ll = ldamodel.logLikelihood(workingTrain3)
lp = ldamodel.logPerplexity(workingTrain3)
print("The lower bound on the log likelihood of the entire corpus: " + str(ll))
print("The upper bound on perplexity: " + str(lp))


# In[160]:


#Ref: https://spark.apache.org/docs/2.2.0/ml-clustering.html#latent-dirichlet-allocation-lda

#Similarly, we can perform topic modelling using other algorithms like K-means
from pyspark.ml.clustering import KMeans

#Get a fresh copy of dataset
workingTrain4 = workingTrain2.selectExpr("ID", "vectorizedFeatures as features")

# Trains a k-means model.
kmeans = KMeans().setK(2).setSeed(1)
kmeansModel = kmeans.fit(workingTrain4)

# Evaluate clustering by computing Within Set Sum of Squared Errors.
wssse = kmeansModel.computeCost(workingTrain4)
print("Within Set Sum of Squared Errors = " + str(wssse))

# Shows the result.
centers = kmeansModel.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)


# In[164]:


#COmmented the below model as GMM   experiencing out of Memory issues.

#Similarly, we can perform topic modelling using other algorithms GaussianMixture with random initialization
from pyspark.ml.clustering import GaussianMixture

#Get a fresh copy of dataset
#workingTrain5 = workingTrain2.selectExpr("ID", "vectorizedFeatures as features")

#gmm = GaussianMixture().setK(2).setSeed(538009467)
#gmmModel = gmm.fit(workingTrain5)

#print("Gaussians shown as a DataFrame: ")
#gmmModel.gaussiansDF.show(truncate=False)

