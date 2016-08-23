
# coding: utf-8

# Cite:
# D. A. Goldstein, et al. 2015 "Automated Transient Identification in the Dark Energy Survey" AJ (accepted).

# # Background

# * We are aiming here to classify two different types of astronomy images: true data, and artificially injected
# 

# First things first, let's get the pyspark kernel. Open up a Cori terminal and type "module load spark"

# Let's grab the data'

# In[ ]:



# In[1]:



# In[ ]:

from skimage.io import imread, imshow

from matplotlib import pyplot as plt

path_to_sample_image = "/project/projectdirs/dasrepo/data_day/astron-images/srch11802308.gif"



# #### Here is a sample astronomy image:

# In[ ]:

#im = imread(path_to_sample_image)

#get an image of the other day

#plt.imshow(im,cmap='gray')


# Instead of running directly on the images, we will run on 40 physics computed features. If we compute pretty discriminating features, this will make it easier for the ML algo to discriminate
# 
# It would interesting to see if a machine learning algorithm could discriminate solely based on the pixels of the image. If you are interested, I can show later applying deep learning to do classification on the raw images

# We have a csv file. Here is what it looks like. Each line represents a single event. Each event consists of 40 numbers which are these physically motivated features from the image. The first row of the file is the header with the name of each feature
# 

# In[4]:



# In[5]:



# Ok, we will use spark, here, so let's load the modules of interest and delete the comments at the beginning.

# In[6]:

from pyspark.sql import SparkSession


# SparkSession is like the workhorse variable here

# In[7]:

spark = SparkSession.builder.getOrCreate()


# In[ ]:




# Now we will read the csv file to a data frame

# In[8]:

df = spark.read.csv('./autoscan_features.2.csv', header=True)


# In[9]:

#ID will not be useful and band is non-numerical
df=df.drop('ID')
df=df.drop('BAND')


# Now let's look at a sample record from the dataset. As we can see, underneath the dataframe is an RDD of rows.

# In[10]:

df.take(1)


# In[11]:

len(df.columns)


# And the schema. As we can see here, there is one label, one ID and 38 other features

# In[12]:

df.printSchema()

#describe a couple of the physics features


# In[13]:

df.groupBy('OBJECT_TYPE').count().show()


# In[14]:

from pyspark.mllib.linalg import Vectors


# In[15]:

from pyspark.sql import Row


# In[16]:

from pyspark.ml.linalg import Vectors, Vector, VectorUDT


# Now the ML algo wants a tuple of label and a vector of the other features. Let's make a little function to convert rows to vectrs

# In[17]:

def convert_row_to_vector(row, lbl_key='OBJECT_TYPE'):
    row = row.asDict()
    lbl = int(row[lbl_key])
    float_list = [0.0 if str(v) == '' else float(v) for k,v in row.iteritems() if k!= lbl_key]
    return (lbl, Vectors.dense(float_list))
    


# Now, we call map on the rdd in the dataframe, converting each row to a vector

# In[18]:

lbl_vec_pairs = df.rdd.map(convert_row_to_vector)


# Now we can create a dataframe

# In[20]:

data = spark.createDataFrame(lbl_vec_pairs, ['label', 'features'])


# In[21]:

from pyspark.sql.types import StructField, IntegerType, StructType

from pyspark.mllib.feature import LabeledPoint


# In[43]:

#data=lbl_vec_pairs.map(lambda (l,v): LabeledPoint(l,v))


# In[22]:

from pyspark.ml.classification import RandomForestClassifier

from pyspark.ml.feature import DecisionTreeParams 
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder

from pyspark.ml.feature import MinMaxScaler

from pyspark.ml import Pipeline

from pyspark.ml.evaluation import BinaryClassificationEvaluator


# In[ ]:




# In[23]:

from pyspark.ml.tuning import TrainValidationSplitModel


# In[24]:

bce = BinaryClassificationEvaluator(metricName='mse')


# In[25]:

tr_data, te_data = data.randomSplit([0.8, 0.2])


# In[26]:

rf = RandomForestClassifier()


# In[27]:

paramGrid = ParamGridBuilder()     .addGrid(rf.numTrees, [50, 100])     .addGrid(rf.maxDepth, [30, 15])     .build()


# In[28]:

tvs = TrainValidationSplit(estimator=rf,
                           estimatorParamMaps=paramGrid,
                           evaluator=bce,
                           trainRatio=0.8)


# In[ ]:

tvs.fit(tr_data)


# In[ ]:

prediction = tvs.transform(test)


# In[5]:

# convert to .py file. Now let's submit to queue


# HW!
# Items to Work on: 3 Options:
# 
# 1. ML
#  * make a logistic regression model
#  * use cross-validation to search a good space of logisitc regression hyoerparams
#  * preprocess all features to mean zero and stdev 1
#  * submit this job to batch
#  
#  
# 2. Data Munging / Saving
#  * find number of columns that have an element over 1
#  * make a new data frame that contains 
#      * the sum of GLUX SNR and GAUSS Columns
#      * a column with the max value from each row from the original data
#      * the mean value from each row
#      * the median
#  * conver this data frame to pandas 
#  * also save this data frame out to JSON
# 
#  
# 3. Deep Learning
#     * Train a convolutional neural network to classify the astronomy images for at least 50 epochs
#     * Submit this job to the quueue
#     * Plot the learning curve and an accuracy curve
