
### 	:sparkling_heart: Table of contents	:sparkling_heart:  :hatched_chick: ### 
1. [Clustering](#clustering)
     1. [Types of Clustering Methods](#types-of-clustering-methods)
     2. [Clustering Algorithms](#clustering-algorithms)
     3. [Applications of Clustering](#applications-of-clustering)
2. [Support Vector Machine](#support-vector-machine)
    1. [Types of Support Vector Machine](#types-of-support-vector-machine)
    2. [How does Support Vector Machine work?](#how-does-support-vector-machine-work)
    3. [Mathematical Intuition behind Support Vector Machine](#mathematical-intuition-behind-support-vector-machine)
    4. [Kernels in Support Vector Machine](#kernels-in-support-vector-machine)



## Clustering ##  
Clustering or cluster analysis is a machine learning technique, which groups the unlabelled dataset.

It does it by finding some similar patterns in the unlabelled dataset such as shape, size, color, behavior, etc., 
and divides them as per the presence and absence of those similar patterns.

It is an **unsupervised learning** method, hence no supervision is provided to the algorithm, and it
deals with the unlabeled dataset.

After applying this clustering technique, each cluster or group is provided with a cluster-ID. 
ML system can use this id to simplify the processing of large and complex datasets.

#### Types of Clustering Methods ####

The clustering methods are broadly divided into Hard clustering (datapoint belongs to only one group) and
Soft Clustering (data points can belong to another group also). But there are also other various approaches 
of Clustering exist. Below are the main clustering methods used in Machine learning:


1. Partitioning Clustering
2. Density-Based Clustering
3. Distribution Model-Based Clustering
4. Hierarchical Clustering
5. Fuzzy Clustering


#### Partitioning Clustering :ribbon: ####

It is a type of clustering that divides the data into non-hierarchical groups.
It is also known as the centroid-based method. The most common example of partitioning 
clustering is the **KMeans Clustering algorithm.**

In this type, the dataset is divided into a set of k groups, where K is used to define the number of
pre-defined groups. The cluster center is created in such a way that the distance between the data
points of one cluster is minimum as compared to another cluster centroid.

#### Density-Based Clustering :ribbon: ####

The density-based clustering method connects the highly-dense areas into clusters, and the
arbitrarily shaped distributions are formed as long as the dense region can be connected. This
algorithm does it by identifying different clusters in the dataset and connects the areas of high
densities into clusters. The dense areas in data space are divided from each other by sparser areas.

These algorithms can face difficulty in clustering the data points if the dataset has varying
densities and high dimensions.

#### Distribution Model-Based Clustering :ribbon: ####

In the distribution model-based clustering method, the data is divided based on the probability
of how a dataset belongs to a particular distribution. The grouping is done by assuming some distributions commonly Gaussian Distribution.

The example of this type is the Expectation-Maximization Clustering algorithm that uses
Gaussian Mixture Models (GMM).

#### Hierarchical Clustering :ribbon: ####

Hierarchical clustering can be used as an alternative for the partitioned clustering as there is no requirement of pre-specifying the number of clusters to be created. In this technique, the dataset is divided into clusters to create a tree-like structure, which is also called a dendrogram. The observations or any number of clusters can be selected by cutting the tree at the correct level. The most common example of this method is the Agglomerative Hierarchical algorithm.

#### Clustering Algorithms ####

Clustering algorithms that are widely used in machine learning:

1.:heavy_check_mark:  **K-Means algorithm:** The k-means algorithm is one of the most popular clustering
algorithms. It classifies the dataset by dividing the samples into different clusters of equal
variances. The number of clusters must be specified in this algorithm. It is fast with fewer
computations required, with the linear complexity of O(n).

2.:heavy_check_mark: **Mean-shift algorithm:** Mean-shift algorithm tries to find the dense areas in the smooth density of data points.
It is an example of a centroid-based model, that works on updating the candidates for centroid to be the center of the
points within a given region

3.:heavy_check_mark: **DBSCAN Algorithm:** It stands for Density-Based Spatial Clustering of Applications
with Noise. It is an example of a density-based model similar to the mean-shift, but with
some remarkable advantages. In this algorithm, the areas of high density are separated by
the areas of low density. Because of this, the clusters can be found in any arbitrary shape.

4.:heavy_check_mark: **Expectation-Maximization Clustering using GMM:** This algorithm can be used as an alternative for the 
k-means algorithm or for those cases where K-means can be failed. In GMM, it is assumed that the data points are Gaussian distributed.

5.:heavy_check_mark: **Agglomerative Hierarchical algorithm:** The Agglomerative hierarchical algorithm performs the bottom-up hierarchical clustering. 
In this, each data point is treated as a single cluster at the outset and then successively merged. The cluster hierarchy can be represented as a tree-structure.

6.:heavy_check_mark:  **Affinity Propagation:** It is different from other clustering algorithms as it does not require to specify the number of clusters.
In this, each data point sends a message between the pair of data points until convergence. It has O(N2T) time complexity, which is the main drawback of this algorithm.

#### Applications of Clustering ####

**In Identification of Cancer Cells** The clustering algorithms are widely used for the
identification of cancerous cells. It divides the cancerous and non-cancerous data sets into
different groups.

**In Search Engines:** Search engines also work on the clustering technique. The search
result appears based on the closest object to the search query. It does it by grouping
similar data objects in one group that is far from the other dissimilar objects. The accurate
result of a query depends on the quality of the clustering algorithm used.

**Customer Segmentation:** It is used in market research to segment the customers based
on their choice and preferences

**In Biology:** It is used in the biology stream to classify different species of plants and
animals using the image recognition technique.


## Support Vector Machine ## 

SVM is a powerful supervised algorithm that works best on smaller datasets but on complex ones. Support
Vector Machine, abbreviated as SVM can be used for both **regression** and **classification** tasks, but
generally, they work best in classification problems.

It is a supervised machine learning problem where we try to find a hyperplane that best separates the two classes. 

**Note**: Don’t get confused between SVM and logistic regression. Both the algorithms try to find the best hyperplane, but the
main difference is logistic regression is a probabilistic approach whereas support vector machine is based on statistical approaches.

### Types of Support Vector Machine ###

#### Linear SVM ####
When the data is perfectly linearly separable only then we can use Linear SVM. Perfectly linearly
separable means that the data points can be classified into 2 classes by using a single straight line(if 2D).

#### Non-Linear SVM ####
When the data is not linearly separable then we can use Non-Linear SVM, which means when the data points cannot be separated into 
2 classes by using a straight line (if 2D) then we use some advanced techniques like kernel tricks to classify them. 
In most real-world applications we do not find linearly separable datapoints hence we use kernel trick to solve them.

**Support Vectors:** These are the points that are closest to the hyperplane. A separating line will be defined with the help of these data points.

**Margin:** it is the distance between the hyperplane and the observations closest to the hyperplane (support vectors). In SVM large margin is considered a good margin. 
There are two types of margins hard margin and soft margin. 

![resim](https://github.com/NurFortuna/Data-Mining/blob/main/pic/1.PNG)

#### How does Support Vector Machine work? ####

SVM is defined such that it is defined in terms of the support vectors only, we don’t have to worry about
other observations since the margin is made using the points which are closest to the hyperplane (support
vectors), whereas in logistic regression the classifier is defined over all the points. Hence SVM enjoys
some natural speed-ups.

![resim](https://github.com/NurFortuna/Data-Mining/blob/main/pic/2.PNG)

The best hyperplane is that plane that has the maximum distance from both the classes, and this is the main aim of SVM. 
This is done by finding different hyperplanes which classify the labels in 
the best way then it will choose the one which is farthest from the data points or the one which has a maximum margin.

#### Mathematical Intuition behind Support Vector Machine ####

##### Dot-Product #####
The dot product can be defined as the projection of one vector along with another, multiply by the product of another vector.

![resim](https://github.com/NurFortuna/Data-Mining/blob/main/pic/3.PNG)

Here a and b are 2 vectors, to find the dot product between these 2 vectors we first find the magnitude of both the vectors 
and to find magnitude we use the Pythagorean theorem or the distance formula.

#### Use of Dot Product in SVM ####

To find this first we assume this point is a vector (X) and then we make a vector (w) which is perpendicular
to the hyperplane. Let’s say the distance of vector w from origin to decision boundary is ‘c’. Now we take
the projection of X vector on w.
![resim](https://github.com/NurFortuna/Data-Mining/blob/main/pic/4.PNG)

![resim](https://github.com/NurFortuna/Data-Mining/blob/main/pic/5.PNG)

We already know that projection of any vector or another vector is called dot-product. Hence, we take the
dot product of x and w vectors. If the dot product is greater than ‘c’ then we can say that the point lies on the right side. If the dot product is less than ‘c’ then the point is on the left side and if the dot product is
equal to ‘c’ then the point lies on the decision boundary.

#### Kernels in Support Vector Machine ####

The most interesting feature of SVM is that it can even work with a non-linear dataset and for this, 
we use “Kernel Trick” which makes it easier to classifies the points. Suppose we have a dataset like this:

![resim](https://github.com/NurFortuna/Data-Mining/blob/main/pic/8.PNG)

Here we see we cannot draw a single line or say hyperplane which can classify the points correctly. So
what we do is try converting this lower dimension space to a higher dimension space using some quadratic
functions which will allow us to find a decision boundary that clearly divides the data points. These
functions which help us do this are called Kernels and which kernel to use is purely determined by
hyperparameter tuning.
![resim](https://github.com/NurFortuna/Data-Mining/blob/main/pic/9.PNG)

#### Different Kernel functions ####

Some kernel functions which you can use in SVM are given below:

##### 1. Polynomial kernel #####

Following is the formula for the polynomial kernel:

![resim](https://github.com/NurFortuna/Data-Mining/blob/main/pic/poli.PNG)

Here d is the degree of the polynomial, which we need to specify manually.

Suppose we have two features X1 and X2 and output variable as Y, so using polynomial kernel we can
write it as:

![resim](https://github.com/NurFortuna/Data-Mining/blob/main/pic/poli2.PNG)

##### 2. Sigmoid kernel #####

We can use it as the proxy for neural networks. Equation is:

![resim](https://github.com/NurFortuna/Data-Mining/blob/main/pic/sigmoid.PNG)

It is just taking your input, mapping them to a value of 0 and 1 so that they can be separated by a simple
straight line.

##### 3. RBF (Radial Basis Function) Kernel #####

What it actually does is to create non-linear combinations of our features to lift your samples onto a higherdimensional 
feature space where we can use a linear decision boundary to separate your classes 
It is the most used kernel in SVM classifications, the following formula explains it mathematically:

![resim](https://github.com/NurFortuna/Data-Mining/blob/main/pic/10.PNG)

1. ‘σ’ is the variance and our hyperparameter
2. ||X₁ – X₂|| is the Euclidean Distance between two points X₁ and X₂


##### 4. Bessel function kernel #####

It is mainly used for eliminating the cross term in mathematical functions. Following is the formula of the Bessel function kernel:

![resim](https://github.com/NurFortuna/Data-Mining/blob/main/pic/bessel.PNG)

##### 5. Anova Kernel #####

It performs well on multidimensional regression problems. The formula for this kernel function is:

![resim](https://github.com/NurFortuna/Data-Mining/blob/main/pic/anova.PNG)

