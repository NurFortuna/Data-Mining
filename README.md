## **Support Vector Machine(SVM)** ##

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

resim gelecek

#### How does Support Vector Machine work? ####

SVM is defined such that it is defined in terms of the support vectors only, we don’t have to worry about
other observations since the margin is made using the points which are closest to the hyperplane (support
vectors), whereas in logistic regression the classifier is defined over all the points. Hence SVM enjoys
some natural speed-ups.

The best hyperplane is that plane that has the maximum distance from both the classes, and this is the main aim of SVM. 
This is done by finding different hyperplanes which classify the labels in 
the best way then it will choose the one which is farthest from the data points or the one which has a maximum margin.

#### Mathematical Intuition behind Support Vector Machine ####

##### Dot-Product #####
The dot product can be defined as the projection of one vector along with another, multiply by the product of another vector.

Here a and b are 2 vectors, to find the dot product between these 2 vectors we first find the magnitude of both the vectors 
and to find magnitude we use the Pythagorean theorem or the distance formula.

#### Use of Dot Product in SVM ####

To find this first we assume this point is a vector (X) and then we make a vector (w) which is perpendicular
to the hyperplane. Let’s say the distance of vector w from origin to decision boundary is ‘c’. Now we take
the projection of X vector on w.

We already know that projection of any vector or another vector is called dot-product. Hence, we take the
dot product of x and w vectors. If the dot product is greater than ‘c’ then we can say that the point lies on the right side. If the dot product is less than ‘c’ then the point is on the left side and if the dot product is
equal to ‘c’ then the point lies on the decision boundary.

#### Kernels in Support Vector Machine ####

The most interesting feature of SVM is that it can even work with a non-linear dataset and for this, 
we use “Kernel Trick” which makes it easier to classifies the points. Suppose we have a dataset like this:

Here we see we cannot draw a single line or say hyperplane which can classify the points correctly. So
what we do is try converting this lower dimension space to a higher dimension space using some quadratic
functions which will allow us to find a decision boundary that clearly divides the data points. These
functions which help us do this are called Kernels and which kernel to use is purely determined by
hyperparameter tuning.

#### Different Kernel functions ####

Some kernel functions which you can use in SVM are given below:

##### 1. Polynomial kernel #####

Following is the formula for the polynomial kernel:


Here d is the degree of the polynomial, which we need to specify manually.

Suppose we have two features X1 and X2 and output variable as Y, so using polynomial kernel we can
write it as:

##### 2. Sigmoid kernel #####

We can use it as the proxy for neural networks. Equation is:

It is just taking your input, mapping them to a value of 0 and 1 so that they can be separated by a simple
straight line.

##### 3. RBF (Radial Basis Function) Kernel #####

What it actually does is to create non-linear combinations of our features to lift your samples onto a higherdimensional 
feature space where we can use a linear decision boundary to separate your classes 
It is the most used kernel in SVM classifications, the following formula explains it mathematically:

1. ‘σ’ is the variance and our hyperparameter
2. ||X₁ – X₂|| is the Euclidean Distance between two points X₁ and X₂


##### 4. Bessel function kernel #####

It is mainly used for eliminating the cross term in mathematical functions. Following is the formula of the Bessel function kernel:

##### 5. Anova Kernel #####

It performs well on multidimensional regression problems. The formula for this kernel function is:

