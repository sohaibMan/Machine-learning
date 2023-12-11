# Supervised Machine Learning:

Supervised learning uses a training set to teach models to yield the desired output. This training dataset includes inputs and correct outputs, which allow the model to learn over time. The algorithm measures its accuracy through the loss function, adjusting until the error has been sufficiently minimized.

One of the most popular supervised machine learning models is : 

**Regression Model  :**

- Machine Learning Regression is a technique for investigating the relationship between independent variables or features and a dependent variable or outcome. It’s used as a method for predictive modeling in [machine learning](https://www.seldon.io/what-is-machine-learning/), in which an algorithm is used to predict continuous outcomes.

What are the types of regression?

- Simple Linear Regression
- Multiple linear regression
- Logistic regression

### **What is simple linear regression?**

Simple Linear regression is a linear regression technique that plots a straight line within data points to minimize error between the line and the data points. The relationship between the independent and dependent variables is assumed to be linear in this case. 

### **What is multiple linear regression?**

Multiple linear regression is a technique used when more than one independent variable is used. Polynomial regression is an example of a multiple linear regression technique. It is a type of multiple linear regression, used when there is more than one independent variable. It achieves a better fit in comparison to simple linear regression when multiple independent variables are involved. The result when plotted on two dimensions would be a curved line fitted to the data points.

### **What is logistic regression?**

Logistic regression is used when the dependent variable can have one of two values, such as true or false, or success or failure. Logistic regression models can be used to predict the probability of a dependent variable occurring. Generally, the output values must be binary. A sigmoid curve can be used to map the relationship between the dependent variable and independent variables.

→ in this lab we had the chance to work with  simple regression (Experience, Salary), multiple regression(Insurance dataset), linear polynomial regression (China GDP)

**Part 1: Simple regression (Experience Salary)**

![Untitled](Supervised%20Machine%20Learning%20a13c8c91a32240a99d095226b9ef42fc/Untitled.png)

From the data, it is evident that there exists a strong positive linear correlation among the observed variables.

**Part 2: Multiple regression (Insurance dataset)**

![Untitled](Supervised%20Machine%20Learning%20a13c8c91a32240a99d095226b9ef42fc/Untitled%201.png)

→bmi, children  data not are strongly correlated with the target column 

**Part 3: Linear polynomial regression (China GDP)**

![Untitled](Supervised%20Machine%20Learning%20a13c8c91a32240a99d095226b9ef42fc/Untitled%202.png)

→ the year and value are strongly correlated but not linear, polynomial 

**Conclusion :**

→Linear regression is a fundamental machine learning algorithm that has been widely used for many years due to its simplicity, interpretability, and efficiency. It is a valuable tool for understanding relationships between variables and making predictions in a variety of applications. However, it is important to be aware of its limitations, such as its assumption of linearity and sensitivity to multicollinearity. When these limitations are carefully considered, linear regression can be a powerful tool for data analysis and prediction.

**References :**

[https://www.seldon.io/machine-learning-regression-explained](https://www.seldon.io/machine-learning-regression-explained)

[https://www.investopedia.com/terms/r/regression.asp](https://www.investopedia.com/terms/r/regression.asp)

[https://www.ibm.com/topics/supervised-learning](https://www.ibm.com/topics/supervised-learning)

[https://www.geeksforgeeks.org/ml-linear-regression/](https://www.geeksforgeeks.org/ml-linear-regression/)