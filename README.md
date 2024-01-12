# CancerDetectML_Monte-Carlo_and_SVM

## Project Description
CancerDetectML is a machine learning project aimed at detecting cancer through the analysis of cell nuclei characteristics in fine needle aspirate (FNA) images of breast masses. Utilizing a comprehensive dataset, the project applies advanced classification methods to differentiate between malignant and benign diagnoses effectively.

## Dataset Information
### Source:
The dataset features are computed from digitized images of FNA of breast masses, describing the characteristics of the cell nuclei present in these images. Some sample images are available at [this link](http://www.cs.wisc.edu/~street/images/).

### Methodology:
A separating plane for classification was derived using the Multisurface Method-Tree (MSM-T), a method that employs linear programming to construct a decision tree. Key features were identified through an exhaustive search in a defined feature and plane space.

The linear program used for the separation in 3-dimensional space is detailed in the paper by K. P. Bennett and O. L. Mangasarian: "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", published in Optimization Methods and Software 1, 1992, 23-34.

### Accessibility:
The database can be accessed via the University of Wisconsin's CS ftp server:
```
ftp ftp.cs.wisc.edu
cd math-prog/cpo-dataset/machine-learn/WDBC/
```
### Missing Values:
The dataset contains no missing values.

## Additional Variable Information
Each record in the dataset includes the following variables:

1. **ID number**
2. **Diagnosis**: M = malignant, B = benign
3. **Features**: Ten real-valued features computed for each cell nucleus:
   - **Radius**: Mean of distances from center to points on the perimeter.
   - **Texture**: Standard deviation of gray-scale values.
   - **Perimeter**
   - **Area**
   - **Smoothness**: Local variation in radius lengths.
   - **Compactness**: Perimeter^2 / area - 1.0.
   - **Concavity**: Severity of concave portions of the contour.
   - **Concave Points**: Number of concave portions of the contour.
   - **Symmetry**
   - **Fractal Dimension**: "Coastline approximation" - 1.
# Final Goal of the Project

The final goal of this project is to explore and compare the effectiveness of different machine learning approaches — namely supervised, semi-supervised, unsupervised, and active learning — in classifying and predicting outcomes in two specific datasets: the Breast Cancer Wisconsin (Diagnostic) dataset and the Banknote Authentication dataset. The project aims to achieve the following objectives:

## Objectives

1. **Evaluate Supervised Learning**
   - Implement and assess an L1-penalized SVM (Support Vector Machine) using the Breast Cancer Wisconsin dataset.
   - Measure performance metrics such as accuracy, precision, recall, F1-score, and AUC (Area Under the Curve).
   - Analyze the ROC (Receiver Operating Characteristic) curve and confusion matrix.

2. **Investigate Semi-Supervised Learning**
   - Apply a self-training approach using an L1-penalized SVM on partially labeled data from the Breast Cancer Wisconsin dataset.
   - Progressively label unlabeled data based on their distance from the SVM decision boundary.
   - Assess the model's performance on various metrics.

3. **Explore Unsupervised Learning**
   - Utilize the k-means clustering algorithm on the Breast Cancer Wisconsin dataset without considering the labels.
   - Label the clusters based on a majority poll from the closest data points to each cluster center.
   - Evaluate performance based on metrics similar to supervised learning.

4. **Examine Spectral Clustering**
   - Implement spectral clustering using the RBF kernel on the Breast Cancer Wisconsin dataset.
   - Compare its effectiveness against other methods.
   - Balance clusters as per the original dataset's class distribution.

5. **Compare Different Learning Approaches**
   - Draw comparisons between supervised, semi-supervised, unsupervised, and spectral clustering methods.
   - Assess their effectiveness and efficiency using the Breast Cancer Wisconsin dataset.

6. **Active Learning Using SVMs**
   - Using the Banknote Authentication dataset, implement both active and passive learning approaches with SVMs.
   - Progressively train SVMs with incrementally larger subsets of the training data.
   - Compare the test errors to illustrate the learning curve through a Monte-Carlo simulation.

7. **Conduct a Monte Carlo Simulation**
   - Perform multiple iterations of the aforementioned procedures to ensure reliability and robustness in the results.
   - Provide a comprehensive comparison of these learning methods.

8. **Visualize and Analyze Results**
   - Plot learning curves and other relevant visualizations.
   - Compare the effectiveness of each method in both datasets.
   - Draw conclusions about the suitability and efficiency of each learning approach for the given classification tasks.

## Conclusion

The overarching goal is to provide a thorough analysis of these machine learning techniques in the context of binary classification problems. This project will offer insights into their practical application, strengths, and limitations in real-world scenarios.

# Necessary Libraries and Frameworks

For the successful implementation of this project, several libraries and frameworks are essential. These are required for data manipulation, machine learning model implementation, and result visualization. The following is a list of the key libraries and frameworks needed:

## Python Libraries

1. **NumPy**
   - For numerical operations and manipulation of arrays.
   - Installation: `pip install numpy`

2. **Pandas**
   - For data handling and manipulation.
   - Installation: `pip install pandas`

3. **Scikit-learn**
   - Provides tools for data mining and data analysis.
   - Contains implementations of various machine learning algorithms including SVM, k-means, etc.
   - Installation: `pip install scikit-learn`

4. **Matplotlib**
   - For creating static, interactive, and animated visualizations in Python.
   - Installation: `pip install matplotlib`

5. **Seaborn**
   - Based on Matplotlib, provides a high-level interface for drawing attractive statistical graphics.
   - Installation: `pip install seaborn`

6. **SciPy**
   - Used for scientific and technical computing.
   - Installation: `pip install scipy`

## Additional Tools

1. **Jupyter Notebook**
   - An open-source web application that allows the creation and sharing of documents containing live code, equations, visualizations, and narrative text.
   - Installation: `pip install notebook`

2. **Git**
   - For version control and efficient handling of the project's development.
   - Installation: Follow the instructions on [Git's official website](https://git-scm.com/downloads).

3. **Python (Version 3.x)**
   - Ensure that Python 3 is installed as the primary Python version as it's required for most libraries.

## Frameworks for Specific Tasks

1. **Libsvm/Liblinear** (Optional)
   - For specialized SVM algorithms, if Scikit-learn's implementation needs supplementation.
   - Can be useful for large-scale or specialized SVM applications.

## Installation
Most of these libraries can be installed using pip, Python’s package installer. Example: `pip install numpy`. Ensure you have pip and Python installed on your system.

## Note
The project may require additional libraries or specific versions of these libraries depending on the dataset's nature and the complexity of the models implemented. Always check for the latest version of these libraries for compatibility and new features.
