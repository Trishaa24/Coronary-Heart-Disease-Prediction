# Coronary Heart Disease Prediction: A Data-Driven Approach to Early Diagnosis

This project leverages machine learning to predict the risk of coronary heart disease (CHD) based on patient data, enabling early diagnosis and proactive health management. By analyzing over 4,240 patient records, the model predicts the likelihood of developing heart disease within the next 10 years. This tool can assist healthcare professionals in identifying high-risk patients and providing personalized care, thereby improving outcomes and reducing healthcare costs.

## Dataset Overview

The dataset used in this project, known as the **Framingham Heart Study**, is publicly available and includes the following features for each patient:

- **Age**: Age of the patient.
- **Sex**: Gender of the patient (1 for male, 0 for female).
- **Chest Pain Type**: Type of chest pain experienced (categorical).
- **Resting Blood Pressure**: Blood pressure at rest (in mm Hg).
- **Cholesterol**: Serum cholesterol levels (mg/dl).
- **Fasting Blood Sugar**: Fasting blood sugar (1 if greater than 120 mg/dl, 0 if not).
- **Resting Electrocardiographic Results**: Electrocardiogram results (categorical).
- **Maximum Heart Rate Achieved**: Maximum heart rate during exercise.
- **Exercise Induced Angina**: Whether the patient experiences angina during exercise (1 if yes, 0 if no).
- **Oldpeak**: Depression induced by exercise relative to rest (in ST depression).
- **Slope of the Peak Exercise ST Segment**: Slope of the ST segment during exercise (categorical).
- **Number of Major Vessels Colored by Fluoroscopy**: Number of major vessels colored by fluoroscopy (numeric).
- **Thalassemia**: A blood disorder (1, 2, 3) indicating the presence of thalassemia.
- **TenYearCHD**: Target variable indicating whether the patient has a 10-year risk of coronary heart disease (1 if yes, 0 if no).

## Objective

The goal of this project is to develop a machine learning model that accurately predicts the risk of coronary heart disease (CHD) in patients. By identifying high-risk individuals early on, healthcare professionals can intervene and provide timely treatments, potentially saving lives and improving long-term health outcomes.

## Steps Performed

1. **Data Preprocessing**: Cleaned and transformed the dataset by handling missing values, outliers, and duplicates.
2. **Exploratory Data Analysis (EDA)**: Conducted exploratory analysis to understand the distribution of features and identify key patterns.
3. **Feature Selection**: Selected the most relevant features that contribute to the prediction of heart disease.
4. **Model Development**: Built multiple machine learning models, including Logistic Regression, k-Nearest Neighbors, Random Forest, Decision Trees, and Gradient Boosting, to predict the 10-year risk of coronary heart disease.
5. **Model Evaluation**: Evaluated the models using metrics like accuracy, precision, recall, F1 score, and confusion matrix.
6. **Hyperparameter Tuning**: Fine-tuned the Random Forest and Gradient Boosting models to achieve the best possible performance.
7. **Ensemble Learning**: Applied ensemble techniques, including stacking, to combine the best models and improve prediction accuracy, achieving a final accuracy of 96.17%.

## Technologies Used

- **Python**: Programming language used for data analysis and machine learning model development.
- **Pandas**: Library used for data manipulation and preprocessing.
- **NumPy**: Library used for numerical computations.
- **Scikit-Learn**: Machine learning library used for building and evaluating models.
- **Matplotlib** & **Seaborn**: Visualization libraries used for data exploration and presenting the results.

## Model Performance

- **Accuracy**: The model achieved a final accuracy of **96.17%** after applying ensemble techniques.
- **Model Comparison**: Evaluated multiple models, with Random Forest and Gradient Boosting being the top performers, achieving accuracies of **90.39%** and **95.21%**, respectively.
- **Evaluation Metrics**: In addition to accuracy, models were evaluated based on F1 score, precision, recall, and confusion matrix to ensure well-rounded performance.

## Conclusion
The Coronary Heart Disease Prediction model provides an efficient, data-driven tool to assess the risk of coronary heart disease, enabling healthcare providers to deliver personalized care to patients. By leveraging machine learning, this tool enhances early diagnosis, which can significantly improve patient outcomes and help manage healthcare resources effectively.
