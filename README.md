# DiagnoseMe 🏥

A machine learning-powered diagnostic assistant that uses Naive Bayes classification to predict the likelihood of heart disease and diabetes based on health-related questions.

---

## 📋 Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Application](#running-the-application)
  - [Heart Disease Diagnosis](#heart-disease-diagnosis)
  - [Diabetes Diagnosis](#diabetes-diagnosis)
- [How It Works](#how-it-works)
- [Technical Details](#technical-details)
- [Data Files](#data-files)
- [Important Disclaimer](#important-disclaimer)
- [Contributing](#contributing)
- [Support](#support)

---

## 📖 Introduction

**DiagnoseMe** is a machine learning model designed to assist in the preliminary diagnosis of heart complications and diabetes. This tool uses a **Naive Bayes classification algorithm** to predict the likelihood of a user having these medical conditions based on a set of specific health-related questions.

By providing accurate answers to these questions, users can receive preliminary insights into their health status and seek appropriate medical advice if necessary.

---

## ✨ Features

- **Dual Diagnosis Capability**: Diagnose both heart disease and diabetes
- **Interactive CLI**: User-friendly command-line interface with guided questions
- **Naive Bayes Classification**: Implements probabilistic machine learning from scratch
- **Probability Analysis**: Provides detailed probability scores for each diagnosis
- **Training Data**: Includes pre-trained models with real health data
- **Data Processing**: Utility scripts for data splitting and preparation
- **Transparent Results**: Generates counts and probability reports for analysis

---

## 📁 Project Structure

```
DiagnoseMe/
├── README.md                 # Project documentation
├── naiveBayes.py            # Main diagnostic application
├── splitData.py             # Data splitting utility
├── cpToAssignment.bash      # Data copying script
├── runAll.bash              # Batch execution script
│
├── heart/                   # Heart disease data
│   ├── train.csv           # Training dataset
│   ├── test.csv            # Test dataset (generated)
│   ├── counts.csv          # Generated counts report
│   └── probabilities.csv   # Generated probabilities report
│
├── diabetes/               # Diabetes data
│   ├── train.csv          # Training dataset
│   ├── test.csv           # Test dataset (generated)
│   ├── counts.csv         # Generated counts report
│   └── probabilities.csv  # Generated probabilities report
│
├── heartBalanced/         # Balanced heart disease dataset
├── heartUnique/           # Unique heart disease dataset
└── .git/                  # Git version control
```

---

## 🔧 Prerequisites

- **Python 3.x** (Python 3.6 or higher recommended)
- **No external dependencies** - Uses only Python standard library

---

## 📥 Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/JugaadSandhu/DiagnoseMe.git
   cd DiagnoseMe
   ```

   Or download the ZIP archive from the GitHub page and extract it.

2. **Verify Installation**:
   ```bash
   python3 naiveBayes.py
   ```

   No additional installation or setup is required!

---

## 🚀 Usage

### Running the Application

1. **Navigate to the DiagnoseMe Directory**:
   ```bash
   cd /path/to/DiagnoseMe
   ```

2. **Execute the Main Script**:
   ```bash
   python3 naiveBayes.py
   ```

3. **Select Diagnosis Type**:
   When prompted, enter either:
   - `heart` - for heart disease diagnosis
   - `diabetes` - for diabetes diagnosis

### Heart Disease Diagnosis

When you select `heart`, you'll be asked 13 questions about:

1. **Smoking History**: Have you smoked at least 100 cigarettes in your entire life?
2. **Alcohol Consumption**: Do you consume alcohol?
3. **Stroke History**: Did you ever have a stroke?
4. **Physical Mobility**: Do you have serious difficulty walking or climbing stairs?
5. **Sex**: What is your sex? (Male/Female)
6. **Age Category**: Select your age range (18-24, 25-29, ..., 80 or older)
7. **Race/Ethnicity**: Select your race/ethnicity
8. **Diabetes Status**: Have you ever had diabetes?
9. **Physical Activity**: Have you done any physical activity in the past 30 days?
10. **General Health**: Rate your general health (Fair, Good, Poor, Very good, Excellent)
11. **Asthma**: Did you ever have asthma?
12. **Kidney Disease**: Were you ever told you had kidney disease?
13. **Skin Cancer**: Did you ever have skin cancer?

**Output**: The model will provide a prediction (Yes/No) and probability scores.

### Diabetes Diagnosis

When you select `diabetes`, you'll be asked 20 questions about:

1. **Blood Pressure**: Have you been told you have high blood pressure?
2. **Cholesterol**: Have you been told you have high cholesterol?
3. **Cholesterol Check**: Have you had a cholesterol check in the past 5 years?
4. **Smoking**: Have you smoked at least 100 cigarettes in your life?
5. **Stroke**: Did you ever have a stroke?
6. **Heart Disease**: Have you ever had coronary heart disease or myocardial infarction?
7. **Physical Activity**: Have you done any physical activity in the past 30 days?
8. **Fruit Consumption**: Do you consume fruit 1 or more times per day?
9. **Vegetable Consumption**: Do you consume vegetables 1 or more times per day?
10. **Heavy Alcohol Consumption**: Are you a heavy drinker?
11. **Health Insurance**: Do you have any kind of health care coverage?
12. **Cost Barrier**: Was there a time you needed to see a doctor but couldn't due to cost?
13. **General Health**: Rate your general health (1.0-5.0)
14. **Mental Health**: Days in past 30 days your mental health was not good (0-30)
15. **Physical Health**: Days in past 30 days your physical health was not good (0-30)
16. **Mobility**: Do you have serious difficulty walking or climbing stairs?
17. **Sex**: What is your sex? (1.0 = Male / 0.0 = Female)
18. **Age**: Select your age range (0.0-12.0)
19. **Education**: Overall education level
20. **Income**: Annual household income range

**Output**: The model will provide a prediction (Yes/No) and probability scores.

---

## 🧠 How It Works

### Naive Bayes Classification

DiagnoseMe implements the Naive Bayes algorithm, which uses Bayes' theorem to calculate conditional probabilities:

```
P(Y=y|X=x) = P(X₁=x₁|Y=y) × P(X₂=x₂|Y=y) × ... × P(Xₙ=xₙ|Y=y) × P(Y=y) / P(X=x)
```

Where:
- **Y** = Target variable (disease present or not)
- **X** = Feature variables (health attributes)
- **P(Y|X)** = Posterior probability (what we want to predict)
- **P(X|Y)** = Likelihood (probability of features given the disease)
- **P(Y)** = Prior probability (base rate of disease)
- **P(X)** = Evidence (probability of observing these features)

### Process Flow

1. **Training Phase**:
   - Load training data from CSV files
   - Calculate feature counts for each disease class
   - Compute conditional probabilities P(X|Y) and prior probabilities P(Y)

2. **Classification Phase**:
   - Accept user input for health attributes
   - Calculate posterior probabilities for each disease class
   - Select the class with the highest probability
   - Return prediction and probability scores

3. **Output Generation**:
   - Generate `counts.csv` with feature-class counts
   - Generate `probabilities.csv` with calculated probabilities
   - Display results to the user

---

## 🔬 Technical Details

### Key Functions

- **`train(train_filename)`**: Trains the Naive Bayes model on the provided dataset
  - Returns: field names, field values, target name, target values, counts, and probabilities

- **`classify(observation, P, target_name, target_values)`**: Classifies a single observation
  - Returns: predicted class and probability scores for all classes

- **`test(target_name, target_values, P, test_filename)`**: Tests the model on a test dataset
  - Returns: classification results and probability strings

- **`report_counts(counts, counts_filename)`**: Exports count statistics to CSV

- **`report_probabilities(P, probabilities_filename)`**: Exports probability statistics to CSV

### Data Format

Training and test CSV files must follow this format:

```
Feature1,Feature2,Feature3,...,FeatureN,TargetClass
value1,value2,value3,...,valueN,classValue
value1,value2,value3,...,valueN,classValue
...
```

---

## 📊 Data Files

### Heart Disease Dataset
- **Location**: `heart/train.csv`
- **Features**: 13 health-related attributes
- **Target**: HeartDisease (Yes/No)
- **Variants**: 
  - `heartBalanced/` - Balanced dataset
  - `heartUnique/` - Unique records dataset

### Diabetes Dataset
- **Location**: `diabetes/train.csv`
- **Features**: 20 health-related attributes
- **Target**: Diabetes (0.0 = No, 1.0 = Yes)

---

## ⚠️ Important Disclaimer

**DiagnoseMe is NOT a substitute for professional medical advice.**

- This tool provides **preliminary insights only** based on the information you provide
- Results should **NOT** be used for self-diagnosis or self-treatment
- Always consult a qualified healthcare professional for:
  - Proper diagnosis
  - Medical advice
  - Treatment recommendations
  - Any health concerns

**Accuracy Limitations**:
- The model's accuracy depends on the quality and completeness of your answers
- Incomplete or inaccurate information may lead to incorrect predictions
- The model is trained on historical data and may not reflect individual variations

---

## 🤝 Contributing

Contributions are welcome! To contribute to DiagnoseMe:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add improvement'`)
5. Push to the branch (`git push origin feature/improvement`)
6. Open a Pull Request

### Areas for Contribution:
- Improving model accuracy
- Adding more diagnostic conditions
- Enhancing user interface
- Improving documentation
- Adding unit tests
- Optimizing performance

---

## 📞 Support

If you encounter any issues or have suggestions for improvements:

1. **Open an Issue**: Visit the [GitHub Issues](https://github.com/JugaadSandhu/DiagnoseMe/issues) page
2. **Provide Details**: Include:
   - Description of the issue
   - Steps to reproduce
   - Expected vs. actual behavior
   - Your Python version and OS

---

## 📝 License

This project is open source and available under the MIT License.

---

## 🙏 Acknowledgments

- Built with Python 3
- Uses Naive Bayes classification algorithm
- Inspired by real-world health diagnostic systems
- Data sourced from health surveys and medical datasets

---

## 📚 Additional Resources

- [Naive Bayes Algorithm](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
- [Bayes' Theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem)
- [Machine Learning Basics](https://en.wikipedia.org/wiki/Machine_learning)

---

**Thank you for using DiagnoseMe!** We hope this tool helps raise awareness of potential health issues and encourages users to seek appropriate medical attention when needed. 🏥❤️

---

*Last Updated: 2024*
*Repository: [DiagnoseMe](https://github.com/JugaadSandhu/DiagnoseMe)*
