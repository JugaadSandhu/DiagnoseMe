# DiagnoseMe

A machine learning project that uses Naive Bayes classification to predict the likelihood of heart disease and diabetes based on health-related questionnaires.

## Overview

**DiagnoseMe** is an educational machine learning tool that demonstrates the application of Naive Bayes classification in healthcare diagnostics. The system analyzes user responses to health-related questions and provides probability-based predictions for heart disease and diabetes.

**⚠️ Important:** This project is for educational purposes only and should not be used as a medical diagnostic tool. Always consult qualified healthcare professionals for medical advice.

## Features

- **Dual Diagnosis Models**: Separate trained models for heart disease and diabetes prediction
- **Interactive CLI**: User-friendly command-line interface for data collection
- **Naive Bayes Classification**: Probabilistic machine learning approach with interpretable results
- **Training & Testing Tools**: Scripts to split data and retrain models
- **Probability Reporting**: Detailed probability outputs for each prediction

## Prerequisites

- Python 3.x
- No external dependencies required (uses only Python standard library)

## Project Structure

```
DiagnoseMe/
├── naiveBayes.py          # Main classification script
├── splitData.py           # Data splitting utility
├── runAll.bash            # Batch testing script
├── diabetes/              # Diabetes model data
│   ├── train.csv          # Training dataset
│   ├── test.csv           # Test dataset
│   ├── counts.csv         # Feature counts
│   └── probabilities.csv  # Computed probabilities
├── heart/                 # Heart disease model data
│   ├── train.csv          # Training dataset
│   ├── test.csv           # Test dataset
│   ├── counts.csv         # Feature counts
│   └── probabilities.csv  # Computed probabilities
├── heartBalanced/         # Balanced heart disease dataset
└── heartUnique/           # Unique heart disease dataset
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/JugaadSandhu/DiagnoseMe.git
   cd DiagnoseMe
   ```

2. Ensure Python 3 is installed:
   ```bash
   python3 --version
   ```

## Usage

### Running the Diagnostic Tool

1. Navigate to the project directory:
   ```bash
   cd /path/to/DiagnoseMe
   ```

2. Run the main script:
   ```bash
   python3 naiveBayes.py
   ```

3. Choose your diagnosis type:
   - Type `heart` for heart disease prediction
   - Type `diabetes` for diabetes prediction

4. Answer the prompted questions accurately

5. Review the results:
   - The model will display a prediction (Yes/No)
   - Probability values for each outcome will be shown

### Example: Heart Disease Diagnosis

When you select `heart`, you'll be asked questions about:
- Smoking history
- Alcohol consumption
- Stroke history
- Physical mobility
- Demographics (age, sex, race)
- Existing conditions (diabetes, asthma, kidney disease)
- Physical activity levels
- General health status

### Example: Diabetes Diagnosis

When you select `diabetes`, you'll be asked about:
- Blood pressure and cholesterol
- Smoking and stroke history
- Heart disease history
- Physical activity and diet
- Alcohol consumption
- Healthcare access
- Mental and physical health
- Demographics and socioeconomic factors

## How It Works

The system implements a **Naive Bayes classifier** that:

1. **Training Phase**:
   - Reads training data from CSV files
   - Calculates prior probabilities P(Y=y) for each class
   - Computes conditional probabilities P(Xi=xi|Y=y) for each feature

2. **Classification Phase**:
   - Collects user input for all features
   - Applies Bayes' theorem: P(Y=y|X=x) ∝ P(Y=y) × ∏P(Xi=xi|Y=y)
   - Returns the class with maximum posterior probability

3. **Output**:
   - Predicted class (Yes/No for disease presence)
   - Raw probability scores for transparency

## Data Splitting

To retrain the model with new data:

```bash
python3 splitData.py
```

This splits `data.csv` into training (80%) and testing (20%) sets randomly.

## Batch Testing

Run all models on their test datasets:

```bash
bash runAll.bash
```

This executes the classifier on all four datasets (diabetes, heart, heartBalanced, heartUnique) and displays results.

## Model Performance

The accuracy of predictions depends on:
- Quality and size of training data
- Feature relevance and independence (Naive Bayes assumption)
- User input accuracy

The model provides probability scores to help users understand prediction confidence.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

## Limitations

- **Educational Tool**: Not validated for clinical use
- **Naive Bayes Assumption**: Assumes feature independence, which may not hold in medical data
- **Data Quality**: Predictions are only as good as the training data
- **No Medical Validation**: Results have not been clinically validated

## License

This project is open source and available for educational and research purposes.

## Disclaimer

**This software is provided for educational purposes only.** It is not intended to diagnose, treat, cure, or prevent any disease. The predictions made by this tool should not be considered medical advice. Always seek the guidance of qualified healthcare professionals with any questions you may have regarding a medical condition.

## Acknowledgments

Built with Python using the Naive Bayes classification algorithm. Training data is based on real-world health datasets to demonstrate machine learning applications in healthcare.

---

**Your health is important. If you have concerns, please consult a healthcare professional.**
