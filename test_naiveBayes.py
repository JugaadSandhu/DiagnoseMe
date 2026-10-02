#!/usr/bin/env python3
"""
Test suite for naiveBayes.py - DiagnoseMe project
Tests the Naive Bayes classifier for heart disease and diabetes prediction
"""

import unittest
import os
import sys
from collections import defaultdict
from naiveBayes import train, classify, report_counts, report_probabilities


class TestNaiveBayesTrain(unittest.TestCase):
    """Test the training function"""
    
    def test_train_heart_dataset(self):
        """Test training on heart disease dataset"""
        field_names, field_values, target_name, target_values, counts, P = train("heart/train.csv")
        
        # Check that field_names is populated
        self.assertIsInstance(field_names, list)
        self.assertGreater(len(field_names), 0)
        
        # Check target name is correct
        self.assertEqual(target_name, "HeartDisease")
        
        # Check target values
        self.assertIsInstance(target_values, list)
        self.assertIn("Yes", target_values)
        self.assertIn("No", target_values)
        
        # Check counts dictionary is populated
        self.assertIsInstance(counts, defaultdict)
        self.assertGreater(len(counts), 0)
        
        # Check probabilities dictionary is populated
        self.assertIsInstance(P, defaultdict)
        self.assertGreater(len(P), 0)
        
        # Check that probabilities are between 0 and 1
        for key, prob in P.items():
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)
    
    def test_train_diabetes_dataset(self):
        """Test training on diabetes dataset"""
        field_names, field_values, target_name, target_values, counts, P = train("diabetes/train.csv")
        
        # Check that field_names is populated
        self.assertIsInstance(field_names, list)
        self.assertGreater(len(field_names), 0)
        
        # Check target name is correct
        self.assertEqual(target_name, "Diabetes_binary")
        
        # Check target values
        self.assertIsInstance(target_values, list)
        self.assertGreater(len(target_values), 0)
        
        # Check counts dictionary is populated
        self.assertIsInstance(counts, defaultdict)
        self.assertGreater(len(counts), 0)
        
        # Check probabilities dictionary is populated
        self.assertIsInstance(P, defaultdict)
        self.assertGreater(len(P), 0)


class TestNaiveBayesClassify(unittest.TestCase):
    """Test the classification function"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Train on heart dataset
        self.heart_data = train("heart/train.csv")
        self.field_names_heart, self.field_values_heart, self.target_name_heart, \
            self.target_values_heart, self.counts_heart, self.P_heart = self.heart_data
        
        # Train on diabetes dataset
        self.diabetes_data = train("diabetes/train.csv")
        self.field_names_diabetes, self.field_values_diabetes, self.target_name_diabetes, \
            self.target_values_diabetes, self.counts_diabetes, self.P_diabetes = self.diabetes_data
    
    def test_classify_heart_positive(self):
        """Test classification with heart disease positive indicators"""
        observation = {
            "Smoking": "Yes",
            "AlcoholDrinking": "No",
            "Stroke": "Yes",
            "DiffWalking": "Yes",
            "Sex": "Male",
            "AgeCategory": "70-74",
            "Race": "White",
            "Diabetic": "Yes",
            "PhysicalActivity": "No",
            "GenHealth": "Poor",
            "Asthma": "Yes",
            "KidneyDisease": "Yes",
            "SkinCancer": "No"
        }
        
        result, Ptop = classify(observation, self.P_heart, self.target_name_heart, self.target_values_heart)
        
        # Check that result is one of the target values
        self.assertIn(result, self.target_values_heart)
        
        # Check that Ptop contains probabilities for all target values
        self.assertEqual(len(Ptop), len(self.target_values_heart))
        
        # Check that all probabilities are non-negative
        for target_value in Ptop:
            self.assertGreaterEqual(Ptop[target_value], 0.0)
    
    def test_classify_heart_negative(self):
        """Test classification with heart disease negative indicators"""
        observation = {
            "Smoking": "No",
            "AlcoholDrinking": "No",
            "Stroke": "No",
            "DiffWalking": "No",
            "Sex": "Female",
            "AgeCategory": "18-24",
            "Race": "White",
            "Diabetic": "No",
            "PhysicalActivity": "Yes",
            "GenHealth": "Excellent",
            "Asthma": "No",
            "KidneyDisease": "No",
            "SkinCancer": "No"
        }
        
        result, Ptop = classify(observation, self.P_heart, self.target_name_heart, self.target_values_heart)
        
        # Check that result is one of the target values
        self.assertIn(result, self.target_values_heart)
        
        # Check that Ptop contains probabilities for all target values
        self.assertEqual(len(Ptop), len(self.target_values_heart))
    
    def test_classify_diabetes(self):
        """Test classification on diabetes dataset"""
        observation = {
            "HighBP": "1.0",
            "HighChol": "1.0",
            "CholCheck": "1.0",
            "BMI": "28.0",
            "Smoker": "1.0",
            "Stroke": "0.0",
            "HeartDiseaseorAttack": "0.0",
            "PhysActivity": "1.0",
            "Fruits": "1.0",
            "Veggies": "1.0",
            "HvyAlcoholConsump": "0.0",
            "AnyHealthcare": "1.0",
            "NoDocbcCost": "0.0",
            "GenHlth": "3.0",
            "MentHlth": "0.0",
            "PhysHlth": "3.0",
            "DiffWalk": "0.0",
            "Sex": "1.0",
            "Age": "11.0",
            "Education": "6.0",
            "Income": "8.0"
        }
        
        result, Ptop = classify(observation, self.P_diabetes, self.target_name_diabetes, self.target_values_diabetes)
        
        # Check that result is one of the target values
        self.assertIn(result, self.target_values_diabetes)
        
        # Check that Ptop contains probabilities for all target values
        self.assertEqual(len(Ptop), len(self.target_values_diabetes))


class TestNaiveBayesReporting(unittest.TestCase):
    """Test the reporting functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_counts_file = "/tmp/test_counts.csv"
        self.test_prob_file = "/tmp/test_probabilities.csv"
        
        # Train on heart dataset
        _, _, _, _, self.counts, self.P = train("heart/train.csv")
    
    def tearDown(self):
        """Clean up test files"""
        if os.path.exists(self.test_counts_file):
            os.remove(self.test_counts_file)
        if os.path.exists(self.test_prob_file):
            os.remove(self.test_prob_file)
    
    def test_report_counts(self):
        """Test that report_counts creates a valid file"""
        report_counts(self.counts, self.test_counts_file)
        
        # Check file was created
        self.assertTrue(os.path.exists(self.test_counts_file))
        
        # Check file has content
        with open(self.test_counts_file, 'r') as f:
            lines = f.readlines()
            self.assertGreater(len(lines), 0)
            
            # Check format of first line
            if len(lines) > 0:
                parts = lines[0].strip().split(',')
                self.assertEqual(len(parts), 5)
    
    def test_report_probabilities(self):
        """Test that report_probabilities creates a valid file"""
        report_probabilities(self.P, self.test_prob_file)
        
        # Check file was created
        self.assertTrue(os.path.exists(self.test_prob_file))
        
        # Check file has content
        with open(self.test_prob_file, 'r') as f:
            lines = f.readlines()
            self.assertGreater(len(lines), 0)
            
            # Check format of first line
            if len(lines) > 0:
                parts = lines[0].strip().split(',')
                self.assertEqual(len(parts), 5)
                # Last part should be a probability (float)
                try:
                    prob = float(parts[4])
                    self.assertGreaterEqual(prob, 0.0)
                    self.assertLessEqual(prob, 1.0)
                except ValueError:
                    self.fail("Last column should be a valid probability")


class TestDatasetIntegrity(unittest.TestCase):
    """Test that the datasets are properly formatted"""
    
    def test_heart_dataset_exists(self):
        """Test that heart training dataset exists"""
        self.assertTrue(os.path.exists("heart/train.csv"))
    
    def test_diabetes_dataset_exists(self):
        """Test that diabetes training dataset exists"""
        self.assertTrue(os.path.exists("diabetes/train.csv"))
    
    def test_heart_dataset_format(self):
        """Test that heart dataset has proper format"""
        with open("heart/train.csv", 'r') as f:
            lines = f.readlines()
            self.assertGreater(len(lines), 1)  # At least header + 1 data row
            
            # Check header
            header = lines[0].strip().split(',')
            self.assertIn("HeartDisease", header)
            
            # Check that data rows have same number of columns as header
            if len(lines) > 1:
                data_row = lines[1].strip().split(',')
                self.assertEqual(len(data_row), len(header))
    
    def test_diabetes_dataset_format(self):
        """Test that diabetes dataset has proper format"""
        with open("diabetes/train.csv", 'r') as f:
            lines = f.readlines()
            self.assertGreater(len(lines), 1)  # At least header + 1 data row
            
            # Check header
            header = lines[0].strip().split(',')
            self.assertIn("Diabetes_binary", header)
            
            # Check that data rows have same number of columns as header
            if len(lines) > 1:
                data_row = lines[1].strip().split(',')
                self.assertEqual(len(data_row), len(header))
    
    def test_heart_dataset_has_data(self):
        """Test that heart dataset has sufficient data"""
        with open("heart/train.csv", 'r') as f:
            lines = f.readlines()
            # Should have at least 100 rows of data (plus header)
            self.assertGreater(len(lines), 100)
    
    def test_diabetes_dataset_has_data(self):
        """Test that diabetes dataset has sufficient data"""
        with open("diabetes/train.csv", 'r') as f:
            lines = f.readlines()
            # Should have at least 100 rows of data (plus header)
            self.assertGreater(len(lines), 100)


class TestProbabilityCalculations(unittest.TestCase):
    """Test probability calculations"""
    
    def test_prior_probabilities_sum_to_one(self):
        """Test that prior probabilities sum to approximately 1"""
        _, _, target_name, target_values, _, P = train("heart/train.csv")
        
        # Sum prior probabilities
        prior_sum = sum(P[("*", "*", target_name, tv)] for tv in target_values)
        
        # Should sum to approximately 1 (allowing for floating point errors)
        self.assertAlmostEqual(prior_sum, 1.0, places=5)
    
    def test_conditional_probabilities_valid(self):
        """Test that conditional probabilities are valid"""
        field_names, _, target_name, target_values, _, P = train("heart/train.csv")
        
        # Check a few conditional probabilities
        for target_value in target_values:
            # Get probabilities for Smoking given target
            smoking_yes = P[("Smoking", "Yes", target_name, target_value)]
            smoking_no = P[("Smoking", "No", target_name, target_value)]
            
            # Both should be between 0 and 1
            self.assertGreaterEqual(smoking_yes, 0.0)
            self.assertLessEqual(smoking_yes, 1.0)
            self.assertGreaterEqual(smoking_no, 0.0)
            self.assertLessEqual(smoking_no, 1.0)
            
            # They should sum to approximately 1 for this feature
            self.assertAlmostEqual(smoking_yes + smoking_no, 1.0, places=5)


def run_tests():
    """Run all tests and print results"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestNaiveBayesTrain))
    suite.addTests(loader.loadTestsFromTestCase(TestNaiveBayesClassify))
    suite.addTests(loader.loadTestsFromTestCase(TestNaiveBayesReporting))
    suite.addTests(loader.loadTestsFromTestCase(TestDatasetIntegrity))
    suite.addTests(loader.loadTestsFromTestCase(TestProbabilityCalculations))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)
    
    if result.wasSuccessful():
        print("\n✓ All tests passed successfully!")
    else:
        print("\n✗ Some tests failed. See details above.")
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())
