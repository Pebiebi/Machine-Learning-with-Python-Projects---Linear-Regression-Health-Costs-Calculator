# Machine-Learning-with-Python-Projects---Linear-Regression-Health-Costs-Calculator

# Predicting Healthcare Costs with Machine Learning

This project is a machine learning solution for predicting healthcare costs using regression algorithms. The model leverages TensorFlow/Keras and a dataset containing information about individuals and their healthcare expenses.

## Project Overview
The main goal of this project is to predict healthcare costs within a margin of error of $3500. The solution involves:
- Preprocessing the dataset.
- Building and training a regression model.
- Evaluating the model's performance.

The final model achieves a **Mean Absolute Error (MAE)** of **3119.54 expenses**, successfully meeting the challenge requirements.

## Dataset
The dataset used in this project was obtained from [freeCodeCamp](https://cdn.freecodecamp.org/project-data/health-costs/insurance.csv). It includes:
- Demographic information (e.g., age, sex, region).
- Behavioral factors (e.g., smoking status).
- Healthcare expenses (target variable).

## Key Steps
1. **Data Preprocessing**:
   - One-hot encoding for categorical variables (`sex`, `smoker`, `region`).
   - Normalization of numerical features using `StandardScaler`.
   - Splitting data into training (80%) and testing (20%) datasets.
2. **Model Building**:
   - A neural network with 3 hidden layers.
   - Activation functions: ReLU for hidden layers.
   - Optimizer: Adam.
   - Loss function: Mean Squared Error (MSE).
   - Metrics: Mean Absolute Error (MAE) and Mean Squared Error (MSE).
3. **Model Training**:
   - Early stopping applied to prevent overfitting.
   - Validation split of 20% from the training dataset.
4. **Model Evaluation**:
   - The model's performance was evaluated using the test dataset.
   - Achieved MAE of 3119.54 expenses.

## Project Files
- `healthcare_costs_regression.ipynb`: Jupyter Notebook containing the complete implementation of the project.
- `README.md`: This documentation file.

## Requirements
The following Python libraries are required to run the notebook:
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

Install the required packages using:
```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```

## How to Run
1. Clone this repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd <repository-directory>
   ```
3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook healthcare_costs_regression.ipynb
   ```
4. Run all cells to train and evaluate the model.

## Results
- **Mean Absolute Error (MAE)**: 3119.54 expenses
- The model successfully predicts healthcare costs within the required error margin.

## License
This project is open-source and available under the [MIT License](LICENSE).

---

Feel free to fork, contribute, or use this project as a reference for your own machine learning implementations!
