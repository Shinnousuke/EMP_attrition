# 🧠 Employee Attrition Predictor

A Streamlit web application that predicts whether an employee may be at risk of leaving an organization based on selected employee and job-related attributes. The app loads a previously trained machine learning model and presents a simple form for entering employee details.

> **Purpose:** This project is an educational machine learning demonstration. Its predictions are estimates from a trained model, not a definitive judgment about an employee or a substitute for a fair, evidence-based HR process.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [How It Works](#-how-it-works)
- [Input Features](#-input-features)
- [Prediction Output](#-prediction-output)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Running the Application](#-running-the-application)
- [Model and Data Files](#-model-and-data-files)
- [Feature Encoding](#-feature-encoding)
- [Troubleshooting](#-troubleshooting)
- [Limitations and Responsible Use](#-limitations-and-responsible-use)
- [Potential Improvements](#-potential-improvements)
- [License](#-license)
- [Author](#-author)

---

## 🔍 Overview

The **Employee Attrition Predictor** is a small interactive application built with Streamlit and Python. It collects selected employee details, converts categorical answers into one-hot encoded features, arranges the resulting values in the column order expected by the trained model, and uses a saved Random Forest-style classifier to generate an attrition prediction and probability.

The application loads two local files at startup:

- `rf_model.pkl` — the serialized trained model.
- `trained_columns.pkl` — the ordered feature-column names used when training the model.

The interface displays one of two outcomes:
- **High Attrition Risk** when the model predicts class `1`.
- **Likely to Stay** when the model predicts class `0`.

The script caption says the model was trained on the IBM HR Analytics dataset from Kaggle. The supplied app code does not include the training pipeline, the original dataset, or model evaluation results, so those details should be documented separately if available.

---

## ✨ Features

- **Simple Streamlit interface:** Employee details are entered through sliders, numeric input, and dropdown menus.
- **Saved-model inference:** Loads a previously trained model instead of training a new model every time the app starts.
- **Numeric and categorical inputs:** Captures job satisfaction, income, age, company tenure, total experience, work-life balance, job role, gender, and marital status.
- **One-hot encoding for selected categories:** Converts gender, job role, and marital status into binary feature columns.
- **Training-column alignment:** Builds the model input vector in the order specified by `trained_columns.pkl`.
- **Prediction probability:** Uses `predict_proba()` to display the probability associated with class `1`.
- **Immediate result display:** Shows an alert or success message based on the predicted class.

---

## ⚙️ How It Works

The app follows this flow:

```text
Start Streamlit App
        |
        v
Load rf_model.pkl
        |
        v
Load trained_columns.pkl
        |
        v
Collect Employee Details
        |
        v
Convert Categories to Binary Features
        |
        v
Arrange Features in Training Column Order
        |
        v
Run model.predict() and model.predict_proba()
        |
        v
Display Attrition Prediction and Probability
```

### 1. Load the trained model

The app uses Python's `pickle` module to load the model object from `rf_model.pkl`.

### 2. Load the training feature columns

The app loads `trained_columns.pkl`, which is expected to contain an ordered list of feature names used during training.

### 3. Collect user inputs

The user enters employee-related values through the Streamlit interface.

### 4. Prepare the feature dictionary

Numeric values are placed directly into a dictionary. Categorical inputs are represented as binary columns such as `Gender_Male`, `JobRole_Manager`, or `MaritalStatus_Single`.

### 5. Align the feature vector

The app creates the input vector using:

```python
input_vector = [input_data.get(col, 0) for col in trained_columns]
```

This means each training column receives the corresponding value from the input dictionary, or `0` if that feature name is not present in the dictionary.

### 6. Generate the prediction

The model returns a predicted class using `predict()`. The app also reads the second entry from `predict_proba()` as the probability of class `1`.

### 7. Display the result

If the predicted class is `1`, the app shows an attrition-risk alert. Otherwise, it displays a likely-to-stay message.

---

## 🧾 Input Features

The application currently exposes these fields:

| Input | UI control | Allowed values / behavior |
|---|---|---|
| Job Satisfaction | Slider | Integer from 1 to 4; default 3 |
| Monthly Income | Number input | Minimum 1,000; default 5,000; step 100 |
| Age | Slider | Integer from 18 to 60; default 30 |
| Years at Company | Slider | Integer from 0 to 40; default 5 |
| Total Experience | Slider | Integer from 0 to 50; default 7 |
| Work-Life Balance | Slider | Integer from 1 to 4; default 3 |
| Job Role | Dropdown | Sales Executive, Research Scientist, Laboratory Technician, Manufacturing Director, Healthcare Representative, Manager, Sales Representative, Research Director, Human Resources |
| Gender | Dropdown | Male or Female |
| Marital Status | Dropdown | Single, Married, or Divorced |

The labels and ranges above reflect the supplied app code. They are not a guarantee that the training dataset used exactly the same definitions or valid ranges.

---

## 📊 Prediction Output

The app produces two pieces of output:

### Predicted class

- `1` → **High Attrition Risk**
- `0` → **Likely to Stay**

This interpretation assumes the saved model was trained with `1` representing employees who left and `0` representing employees who stayed. Confirm that mapping in the training code before relying on the labels.

### Probability

The app displays:

```python
proba = model.predict_proba([input_vector])[0][1]
```

The value is formatted to two decimal places. For example, `0.73` is displayed as `0.73`.

This value should be interpreted as the model's estimated probability for class `1` only if the model's class ordering and training labels match that assumption. A probability from a model is not automatically well-calibrated or a measure of certainty.

---

## 🧰 Technology Stack

| Technology | Role |
|---|---|
| Python | Application logic |
| Streamlit | Web interface and input controls |
| Pickle | Loading serialized model and feature-column objects |
| NumPy | Imported numerical computing library; not directly used elsewhere in the supplied script |
| Trained classifier | Generates the attrition prediction; expected to support `predict()` and `predict_proba()` |

The filename `rf_model.pkl` suggests a Random Forest model, but the actual estimator type should be confirmed from the model-training code or by inspecting the saved object.

---

## 📁 Project Structure

The app expects the following files to be in the working directory:

```text
employee-attrition-predictor/
├── app.py
├── rf_model.pkl
├── trained_columns.pkl
├── requirements.txt
└── README.md
```

The model and column files are required at runtime. The supplied script does not create them; they must be generated by a separate training workflow or obtained from the project maintainer.

---

## ✅ Prerequisites

- Python installed on your computer.
- `pip` for installing Python packages.
- The application source file `app.py`.
- A compatible `rf_model.pkl` model file.
- A compatible `trained_columns.pkl` file.
- A browser to open the Streamlit interface.

Use a Python and scikit-learn environment compatible with the versions used to serialize the model. Serialized Python models can fail to load when package versions or class definitions differ.

---

## 📦 Installation

### 1. Get the project files

Clone your repository if the project is hosted on GitHub:

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_PROJECT_DIRECTORY>
```

Replace the placeholders with your actual repository URL and folder name.

Alternatively, download and extract the repository ZIP file.

### 2. Create and activate a virtual environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

Create a `requirements.txt` file containing:

```text
streamlit
numpy
scikit-learn
```

Then install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`pickle` is part of Python's standard library and does not need to be installed separately. NumPy is imported by the supplied code, although it is not directly used in the visible script. Scikit-learn is needed if the saved model relies on scikit-learn classes.

For reproducible deployment, pin package versions after confirming which versions were used to train and serialize the model.

---

## ▶️ Running the Application

From the project directory, run:

```bash
streamlit run app.py
```

Streamlit prints a local URL in the terminal, commonly:

```text
http://localhost:8501
```

Open that address in a browser and enter the employee details.

To stop the application, return to the terminal and press `Ctrl + C`.

> **File location:** Because the script opens `rf_model.pkl` and `trained_columns.pkl` using relative paths, run Streamlit from a directory where both files can be found, or update the script to resolve paths relative to `app.py`.

---

## 🗂️ Model and Data Files

### `rf_model.pkl`

Contains a serialized trained model. At prediction time, the object must provide these methods:

- `predict(X)` — returns the predicted class.
- `predict_proba(X)` — returns class probabilities.

The supplied script does not show how the model was trained, what hyperparameters were used, or how well it performs.

### `trained_columns.pkl`

Contains the ordered feature names expected by the model. The feature vector is assembled in precisely this order, so the column list must match the training pipeline.

If the model was trained using a scikit-learn pipeline or a different encoding procedure, the current manual input encoding may not match it. The training and inference transformations must be consistent.

### Security warning about pickle files

Only load pickle files from a trusted source. Python pickle deserialization can execute code. Do not accept arbitrary `.pkl` uploads from users or load untrusted model files.

---

## 🔤 Feature Encoding

The application manually encodes selected categorical fields.

### Gender

```python
"Gender_Male": 1 if gender == "Male" else 0
```

Selecting Male sets `Gender_Male` to `1`; selecting Female sets it to `0`.

### Job role

```python
"JobRole_" + job_role: 1
```

The selected role receives a value of `1`. Other job-role columns in `trained_columns` receive `0` because they are absent from the input dictionary.

### Marital status

```python
"MaritalStatus_" + marital_status: 1
```

The selected marital-status column receives a value of `1`; other marital-status columns receive `0`.

### Numeric inputs

The values for job satisfaction, monthly income, age, years at company, total working years, and work-life balance are passed through as numeric values.

### Why training alignment matters

The feature names and encoding scheme must exactly match the data used during training. A mismatch may cause incorrect predictions even when the app runs without errors.

Before deployment, verify:
- The exact spelling and capitalization of every feature name.
- Whether the training process used one-hot encoding, ordinal encoding, or another transformation.
- Whether the model expects the same feature order.
- Whether all expected features are represented.
- Whether the target class mapping is `1 = left` and `0 = stayed`.

---

## 🛠️ Troubleshooting

### `FileNotFoundError: rf_model.pkl`

Ensure that `rf_model.pkl` exists and is accessible from the application's working directory. The app cannot run without the trained model file.

### `FileNotFoundError: trained_columns.pkl`

Ensure that the feature-column file exists alongside the app, or update the path in the script.

### `ModuleNotFoundError: No module named 'streamlit'`

Activate the intended virtual environment and install the dependencies:

```bash
pip install -r requirements.txt
```

### The model fails to load

The saved model may require a different Python or scikit-learn version, or the file may be damaged. Use the environment associated with the training workflow and only load trusted files.

### `AttributeError` involving `predict_proba`

The saved estimator may not support probability prediction. Confirm the model type and whether probability estimates were enabled during training.

### The application reports a feature-count or feature-name error

Check `trained_columns.pkl` against the exact feature names and encoding used when training the model. Ensure that the number and order of values match what the model expects.

### Probability or prediction seems incorrect

Verify the target label encoding, model training procedure, input ranges, feature preprocessing, and probability calibration. The app's displayed labels assume a particular class mapping that is not established by the supplied inference script alone.

---

## ⚖️ Limitations and Responsible Use

Employee attrition prediction involves people and workplace decisions. A model trained on historical HR data can reproduce biases, historical inequities, data-quality problems, or patterns that do not generalize to a new workforce.

- The app does not explain why a prediction was made.
- The supplied code does not show training data preparation, evaluation, calibration, or fairness testing.
- The user-input fields may not represent all factors relevant to employee experience.
- The app's labels depend on the target-class encoding used during training.
- The probability output should not be treated as certainty.
- A high-risk prediction does not mean an employee will leave.
- Age, gender, and marital status are sensitive personal attributes in many employment contexts and can introduce fairness, privacy, and legal concerns.

**Responsible-use recommendation:** Treat the output as an exploratory analytical signal, not as a basis for punishment, exclusion, termination, or other consequential employment decisions. Any real-world use should involve appropriate privacy safeguards, fairness and validity testing, human oversight, and review of applicable laws and organizational policies.

---

## 🚀 Potential Improvements

The following are proposed extensions, not features currently present in the supplied code.

- Add a training script with documented preprocessing and target-label definitions.
- Save the complete preprocessing-and-model pipeline to reduce manual encoding mismatches.
- Validate input ranges and relationships between inputs.
- Verify the expected feature schema before making predictions.
- Use named pandas DataFrames when the estimator was trained with feature names.
- Add a clear explanation of what the predicted classes mean.
- Evaluate the model using a held-out test set and appropriate classification metrics.
- Assess calibration and choose probability thresholds based on validated criteria.
- Perform fairness and subgroup performance analysis before any real-world use.
- Add model explainability, with appropriate cautions about explanation limitations.
- Add input validation and more helpful error messages for missing or incompatible files.
- Resolve model paths relative to the script directory.
- Add automated tests for feature encoding and inference.
- Pin dependency versions for reproducible deployment.
- Add a deployment guide and configuration for hosting.
- Add privacy guidance and avoid collecting unnecessary employee-identifying information.

---

## 📄 License

No license was provided in the supplied code. Add a `LICENSE` file and update this section once the project's intended license has been chosen. Until then, do not assume that the code or model artifacts are licensed for unrestricted reuse.

---

## 👩‍💻 Author

**Project:** Employee Attrition Predictor

Add your name, GitHub profile, LinkedIn profile, repository URL, and contact information here if you want to identify the project author.

---

## ⭐ Final Note

The Employee Attrition Predictor demonstrates how a saved machine learning classifier can be connected to an interactive Streamlit interface. For a reliable and responsible application, ensure that inference preprocessing matches training, validate the model on appropriate data, confirm the class-label mapping, and assess fairness and privacy risks before interpreting predictions in an employment setting.
