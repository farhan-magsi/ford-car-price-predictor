# Ford Car Price Prediction 🚗💰

This repository contains a Machine Learning project focused on predicting the prices of used Ford cars based on various vehicular attributes. The project handles data preprocessing, exploratory data analysis (EDA), and builds a regression model to estimate car prices accurately.

---

## 📊 Dataset Overview
The dataset contains information on approximately **17,966 used Ford cars** with **9 distinct features**:
*   **model**: The specific model of the Ford car.
*   **year**: The registration year of the vehicle.
*   **price**: The target variable (price of the car in R).
*   **transmission**: Type of gearbox (Manual, Automatic, Semi-Auto).
*   **mileage**: Total miles driven by the car.
*   **fuelType**: Type of fuel used (Petrol, Diesel, Hybrid, etc.).
*   **tax**: Road tax amount.
*   **mpg**: Miles per gallon (fuel efficiency).
*   **engineSize**: Capacity of the engine (in liters).

---

## 🛠️ Tech Stack & Libraries
The project is implemented in Python using a Jupyter Notebook. The following key libraries were utilized:
*   **Data Manipulation:** `pandas`, `numpy`
*   **Data Visualization:** `matplotlib`, `seaborn`
*   **Machine Learning:** `scikit-learn`

---

## 🚀 Key Features of the Project
*   **Data Cleaning:** Handled missing values, duplicates, and verified logical data consistency.
*   **Exploratory Data Analysis (EDA):** Analyzed correlations between features (like how `year` and `mileage` affect the `price`) using heatmaps, scatter plots, and distribution charts.
*   **Feature Engineering:** Encoded categorical variables (`model`, `transmission`, `fuelType`) to make them compatible with machine learning algorithms.
*   **Model Building:** Implemented Regression Analysis to train the model on the vehicle dataset.

---

## ⚙️ How to Run the Project Local System

1. **Clone the Repository:**
```bash
   git clone [https://github.com/Farhan-magsi/Ford-Car-Price-Prediction.git](https://github.com/Farhan-magsi/Ford-Car-Price-Prediction.git)
   cd Ford-Car-Price-Prediction