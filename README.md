# 📘 Sales Prediction

A machine learning model to forecast units sold based on advertising budget in TV, Radio and Newspaper

## 🔍 Overview

This project focuses on building a predictive model using the Advertising dataset, a well-known dataset used in marketing and data science domains. The dataset contains data on advertising expenditures across three media channels—TV, Radio, and Newspaper—and the corresponding Sales figures for a product.

The goal of this project is to analyze the relationship between advertising budgets and sales performance, and to predict future sales based on given advertising investments. The dataset enables us to explore how different marketing channels contribute to revenue and to build a regression model that can assist in decision-making and budget allocation.

## 📁 Project Structure
```
├── data/                # Raw and processed
├── models/              # Trained machine learning model
├── notebooks/           # Jupyter notebooks for exploratory data analysis (EDA) and main pipeline
├── src/                 # Source code for data preprocessing, model training & evaluation along with entire pipeline file
├── tests/               # Unit tests for individual components
├── .gitignore/          # Untracked files that Git ignore
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation
└── streamlit-app.py     # Streamlit application
```
## 📊 Data

-- Source: Advertising dataset referred by CodeAlpha.(Downloaded from Kaggle) https://www.kaggle.com/datasets/bumba5341/advertisingcsv

-- Columns: ID, TV, Radio, Newspaper, Sales

-- Preprocessing: Data cleaning, Removing unwanted columns and Scaling

## 🧠 Modeling

This project uses Linear Regression method to predict the amount of units that will be sold based on the advertising budget in each sector ; TV, Radio & Newspaper.

## 🚀 Installation & Usage

### Install dependencies

pip install -r requirements.txt

### Run the project using Streamlit
To use the streamlit application run the below in terminal:

streamlit run .\streamlit-app.py

### To run Jupyter notebooks for experimentation:

jupyter notebook

## ✅ Testing

To ensure the quality of the code, run the unit tests from terminal using:

pytest

## 📈 Results

Exploratory data analysis with some visualization shows clearly that TV advertising is the feature/variable that has the highest impact on units sold followed by Radio Advertising. However, Newspaper has the lowest impact on future sales. This really reflect the real world situation nowadays as visual media is much more important and attractive and has audience more than written media.

The final model achieves a root mean squared error of 1.6, which means on average the model’s predictions are off by about 1.6 thousand. In terms of percentage, Mean Absolute Error Percenatge was calculated to show the average error of the model's prediction which is around 14%.

## 🗒️ Notes / Future Work

The advertising dataset is a small and old dataset compared to real-world advertising and marketing, an interesting real future world should take into consideration social media instead of all sectors mentioned in this project. Comparing future sales based on advertising and marketing budget on 'Facebook vs. Whatsapp vs. Instagram vs. Tiktok' could be a needed interesting project.


## 👤 Author
Mohammad Ezzeddine

github.com/Mhmd-14 | in/mohammad-ezzeddine-14ae1
