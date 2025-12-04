# 🔥 Calories Burnt Prediction App

A Machine Learning web application built using **Linear Regression** to predict the approximate calories burnt during exercise based on various physiological and workout-related factors.  
This project was developed as part of the **Samsung Innovation Campus – Artificial Intelligence Program** and completed as a **team project**.

The application is deployed using **Streamlit**, enabling an interactive and user-friendly interface.

---

## 📌 Project Overview

This app predicts the number of calories burnt during a workout session using user inputs such as:

- Gender  
- Age  
- Height  
- Weight  
- Exercise Duration  
- Heart Rate  
- Body Temperature  

📊 Dataset:
The model is trained on the [**Calories Burnt Prediction Dataset**](https://www.kaggle.com/datasets/ruchikakumbhar/calories-burnt-prediction/data
) from Kaggle.

---

## 🧠 Machine Learning Model

The project uses:

- **Linear Regression Model**
- **Standard Scaler** for feature scaling

**Training Pipeline:**

1. Data Cleaning  
2. Encoding (Gender → Binary Encoding)  
3. Train-test split  
4. Scaling numerical features  
5. Training Linear Regression  
6. Evaluation using regression metrics  
7. Exporting `calories_model.pkl` and `scaler.pkl` using joblib  

---

## 📊 Model Evaluation

| Metric | Train Score | Test Score |
|--------|-------------|-------------|
| **R² Score** | **0.96716** | **0.96729** |
| **Mean Squared Error (MSE)** | 126.94 | 131.99 |

➡️ The model generalizes extremely well with **~96.7% accuracy (R² score)** on both training and test sets.

---

## 📂 Project Structure

```
Calories-Burnt-Prediction/
│
├── app.py # Main Streamlit web app
├── calories_model.pkl # Trained ML model
├── scaler.pkl # Fitted scaler for preprocessing
├── requirements.txt # Project dependencies
└── README.md 
```

---

## 🖥️ Tech Stack

- **Python**
- **Streamlit**
- **NumPy**
- **scikit-learn**
- **Joblib**
- **Pandas** (for preprocessing)
- **Linear Regression Algorithm**

---

## 🎮 Features of the Web App

- Simple, clean, and intuitive interface  
- Predicts calories burnt instantly  
- Real-time input fields  
- Uses gender encoding internally  
- Scales input using trained StandardScaler  
- Lightweight, fast, and accurate  

---

## 🛠️ Run This Project Locally

Follow the steps below to run the project on your machine.

### **1️⃣ Clone the Repository**

```
git clone https://github.com/InsaneIshita/Calories-Burnt-Prediction
cd Calories-Burnt-Prediction
```

### **2️⃣ Create a Virtual Environment (Recommended)**

```
python -m venv env
```

Activate it:

**Windows**
```
env\Scripts\activate
```

**Mac/Linux**
```
source env/bin/activate
```

### **3️⃣ Install Required Libraries**

```
pip install -r requirements.txt
```

### **4️⃣ Run the Streamlit App**

```
streamlit run app.py
```

---

## 👥 Team & Acknowledgement

This project is developed as part of the **Samsung Innovation Campus – AI Course** by our team:

- **Ishita Singh**
- **Anchal Vishwakarma**
- **Prachi Verma**
- **Aditya Yadav**

We thank Samsung Innovation Campus mentors for guiding us throughout the course.

---

## ❤️ Contribution

Pull requests and suggestions are welcome!  
If you find issues, feel free to create an issue in the repository.

---

## ⭐ Support

If you like this project, don't forget to **star ⭐ the repository**!
