# 🏥 Medical Insurance Price Prediction (Deep Learning Project)

## 📌 Overview
This project is a **Deep Learning-based web application** that predicts medical insurance costs based on user inputs such as age, BMI, smoking status, and lifestyle factors.

The system uses an **Artificial Neural Network (ANN)** trained on an insurance dataset and deployed using **Streamlit** for real-time predictions.

---

## 🚀 Features
- 🔹 Real-time insurance cost prediction  
- 🔹 Deep Learning model (ANN)  
- 🔹 User-friendly web interface (Streamlit)  
- 🔹 Input validation (name, phone number)  
- 🔹 Lifestyle-based cost adjustments  
- 🔹 USD → INR conversion  
- 🔹 Fast and interactive UI  

---

## 🧠 Tech Stack
- **Programming Language:** Python  
- **Deep Learning:** TensorFlow / Keras  
- **Machine Learning Tools:** Scikit-learn  
- **Data Handling:** Pandas, NumPy  
- **Web Framework:** Streamlit  

---

## 📊 Dataset
- **Dataset Name:** Insurance Dataset  
- **Source:** Kaggle  
- **Features:**
  - Age  
  - Sex  
  - BMI  
  - Children  
  - Smoker  
  - Region  
- **Target:**
  - Charges (Insurance Cost)  

---

## ⚙️ Data Preprocessing
- **Label Encoding:**
  - Sex → Male (1), Female (0)  
  - Smoker → Yes (1), No (0)  

- **One-Hot Encoding:**
  - Region converted into multiple columns  

- **Feature Scaling:**
  - StandardScaler used to normalize data  

---

## 🧠 Model Architecture (ANN)

Artificial Neural Network (Deep Learning Model):

Input Layer  
↓  
Dense (128 neurons, ReLU)  
↓  
Dense (64 neurons, ReLU)  
↓  
Dense (32 neurons, ReLU)  
↓  
Output Layer (1 neuron, Linear Activation)  

**Model Configuration:**
- Optimizer: Adam  
- Loss Function: Mean Squared Error (MSE)  
- Evaluation Metric: Mean Absolute Error (MAE)  
- Epochs: 100  
- Batch Size: 32  

---

## 📈 Model Performance
- Mean Absolute Error (MAE) ≈ 2300  
- Mean Squared Error (MSE) ≈ 18,000,000  

These values indicate that the model provides reasonably accurate predictions.

---

## 🌐 Deployment (Streamlit App)

The trained model is deployed using Streamlit for real-time user interaction.

### 🔄 Workflow:
1. User enters personal & health details  
2. Input data is validated  
3. Encoding and scaling are applied  
4. Data is passed to the ANN model  
5. Prediction is generated  
6. Converted from USD → INR  
7. Additional adjustments applied (lifestyle, medical history)  
8. Final cost displayed  

---

## 📂 Project Structure
📁 Medical-Insurance-Prediction
│
├── app.py # Streamlit application
├── train_model.py # Model training script
├── insurance.csv # Dataset
├── model.pkl # Trained ANN model
├── scaler.pkl # Feature scaler
├── requirements.txt # Dependencies
└── README.md # Project documentation


---

## ▶️ How to Run the Project

### 1️⃣ Clone the Repository

git clone https://github.com/your-username/medical-insurance-prediction.git
cd medical-insurance-prediction

2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run the Application
streamlit run app.py

## 🔮 Future Improvements
- Use larger and updated datasets  
- Perform hyperparameter tuning  
- Add more health-related features  
- Deploy on cloud platforms (AWS, Heroku)  
- Improve UI/UX design  

---

## 🎯 Conclusion
This project demonstrates the practical application of **Deep Learning using an Artificial Neural Network (ANN)** for solving a real-world problem.

It integrates:
- Data preprocessing  
- Model training  
- Model deployment  

into a complete, working system for insurance cost prediction.

---

## 👨‍💻 Authors
- Shirdi Sai K  
- Kasu Sai Kiran Reddy  

---

## 📚 References
- TensorFlow Documentation  
- Keras Documentation  
- Scikit-learn Documentation  
- Streamlit Documentation  
- Kaggle Insurance Dataset  
