# 🎓 Student Performance Predictor
machine learning project using linear regression and random forest

A Flask + Machine Learning web app that predicts **student performance** using:  
- **Random Forest Classifier** → Predicts **Pass/Fail**  
- **Linear Regression** → Predicts the **average score**  

The app includes a modern TailwindCSS UI and stores prediction history.

---

## ⚡ Features
- 📊 Predict Pass/Fail from student details  
- 📈 Predict average score using regression  
- 🖼️ Beautiful UI (Tailwind + custom static folder)  
- 📝 Prediction history page  

---

## 🚀 Setup & Installation 

1️⃣ Clone the Repository  
```bash
git clone https://github.com/<your-username>/student-performance-predictor.git
cd student-performance-predictor

2️⃣ Create Virtual Environment & Install Dependencies

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

pip install -r requirements.txt

3️⃣ Train Models
python train_model.py

4️⃣ Run Flask App
python app.py

Then open :  http://127.0.0.1:5000
