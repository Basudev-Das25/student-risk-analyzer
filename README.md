# 🎓 Student Risk Analyzer (Random Forest ML Project)

This project is a machine learning–based academic risk prediction system that classifies students into **Low**, **Medium**, or **High** risk categories based on study behavior and academic performance indicators.

The project demonstrates the complete machine learning lifecycle, including data preprocessing, feature engineering, multi-class classification using Random Forest, model evaluation, explainability through feature importance, and deployment readiness.

## 🚀 Running & Deploying the Application

### Run Locally
```bash
git clone https://github.com/YOUR_USERNAME/student-risk-analyzer.git
cd student-risk-analyzer
pip install -r requirements.txt
cd app
python app.py

Then open:
http://127.0.0.1:5000

Deploy on Render (Optional)

Create a new Web Service on Render

Connect the GitHub repository

Use the following settings:

Root Directory: (leave empty)

Build Command:
pip install -r requirements.txt

Start Command:
gunicorn app.app:app

The application is production-ready and uses deployment-safe paths.

This is **huge** for recruiters and collaborators.

---

## 5️⃣ Add a “Why This Repo Is Deployable” Section (Very Strong)

Add this to README:

```markdown
## ✅ Deployment-Ready Design

- Training and inference are fully separated
- Model artifacts are versioned and included
- Absolute file paths ensure environment independence
- Minimal dependency specification
- Production-compatible WSGI server (Gunicorn)

The project can be deployed on any standard Python hosting platform.

