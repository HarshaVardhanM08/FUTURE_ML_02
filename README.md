# 🎫 AI-Powered Support Ticket Classification System
### Machine Learning Project – FUTURE_ML_02

---

# 📌 Project Overview

The **AI-Powered Support Ticket Classification System** is a machine learning application designed to automatically analyze, categorize, and prioritize customer support tickets.

Modern businesses receive **thousands of customer support requests daily** through email, chat systems, and helpdesk platforms. Manually reviewing and categorizing these tickets can be **time-consuming, inefficient, and prone to errors**.

This project leverages **Natural Language Processing (NLP)** and **Machine Learning** to build an intelligent system that can:

- Automatically classify support tickets into categories
- Estimate the priority level of each ticket
- Provide an interactive analytics dashboard
- Train a machine learning model for improved accuracy
- Suggest responses for faster customer service

The application is built using **Python, Streamlit, Scikit-Learn, Pandas, and Plotly**, making it a powerful and easy-to-use AI-powered helpdesk analytics tool.

---

# 🚀 Key Features

## 📊 Interactive Support Ticket Dashboard

The application includes a **real-time analytics dashboard** that visualizes support ticket data and insights.

Dashboard metrics include:

- Total number of tickets
- Number of ticket categories
- High-priority ticket count
- Average ticket text length

Visualizations include:

- Category distribution chart
- Priority breakdown chart
- Category vs Priority heatmap
- Raw dataset preview

These insights help support teams **identify patterns and manage workloads efficiently**.

---

## 📁 Dataset Upload and Processing

Users can upload their own **CSV dataset of support tickets**.

The system automatically:

- Detects text columns containing ticket descriptions
- Allows users to select the appropriate column
- Classifies all tickets automatically
- Displays classification results in real-time

This enables quick processing of **large-scale customer support datasets**.

---

# 🧠 Ticket Classification System

The project supports **two classification approaches**.

---

## 1️⃣ Keyword-Based Classification

When no machine learning model is trained, the system uses a **rule-based keyword classifier**.

It detects important keywords in the ticket text and assigns categories accordingly.

Example keyword mappings:

| Category | Example Keywords |
|--------|--------|
| Billing | bill, payment, invoice, refund, charge |
| Technical | error, bug, crash, not working |
| Account | login, password, account access |
| Shipping | delivery, package, tracking |
| Returns | refund, exchange, damaged item |
| General Inquiry | question, help, information |

This ensures the system **works immediately even without training a model**.

---

## 2️⃣ Machine Learning Classification

For higher accuracy, the system allows training a **machine learning classifier**.

The model pipeline includes:

```
Ticket Text
     ↓
TF-IDF Vectorization
     ↓
Logistic Regression Classifier
     ↓
Predicted Ticket Category
```

This approach allows the model to **learn patterns from historical ticket data** and improve prediction accuracy.

---

# 🔬 Machine Learning Pipeline

The machine learning pipeline includes several important steps.

### Text Preprocessing

The ticket text is cleaned and prepared for machine learning using:

- Lowercasing
- Tokenization
- Stop-word removal
- Text vectorization

---

### Feature Engineering

Text features are extracted using **TF-IDF (Term Frequency – Inverse Document Frequency)**.

TF-IDF helps identify **important words in ticket descriptions**.

Configuration includes:

- Maximum feature size
- N-gram range (1,2)
- Sublinear term frequency scaling

---

### Classification Model

The project uses **Logistic Regression** as the classifier.

Advantages of Logistic Regression for text classification:

- Fast training
- Good accuracy
- Interpretable model coefficients
- Works well with TF-IDF features

The classifier predicts the **category of each support ticket**.

---

# 🏷️ Ticket Categories

The system supports the following categories:

- Billing Issues
- Technical Problems
- Account Management
- Shipping & Delivery
- Returns and Refunds
- General Inquiry
- Other Issues

These categories help organize support requests and route them to the appropriate teams.

---

# ⚡ Priority Detection System

The system also estimates the **priority level of each support ticket**.

Priority levels include:

### High Priority

Detected using keywords such as:

- urgent
- critical
- immediately
- emergency
- cannot access
- fraud

These tickets require **immediate attention**.

---

### Medium Priority

Default level for standard support requests.

---

### Low Priority

Detected using phrases such as:

- just asking
- no rush
- whenever possible
- curious about

These tickets can be handled **after urgent requests**.

---

# 📈 Model Evaluation

After training the machine learning model, the system displays detailed evaluation metrics.

Evaluation includes:

### Classification Report

- Accuracy
- Precision
- Recall
- F1 Score

These metrics measure how well the model predicts ticket categories.

---

### Confusion Matrix

A confusion matrix visualizes **prediction performance across categories**, showing where the model performs well or makes mistakes.

---

### Top Predictive Words

The system identifies the **most important words influencing predictions**.

Example:

| Category | Important Terms |
|--------|--------|
| Billing | payment, invoice, charge |
| Technical | error, bug, crash |
| Shipping | delivery, tracking |
| Account | login, password |

This helps interpret how the machine learning model works.

---

# 💬 Suggested Customer Responses

The system also provides **automated response suggestions** for different ticket categories.

Example:

Billing Issue Response

```
Hi! I can see you have a billing concern.
I'll review your account charges and follow up within 24 hours.
```

Technical Issue Response

```
Thanks for reaching out!
Our technical team has been notified. Could you provide screenshots or error messages?
```

These responses help **reduce support response time**.

---

# ✏️ Single Ticket Prediction

Users can also paste **individual support messages** into the system.

Example input:

```
I was charged twice for my subscription and need a refund immediately.
```

Example output:

```
Predicted Category: Billing
Estimated Priority: High
```

The system also displays **confidence scores for predictions**.

---

# 🖥️ Application Pages

The Streamlit application contains four main sections.

### 📊 Dashboard

Displays analytics and visualizations for support tickets.

---

### 📁 Upload & Classify

Allows users to upload CSV datasets and automatically classify tickets.

---

### 🤖 Train Model

Users can train a machine learning model using labelled data.

The training interface allows configuration of:

- Test split percentage
- TF-IDF feature size
- Training dataset columns

---

### ✏️ Classify Single Ticket

Users can manually enter ticket text to get instant predictions.

---

# 📂 Project Structure

```
FUTURE_ML_02
│
├── Support Ticket Classification.ipynb
├── Ticket_app.py
├── customer_support_tickets.csv
├── README.md
├── LICENSE
```

File descriptions:

Support Ticket Classification.ipynb  
Notebook used for experimentation and model development.

Ticket_app.py  
Main Streamlit application.

customer_support_tickets.csv  
Sample dataset used for training and testing.

README.md  
Project documentation.

LICENSE  
MIT License for open-source usage.

---

# ⚙️ Installation

## 1️⃣ Clone the Repository

```
git clone https://github.com/HarshaVardhanM08/FUTURE_ML_02.git
cd FUTURE_ML_02
```

---

## 2️⃣ Install Dependencies

```
pip install streamlit pandas scikit-learn plotly
```

---

## 3️⃣ Run the Application

```
streamlit run Ticket_app.py
```

The application will open in your browser at:

```
http://localhost:8501
```

---

# 🧰 Technologies Used

| Technology | Purpose |
|--------|--------|
| Python | Core programming language |
| Streamlit | Web application framework |
| Pandas | Data manipulation |
| Scikit-Learn | Machine learning algorithms |
| Plotly | Interactive data visualization |
| TF-IDF | Feature extraction |
| Logistic Regression | Text classification |

---

# 🌍 Real-World Applications

This system can be used in:

- Customer Support Platforms
- Helpdesk Automation Systems
- IT Service Management
- SaaS Customer Support
- AI Chatbots
- CRM systems

It can significantly improve:

- Ticket routing
- Response time
- Customer satisfaction
- Support team efficiency

---

# 🔮 Future Improvements

Possible improvements include:

- Deep Learning models (BERT, Transformers)
- Multi-language ticket support
- Integration with CRM systems
- Real-time email ticket processing
- AI-generated automated replies
- Cloud deployment for production use

---

# 📜 License

This project is licensed under the **MIT License**.

---

# 👨‍💻 Author

Harsha Vardhan Maradana

Python Full Stack Developer (In Progress) | Game Development Enthusiast | Focused on Creating Immersive & Innovative Digital Experiences 

GitHub  
https://github.com/HarshaVardhanM08

---

# ⭐ Support

If you found this project helpful:

⭐ Star the repository  
🍴 Fork the project  
📢 Share it with others
