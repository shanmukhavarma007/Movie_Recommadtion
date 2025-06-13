# 🎬 Movie Recommendation System

This is a Movie Recommendation System built using Python, Streamlit, and Machine Learning. It supports:

- ✅ Content-Based Filtering (based on genres)
- ✅ Collaborative Filtering (based on user ratings using SVD)
- ✅ Interactive Web UI using Streamlit

---

## 📁 Dataset

Uses [MovieLens 100k Dataset](https://grouplens.org/datasets/movielens/100k/) with the following files:
- `movies.csv`
- `ratings.csv`

---

## 🚀 Installation

1. Clone the repo:
```bash
git clone https://github.com/yourusername/movie-recommendation-system.git
cd movie-recommendation-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run app.py
```

---

## 🖥️ Features

- Recommends similar movies using **genres** (TF-IDF + Cosine Similarity)
- Suggests personalized recommendations based on a **user's past ratings**
- Clean, interactive interface using **Streamlit**

---

## 📦 Deployment

You can deploy this project for free using [Streamlit Cloud](https://streamlit.io/cloud). Just connect your GitHub repo and click “Deploy”.

---

## 📜 License

MIT License © Your Name
