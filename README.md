# TrendScope Backend

TrendScope Backend is the API and machine learning engine for a YouTube trend prediction platform. It combines historical YouTube datasets with live YouTube Data API v3 data to generate trend predictions, topic clusters, sentiment insights and creator focused recommendations.

## Features
- YouTube trend data ingestion
- NLP preprocessing and TF-IDF feature extraction
- K-Means topic clustering
- Random Forest based viral potential prediction
- Sentiment analysis and LLM-based strategy generation
- REST API endpoints for frontend integration
- PostgreSQL database integration

## Tech Stack
Python, Flask, PostgreSQL, SQLAlchemy, scikit-learn, Pandas, NumPy, TF-IDF, K-Means, Random Forest

## Project Structure
```text
backend/
├── app/
│   ├── api/
│   ├── ml/
│   ├── models/
│   ├── services/
│   └── utils/
├── docs/
├── migrations/
├── notebooks/
├── scripts/
├── .env.example
├── requirements.txt
└── run.py
```

## Setup Instructions
```bash
git clone https://github.com/Samadhi-Kandewela/trendscope-backend.git
cd trendscope-backend/backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

## Environment Variables

Create a `.env` file using `.env.example`.

```env
DATABASE_URL=your_database_url
YOUTUBE_API_KEY=your_youtube_api_key
GROQ_API_KEY=your_groq_api_key
```
