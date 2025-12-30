# 📚 Semantic Book Recommender

An **AI-powered semantic book recommendation system** that suggests books based on **natural language descriptions**, **genre preferences**, and **emotional tone**.  
The system leverages **Hugging Face sentence embeddings**, **LangChain**, **ChromaDB**, and an interactive **Gradio dashboard** — with **no external API keys required**.

---

## 🚀 Project Overview

Traditional recommendation systems rely on keywords or ratings.  
This project goes beyond that by using **semantic search** to understand *what the user means*, not just what they type.

Users can:
- Describe the kind of book they want in plain English
- Filter by category (genre)
- Select an emotional tone (happy, sad, suspenseful, etc.)
- Instantly receive visually rich book recommendations

---

## 🧠 How It Works

1. **Book Descriptions Processing**
   - Book metadata and descriptions are preprocessed and stored.
   - Descriptions are tagged and chunked for efficient semantic retrieval.

2. **Embeddings**
   - Uses the Hugging Face model:
     ```
     sentence-transformers/all-MiniLM-L6-v2
     ```
   - Converts text into dense vector embeddings.

3. **Vector Database**
   - Embeddings are stored in **ChromaDB**
   - Enables fast semantic similarity search

4. **Recommendation Logic**
   - Retrieves top semantically similar books
   - Applies category filtering (optional)
   - Ranks results based on emotional tone scores (joy, fear, anger, etc.)

5. **User Interface**
   - Built with **Gradio**
   - Clean, responsive, and interactive UI

---

## 🛠️ Tech Stack

- **Python**
- **Pandas & NumPy** – data handling
- **LangChain** – document loading & orchestration
- **ChromaDB** – vector storage
- **Hugging Face Transformers** – embeddings
- **Gradio** – web-based dashboard
- **dotenv** – environment management

---

## 📂 Project Structure

├── gradio_dashboard.py -Main application file (Gradio UI + recommendation logic)

├── books_with_emotions.csv -Book metadata with emotion scores

├── books_with_categories.csv -Genre/category information

├── books_cleaned.csv -Cleaned and preprocessed book dataset

├── tagged_description.txt -Tagged book descriptions for semantic search

├── cover-not-found.jpg -Fallback image for missing book covers

├── requirements.txt -Python dependencies

├── README.md -Project documentation

├── *.ipynb -Jupyter notebooks for EDA and experiments


---

## 🎨 Features

- 🔍 **Semantic Search**  
  Understands user intent using vector embeddings rather than keyword matching.

- 😊 **Emotion-Aware Recommendations**  
  Books are ranked based on emotional tone such as happy, sad, suspenseful, or surprising.

- 📚 **Category Filtering**  
  Allows users to filter recommendations by book genre.

- 🖼️ **Book Cover Previews**  
  Displays book thumbnails with a fallback image when unavailable.

- ⚡ **Fast & Lightweight Embeddings**  
  Powered by `sentence-transformers/all-MiniLM-L6-v2` for efficient semantic search.

- 🔐 **No API Keys Required**  
  Fully open-source solution using Hugging Face models.

---

## 📈 Example Use Cases

- *“A heartwarming story about friendship and personal growth”*  
- *“Dark, suspenseful novels with mystery elements”*  
- *“Light and happy books for casual reading”*

---

## 🔮 Future Improvements

- User authentication and personalized recommendations
- Recommendation history and bookmarking
- Advanced ranking and reranking strategies
- Deployment on Hugging Face Spaces or cloud platforms
- Multilingual semantic book search

  ---

  ## Demo Video

[![Watch the video](https://img.youtube.com/vi/Ui0UJn31VRk/0.jpg)](https://youtu.be/Ui0UJn31VRk)

