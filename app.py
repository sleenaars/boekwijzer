# app.py
from flask import Flask, request, jsonify
from pinecone import Pinecone
from openai import OpenAI
from flask_cors import CORS
import os

# Import memory and query function
from utils_books import answer_query, conversation_history

app = Flask(__name__)
CORS(app)

# Initialize APIs
pc = Pinecone()
index = pc.Index("boekwijzer-index")
client = OpenAI()

#reset when page is reloaded
conversation_history.clear()
print("Conversation memory reset on server startup.")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("query")
    selected_titles = data.get("selected_titles")  # <-- nieuwe regel

    if not query:
        return jsonify({"answer": "No query provided."})

    # geef selected_titles door
    answer = answer_query(query, index, client, selected_titles=selected_titles)
    return jsonify({"answer": answer})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
