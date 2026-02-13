from flask import Flask, request, jsonify
from flask_cors import CORS
from chat_with_llm import KGRAG

app = Flask(__name__)
CORS(app)

chatbot = KGRAG()

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    query = data.get("query")
    answer = chatbot.chat(query)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
