from flask import Flask, request, jsonify, send_from_directory
import sys
import os

# Add the project root to the Python path to import the agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent import AutoStreamAgent

app = Flask(__name__)

# Initialize agent (API keys must be set in Vercel Environment Variables)
agent = AutoStreamAgent()

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    if not data or 'message' not in data:
        return jsonify({"error": "Please provide a 'message' field in the JSON body."}), 400
        
    user_message = data['message']
    response = agent.chat(user_message)
    
    return jsonify({"response": response})

# Serve the frontend locally
@app.route('/')
def serve_index():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return send_from_directory(root_dir, 'index.html')

if __name__ == '__main__':
    print("Starting local server at http://localhost:5000")
    app.run(port=5000, debug=True)