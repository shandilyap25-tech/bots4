from flask import Flask, request, jsonify, send_from_directory
import sys
import os
from langchain_core.messages import HumanMessage, AIMessage

# Add the project root to the Python path to import the agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent import AutoStreamAgent

app = Flask(__name__)

# Initialize agent (API keys must be set in Vercel Environment Variables)
agent = AutoStreamAgent()

# In-memory session storage (For local testing & demo)
user_sessions = {}

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    if not data or 'message' not in data:
        return jsonify({"error": "Please provide a 'message' field in the JSON body."}), 400
        
    user_message = data['message']
    session_id = data.get('session_id', 'default_session')
    
    # Initialize state for new sessions
    if session_id not in user_sessions:
        user_sessions[session_id] = {
            "messages": [],
            "intent": None,
            "user_name": None,
            "user_email": None,
            "user_platform": None,
            "lead_captured": False,
            "conversation_turn": 0,
            "knowledge_context": None
        }
        
    state = user_sessions[session_id]
    state["messages"].append(HumanMessage(content=user_message))
    
    # Run the graph and update state
    result = agent.graph.invoke(state)
    user_sessions[session_id] = result
    
    # Extract response
    if result["messages"]:
        last_message = result["messages"][-1]
        response = last_message.content if isinstance(last_message, AIMessage) else str(last_message)
    else:
        response = "I couldn't generate a response."
    
    return jsonify({"response": response})

# Serve the frontend locally
@app.route('/')
def serve_index():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return send_from_directory(root_dir, 'index.html')

if __name__ == '__main__':
    print("Starting local server at http://localhost:5000")
    app.run(port=5000, debug=True)