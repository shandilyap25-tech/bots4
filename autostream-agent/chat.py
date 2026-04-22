#!/usr/bin/env python
"""
Interactive CLI for AutoStream Agent
This allows you to chat with the agent in real-time
"""

import sys
from agent import AutoStreamAgent


def print_banner():
    """Print welcome banner"""
    print("\n" + "="*70)
    print("  🤖 AutoStream Sales Agent - Interactive Chat")
    print("="*70)
    print("""
Welcome to AutoStream Agent! This is an interactive chat interface.
Type your messages to converse with the AI sales agent.

Commands:
  /quit       - Exit the chat
  /clear      - Start a new conversation
  /help       - Show this help message
  /demo       - Run a demo conversation

Example conversation flow:
  1. Ask about pricing
  2. Show interest in a plan
  3. Provide your information (name, email, platform)
  4. Complete the lead capture
""")
    print("="*70 + "\n")


def print_help():
    """Print help message"""
    print("""
Commands:
  /quit       - Exit the chat
  /clear      - Start a new conversation
  /help       - Show this help message
  /demo       - Run a demo conversation
  /status     - Show conversation status

Tips:
  • Be natural in your conversation
  • The agent will detect your intent automatically
  • If you show high interest, the agent will ask for your details
  • All information is needed: name, email, and platform
""")


def run_demo():
    """Run a quick demo conversation"""
    print("\n📺 Running Demo Conversation...\n")
    
    agent = AutoStreamAgent()
    
    demo_messages = [
        "Hi, I'm interested in video editing software",
        "Can you tell me about your pricing?",
        "That sounds great! I want to try the Pro plan for my YouTube channel",
        "My name is Sarah Johnson and my email is sarah.johnson@email.com"
    ]
    
    for message in demo_messages:
        print(f"👤 You: {message}")
        response = agent.chat(message)
        print(f"🤖 Agent: {response}\n")
        input("Press Enter to continue...")


def main():
    """Main CLI loop"""
    print_banner()
    
    try:
        # Initialize agent
        print("🔄 Initializing AutoStream Agent...", end="", flush=True)
        agent = AutoStreamAgent()
        print(" ✅\n")
        
        conversation_history = []
        
        while True:
            try:
                # Get user input
                user_input = input("👤 You: ").strip()
                
                # Handle empty input
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith("/"):
                    command = user_input.lower()
                    
                    if command == "/quit" or command == "/exit":
                        print("\n👋 Thank you for using AutoStream Agent! Goodbye!\n")
                        break
                    
                    elif command == "/clear":
                        conversation_history = []
                        print("\n✅ Conversation cleared. Starting fresh!\n")
                        continue
                    
                    elif command == "/help":
                        print_help()
                        continue
                    
                    elif command == "/demo":
                        run_demo()
                        continue
                    
                    elif command == "/status":
                        print(f"\n📊 Conversation Status:")
                        print(f"   Turns: {len(conversation_history) // 2}")
                        print(f"   Messages: {len(conversation_history)}\n")
                        continue
                    
                    else:
                        print("❌ Unknown command. Type /help for available commands.\n")
                        continue
                
                # Get agent response
                print("🤖 Agent: ", end="", flush=True)
                response = agent.chat(user_input)
                print(response + "\n")
                
                # Store in history
                conversation_history.append({"role": "user", "content": user_input})
                conversation_history.append({"role": "assistant", "content": response})
            
            except KeyboardInterrupt:
                print("\n\n⏹️  Interrupted. Type /quit to exit properly.\n")
            
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                print("Please try again or type /help for commands.\n")
    
    except Exception as e:
        print(f"\n❌ Failed to initialize agent: {str(e)}")
        print("\nMake sure to:")
        print("1. Set your ANTHROPIC_API_KEY environment variable")
        print("2. Run: pip install -r requirements.txt")
        print("3. Check that knowledge_base/autostream_kb.json exists")
        sys.exit(1)


if __name__ == "__main__":
    main()
