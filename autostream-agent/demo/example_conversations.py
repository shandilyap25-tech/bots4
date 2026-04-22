"""
Demo Script for AutoStream Agent
This file demonstrates the agent in action with different conversation scenarios
"""

from agent import AutoStreamAgent
import time


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def print_turn(turn_num, role, message):
    """Print a conversation turn"""
    role_emoji = "👤 User" if role == "user" else "🤖 Agent"
    print(f"\n[Turn {turn_num}] {role_emoji}:")
    print(f"  {message}")
    time.sleep(1)  # Pause for readability


def demo_1_pricing_inquiry():
    """Demo 1: User asks about pricing"""
    print_section("DEMO 1: Pricing Inquiry")
    print("Scenario: User wants to know about pricing options")
    
    agent = AutoStreamAgent()
    
    conversation = [
        "Hi, I'm interested in your video editing tool",
        "Can you tell me about your pricing options?",
        "What's the difference between Basic and Pro?",
        "Thanks for the info!"
    ]
    
    result = agent.multi_turn_chat(conversation)
    
    for turn_num, exchange in enumerate(result, 1):
        role = exchange["role"]
        content = exchange["content"]
        print_turn(turn_num, role, content)
    
    print("\n✅ Demo 1 Complete: Agent successfully answered pricing questions\n")


def demo_2_high_intent_lead_capture():
    """Demo 2: High-intent lead capture"""
    print_section("DEMO 2: High-Intent Lead Capture")
    print("Scenario: User shows high intent and becomes a qualified lead")
    
    agent = AutoStreamAgent()
    
    conversation = [
        "Hello! Tell me about your platform",
        "I'm impressed! I want to try the Pro plan for my YouTube channel",
        "My name is Alex Rivera",
        "My email is alex.rivera@email.com",
        "YouTube"
    ]
    
    result = agent.multi_turn_chat(conversation)
    
    for turn_num, exchange in enumerate(result, 1):
        role = exchange["role"]
        content = exchange["content"]
        print_turn(turn_num, role, content)
    
    print("\n✅ Demo 2 Complete: Lead successfully captured!\n")


def demo_3_feature_inquiry():
    """Demo 3: Feature-specific inquiry"""
    print_section("DEMO 3: Feature-Specific Inquiry")
    print("Scenario: User asks detailed questions about features")
    
    agent = AutoStreamAgent()
    
    conversation = [
        "Do you support 4K resolution?",
        "Can I get unlimited video exports?",
        "What about AI captions?",
        "This sounds perfect for my content creation"
    ]
    
    result = agent.multi_turn_chat(conversation)
    
    for turn_num, exchange in enumerate(result, 1):
        role = exchange["role"]
        content = exchange["content"]
        print_turn(turn_num, role, content)
    
    print("\n✅ Demo 3 Complete: Feature questions addressed\n")


def demo_4_policy_inquiry():
    """Demo 4: Policy-related questions"""
    print_section("DEMO 4: Policy Questions")
    print("Scenario: User asks about refund and support policies")
    
    agent = AutoStreamAgent()
    
    conversation = [
        "What's your refund policy?",
        "Do you offer 24/7 support?",
        "Is there a free trial?",
        "Great! I'm ready to get started"
    ]
    
    result = agent.multi_turn_chat(conversation)
    
    for turn_num, exchange in enumerate(result, 1):
        role = exchange["role"]
        content = exchange["content"]
        print_turn(turn_num, role, content)
    
    print("\n✅ Demo 4 Complete: Policies explained\n")


def demo_5_creator_use_case():
    """Demo 5: Creator-specific use case"""
    print_section("DEMO 5: Creator Use Case Consultation")
    print("Scenario: User asks about use case for Instagram creators")
    
    agent = AutoStreamAgent()
    
    conversation = [
        "I'm an Instagram content creator, can your tool help me?",
        "What features are best for vertical video content?",
        "This is exactly what I need! Where do I sign up?",
        "I'm Sarah and you can reach me at sarah.m@email.com",
        "Instagram is my main platform"
    ]
    
    result = agent.multi_turn_chat(conversation)
    
    for turn_num, exchange in enumerate(result, 1):
        role = exchange["role"]
        content = exchange["content"]
        print_turn(turn_num, role, content)
    
    print("\n✅ Demo 5 Complete: Creator converted to lead!\n")


def demo_single_turn():
    """Demo: Single turn conversations"""
    print_section("SINGLE TURN EXAMPLES")
    print("Quick examples of single-turn interactions\n")
    
    agent = AutoStreamAgent()
    
    test_queries = [
        "Hi there!",
        "What's your pricing?",
        "I want to subscribe to the Pro plan",
        "How much does it cost for unlimited videos?"
    ]
    
    for query in test_queries:
        print(f"📝 User: {query}")
        response = agent.chat(query)
        print(f"🤖 Agent: {response}\n")


def print_demo_menu():
    """Print available demos"""
    print("\n" + "="*70)
    print("  AutoStream Agent - Demo Menu")
    print("="*70)
    print("""
    1. Pricing Inquiry Demo
    2. High-Intent Lead Capture Demo
    3. Feature Inquiry Demo
    4. Policy Questions Demo
    5. Creator Use Case Demo
    6. Single Turn Examples
    7. Run All Demos
    0. Exit
    """)


def main():
    """Main demo runner"""
    print("\n🚀 Welcome to AutoStream Agent Demo")
    print("="*70)
    
    demos = {
        '1': ("Pricing Inquiry", demo_1_pricing_inquiry),
        '2': ("High-Intent Lead Capture", demo_2_high_intent_lead_capture),
        '3': ("Feature Inquiry", demo_3_feature_inquiry),
        '4': ("Policy Questions", demo_4_policy_inquiry),
        '5': ("Creator Use Case", demo_5_creator_use_case),
        '6': ("Single Turn Examples", demo_single_turn),
    }
    
    while True:
        print_demo_menu()
        choice = input("Select a demo (0-7): ").strip()
        
        if choice == '0':
            print("\n👋 Thank you for using AutoStream Agent Demo!")
            break
        elif choice == '7':
            # Run all demos
            print("\n▶️  Running all demos in sequence...\n")
            for key in sorted(demos.keys()):
                if key != '6':  # Skip single turn for batch run
                    title, demo_func = demos[key]
                    try:
                        demo_func()
                        time.sleep(2)  # Pause between demos
                    except Exception as e:
                        print(f"\n❌ Error in {title}: {str(e)}\n")
        elif choice in demos:
            title, demo_func = demos[choice]
            try:
                demo_func()
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                print("Make sure you have set your ANTHROPIC_API_KEY environment variable\n")
        else:
            print("❌ Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    # Run the default demo sequence for quick testing
    print("\n🎬 AutoStream Agent - Quick Demo")
    print("="*70)
    
    try:
        # Run demo 2 (high-intent lead capture) as default
        print("\nRunning High-Intent Lead Capture Demo...\n")
        demo_2_high_intent_lead_capture()
        
        print("\n" + "="*70)
        print("✅ Demo completed successfully!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nMake sure to:")
        print("1. Set your ANTHROPIC_API_KEY environment variable")
        print("2. Install required packages: pip install -r requirements.txt")
        print("3. Ensure the knowledge_base/autostream_kb.json file exists")
