"""
Setup verification script for AutoStream Agent
Run this to verify your installation is complete
"""

import sys
import os
from pathlib import Path


def check_python_version():
    """Check if Python version is 3.9+"""
    print("✓ Checking Python version...", end=" ")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 9:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor} (requires 3.9+)")
        return False


def check_project_structure():
    """Check if all required files exist"""
    print("\n✓ Checking project structure...")
    
    required_files = [
        "agent.py",
        "knowledge_base/autostream_kb.json",
        "requirements.txt",
        "README.md",
        "tests/test_intent_detection.py",
        "tests/test_rag_retrieval.py",
        "tests/test_lead_capture.py",
        "demo/example_conversations.py"
    ]
    
    all_exist = True
    for file_path in required_files:
        full_path = Path(file_path)
        if full_path.exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} - MISSING")
            all_exist = False
    
    return all_exist


def check_dependencies():
    """Check if required packages are installed"""
    print("\n✓ Checking dependencies...")
    
    required_packages = [
        "langchain",
        "langgraph",
        "langchain_core",
        "langchain_openai",
        "langchain_anthropic",
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("_", "-"))
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - NOT INSTALLED")
            missing_packages.append(package)
    
    if missing_packages:
        print("\n  To install missing packages, run:")
        print("  pip install -r requirements.txt")
        return False
    
    return True


def check_api_key():
    """Check if API key is configured"""
    print("\n✓ Checking API configuration...")
    
    # Check various API keys
    api_keys = {
        "ANTHROPIC_API_KEY": "Claude (Recommended)",
        "OPENAI_API_KEY": "OpenAI",
        "GOOGLE_API_KEY": "Google Gemini"
    }
    
    found_key = False
    for env_var, provider in api_keys.items():
        if os.getenv(env_var):
            print(f"  ✅ {env_var} found ({provider})")
            found_key = True
        else:
            print(f"  ⚠️  {env_var} not set ({provider})")
    
    if not found_key:
        print("\n  ❌ No API key found!")
        print("  You need to set one of the following environment variables:")
        print("    ANTHROPIC_API_KEY (recommended)")
        print("    OPENAI_API_KEY")
        print("    GOOGLE_API_KEY")
        print("\n  To set it, run:")
        print("    export ANTHROPIC_API_KEY='your-key-here'  # Linux/Mac")
        print("    set ANTHROPIC_API_KEY=your-key-here       # Windows")
        return False
    
    return True


def check_kb_valid():
    """Check if knowledge base is valid JSON"""
    print("\n✓ Checking knowledge base validity...")
    
    import json
    
    kb_path = "knowledge_base/autostream_kb.json"
    try:
        with open(kb_path, 'r') as f:
            kb = json.load(f)
        
        required_sections = ["company", "pricing", "features", "policies"]
        missing = [s for s in required_sections if s not in kb]
        
        if missing:
            print(f"  ❌ Missing sections: {', '.join(missing)}")
            return False
        else:
            print(f"  ✅ Knowledge base valid with {len(kb)} sections")
            return True
    
    except json.JSONDecodeError as e:
        print(f"  ❌ Invalid JSON: {str(e)}")
        return False
    except FileNotFoundError:
        print(f"  ❌ Knowledge base not found at {kb_path}")
        return False


def run_quick_test():
    """Try to initialize the agent"""
    print("\n✓ Running quick agent test...")
    
    try:
        from agent import AutoStreamAgent
        
        print("  Initializing agent...", end=" ")
        agent = AutoStreamAgent()
        print("✅")
        
        print("  Testing single turn...", end=" ")
        response = agent.chat("Hi!")
        if response and len(response) > 0:
            print("✅")
            return True
        else:
            print("❌ Empty response")
            return False
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def main():
    """Run all checks"""
    print("\n" + "="*60)
    print("  AutoStream Agent - Setup Verification")
    print("="*60 + "\n")
    
    checks = [
        ("Python Version", check_python_version),
        ("Project Structure", check_project_structure),
        ("Dependencies", check_dependencies),
        ("API Configuration", check_api_key),
        ("Knowledge Base", check_kb_valid),
        ("Quick Test", run_quick_test),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Error in {name}: {str(e)}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("  Verification Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nResult: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All checks passed! You're ready to use AutoStream Agent!")
        print("\nNext steps:")
        print("  1. Run: python demo/example_conversations.py")
        print("  2. Or run: python agent.py")
        print("  3. Or use: from agent import AutoStreamAgent")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
