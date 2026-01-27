#!/usr/bin/env python3
"""Test Gemini API to list available models"""
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    print("❌ No API key found in .env file")
    exit(1)

print(f"✅ API Key found: {GEMINI_API_KEY[:20]}...")

try:
    genai.configure(api_key=GEMINI_API_KEY)
    print("\n✅ Gemini configured successfully\n")
    
    print("📋 Available models for generateContent:")
    print("=" * 60)
    
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            print(f"  ✓ {model.name}")
    
    print("\n" + "=" * 60)
    print("\n🧪 Testing model...")
    
    # Try the most common model
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content("Say hello in Hindi")
    print(f"✅ Model works! Response: {response.text}")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
