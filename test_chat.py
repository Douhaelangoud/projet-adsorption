#!/usr/bin/env python
import requests
import json
import time
from requests.cookies import RequestsCookieJar

# Test Ollama directly first
print("=" * 60)
print("[TEST 1] Testing Ollama directly...")
print("=" * 60)
try:
    import ollama
    print("Testing Ollama chat...")
    response = ollama.chat(
        model='llama3',
        messages=[{'role': 'user', 'content': 'Say hi'}]
    )
    print(f"✅ Ollama Response: {response['message']['content'][:50]}")
except Exception as e:
    print(f"❌ Ollama Error: {e}")

# Test Flask chat API without auth (should fail with 401)
print("\n" + "=" * 60)
print("[TEST 2] Testing Chat API without authentication...")
print("=" * 60)

headers = {"Content-Type": "application/json"}
payload = {"message": "Hello, this is a test message"}

print(f"Sending POST to http://localhost:5000/chat-api (no auth)")
try:
    response = requests.post("http://localhost:5000/chat-api", json=payload, headers=headers, timeout=10)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    if response.status_code == 401:
        print("✅ Correctly requires authentication")
except Exception as e:
    print(f"❌ Request Error: {e}")

# Test Flask chat API with simulated session
print("\n" + "=" * 60)
print("[TEST 3] Testing Chat API with session cookie...")
print("=" * 60)

# First, get a session cookie by visiting the login page
session = requests.Session()
print("Getting session cookie from home page...")
try:
    home_response = session.get("http://localhost:5000/", timeout=10)
    print(f"Got session, cookies: {session.cookies}")
    
    # Now try to access chat-api with the session
    print("\nAttempting to call chat-api with session (should fail - not logged in)...")
    api_response = session.post("http://localhost:5000/chat-api", json=payload, headers=headers, timeout=10)
    print(f"Status Code: {api_response.status_code}")
    print(f"Response: {api_response.text}")
    
except Exception as e:
    print(f"❌ Request Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("[INFO] To test with a logged-in user, login via the web interface first")
print("[INFO] then the chat will work when accessed from that browser session")
print("=" * 60)

