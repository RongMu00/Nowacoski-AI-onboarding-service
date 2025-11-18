#!/usr/bin/env python3
"""
MongoDB Atlas SSL/TLS Troubleshooting Script
Diagnoses and fixes SSL connection issues
"""

import os
import ssl
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

print("╔" + "="*68 + "╗")
print("║  MongoDB Atlas SSL/TLS Diagnostic & Fix Tool                     ║")
print("╚" + "="*68 + "╝")
print()

load_dotenv()

# 1. Check Python SSL capabilities
print("1️⃣  Checking Python SSL Setup...")
print("-" * 70)

try:
    import certifi
    cert_path = certifi.where()
    print(f"✅ certifi installed")
    print(f"   Certificate path: {cert_path}")
    print()
except ImportError:
    print("❌ certifi NOT installed")
    print("   Run: pip install certifi")
    print()

# 2. Check if macOS needs certificate installation
print("2️⃣  Checking macOS Certificate Installation...")
print("-" * 70)

if sys.platform == "darwin":  # macOS
    print("🍎 Detected macOS")

    # Find Python installation
    python_path = subprocess.check_output(["which", "python3"]).decode().strip()
    print(f"   Python: {python_path}")

    # Find corresponding certificate installer
    # Python from Homebrew: /opt/homebrew/bin/python3.x
    # Python from official installer: /usr/local/bin/python3.x or /Applications/Python x.x/

    if "homebrew" in python_path or "opt/homebrew" in python_path:
        print("   Source: Homebrew")
        cert_installer = python_path.replace("python3", "").replace("/bin/", "/Cellar/")
        print(f"   Run: {python_path} -m certifi")
    elif "Applications/Python" in python_path:
        print("   Source: Official Python Installer")
        base = "/Applications/Python 3.12"  # Adjust version as needed
        print(f"   Run: {base}/Install\\ Certificates.command")

    print()
    print("   To fix SSL on macOS:")
    print(f"     1. Run: {python_path} -m pip install --upgrade certifi")
    print(f"     2. Run: {python_path} -m certifi")
    print()
else:
    print(f"✅ Not macOS (running {sys.platform})")
    print()

# 3. Check MongoDB connection
print("3️⃣  Testing MongoDB Connection...")
print("-" * 70)

mongo_uri = os.getenv("MONGODB_URI")

if not mongo_uri:
    print("❌ MONGODB_URI not found in .env")
    sys.exit(1)

# Mask the password for display
display_uri = mongo_uri[:50] + "..." if len(mongo_uri) > 50 else mongo_uri
print(f"Connection string: {display_uri}")
print()

try:
    from pymongo import MongoClient

    # Try default SSL
    print("   Attempting: Default SSL settings...")
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10000)
    client.admin.command('ping')
    print("   ✅ Connected with default SSL!")
    print()

except Exception as e:
    error_str = str(e)
    print(f"   ❌ Failed: {error_str[:80]}...")
    print()

    # Check error type and suggest fix
    if "SSL" in error_str or "tlsv1" in error_str or "certificate" in error_str:
        print("   💡 This is an SSL/TLS certificate issue")
        print()
        print("   FIXES TO TRY (in order):")
        print("   1. Install Python certificates:")
        print(f"      {python_path} -m pip install --upgrade certifi")
        print()
        print("   2. If that doesn't work, add to .env temporarily:")
        print("      MONGODB_URI=mongodb+srv://...?tlsInsecure=true")
        print()
        print("   3. Or update connection code with tlsInsecure=True parameter")
        print()
    elif "Connection refused" in error_str or "Timeout" in error_str:
        print("   💡 This is a network/connectivity issue")
        print()
        print("   FIXES TO TRY:")
        print("   1. Check your internet connection")
        print("   2. Verify MongoDB Atlas is running")
        print("   3. Check if IP is whitelisted (Atlas > Network Access)")
        print("   4. Try increasing timeout: ?serverSelectionTimeoutMS=30000")
        print()
    elif "Authentication failed" in error_str:
        print("   💡 This is an authentication issue")
        print()
        print("   FIXES TO TRY:")
        print("   1. Verify username and password")
        print("   2. Check if special characters are URL-encoded")
        print("   3. Verify user exists in MongoDB Atlas")
        print()

# 4. Recommended fix
print("4️⃣  Recommended Fix Steps:")
print("-" * 70)

print("""
For macOS (most likely your issue):
  1. Update certifi:
     pip install --upgrade certifi

  2. Reinstall Python certificates:
     python3 -m certifi

  3. If still fails, add to .env:
     MONGODB_URI=mongodb+srv://...?tlsInsecure=true

For Linux/Windows:
  1. pip install --upgrade certifi

  2. If that fails, add to .env:
     MONGODB_URI=mongodb+srv://...?tlsInsecure=true
""")

print()
print("5️⃣  After Fixing:")
print("-" * 70)
print("   1. Update .env if needed")
print("   2. Restart Streamlit: uv run streamlit run ui.py")
print("   3. You should see: Connected to MongoDB Atlas!")
print()
