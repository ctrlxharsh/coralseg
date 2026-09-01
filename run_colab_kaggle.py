"""
Helper script to run the Streamlit CoralSCOP dashboard on Google Colab or Kaggle.
Exposes the local Streamlit port via localtunnel or pyngrok.
"""

import subprocess
import time
import os
import sys

def main():
    print("=" * 60)
    print("🪸 CoralSCOP Streamlit Cloud / Colab / Kaggle Runner")
    print("=" * 60)

    # 1. Start Streamlit in background
    print("\n[1/2] Starting Streamlit server...")
    streamlit_cmd = [
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port", "8501",
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false"
    ]
    st_proc = subprocess.Popen(streamlit_cmd)
    time.sleep(3)

    # 2. Start localtunnel or print public access info
    print("\n[2/2] Exposing public URL via localtunnel (npx)...")
    try:
        # Get public IP for localtunnel password
        import urllib.request
        ipv4 = urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode('utf8').strip()
        print(f"\n🔑 Localtunnel Password (your public IP): {ipv4}")
        print("Click the link below and paste this IP if prompted:\n")
    except Exception:
        pass

    lt_proc = subprocess.Popen(["npx", "localtunnel", "--port", "8501"])
    
    try:
        lt_proc.wait()
    except KeyboardInterrupt:
        print("\nStopping servers...")
        st_proc.terminate()
        lt_proc.terminate()

if __name__ == "__main__":
    main()
