"""
Helper script to run the Streamlit CoralSCOP dashboard on Google Colab or Kaggle.
Exposes the local Streamlit port via localtunnel.
"""

import subprocess
import time
import os
import sys
import urllib.request

def main():
    print("=" * 60, flush=True)
    print("🪸 CoralSCOP Streamlit Cloud / Colab / Kaggle Runner", flush=True)
    print("=" * 60, flush=True)

    # 1. Start Streamlit in background
    print("\n[1/2] Starting Streamlit server...", flush=True)
    streamlit_cmd = [
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port", "8501",
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false"
    ]
    st_proc = subprocess.Popen(streamlit_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)

    if st_proc.poll() is not None:
        print("❌ Error: Streamlit server failed to start.", flush=True)
        return

    # 2. Start localtunnel or print public access info
    print("\n[2/2] Exposing public URL via localtunnel...", flush=True)
    try:
        ipv4 = urllib.request.urlopen('https://ipv4.icanhazip.com', timeout=5).read().decode('utf8').strip()
        print(f"🔑 Localtunnel Password (your public IP): {ipv4}", flush=True)
        print("Click the URL below and paste this password/IP if prompted:\n", flush=True)
    except Exception:
        pass

    # Use 'npx -y' to prevent interactive prompt hangs
    lt_proc = subprocess.Popen(
        ["npx", "-y", "localtunnel", "--port", "8501"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    try:
        for line in iter(lt_proc.stdout.readline, ''):
            if line:
                print(line.strip(), flush=True)
        lt_proc.wait()
    except KeyboardInterrupt:
        print("\nStopping servers...", flush=True)
    finally:
        st_proc.terminate()
        try:
            lt_proc.terminate()
        except Exception:
            pass

if __name__ == "__main__":
    main()
