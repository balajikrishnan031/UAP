"""
UAP 4.0 Unified Launcher & Production CLI.
Provides a single, intuitive entrypoint for users, researchers, and enterprises to run:
1. Interactive Web Application (Streamlit UI)
2. Production REST API Server (FastAPI + Swagger Docs)
3. Universal Benchmark Suite (12 Domains vs 6 Baselines)
4. Infinite-Scale Streaming Convergence Demo (5,000,000 samples in O(d) RAM)
5. Automated Test Suite (Pytest)
"""

import sys
import subprocess
from pathlib import Path


def print_banner():
    print("""
    ====================================================================
      🚀 UAP 4.0: RELIABLE ADAPTIVE PREDICTION PLATFORM
      Goal-Aware | Conformal Safe | Causal SCM | O(d) Streaming
    ====================================================================
    [1] 🌐 Launch Interactive Web Application (Streamlit)
    [2] ⚡ Launch Production REST API Server (FastAPI on Port 8000)
    [3] 📊 Run Master 12-Domain Universal Benchmark
    [4] 🌊 Run Infinite Streaming Engine Demo (5M Samples, O(d) RAM)
    [5] 🧪 Run Complete Test Suite (16 Automated Unit Tests)
    [6] ❌ Exit
    ====================================================================
    """)


def main():
    while True:
        print_banner()
        choice = input("Enter your choice (1-6): ").strip()

        if choice == "1":
            print("\n[UAP 4.0] Launching Streamlit Web App on http://localhost:8501 ...")
            subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
        elif choice == "2":
            print("\n[UAP 4.0] Launching FastAPI REST API on http://127.0.0.1:8000/docs ...")
            subprocess.run([sys.executable, "-m", "uvicorn", "api:app", "--reload", "--port", "8000"])
        elif choice == "3":
            print("\n[UAP 4.0] Executing Universal Benchmark across 12 Real-World Domains...")
            subprocess.run([sys.executable, "uap_universal_benchmark.py"])
        elif choice == "4":
            print("\n[UAP 4.0] Executing 5,000,000 Sample Asymptotic Streaming Engine...")
            subprocess.run([sys.executable, "train_infinite_stream.py"])
        elif choice == "5":
            print("\n[UAP 4.0] Executing Pytest Suite across all modules...")
            subprocess.run([sys.executable, "-m", "pytest", "-v"])
        elif choice == "6":
            print("\n[UAP 4.0] Exiting. Thank you!")
            break
        else:
            print("\n⚠️ Invalid selection. Please enter 1, 2, 3, 4, 5, or 6.")


if __name__ == "__main__":
    main()
