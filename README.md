# Local-financial-dashboard
This is financial dashboard for Stocks analysis 
Step A: Prepare LM Studio
1. Open LM Studio.
2. Go to the Chat tab and load your Qwen3-VL-8B model.
3. Click the Developer/Server icon (usually looks like <-> on the left sidebar).
4. Click Start Server.
5. Ensure the server URL is http://localhost:1234

Step B: Install Python Dependencies
You need to remove groq and install openai (which works with LM Studio). Open your terminal/command prompt:
code
Bash >>
pip install streamlit yfinance pandas numpy openai

4. How to Run It
Make sure LM Studio server is running (Green "Start Server" button clicked).
Open your command prompt in the folder where you saved local_app.py.
Run the command:
code
Bash

streamlit run local_dashboard_app.py


Access outside home
Method 1: Download the Windows Binary (Recommended)
Step 1: Download cloudflared

Go to: https://github.com/cloudflare/cloudflared/releases/latest
Download cloudflared-windows-amd64.exe

Step 2: Rename and move it

Rename the file to cloudflared.exe
Move it to a folder like C:\git-repo\cloudflared\

C:\git-repo\cloudflared\cloudflared.exe tunnel --url http://localhost:8501


Test: LLY, MEDP, NU, NVDA, AMGN, MPWR, LRCX, ASML, FIX, SCCO, VRT, CLS, GOOGL, NXT, TSM, APH, INCY, HOOD, AVGO

Test  CNQ, SHOP, ENB, MFC, SLF, RY, CM, TD, BNS, TRI, PXT, TMQ

Test:  AAPL, MSFT, TSLA, NVDA, GOOGL, META, HOOD, PLTR, SNDK, MU

Test: 0001, 0005, 0272, 0330, 0939, 0941, 0981, 2388, 2628
