import streamlit as st
import streamlit.components.v1 as components # Required for JS execution
import yfinance as yf
import pandas as pd
import numpy as np
from openai import OpenAI
import concurrent.futures
import json
import re
from datetime import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Stock Comparison Dashboard", layout="wide", page_icon="📊")

# --- LOCAL AI CLIENT ---
try:
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    client.models.list() 
    AI_CONNECTED = True
except:
    AI_CONNECTED = False

# --- HELPER FUNCTIONS ---

def fmt_num(val, is_pct=False, is_currency=False, decimals=2):
    if val is None or pd.isna(val) or val == "N/A": return "-"
    if is_pct: return f"{val * 100:.{decimals}f}%"
    if is_currency:
        if abs(val) > 1e12: return f"{val/1e12:.2f}T"
        if abs(val) > 1e9: return f"{val/1e9:.2f}B"
        if abs(val) > 1e6: return f"{val/1e6:.2f}M"
    return f"{val:.{decimals}f}"

def get_arrow(curr, prev):
    if pd.isna(curr) or pd.isna(prev) or prev == 0: return "-", "-"
    pct = ((curr - prev) / abs(prev))
    arrow = "▲" if pct > 0 else "▼"
    # Use simple class names for the iframe CSS
    color = "txt-green" if pct > 0 else "txt-red"
    return f"<span class='{color}'>{arrow} {fmt_num(curr, is_currency=True)}</span>", f"<span class='{color}'>{fmt_num(pct, is_pct=True)}</span>"

def format_surp(val):
    if val is None or pd.isna(val): return "-"
    arrow = "▲" if val > 0 else "▼"
    color = "txt-green" if val > 0 else "txt-red"
    return f"<span class='{color}'>{arrow} {val * 100:.2f}%</span>"

def extract_json(text):
    try:
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = re.sub(r'```json', '', text)
        text = re.sub(r'```', '', text)
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            json_str = text[start:end+1]
            json_str = re.sub(r',\s*}', '}', json_str)
            return json.loads(json_str)
        return None
    except:
        return None

# --- DATA FETCHING ---

def fetch_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="1y")
        
        price = hist['Close'].iloc[-1] if not hist.empty else 0
        eps = info.get('forwardEps', info.get('trailingEps'))
        pe = info.get('forwardPE', info.get('trailingPE'))
        if (not pe or pd.isna(pe)) and eps and price: pe = price / eps
            
        min_pe, max_pe = 0, 0
        if eps and not hist.empty and eps > 0:
            pe_series = hist['Close'] / eps
            min_pe, max_pe = pe_series.min(), pe_series.max()

        q_stmts = stock.quarterly_income_stmt
        
        try:
            earnings = stock.earnings_dates
            if earnings is not None and not earnings.empty:
                now = pd.Timestamp.now(tz=earnings.index.tz)
                past = earnings[earnings.index < now]
                latest_earn = past.iloc[0] if not past.empty else None
                earn_date = past.index[0].strftime('%Y-%m-%d') if not past.empty else "-"
            else:
                latest_earn, earn_date = None, "-"
        except:
            latest_earn, earn_date = None, "-"
        
        div_yield = info.get('dividendYield')
        if div_yield and div_yield > 0.15: 
            div_yield = div_yield / 100.0
        info['dividendYield'] = div_yield

        return {
            "ticker": ticker, "price": price, "currency": info.get('currency', 'USD'),
            "info": info, "history": hist, "pe": pe, "min_pe": min_pe, "max_pe": max_pe,
            "quarterly": q_stmts, "latest_earn": latest_earn, "earn_date": earn_date,
            "summary": info.get('longBusinessSummary', '')
        }
    except Exception as e:
        return None

def process_technicals(df):
    if df.empty or len(df) < 200: 
        return {"trend": "-", "rsi": 50, "vol_ratio": 0, "verdict": "-", "reason": "-"}
    
    close = df['Close']
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs)).iloc[-1]
    
    sma50 = close.rolling(50).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1]
    curr = close.iloc[-1]
    
    avg_vol = df['Volume'].rolling(20).mean().iloc[-1]
    vol_ratio = (df['Volume'].iloc[-1] / avg_vol) if avg_vol > 0 else 1.0
    
    trend = "Uptrend" if curr > sma200 and curr > sma50 else "Downtrend"
    
    verdict = "HOLD"
    reason = "Neutral"
    if trend == "Uptrend":
        if rsi < 40: verdict, reason = "BUY", "Dip"
        elif vol_ratio > 1.5: verdict, reason = "STRONG BUY", "Volume"
        else: verdict, reason = "BUY", "Trend"
    else:
        if rsi < 30: verdict, reason = "WATCH", "Oversold"
        else: verdict, reason = "AVOID", "Down"

    return {
        "trend": trend, "rsi": rsi, "vol_ratio": vol_ratio,
        "verdict": verdict, "reason": reason
    }

def get_ai_analysis(ticker, summary, pe_ctx, earn_ctx):
    if not AI_CONNECTED: return {}, "AI Offline"
    
    prompt = f"""
    Analyze {ticker}. Context: {summary[:800]}...
    Valuation: {pe_ctx}
    Earnings: {earn_ctx}
    
    Respond ONLY with valid JSON.
    Score each category from 0.0 to 4.0.
    
    Keys:
    {{
        "moat_score": 0.0, "moat_reason": "short string",
        "growth_score": 0.0, "growth_reason": "short string",
        "advantage_score": 0.0, "advantage_reason": "short string",
        "stability_score": 0.0, "stability_reason": "short string",
        "management_score": 0.0, "management_reason": "short string",
        "valuation_verdict": "Cheap/Fair/Exp",
        "earnings_summary": "Recent financial highlights. IF DATA MISSING: Provide a 1-sentence business summary. (Must not be empty)"
    }}
    """
    try:
        resp = client.chat.completions.create(
            model="local-model",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=2500 
        )
        raw_text = resp.choices[0].message.content
        data = extract_json(raw_text)
        return data, raw_text 
    except Exception as e:
        return None, str(e)

def process_ticker(ticker):
    d = fetch_stock_data(ticker)
    if not d: return None
    
    t = process_technicals(d['history'])
    
    pe = d['pe']
    min_pe, max_pe = d['min_pe'], d['max_pe']
    mult = 1.0
    val_ctx = "N/A"
    
    if pe and min_pe and max_pe and max_pe > min_pe:
        pos = (pe - min_pe) / (max_pe - min_pe)
        if pos < 0.25: mult, val_ctx = 5, "Undervalued"
        elif pos < 0.50: mult, val_ctx = 4, "Fair-Low"
        elif pos < 0.75: mult, val_ctx = 3, "Fair-High"
        elif pos < 1.0: mult, val_ctx = 2, "Overvalued"
        else: mult, val_ctx = 1, "Expensive"
        
    pe_str = f"{pe:.2f}" if pe else "N/A"
    ai, ai_raw = get_ai_analysis(ticker, d['summary'], pe_str, d['earn_date'])
    if not ai: ai = {}
    
    def get_s(k): 
        val = ai.get(k, 0)
        if val is None: return 0 
        if val > 40: return val / 25.0 
        if val > 4: return val / 2.5 
        return val 

    def get_r(k): return ai.get(k, "-")
    
    q_total = sum([get_s(k) for k in ["moat_score","growth_score","advantage_score","stability_score","management_score"]])
    final_score = q_total * mult
    
    q = d['quarterly']
    q_dat = {}
    cols = ['Rev', 'OpInc', 'NetInc', 'OpExp', 'EPS']
    fields = ['Total Revenue', 'Operating Income', 'Net Income', 'Operating Expense', 'Basic EPS']
    
    if q is not None and q.shape[1] >= 2:
        c, p = q.iloc[:, 0], q.iloc[:, 1]
        for k, f in zip(cols, fields):
            q_dat[k] = get_arrow(c.get(f), p.get(f))
        try:
            gmc = c.get('Gross Profit')/c.get('Total Revenue')
            gmp = p.get('Gross Profit')/p.get('Total Revenue')
            color = "txt-green" if gmc > gmp else "txt-red"
            arrow = "▲" if gmc > gmp else "▼"
            q_dat['GM'] = (f"{gmc*100:.1f}%", f"<span class='{color}'>{arrow} {(gmc-gmp)*100:.2f}bps</span>")
        except:
            q_dat['GM'] = ("-", "-")
    else:
        for k in cols + ['GM']: q_dat[k] = ("-", "-")

    le = d['latest_earn']
    est = le.get('EPS Estimate') if le is not None else None
    act = le.get('Reported EPS') if le is not None else None
    
    surp = None
    if est is not None and act is not None and est != 0:
        surp = (act - est) / abs(est)
    
    return {
        "T": ticker, "Score": final_score, "ValCtx": val_ctx,
        "Moat": (get_s("moat_score"), get_r("moat_reason")),
        "Grow": (get_s("growth_score"), get_r("growth_reason")),
        "Adv": (get_s("advantage_score"), get_r("advantage_reason")),
        "Stab": (get_s("stability_score"), get_r("stability_reason")),
        "Mgmt": (get_s("management_score"), get_r("management_reason")),
        "Tech": t,
        "Fin": d['info'],
        "Earn": {"Date": d['earn_date'], "Est": est, "Act": act, "Surp": surp},
        "Q": q_dat,
        "AISum": ai.get("earnings_summary", "-"),
        "DebugRaw": ai_raw
    }

# --- MAIN UI ---

with st.sidebar:
    st.header("Stock Analysis Dashboard")
    if AI_CONNECTED: st.success("🟢 AI Online")
    else: st.error("🔴 AI Offline")
    
    # --- MARKET SELECTION ---
    market = st.selectbox("Select Market", ["US", "Canada (TSX)", "HK (HKEX)"])
    
    # Dynamic Defaults based on Market
    if market == "US":
        def_tickers = "AAPL, MSFT, TSLA, NVDA"
    elif market == "Canada (TSX)":
        def_tickers = "TD, RY, SHOP, CSU"
    else:
        def_tickers = "0700, 9988, 1299, 0005"

    input_tickers = st.text_area("Tickers", def_tickers, height=100)
    st.caption("Note: For HK, you can type '700' or '0700'. For Canada, 'TD' or 'TD.TO'.")
    
    go = st.button("Analyze", type="primary")
    
    st.divider()
    st.markdown("### 🐞 AI Debugger")
    debug_exp = st.expander("View Raw AI Output", expanded=False)

if go:
    # --- TICKER PROCESSING LOGIC ---
    raw_list = [x.strip().upper() for x in input_tickers.split(',') if x.strip()]
    final_tickers = []
    
    for t in raw_list:
        if market == "Canada (TSX)":
            # If user didn't type .TO, add it
            if not t.endswith(".TO") and not t.endswith(".V"):
                t += ".TO"
        elif market == "HK (HKEX)":
            # Remove .HK if typed, to handle padding first
            clean_t = t.replace(".HK", "")
            # Check if it's digit-based
            if clean_t.isdigit():
                # Zero pad to 4 digits (e.g. 700 -> 0700)
                clean_t = clean_t.zfill(4)
            t = clean_t + ".HK"
        
        final_tickers.append(t)
    
    rows = []
    
    with st.spinner("🤖 AI Analyst is studying the market... Please wait."):
        bar = st.progress(0)
        status_text = st.empty()
        
        with concurrent.futures.ThreadPoolExecutor() as exe:
            fut = {exe.submit(process_ticker, t): t for t in final_tickers}
            for i, f in enumerate(concurrent.futures.as_completed(fut)):
                finished_ticker = fut[f]
                status_text.caption(f"✅ Analyzed {finished_ticker}...")
                try:
                    r = f.result()
                    if r: rows.append(r)
                except Exception as e:
                    st.error(f"Err: {e}")
                bar.progress((i+1)/len(final_tickers))
        
        bar.empty()
        status_text.empty()
    
    # --- RENDER TABLE (Iframe Height Doubled to 1300) ---
    
    table_html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {
                background-color: #0e1117;
                color: #fafafa;
                font-family: 'Segoe UI', sans-serif;
                margin: 0;
                padding: 10px;
            }
            .main-table {
                border-collapse: collapse;
                width: 100%;
                font-size: 13px;
            }
            
            /* HEADERS */
            th {
                position: sticky;
                z-index: 20;
                border: 1px solid #444;
                text-align: center;
                vertical-align: middle;
                cursor: pointer;
            }
            th:hover { background-color: #444; }
            
            /* Row 1 Headers */
            .h-row-1 {
                top: 0;
                height: 40px;
                font-size: 14px;
                font-weight: 800;
                text-transform: uppercase;
                color: #ffffff;
            }
            
            /* Row 2 Headers */
            .h-row-2 {
                top: 40px;
                z-index: 15;
                background-color: #262730;
                color: #eee;
                padding: 8px 4px;
                font-size: 12px;
            }

            /* Colors */
            .h-val { background-color: #1e3a8a; } 
            .h-tech { background-color: #064e3b; }
            .h-fin { background-color: #713f12; }
            .h-news { background-color: #4c1d95; }
            
            td {
                border: 1px solid #333;
                padding: 8px 5px;
                text-align: center;
                color: #ddd;
            }
            
            /* Sticky Ticker Column */
            .col-ticker {
                position: sticky;
                left: 0;
                background-color: #111;
                z-index: 30;
                border-right: 2px solid #555;
                min-width: 80px;
                font-weight: bold;
            }
            
            /* Utilities */
            .txt-green { color: #4ade80 !important; font-weight:bold; }
            .txt-red { color: #f87171 !important; font-weight:bold; }
            
            .grade-buy { background-color: rgba(74, 222, 128, 0.2); color: #4ade80; font-weight: bold; }
            .grade-hold { background-color: rgba(250, 204, 21, 0.2); color: #facc15; font-weight: bold; }
            .grade-sell { background-color: rgba(248, 113, 113, 0.2); color: #f87171; font-weight: bold; }
            
            .ai-tooltip { border-bottom: 1px dotted #888; cursor: help; }
            a { color: #60a5fa; text-decoration: none; font-size: 11px; }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
    <table id="stockTable" class="main-table">
        <thead>
            <tr>
                <th class="col-ticker h-row-1" rowspan="2" onclick="sortTable(0)">Ticker</th>
                <th colspan="7" class="h-val h-row-1">1. Value Analysis (AI Rated)</th>
                <th colspan="4" class="h-tech h-row-1">2. Technical Analysis</th>
                <th colspan="8" class="h-fin h-row-1">3. Financials</th>
                <th colspan="11" class="h-news h-row-1">4. News & Earnings</th>
            </tr>
            <tr class="h-row-2">
                <th onclick="sortTable(1)">Moat</th>
                <th onclick="sortTable(2)">Grow</th>
                <th onclick="sortTable(3)">Adv</th>
                <th onclick="sortTable(4)">Stab</th>
                <th onclick="sortTable(5)">Mgmt</th>
                <th onclick="sortTable(6)">Score</th>
                <th onclick="sortTable(7)">Ctx</th>
                <th onclick="sortTable(8)">Trend</th>
                <th onclick="sortTable(9)">RSI</th>
                <th onclick="sortTable(10)">Vol</th>
                <th onclick="sortTable(11)">Verdict</th>
                <th onclick="sortTable(12)">Mkt Cap</th>
                <th onclick="sortTable(13)">P/E</th>
                <th onclick="sortTable(14)">Fwd P/E</th>
                <th onclick="sortTable(15)">Beta</th>
                <th onclick="sortTable(16)">PM</th>
                <th onclick="sortTable(17)">GM</th>
                <th onclick="sortTable(18)">ROE</th>
                <th onclick="sortTable(19)">Div</th>
                <th onclick="sortTable(20)">Date</th>
                <th onclick="sortTable(21)">Est</th>
                <th onclick="sortTable(22)">Act</th>
                <th onclick="sortTable(23)">Surp</th>
                <th onclick="sortTable(24)">Rev Q/Q</th>
                <th onclick="sortTable(25)">OpInc Q/Q</th>
                <th onclick="sortTable(26)">NetInc Q/Q</th>
                <th onclick="sortTable(27)">OpExp Q/Q</th>
                <th onclick="sortTable(28)">EPS Q/Q</th>
                <th onclick="sortTable(29)">GM Q/Q</th>
                <th onclick="sortTable(30)">AI Sum</th>
            </tr>
        </thead>
        <tbody>
    """
    
    # Generate Rows
    for r in rows:
        s = r['Score']
        cls = "grade-buy" if s >= 75 else "grade-hold" if s >= 45 else "grade-sell"
        score_txt = f"{s:.0f}" if s > 0 else "-"
        
        def tip(x): return f"<span class='ai-tooltip' title='{x[1]}'>{x[0]:.1f}</span>"
        
        t = r['Tech']
        rsi_c = "" 
        
        f = r['Fin']
        e = r['Earn']
        q = r['Q']

        # Fwd P/E Green Logic
        pe_v = f.get('trailingPE')
        fwd_v = f.get('forwardPE')
        fwd_cls = ""
        if (pe_v is not None and fwd_v is not None and 
            not pd.isna(pe_v) and not pd.isna(fwd_v) and 
            fwd_v > 0 and fwd_v < pe_v):
            fwd_cls = "txt-green"

        # PM Green Logic
        pm_v = f.get('profitMargins')
        pm_cls = ""
        if pm_v is not None and not pd.isna(pm_v) and pm_v > 0.20:
            pm_cls = "txt-green"

        row = "<tr>"
        row += f"<td class='col-ticker'><b>{r['T']}</b><br><a href='https://finance.yahoo.com/quote/{r['T']}' target='_blank'>Chart</a></td>"
        row += f"<td>{tip(r['Moat'])}</td><td>{tip(r['Grow'])}</td><td>{tip(r['Adv'])}</td><td>{tip(r['Stab'])}</td><td>{tip(r['Mgmt'])}</td>"
        row += f"<td class='{cls}'>{score_txt}</td><td style='font-size:11px'>{r['ValCtx']}</td>"
        row += f"<td>{t['trend']}</td><td>{t['rsi']:.1f}</td><td>{t['vol_ratio']:.2f}</td><td title='{t['reason']}' style='cursor:help'>{t['verdict']}</td>"
        row += f"<td>{fmt_num(f.get('marketCap'), is_currency=True)}</td><td>{fmt_num(f.get('trailingPE'))}</td><td class='{fwd_cls}'>{fmt_num(f.get('forwardPE'))}</td><td>{fmt_num(f.get('beta'))}</td>"
        row += f"<td class='{pm_cls}'>{fmt_num(f.get('profitMargins'), True)}</td><td>{fmt_num(f.get('grossMargins'), True)}</td><td>{fmt_num(f.get('returnOnEquity'), True)}</td><td>{fmt_num(f.get('dividendYield'), True)}</td>"
        row += f"<td>{e['Date']}</td><td>{fmt_num(e['Est'])}</td><td>{fmt_num(e['Act'])}</td><td>{format_surp(e['Surp'])}</td>"
        row += f"<td>{q['Rev'][0]}<br>{q['Rev'][1]}</td><td>{q['OpInc'][0]}<br>{q['OpInc'][1]}</td><td>{q['NetInc'][0]}<br>{q['NetInc'][1]}</td>"
        row += f"<td>{q['OpExp'][0]}<br>{q['OpExp'][1]}</td><td>{q['EPS'][0]}<br>{q['EPS'][1]}</td><td>{q['GM'][0]}<br>{q['GM'][1]}</td>"
        row += f"<td style='font-size:10px; text-align:left; min-width:150px'>{r['AISum']}</td></tr>"
        table_html_content += row

    # Close Table and Add Script
    table_html_content += """
        </tbody>
    </table>
    <script>
    function sortTable(n) {
      var table, rows, switching, i, x, y, shouldSwitch, dir, switchcount = 0;
      table = document.getElementById("stockTable");
      switching = true;
      dir = "asc"; 
      while (switching) {
        switching = false;
        rows = table.rows;
        // Headers are row 0 and 1. Data starts at row 2.
        for (i = 2; i < (rows.length - 1); i++) {
          shouldSwitch = false;
          x = rows[i].getElementsByTagName("TD")[n];
          y = rows[i + 1].getElementsByTagName("TD")[n];
          
          function getVal(cell) {
             var s = cell.innerText.trim().toUpperCase();
             if (s === "-" || s === "N/A") return -999999999;
             var mult = 1;
             if (s.endsWith('T')) mult = 1e12;
             else if (s.endsWith('B')) mult = 1e9;
             else if (s.endsWith('M')) mult = 1e6;
             else if (s.endsWith('K')) mult = 1e3;
             else if (s.endsWith('%')) mult = 0.01;
             var clean = s.replace(/[^0-9.-]/g, '');
             var num = parseFloat(clean);
             if (isNaN(num)) return s;
             return num * mult;
          }
          
          var xVal = getVal(x);
          var yVal = getVal(y);
          
          if (typeof xVal === 'string' && typeof yVal === 'string') {
              if (dir == "asc") {
                if (xVal.toLowerCase() > yVal.toLowerCase()) { shouldSwitch = true; break; }
              } else {
                if (xVal.toLowerCase() < yVal.toLowerCase()) { shouldSwitch = true; break; }
              }
          } else {
              if (dir == "asc") {
                if (xVal > yVal) { shouldSwitch = true; break; }
              } else {
                if (xVal < yVal) { shouldSwitch = true; break; }
              }
          }
        }
        if (shouldSwitch) {
          rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
          switching = true;
          switchcount ++; 
        } else {
          if (switchcount == 0 && dir == "asc") {
            dir = "desc";
            switching = true;
          }
        }
      }
    }
    </script>
    </body>
    </html>
    """
    
    # Debug
    if r['DebugRaw'] != "AI Offline":
         debug_exp.text_area(f"Raw Output: {r['T']}", r['DebugRaw'], height=100)

    # Render Iframe (Height 1300px, Scrolling enabled)
    components.html(table_html_content, height=1300, scrolling=True)