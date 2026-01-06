# 🎯 AI Lead Generation Agent

An intelligent lead generation tool that discovers and qualifies potential leads from Google search results using AI-powered analysis. Built with an optimized architecture for speed and efficiency.

![AI Lead Generation Agent](https://img.shields.io/badge/Status-Active-brightgreen) ![Python](https://img.shields.io/badge/Python-3.8+-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red)

## 🚀 What It Does

- **🔍 Smart Search**: Uses Serper.dev to search Google and extract relevant snippets
- **📋 Rule-Based Filtering**: Pre-filters results to reduce AI API calls by 60%
- **🎯 Batch AI Analysis**: Processes leads in batches for qualification and scoring
- **📊 Intelligent Ranking**: Scores leads based on relevance and intent signals
- **💾 CSV Export**: Exports qualified leads in CRM-ready format
- **⚡ Optimized Architecture**: 10x faster with fewer API calls

## 🏗️ Architecture Highlights

- **No Web Scraping**: Uses Google search snippets only (legal and fast)
- **Batch Processing**: Analyzes 3-5 leads per API call instead of 1
- **Model Fallback**: Automatically tries different Gemini models on quota limits
- **Smart Orchestration**: AI planner coordinates the entire workflow
- **Fallback Logic**: Graceful degradation when APIs are unavailable

## 🛠️ Tech Stack

- **🔍 Serper.dev**: Google search API (2,500 free searches/month)
- **🤖 Google Gemini**: AI analysis with automatic model fallback
- **📊 Streamlit**: Modern web UI
- **🐍 Python**: Core logic with type hints
- **📈 CSV Export**: CRM-ready lead data

## ⚡ Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/shrushtijadhavv/AI-Lead-Generation-Agent.git
cd AI-Lead-Generation-Agent
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Get API Keys

#### Google Gemini API Key
- Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
- Create a new API key
- Copy the key

#### Serper API Key
- Visit [Serper.dev](https://serper.dev)
- Sign up for a free account (2,500 searches/month)
- Get your API key

### 4. Configure Environment
Create a `.env` file in the project root (copy from `.env.example`):
```bash
cp .env.example .env
```

Then edit `.env` with your actual API keys:
```env
GOOGLE_API_KEY=your_google_gemini_api_key_here
SERPER_API_KEY=your_serper_api_key_here
```

### 5. Run the App
```bash
streamlit run app.py
```

## 🎯 How to Use

1. **Open the Streamlit app** in your browser
2. **Enter API keys** in the sidebar (or use .env file)
3. **Describe your ideal leads** in the text area
4. **Click "🚀 Generate Leads"**
5. **Download results** as CSV

### Example Lead Descriptions:
- "Find startup founders discussing customer acquisition challenges"
- "Locate marketing managers looking for growth strategies"
- "Discover CTOs interested in AI implementation"

## 📊 Workflow

1. **🧠 Orchestrator**: AI plans search strategy and qualification criteria
2. **🔍 Search**: Multi-query Google search via Serper
3. **📋 Filter**: Rule-based pre-filtering removes obvious spam
4. **🎯 Analyze**: Batch AI qualification and scoring
5. **📊 Process**: Deduplication, ranking, and enhancement
6. **💾 Export**: CSV download for CRM import

## ⚠️ Rate Limits & Quotas

### Google Gemini
- **Free Tier**: 1,500 requests/day, 15 RPM
- **Paid**: Higher limits available
- **Fallback**: App tries different models automatically

### Serper.dev
- **Free Tier**: 2,500 searches/month
- **Paid**: Higher limits available

## 🔧 Configuration

### Settings (via Streamlit sidebar):
- **Results per query**: 3-15 (default: 8)
- **API Keys**: Override environment variables

### Advanced Tuning:
- Modify `batch_size` in `batch_analyze_leads()` for different batch sizes
- Adjust spam/quality keywords in `rule_based_filter()`
- Customize qualification prompts in agent system messages

### Model Utility:
Run `python model.py` to list all available Google Gemini models and check API connectivity.

## 📁 Project Structure

```
AI-Lead-Generation-Agent/
├── app.py                 # Main Streamlit application
├── model.py              # Utility script to list available Gemini models
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── LICENSE              # MIT License
├── .env.example         # Environment variables template
├── .env                 # Environment variables (create from .env.example)
├── .gitignore          # Git ignore rules
└── __pycache__/        # Python cache (auto-generated)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 🙏 Acknowledgments

- **Serper.dev** for reliable Google search API
- **Google Gemini** for powerful AI analysis
- **Streamlit** for the amazing web app framework
- **Phi** for the agent orchestration framework

---

**⭐ If you find this project helpful, please give it a star!**

**Follow for more AI automation projects**
