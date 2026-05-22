# 🤖 Smart Dentsu Buddy - Agentic RAG Chatbot

A production-ready **Agentic Retrieval-Augmented Generation (RAG)** system for marketing and advertising research at Dentsu. Powered by **LangGraph** and **Azure OpenAI**, this chatbot automatically retrieves from multiple knowledge sources, validates relevance, and self-corrects when needed.

---

## 🌟 Features

### Core Capabilities
- ✅ **Agentic Reasoning** - Autonomous agent decides which knowledge source to query
- ✅ **Multi-Source RAG** - Campaign database (JSON), research PDFs, marketing law articles, live web search
- ✅ **Marketing Guardrails** - Automatically rejects off-topic queries before processing
- ✅ **Self-Correcting Agent** - Rewrites queries and retries if retrieval fails
- ✅ **Relevance Grading** - Validates retrieved documents before generating answers
- ✅ **Chat History** - Persistent conversation tracking with auto-save to JSON
- ✅ **File Uploads** - Support for PDF, images (PNG/JPG), and Word documents (DOC/DOCX)
- ✅ **No .env File Required** - All credentials configured via Streamlit UI

### Knowledge Sources

| Source | Type | Tool Name | Use Case |
|--------|------|-----------|----------|
| **Campaign Database** | JSON | `search_campaigns_db` | Campaign search, channel recommendations, performance benchmarks |
| **Research PDFs** | 2 PDFs | `search_marketing_research` | Content effectiveness, digital transformation insights |
| **Marketing Law Articles** | Web URLs | `search_marketing_law_articles` | Legal compliance, advertising rules, consumer protection |
| **Live Web Search** | Tavily API | `search_web` | Current trends, platform updates, breaking news |

---

## 📦 Installation

### Prerequisites
- Python 3.11+
- Azure OpenAI account (with API key and endpoint)
- Tavily API key (optional, for live web search)
- Campaign database file (`campaigns_db.json`)
- Research PDFs: `Content_Effects_Advertising_Marketing.pdf`, `Digital_Transformation_in_Marketing.pdf`

### Step 1: Clone or Create Project Directory

```bash
mkdir Smart_Dentsu_Buddy
cd Smart_Dentsu_Buddy
```

### Step 2: Create Virtual Environment

```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Prepare Knowledge Base Files

Place these files in the project root directory:
- `campaigns_db.json` - Campaign profiles (structured data)
- `Content_Effects_Advertising_Marketing.pdf` - Research paper 1
- `Digital_Transformation_in_Marketing.pdf` - Research paper 2

---

## 🚀 Quick Start

### Run the Streamlit App

```bash
streamlit run main.py
```

The app will open at `http://localhost:8501`

### Configuration (First Time)

1. **Sidebar → API Keys Tab**
   - Enter Azure OpenAI endpoint (e.g., `https://resource.openai.azure.com/`)
   - Enter Azure OpenAI API key
   - Enter Chat model name (e.g., `gpt-4o`)
   - Enter Embedding model name (e.g., `text-embedding-ada-002`)
   - Enter API version (e.g., `2024-02-15-preview`)
   - (Optional) Enter Tavily API key for live web search

2. **Click "Initialize Agent"**
   - Loads all vector stores
   - Sets up LLM and embeddings
   - Builds the LangGraph workflow

3. **Start Chatting**
   - Ask marketing/advertising questions
   - Guardrail automatically validates scope
   - Agent routes to appropriate knowledge source

---

## 💬 Example Queries

### Campaign Database Queries
```
"Find me a successful digital campaign for CPG brands"
"What channels were used in the Toyota electric vehicle campaign?"
"Show me campaigns with ROAS above 3x"
```

### Research & Insights Queries
```
"What are the key findings on content effectiveness in advertising?"
"How does digital transformation impact marketing ROI?"
"Explain the relationship between audience engagement and campaign performance"
```

### Legal & Compliance Queries
```
"What are marketing law basics for in-house counsel?"
"Explain truthful and non-misleading claims in advertising"
"What are consumer protection considerations for marketing in India?"
```

### Current Events Queries
```
"What are the latest Google Ads platform updates in 2025?"
"What new features did Meta introduce for advertisers recently?"
```

### Direct Knowledge Queries
```
"What does CPM stand for in digital advertising?"
"Explain programmatic advertising"
"What is the difference between brand and performance marketing?"
```

---

## 🏗️ Architecture

### LangGraph Workflow

```
                    AGENTIC RAG FLOW

User Question
    ↓
[GUARDRAIL] ← Validates marketing relevance
    ↓
[AGENT] ← Decides which tool to use
    ↓
    ├─→ search_campaigns_db → [RETRIEVE]
    ├─→ search_marketing_research → [RETRIEVE]
    ├─→ search_marketing_law_articles → [RETRIEVE]
    ├─→ search_web → [RETRIEVE]
    └─→ (Direct Answer) → END
    ↓
[GRADE_DOCUMENTS] ← Checks relevance
    ├─→ YES → [GENERATE] → END
    └─→ NO → [REWRITE] → [AGENT] (LOOP)
```

### Components

| Component | Role |
|-----------|------|
| **Guardrail** | Pre-processes query; rejects non-marketing topics |
| **Agent Node** | LLM with tool bindings; decides action |
| **Retrieval Node** | Executes tool calls; fetches documents |
| **Grading Node** | Evaluates document relevance |
| **Generation Node** | Writes final answer with disclaimers |
| **Rewrite Node** | Improves failed queries; triggers retry loop |

### Knowledge Base Architecture

| Store | Type | Retriever | Persistence |
|-------|------|-----------|-------------|
| Campaign DB | Vector | ChromaDB | `campaigns_db_vectorstore/` |
| Research PDFs | Vector | ChromaDB | `marketing_pdf_db/` |
| Law Articles | Vector | ChromaDB | `marketing_law_articles_db/` |
| Uploaded Docs | Vector | ChromaDB | `uploaded_docs_vectorstore/` |

---

## 📁 Project Structure

```
Smart_Dentsu_Buddy/
├── main.py                                    # Streamlit app (main entry point)
├── requirements.txt                           # Python dependencies
├── README.md                                  # This file
│
├── campaigns_db.json                          # Campaign profiles (structured data)
├── Content_Effects_Advertising_Marketing.pdf  # Research paper 1
├── Digital_Transformation_in_Marketing.pdf    # Research paper 2
│
├── campaigns_db_vectorstore/                  # ChromaDB: Campaign embeddings
├── marketing_pdf_db/                          # ChromaDB: Research PDF embeddings
├── marketing_law_articles_db/                 # ChromaDB: Law article embeddings
├── uploaded_docs_vectorstore/                 # ChromaDB: User uploads
│
├── chat_history.json                          # Auto-saved conversation history
└── venv/                                      # Python virtual environment
```

---

## ⚙️ Configuration

### Environment Variables (Via Streamlit UI)

All credentials are entered through the Streamlit sidebar:

```
AZURE_OPENAI_ENDPOINT       = User Input → "https://resource.openai.azure.com/"
AZURE_OPENAI_API_KEY        = User Input → "sk-xxxxx"
CHAT_MODEL_NAME             = User Input → "gpt-4o"
EMBEDDING_MODEL_NAME        = User Input → "text-embedding-ada-002"
API_VERSION                 = User Input → "2024-02-15-preview"
TAVILY_API_KEY              = User Input (Optional) → "tvly-xxxxx"
```

**No .env file is created or required.** All secrets stay in Streamlit's session state.

---

## 📊 Chat History

### Auto-Save Behavior
- Every conversation message is automatically saved to `chat_history.json`
- Format includes: role, content, timestamp, conversation ID

### Manual Save
- Click **"Save History"** button in sidebar to explicitly save
- Useful for backup or sharing

### New Conversation
- Click **"New Conversation"** button to clear chat history
- Auto-generates new conversation ID

### Chat History Format

```json
{
  "conversation_id": "20250522_145530",
  "timestamp": "2025-05-22T14:55:30.123456",
  "messages": [
    {
      "role": "user",
      "content": "Find me a successful digital campaign",
      "timestamp": "2025-05-22T14:55:31"
    },
    {
      "role": "assistant",
      "content": "Based on the campaign database...",
      "timestamp": "2025-05-22T14:55:40"
    }
  ]
}
```

---

## 📤 File Uploads

### Supported Formats
- **PDF** (`.pdf`) - Auto-chunked and embedded
- **Images** (`.png`, `.jpg`, `.jpeg`) - Stored with metadata
- **Word** (`.doc`, `.docx`) - Text extracted and embedded

### Upload Workflow
1. Click **"Upload Documents"** in sidebar
2. Select one or more files
3. Click **"Process Uploads"** button
4. Wait for progress bar to complete
5. Files are embedded into `uploaded_docs_vectorstore/`
6. Agent can now query uploaded content

### Limitations
- Max file size: Depends on Streamlit config (default ~200MB)
- PDFs are chunked at 600 characters with 100-character overlap
- Images are stored as metadata (not OCR'd by default)
- Word docs are plain text extracted

---

## 🔍 Search Quality Tips

### For Best Results

1. **Be Specific**
   - ❌ Bad: "Tell me about campaigns"
   - ✅ Good: "Find successful digital campaigns for CPG brands with high ROI"

2. **Include Industry Context**
   - ❌ Bad: "What channels work best?"
   - ✅ Good: "What channels performed best for automotive brands in the Drive Electric campaign?"

3. **Reference Research Papers**
   - ❌ Bad: "How does content work?"
   - ✅ Good: "What does the research say about content effectiveness in video advertising?"

4. **Ask for Specific Metrics**
   - ❌ Bad: "How did the campaign perform?"
   - ✅ Good: "What was the ROAS and brand lift for the Coca-Cola Summer Refresh campaign?"

---

## ⚠️ Troubleshooting

### Issue: "Please initialize the agent first"
**Solution:** Enter API credentials in sidebar and click "Initialize Agent"

### Issue: "No results found" or "Documents not relevant"
**Solution:** The agent will automatically rewrite your query and retry. Wait for the retry loop.

### Issue: "OpenAI API Error: Invalid API key"
**Solution:** Check that your Azure OpenAI API key is correct and has access to the specified model.

### Issue: Vector store not loading
**Solution:** Ensure you're in the correct directory with `campaigns_db.json` and PDF files present.

### Issue: "Tavily API not configured"
**Solution:** Live web search is optional. Skip if you only need internal knowledge bases.

### Issue: File upload fails
**Solution:** Ensure file is < 200MB and format is PDF, PNG, JPG, DOC, or DOCX.

### Issue: Chat history not saving
**Solution:** Ensure write permissions in project directory. Check `chat_history.json` for errors.

---

## 🎯 Performance Notes

### Response Time
- **Local docs (campaigns/PDFs)**: 5-15 seconds (depends on retrieval + LLM)
- **Web search**: 10-20 seconds (includes API latency)
- **Direct answers**: 3-8 seconds (LLM only)

### Optimization Tips
1. **Pin vector stores to memory** for faster retrieval (if enough RAM)
2. **Batch queries** to reduce API calls
3. **Cache embeddings** at the Chroma level (automatic)
4. **Use similarity score threshold** to filter poor matches (default: 0.5)

### Token Usage
- Campaign queries: ~500-1500 tokens
- Research queries: ~1000-2500 tokens
- Web search: ~2000-3000 tokens

---

## 🔐 Security & Compliance

### Best Practices
- ✅ API keys entered via Streamlit (not hardcoded)
- ✅ Session state is ephemeral (clears on app restart)
- ✅ Chat history saved locally (not in cloud by default)
- ✅ No sensitive data in vector stores
- ✅ GDPR-compliant document handling

### For Production Deployment
- Use **Streamlit Cloud** or **AWS/Azure** with secure secret management
- Enable **authentication** (Streamlit Cloud native)
- Add **rate limiting** to prevent API abuse
- Set up **audit logging** for compliance
- Use **TLS encryption** for all API calls

---

## 📈 Roadmap

### Planned Enhancements
- [ ] Multi-user authentication
- [ ] Database integration for persistent chat history
- [ ] Real-time streaming responses
- [ ] Advanced analytics dashboard
- [ ] Custom knowledge base management UI
- [ ] Export conversations to PDF/Word
- [ ] Multi-language support
- [ ] Voice input/output
- [ ] Integration with Dentsu's internal systems

---

## 📞 Support & Contributing

### Issues & Feedback
Report issues or suggest improvements:
- GitHub Issues: (Create a repo to use this)
- Email: (Add support email)

### Contributing
Pull requests welcome! Please ensure:
- ✅ Code follows PEP 8
- ✅ All dependencies are in `requirements.txt`
- ✅ README is updated with changes
- ✅ Test with Python 3.11+

---

## 📜 License

Proprietary - Dentsu Group

---

## 🙏 Acknowledgments

Built with:
- [Streamlit](https://streamlit.io) - Web app framework
- [LangChain](https://langchain.com) - LLM orchestration
- [LangGraph](https://github.com/langchain-ai/langgraph) - Agentic workflows
- [ChromaDB](https://www.trychroma.com) - Vector database
- [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service/) - LLM backbone
- [Tavily](https://tavily.com) - Web search API

---

**Last Updated:** May 22, 2025  
**Python Version:** 3.11+  
**Status:** Production Ready ✅