# Nowacoski: AI Onboarding Service

Nowacoski is an AI-powered onboarding orchestrator that builds personalized, high-context onboarding plans for new hires. It pulls from scattered organizational knowledge sources — Google Drive, GitHub, Slack, and the web — and synthesizes them into actionable onboarding roadmaps.

Built for the **AWS GenAI Hackathon**.

## Demo

[YouTube video](https://youtu.be/9EpYrJlqZHw)

## Architecture Overview

Nowacoski uses a **Plan-Action-Reflect** agentic loop with pure worker agents and centralized caching:

```
User Query
    │
    ▼
┌─────────────────────────────────────────────┐
│             OrchestratorAgent               │
│                                             │
│  ┌─────────────────────────────────────┐    │
│  │  ConversationMemory                 │    │
│  │  (repo URLs, drive folders,         │    │
│  │   slack channels, turn summaries)   │    │
│  └─────────────────────────────────────┘    │
│                                             │
│  1. PLAN    ─ Decompose query + select tools│
│  2. ACTION  ─ Check cache → fetch if needed │
│  3. REFLECT ─ Evaluate completeness         │
│  4. FUSE    ─ Deduplicate + merge results   │
│  5. ANSWER  ─ Generate final response       │
│  6. RECORD  ─ Store sources for next turn   │
│                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Codebase │  │  Drive   │  │  Slack   │   │
│  │  Agent   │  │  Agent   │  │  Agent   │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│       │              │              │       │
│  Pure workers: fetch only, no cache logic   │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
         ┌─────────────────-┐
         │ MongoDB Atlas    │
         │ VectorDB         │
         │ (embeddings +    │
         │  query_embedding)│
         └─────────────────-┘
```

### Batch Onboarding (Temporal)

For onboarding multiple hires at once, Nowacoski uses **Temporal durable workflows**:

```
Streamlit UI (Batch Mode)
    │  upload CSV → start workflow → poll progress
    ▼
Temporal Server (PostgreSQL-backed)
    │  persists workflow state, handles retries
    ▼
BatchOnboardingWorkflow
    ├─ SingleHireOnboardingWorkflow (hire #1)
    │   └─ run_onboarding_activity → OrchestratorAgent → plan
    ├─ SingleHireOnboardingWorkflow (hire #2)
    │   └─ run_onboarding_activity → OrchestratorAgent → plan
    └─ ... (max 3 concurrent via worker config)
```

- **Fault tolerance** — Worker crashes mid-batch → restart → Temporal resumes from where it left off
- **Automatic retry** — Failed hires retry with exponential backoff (3 attempts, 30s → 60s)
- **Parallel processing** — Up to 3 hires processed concurrently per worker
- **Progress tracking** — Real-time progress bar + per-hire results in Streamlit UI

### Key Design Decisions

- **Pure worker agents** — Each agent (codebase, drive, slack, tavily) only fetches and analyzes. They never touch the cache. All cache-vs-fetch decisions are centralized in the orchestrator.
- **ConversationMemory** — Tracks GitHub repos, Drive folders, and Slack channels across turns so follow-up questions resolve to the correct source without re-providing URLs.
- **Dual embeddings** — Documents store both a full content embedding and a separate `query_embedding` (from the original short query) to avoid embedding dilution during cache lookups.
- **Centralized cache** — The orchestrator checks VectorDB before calling any agent. If cached content is sufficiently relevant (cosine similarity above threshold), the agent call is skipped entirely.

## Agents

| Agent | Source | Purpose |
|-------|--------|---------|
| **CodebaseAgent** | GitHub API | Analyze repo structure, technologies, dependencies, and generate setup instructions. Supports targeted file fetching for specific questions. |
| **DriveAgent** | Google Drive API | Fetch and analyze onboarding documents from shared Drive folders. |
| **SlackAgent** | Slack API | Fetch channel history and extract team norms, projects, and key decisions. |
| **TavilyAgent** | Tavily API (MCP) | Web search for current best practices, tools, and external knowledge. |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Amazon Bedrock (Claude 3.5 Sonnet) |
| Agent Framework | [Strands Agents SDK](https://github.com/strands-agents/sdk-python) |
| Vector Store | MongoDB Atlas + sentence-transformers (`all-MiniLM-L6-v2`) |
| Web Search | Tavily API via MCP |
| Frontend | Streamlit |
| Workflow Engine | Temporal (durable execution, retry, crash recovery) |
| Infrastructure | Docker Compose (5 services), Kubernetes (k8s manifests included) |

## Setup Instructions

### Prerequisites

- Python 3.13+
- AWS credentials with Bedrock access
- MongoDB Atlas cluster
- API keys for GitHub, Tavily, and optionally Slack and Google Drive

### 1. Clone and install

```bash
git clone https://github.com/RongMu00/Nowacoski-AI-onboarding-service.git
cd Nowacoski-AI-onboarding-service
pip install -e .
```

### 2. Configure environment

Create a `.env` file in the project root:

```env
# AWS Bedrock
BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_DEFAULT_REGION=us-west-2

# MongoDB Atlas
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/nowacoski

# APIs
GITHUB_TOKEN=ghp_...
TAVILY_KEY=tvly-...

# Optional
SLACK_BOT_TOKEN=xoxb-...
SERVICE_ACCOUNT_PATH=/path/to/service-account.json
```

### 3. Run

```bash
streamlit run src/enterprise_ai/ui.py
```

### 4. Run with Batch Mode (Temporal)

```bash
# Terminal 1: Start Temporal dev server
temporal server start-dev

# Terminal 2: Start the batch worker
python -m enterprise_ai.batch.worker

# Terminal 3: Start Streamlit
streamlit run src/enterprise_ai/ui.py
```

Or with Docker Compose (all-in-one):

```bash
docker compose -f docker/docker-compose.yml up --build
# Streamlit: http://localhost:8501
# Temporal UI: http://localhost:8080
```

### 5. (Optional) Google Drive OAuth

For user-level Drive access:

```bash
python src/enterprise_ai/agents/authorize_drive.py
```

## Project Structure

```
src/enterprise_ai/
├── ui.py                          # Streamlit frontend (Chat + Batch modes)
├── batch_ui.py                    # Batch mode UI (upload, progress, results)
├── agents/
│   ├── orchestrator_caching.py    # Plan-Action-Reflect orchestrator + ConversationMemory
│   ├── codebase_agent_caching.py  # GitHub repo analysis (pure worker)
│   ├── drive_agent_caching.py     # Google Drive analysis (pure worker)
│   ├── slack_agent_caching.py     # Slack channel analysis (pure worker)
│   ├── tavily_agent_caching.py    # Web search via Tavily MCP (pure worker)
│   └── tools_caching.py           # @tool wrappers + raw search helpers
├── batch/
│   ├── models.py                  # NewHire dataclass + CSV parser
│   ├── activities.py              # Temporal activity wrapping OrchestratorAgent
│   ├── workflows.py               # BatchOnboarding + SingleHireOnboarding workflows
│   └── worker.py                  # Temporal worker entry point
└── storage/
    └── vector_store.py            # MongoDB VectorDB with dual embeddings
```

## License

MIT
