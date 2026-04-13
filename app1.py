from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, TypedDict

from langgraph.graph import StateGraph, END

import streamlit as st
import requests
# Load local .env explicitly from the project directory (next to this file)
# This avoids Streamlit cwd surprises.
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).resolve().with_name(".env")
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
    else:
        # Fallback: try current working directory
        load_dotenv(override=True)
except Exception:
    pass

# If OPENAI_API_KEY is still missing, parse .env manually (works even without python-dotenv)
try:
    env_path = Path(__file__).resolve().with_name(".env")
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and v:
                os.environ[k] = v
except Exception:
    pass

# OpenAI client (optional)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

# NOTE: set_page_config must be the FIRST Streamlit command.
st.set_page_config(page_title="NewsGenie", page_icon="📰")

# -----------------------------
# 1) Framework: Core Types
# -----------------------------
Role = Literal["user", "assistant", "system"]


class ChatMessage(TypedDict):
    role: Role
    content: str


# LangGraph state for NewsGenie
class NewsGenieState(TypedDict):
    user_text: str
    query_type: str
    category: str
    response: str
    location: str
    search_query: str


@dataclass(frozen=True)
class BotConfig:
    """Framework config you can reuse across projects."""

    app_title: str = "🧠 Chatbot"
    app_caption: str = "Reusable Streamlit chatbot framework"

    # Session key for conversation storage
    session_messages_key: str = "messages"

    # Optional system prompt to seed the chat (useful for specialized bots)
    system_prompt: Optional[str] = None

    # UI
    user_input_placeholder: str = "Ask me anything…"

    # LLM (optional)
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.4
    max_history_messages: int = 20


# -----------------------------
# 2) Framework: Memory Layer
# -----------------------------
class MessageStore:
    """Thin wrapper around st.session_state to manage chat memory."""

    def __init__(self, key: str, system_prompt: Optional[str] = None):
        self.key = key
        if key not in st.session_state:
            st.session_state[key] = []  # type: ignore

        # Seed system prompt once (optional)
        if system_prompt and not self._has_system_prompt():
            self.append({"role": "system", "content": system_prompt})

    def _has_system_prompt(self) -> bool:
        msgs: List[ChatMessage] = st.session_state[self.key]
        return any(m["role"] == "system" for m in msgs)

    def get(self) -> List[ChatMessage]:
        return st.session_state[self.key]

    def append(self, msg: ChatMessage) -> None:
        st.session_state[self.key].append(msg)

    def clear(self) -> None:
        st.session_state[self.key] = []


# -----------------------------
# 3.5) Framework: LLM Provider (optional)
# -----------------------------
class LLMProvider:
    """Swap implementations (OpenAI, local model, etc.) without changing the app."""

    def generate(self, messages: List[ChatMessage], *, model: str, temperature: float) -> str:
        raise NotImplementedError


class OpenAIChatProvider(LLMProvider):
    def __init__(self, api_key: str):
        if OpenAI is None:
            raise RuntimeError("openai package not installed")
        self.client = OpenAI(api_key=api_key)

    def generate(self, messages: List[ChatMessage], *, model: str, temperature: float) -> str:
        oa_messages = [{"role": m["role"], "content": m["content"]} for m in messages]
        resp = self.client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=oa_messages,
        )
        return resp.choices[0].message.content or ""


def build_llm_provider() -> Optional[LLMProvider]:
    # Clear prior error
    st.session_state["_llm_error"] = ""

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        st.session_state["_llm_error"] = "OPENAI_API_KEY not found in environment"
        return None

    if OpenAI is None:
        st.session_state["_llm_error"] = "openai package not importable in this Python environment"
        return None

    try:
        return OpenAIChatProvider(api_key=api_key)
    except Exception as e:
        st.session_state["_llm_error"] = repr(e)
        return None


# -----------------------------
# 4) Framework: Handler Registry
# -----------------------------
HandlerFn = Callable[[str, List[ChatMessage]], str]


def llm_general_handler(user_text: str, history: List[ChatMessage]) -> str:
    """General Q&A powered by an LLM when configured, otherwise falls back."""

    provider: Optional[LLMProvider] = st.session_state.get("_llm_provider")
    config: Optional[BotConfig] = st.session_state.get("_bot_config")

    if provider is None or config is None:
        return (
            "(GENERAL MODE) LLM is not configured. "
            "Make sure your .env contains OPENAI_API_KEY=... and restart Streamlit.\n\n"
            f"You said: {user_text}"
        )

    # Build a bounded prompt: system (optional) + last N turns + current user
    system_msgs = [m for m in history if m["role"] == "system"]
    convo_msgs = [m for m in history if m["role"] != "system"]

    recent = convo_msgs[-config.max_history_messages :] if config.max_history_messages > 0 else convo_msgs

    messages: List[ChatMessage] = []
    if system_msgs:
        messages.append(system_msgs[0])
    messages.extend(recent)
    messages.append({"role": "user", "content": user_text})

    try:
        answer = provider.generate(
            messages,
            model=config.llm_model,
            temperature=config.llm_temperature,
        ).strip()
        return answer or "(GENERAL MODE) I didn’t generate a response."
    except Exception as e:
        return f"(GENERAL MODE) LLM call failed: `{e}`"


# -----------------------------
# NewsGenie LangGraph Functions
# -----------------------------

def classify_query_node(state: NewsGenieState) -> NewsGenieState:
    text = state["user_text"].lower()

    news_words = [
        "news", "latest", "headline", "headlines", "update", "updates", "breaking",
        "current", "today", "recent", "trending", "report", "reports", "coverage",
        "what are the news", "what's the news", "what is the news"
    ]
    tech_words = [
        "tech", "technology", "ai", "artificial intelligence", "machine learning", "ml",
        "robotics", "robot", "chip", "chips", "semiconductor", "semiconductors", "software",
        "hardware", "startup", "startups", "app", "apps", "cybersecurity", "cloud", "device", "devices"
    ]
    finance_words = [
        "finance", "financial", "market", "markets", "stock", "stocks", "wall street", "economy",
        "economic", "inflation", "interest rate", "interest rates", "fed", "banking", "bank", "banks",
        "investment", "investments", "investor", "investors", "crypto", "bitcoin", "ethereum", "trading",
        "business", "earnings", "recession"
    ]
    sports_words = [
        "sports", "sport", "game", "games", "athlete", "athletes", "playoff", "playoffs",
        "score", "scores", "match", "matches", "team", "teams", "season", "league", "tournament",
        "football", "basketball", "baseball", "soccer", "hockey", "tennis", "golf", "ufc", "mma",
        "nfl", "nba", "mlb", "nhl"
    ]

    location = extract_location(state["user_text"])
    state["location"] = location
    state["search_query"] = ""

    looks_like_news = any(word in text for word in news_words) or bool(location)
    if looks_like_news:
        state["query_type"] = "news"

        if any(word in text for word in tech_words):
            state["category"] = "technology"
        elif any(word in text for word in finance_words):
            state["category"] = "finance"
        elif any(word in text for word in sports_words):
            state["category"] = "sports"
        else:
            state["category"] = "general"

        if location and state["category"] != "general":
            state["search_query"] = f"{state['category']} news {location}"
        elif location:
            state["search_query"] = f"news {location}"
        else:
            state["search_query"] = text
    else:
        state["query_type"] = "general"
        state["category"] = ""
        state["location"] = ""
        state["search_query"] = ""

    return state


def extract_location(text: str) -> str:
    cleaned = re.sub(r"[?.,!]", "", text)
    patterns = [
        r"\bin\s+([A-Za-z]+(?:\s+[A-Za-z]+){0,2})",
        r"\bfor\s+([A-Za-z]+(?:\s+[A-Za-z]+){0,2})",
        r"\bnear\s+([A-Za-z]+(?:\s+[A-Za-z]+){0,2})",
        r"\bfrom\s+([A-Za-z]+(?:\s+[A-Za-z]+){0,2})",
    ]
    blacklist = {"space", "industry", "sector", "market", "markets", "world", "global", "news"}

    for pattern in patterns:
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            words = candidate.split()[:3]
            if any(word.lower() in blacklist for word in words):
                return ""
            return " ".join(word.capitalize() for word in words)
    return ""


def format_location_prefix(location: str) -> str:
    return f" for {location}" if location else ""


TRUSTED_SOURCE_KEYWORDS = [
    "ap", "associated press", "reuters", "bbc", "cnn", "cnbc", "npr", "the new york times",
    "washington post", "the guardian", "abc", "nbc", "cbs", "espn", "bloomberg", "forbes",
    "techcrunch", "the verge", "wired", "ars technica", "wsj", "wall street journal"
]
BLOCKED_SOURCE_KEYWORDS = [
    "unknown", "pr newswire", "globenewswire", "accesswire", "ein presswire", "benzinga sponsored"
]


def is_reliable_source(source_name: str) -> bool:
    lowered = source_name.lower().strip()
    if not lowered:
        return False
    if any(bad in lowered for bad in BLOCKED_SOURCE_KEYWORDS):
        return False
    if any(good in lowered for good in TRUSTED_SOURCE_KEYWORDS):
        return True
    # Allow university, local TV, and newspaper-style sources when location-specific search is used.
    local_source_signals = [
        ".com", "times", "post", "tribune", "herald", "gazette", "journal",
        "news", "tv", "fm", "radio", "chronicle", "blade", "sentinel"
    ]
    return any(sig in lowered for sig in local_source_signals)


def build_search_query(category: str, location: str, raw_text: str) -> str:
    lowered = raw_text.lower().strip()
    quoted_location = f'"{location}"' if location else ""

    if location and category == "sports":
        return f"{quoted_location} sports OR athletics OR team OR game"
    if location and category == "technology":
        return f"{quoted_location} technology OR tech OR startup OR AI"
    if location and category == "finance":
        return f"{quoted_location} business OR economy OR market OR development"
    if location:
        return f"{quoted_location} local news OR city council OR development OR community"
    if category == "finance":
        return "business markets economy"
    if category == "technology":
        return "technology AI software chips"
    if category == "sports":
        return "sports league team game"
    return lowered or "world news"


def format_articles(articles: List[dict]) -> str:
    lines: List[str] = []
    for i, article in enumerate(articles, 1):
        title = article.get("title", "No title").strip()
        title = re.sub(r"\s+", " ", title)
        description = (article.get("description") or "").strip()
        description = re.sub(r"\s+", " ", description)
        if len(description) > 180:
            description = description[:177].rstrip() + "..."
        source = article.get("source", {}).get("name", "Unknown source").strip()
        url = article.get("url", "").strip()
        published = (article.get("publishedAt") or "")[:10]

        line = f"{i}. **{title}**"
        meta_bits = []
        if source:
            meta_bits.append(source)
        if published:
            meta_bits.append(published)
        if meta_bits:
            line += f" — {' | '.join(meta_bits)}"
        if description:
            line += f"\n   {description}"
        if url:
            line += f"\n   [Read more]({url})"
        lines.append(line)
    return "\n\n".join(lines)


def fetch_news(category: str, location: str = "", raw_text: str = "") -> str:
    api_key = os.getenv("GNEWS_API_KEY", "").strip()
    if not api_key:
        return "Error: GNEWS_API_KEY not found."

    query = build_search_query(category, location, raw_text)
    url = "https://gnews.io/api/v4/search"
    params = {
        "q": query,
        "lang": "en",
        "max": 10,
        "token": api_key,
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        raw_articles = data.get("articles", [])

        seen_titles = set()
        filtered_articles = []
        for article in raw_articles:
            title = (article.get("title") or "").strip()
            source = article.get("source", {}).get("name", "")
            description = (article.get("description") or "").lower()
            haystack = f"{title.lower()} {description}"
            if not title or title.lower() in seen_titles:
                continue
            if not is_reliable_source(source):
                continue
            if location:
                location_words = [w.lower() for w in location.split() if w.strip()]
                if location_words and not any(word in haystack for word in location_words):
                    continue
            seen_titles.add(title.lower())
            filtered_articles.append(article)
            if len(filtered_articles) == 3:
                break

        if not filtered_articles:
            return "No news found."

        return format_articles(filtered_articles)
    except Exception as e:
        return f"Error fetching news: {e}"


def fetch_external_context(category: str, location: str = "", raw_text: str = "") -> str:
    api_key = os.getenv("SERP_API_KEY", "").strip()
    if not api_key:
        return ""

    query = build_search_query(category, location, raw_text)

    url = "https://serpapi.com/search.json"
    params = {
        "q": query,
        "engine": "google",
        "api_key": api_key,
        "num": 5,
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        results = data.get("organic_results", [])
        if not results:
            return ""

        lines = []
        count = 0

        for result in results:
            title = (result.get("title") or "").strip()
            snippet = (result.get("snippet") or "").strip()
            link = (result.get("link") or "").strip()

            if not title or not link:
                continue

            snippet = re.sub(r"\s+", " ", snippet)
            if len(snippet) > 180:
                snippet = snippet[:177].rstrip() + "..."

            lines.append(
                f"{count+1}. **{title}**\n   {snippet}\n   [Read more]({link})"
            )
            count += 1

            if count == 2:
                break

        if not lines:
            return ""

        return "Web search results:\n\n" + "\n\n".join(lines)

    except Exception:
        return ""


def news_node(state: NewsGenieState) -> NewsGenieState:
    raw_text = state.get("user_text", "")
    location = state.get("location", "") or extract_location(raw_text)
    location_suffix = format_location_prefix(location)
    category = state.get("category", "general")

    api_news = fetch_news(category, location=location, raw_text=raw_text)
    context_note = fetch_external_context(category, location=location, raw_text=raw_text)
    if not api_news.startswith("Error") and api_news != "No news found.":
        heading = {
            "technology": "Latest Technology News",
            "finance": "Latest Finance News",
            "sports": "Latest Sports News",
            "general": "General News",
        }.get(category, "General News")
        state["response"] = f"{heading}{location_suffix}:\n\n{api_news}"
        if context_note:
            state["response"] += f"\n\n### External Web Context\n{context_note}"
        return state

    fallback_note = "\n\n_Using demo fallback data because live news could not be retrieved._"

    if category == "technology":
        state["response"] = (
            f"Latest Technology News{location_suffix}:\n\n"
            f"1. AI tools continue expanding across business, education, and software development{location_suffix}.\n"
            f"2. Chipmakers remain a major focus as companies push for better speed, efficiency, and AI performance{location_suffix}.\n"
            f"3. Robotics and automation startups are attracting increased investor attention{location_suffix}."
            f"{fallback_note}"
        )
    elif category == "finance":
        state["response"] = (
            f"Latest Finance News{location_suffix}:\n\n"
            f"1. Markets remain mixed as investors react to inflation, interest rates, and global uncertainty{location_suffix}.\n"
            f"2. Central bank policy discussions continue to influence stocks, bonds, and currency movement{location_suffix}.\n"
            f"3. Cryptocurrency remains volatile as traders respond to macroeconomic pressure and sentiment shifts{location_suffix}."
            f"{fallback_note}"
        )
    elif category == "sports":
        state["response"] = (
            f"Latest Sports News{location_suffix}:\n\n"
            f"1. Teams across major leagues are positioning themselves for playoff contention{location_suffix}.\n"
            f"2. Star athletes continue to dominate headlines with major performances and milestone achievements{location_suffix}.\n"
            f"3. Trade speculation is increasing as organizations look to strengthen their rosters{location_suffix}."
            f"{fallback_note}"
        )
    else:
        state["response"] = (
            f"General News{location_suffix}:\n\n"
            f"1. Major developments continue across technology, finance, sports, and global affairs{location_suffix}.\n"
            f"2. For more targeted updates, ask for technology news, finance news, or sports headlines{location_suffix}."
            f"{fallback_note}"
        )

    if context_note:
        state["response"] += f"\n\n### External Web Context\n{context_note}"

    return state


def general_node(state: NewsGenieState) -> NewsGenieState:
    history = st.session_state.get("messages", [])
    state["response"] = llm_general_handler(state["user_text"], history)
    return state


def fallback_node(state: NewsGenieState) -> NewsGenieState:
    state["response"] = "Sorry, I couldn't process that request. Please try again."
    return state


# -----------------------------
# NewsGenie LangGraph Builder
# -----------------------------
def build_newsgenie_graph():
    graph = StateGraph(NewsGenieState)

    graph.add_node("classify", classify_query_node)
    graph.add_node("news", news_node)
    graph.add_node("general", general_node)
    graph.add_node("fallback", fallback_node)

    graph.set_entry_point("classify")

    def route(state: NewsGenieState):
        if state["query_type"] == "news":
            return "news"
        elif state["query_type"] == "general":
            return "general"
        else:
            return "fallback"

    graph.add_conditional_edges("classify", route)

    graph.add_edge("news", END)
    graph.add_edge("general", END)
    graph.add_edge("fallback", END)

    return graph.compile()


newsgenie_graph = build_newsgenie_graph()

def langgraph_newsgenie_handler(user_text: str, history: List[ChatMessage]) -> str:
    followup_phrases = {
        "tell me more", "more", "expand", "expand on that", "go deeper", "more detail", "more details"
    }
    normalized = user_text.lower().strip()

    if normalized in followup_phrases:
        last_assistant = next((m["content"] for m in reversed(history) if m["role"] == "assistant"), "")
        if last_assistant.startswith("Latest ") or last_assistant.startswith("General News"):
            if "Technology" in last_assistant:
                seed_query = "technology news"
            elif "Finance" in last_assistant:
                seed_query = "finance news"
            elif "Sports" in last_assistant:
                seed_query = "sports news"
            else:
                seed_query = "general news"

            loc_match = re.search(r"for ([A-Za-z]+(?:\s+[A-Za-z]+){0,2})", last_assistant)
            if loc_match:
                seed_query += f" for {loc_match.group(1)}"

            state: NewsGenieState = {
                "user_text": seed_query,
                "query_type": "",
                "category": "",
                "response": "",
                "location": "",
                "search_query": "",
            }
            result = newsgenie_graph.invoke(state)
            return result["response"]

    state: NewsGenieState = {
        "user_text": user_text,
        "query_type": "",
        "category": "",
        "response": "",
        "location": "",
        "search_query": "",
    }

    result = newsgenie_graph.invoke(state)
    return result["response"]


# -----------------------------
# 5) Framework: App Orchestrator
# -----------------------------
class ChatbotApp:
    def __init__(
        self,
        config: BotConfig,
        handler: HandlerFn,
    ):
        self.config = config
        self.handler = handler
        self.store = MessageStore(
            key=self.config.session_messages_key,
            system_prompt=self.config.system_prompt,
        )

        # Expose config/provider to handlers without global variables
        st.session_state["_bot_config"] = self.config
        # (Re)build provider at startup so .env changes take effect after a restart
        st.session_state["_llm_provider"] = build_llm_provider()

    def render_header(self) -> None:
        st.title(self.config.app_title)
        st.caption(self.config.app_caption)

    def render_sidebar(self) -> None:
        with st.sidebar:
            st.subheader("Credits")
            st.caption("Created by Connor Applin for UMich Applied Generative AI Specialization.")
            st.subheader("Controls")
            st.caption("Live news is fetched from GNews when available.")
            st.caption("External web context is retrieved via MediaWiki search.")
            st.caption("If live retrieval fails, the app falls back to demo data.")
            st.markdown("**Try these sample prompts:**")
            st.code("What is machine learning?")
            st.code("latest technology news in Japan")
            st.code("latest finance news for New York")
            st.code("latest sports headlines near Los Angeles")
            st.code("Fallback demo-latest sports headlines near Neverland")
            if st.button("Reset chat"):
                self.store.clear()
                st.rerun()

    def render_history(self) -> None:
        for msg in self.store.get():
            # Don't render system messages in the UI
            if msg["role"] == "system":
                continue
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    def run_once(self) -> None:
        self.render_header()
        self.render_sidebar()
        self.render_history()

        user_text = st.chat_input(self.config.user_input_placeholder)
        if not user_text:
            st.divider()
            st.caption("System status: active")
            return

        # 1) Persist + show user message
        self.store.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)

        # 2) Generate response via handler
        assistant_text = self.handler(user_text, self.store.get())

        # 3) Persist + show assistant response
        self.store.append({"role": "assistant", "content": assistant_text})
        with st.chat_message("assistant"):
            st.markdown(assistant_text)

        st.divider()
        st.caption("System status: active")


# -----------------------------
# 6) Instantiate a bot (this is the part you swap per project)
# -----------------------------
newsgenie_config = BotConfig(
    app_title="📰 NewsGenie",
    app_caption="AI-Powered Information and News Assistant",
    system_prompt=(
        "You are NewsGenie, an AI-powered information and news assistant. "
        "You help users with general questions and provide categorized news updates in technology, finance, and sports. "
        "Be clear, concise, and helpful."
    ),
    user_input_placeholder="Ask a question or request the latest news…",
)

app = ChatbotApp(
    config=newsgenie_config,
    handler=langgraph_newsgenie_handler,
)

app.run_once()
