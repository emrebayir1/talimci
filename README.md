# 🎓 Talimci - Personalized Course Recommendation Assistant

Talimci is an AI-powered chat assistant that helps users find the most suitable courses for their learning goals. It builds a short profile through a guided conversation (goal, optional resume, optional target job posting, and price preference), searches Udemy, Coursera, and YouTube for relevant courses, and then recommends the best matches with the reasoning behind them.

## 🌐 Demo

🔗 **[Try Talimci Live Demo](https://talimci.emrebayir.com)**

## ✨ Features

- **Guided Profile Wizard**: Asks one question at a time (learning goal, resume, target job posting, paid/free preference) and adapts based on previous answers
- **Multi-provider LLM Support**: Works with Google Gemini, Groq, or OpenRouter — pick whichever provider/model fits your needs and budget
- **Smart Course Retrieval**: Searches and aggregates courses from Udemy, Coursera, and YouTube into a structured dataset
- **Personalized Recommendations**: Matches courses to the user's goal, background, and target role, with explanations
- **Conversational Follow-ups**: Users can ask for more detail, request different courses, or change their preferences at any point in the chat
- **Chat-based Interface**: Built with [Chainlit](https://chainlit.io/), including button-based quick replies for closed-choice questions

## 🏗️ How It Works

1. **Profile wizard** (`core/user_orienter.py`) — an LLM-guided, step-by-step wizard collects the learning goal and (optionally) a resume, a job posting, and a subscription preference (paid/free/either).
2. **Keyword & title generation** (`core/keywords_generator.py`, `core/course_titles_generator.py`) — the profile is turned into concrete search keywords and candidate course titles.
3. **Course retrieval** (`core/courses_dataframe_generator.py`, `core/courses_retreiver.py`) — courses are fetched from Udemy, Coursera, and YouTube (via the YouTube Data API) and cleaned into a single pandas DataFrame.
4. **Recommendation** (`core/course_recommender.py`) — the LLM reviews the catalog and the conversation so far, and returns a personalized shortlist with explanations. Every follow-up message goes through this step again, so the recommendations keep adapting as the conversation continues.

## 🚀 Getting Started

### Prerequisites

- Python 3.12 (or Docker, if you prefer not to install Python locally)
- An API key for **at least one** LLM provider: Gemini, Groq, or OpenRouter
- A YouTube Data API v3 key (used to fetch YouTube courses)

### Option A — Local setup with venv

1. **Clone the repository**
   
   ```bash
   git clone https://github.com/emrebayir1/talimci
   cd talimci
   ```

2. **Create and activate a virtual environment**
   
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # on Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   
   Copy the example file and fill in your keys:
   
   ```bash
   cp .env.Example .env
   ```
   
   See the [API Keys Setup](#-api-keys-setup) section below for where to get each key.

5. **Run the app**
   
   ```bash
   chainlit run app.py -w
   ```

6. **Open the interface**
   
   Chainlit will print a local URL in the terminal (typically `http://localhost:8000`). Open it in your browser to start chatting.

### Option B — Docker

1. **Clone the repository and configure your `.env`**
   
   ```bash
   git clone https://github.com/emrebayir1/talimci
   cd talimci
   cp .env.Example .env
   ```
   
   Fill in your API keys in `.env` as described below.

2. **Build the image**
   
   ```bash
   docker build -f .Dockerfile -t talimci .
   ```

3. **Run the container**
   
   ```bash
   docker run --env-file .env -p 8000:8000 talimci
   ```

4. **Open the interface**
   
   Visit `http://localhost:8000` in your browser.

## 🔑 API Keys Setup

Talimci needs an LLM provider key (choose one or more — you select which one is active via `LLM_PROVIDER`) and a YouTube Data API key. Add whichever keys you obtain to your `.env` file.

### Google Gemini API Key (`GEMINI_API`)

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API key"
4. Copy the key into `GEMINI_API` in your `.env` file

### Groq API Key (`GROQ_API`)

1. Go to [Groq Console](https://console.groq.com/keys)
2. Sign in or create a free account
3. Create a new API key
4. Copy the key into `GROQ_API` in your `.env` file

### OpenRouter API Key (`OPENROUTER_API_KEY`)

1. Go to [OpenRouter](https://openrouter.ai/keys)
2. Sign in or create an account
3. Create a new API key (OpenRouter offers several `:free` models, so no payment method is required to get started)
4. Copy the key into `OPENROUTER_API_KEY` in your `.env` file

### YouTube Data API Key (`YOUTUBE_API_KEYS`)

1. Open the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (or select an existing one)
3. Go to **APIs & Services → Library** and enable **YouTube Data API v3**
4. Go to **APIs & Services → Credentials** and create an **API key**
5. Copy the key into `YOUTUBE_API_KEYS` in your `.env` file

> `.env.Example` also contains an `OPENAI_API_KEY` placeholder for future use; it isn't required by the current codebase.

## 🛠️ Configuration

Below are the environment variables read by the app (see `.env.Example` for the full template):

| Variable             | Required                 | Description                                                                                                                                            |
| -------------------- | ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `LLM_PROVIDER`       | No                       | Which LLM provider to use: `gemini`, `groq`, or `openrouter`. Default: `gemini`                                                                        |
| `CHAT_MODEL`         | No                       | Model name for the selected provider (e.g. `gemini-2.5-flash`, a Groq model, or an OpenRouter model id). Falls back to a sensible default per provider |
| `GEMINI_API`         | Only if using Gemini     | Google Gemini API key                                                                                                                                  |
| `GROQ_API`           | Only if using Groq       | Groq API key                                                                                                                                           |
| `OPENROUTER_API_KEY` | Only if using OpenRouter | OpenRouter API key                                                                                                                                     |
| `YOUTUBE_API_KEYS`   | Yes                      | YouTube Data API v3 key used to fetch YouTube courses                                                                                                  |
| `GEMINI_RPM_LIMIT`   | No                       | Requests-per-minute throttle for Gemini calls (default: `4`)                                                                                           |
| `GROQ_TPM_LIMIT`     | No                       | Tokens-per-minute throttle for Groq calls (default: `8000`)                                                                                            |
| `COURSE_TOP_N`       | No                       | How many courses to keep per search when building the catalog                                                                                          |

## 💬 Example Interactions

**Basic learning goal:**

```
Assistant: What's your learning goal?
User: I want to learn web development. I'm a complete beginner.
```

**Career-focused request, with resume/job posting:**

```
Assistant: Would you like to share your resume?
User: Sure — [pastes resume]
Assistant: Do you have a target job posting in mind?
User: Yes — [pastes job posting]
```

**Follow-up refinement, after the first recommendations:**

```
User: Can you recommend something more advanced in React instead?
User: Actually, show me free courses only.
```

## 🏗️ Project Structure

```
├── app.py                               # Chainlit app: profile wizard + chat loop
├── requirements.txt                     # Project dependencies
├── .Dockerfile                          # Docker build definition
├── .env.Example                         # Environment variable template (copy to .env)
├── chainlit.md                          # Welcome text shown in the Chainlit UI
├── README.md                            # Project description
│
├── core/                                # Application logic
│   ├── learning_session.py              # LearningSession data model (Pydantic)
│   ├── user_orienter.py                 # Profile wizard: asks for goal/resume/job/preference
│   ├── keywords_generator.py            # Generates search keywords from the profile
│   ├── course_titles_generator.py       # Generates candidate course titles
│   ├── courses_dataframe_generator.py   # Fetches & cleans course data (Udemy, Coursera, YouTube) into a DataFrame
│   ├── courses_retreiver.py             # Orchestrates course search and filtering
│   └── course_recommender.py            # Builds the final recommendation using the LLM
│
├── utils/
│   ├── models.py                        # LLM provider abstraction (Gemini / Groq / OpenRouter)
│   └── utilities.py                     # Async helpers, retry logic
│
└── public/                              # Static assets (avatars, favicon, stylesheet) for the Chainlit UI
```

## 📋 Key Dependencies

- `chainlit` — chat UI framework powering the interface
- `pandas` — building and manipulating the course catalog
- `ddgs` — web search used during course discovery
- `google-api-python-client` — YouTube Data API client
- `google-genai` — Gemini API client
- `lingua-language-detector` — language detection for user input
- `pydantic` — data validation for the `LearningSession` model
- `requests` / `httpx` / `httpcore` — HTTP calls to LLM providers

## 🚨 Troubleshooting

**API key errors:**

- Make sure `.env` is in the project root (not `.env.Example`)
- Double check `LLM_PROVIDER` matches the key you actually filled in
- Make sure there are no extra spaces or quotes around key values

**No courses returned / empty catalog:**

- Verify `YOUTUBE_API_KEYS` is valid and the YouTube Data API v3 is enabled on that Google Cloud project
- Check your terminal logs — provider rate limits (`GEMINI_RPM_LIMIT`, `GROQ_TPM_LIMIT`) may be throttling requests

**Docker container exits immediately:**

- Confirm you passed `--env-file .env` when running `docker run`
- Check `docker logs <container_id>` for the underlying error

## 🧭 Next Steps

Planned improvements for upcoming versions:

- **File upload support** — letting users upload their CV and the target job posting as files (PDF/DOCX), instead of pasting text
- **UI improvements** — general polish of the chat experience and quick-reply widgets
- **Excel export** — allowing users to download the full list of found courses (not just the shortlist shown in chat) as an Excel file, likely by first presenting it as a proper table in the UI instead of a raw DataFrame

## 🤝 Contributing

Contributions are welcome! Here's how to help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Make your changes** and test thoroughly
4. **Commit your changes**: `git commit -am 'Add new feature'`
5. **Push to the branch**: `git push origin feature-name`
6. **Open a Pull Request**

## 📄 License

This project is licensed under the MIT License — see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [Chainlit](https://chainlit.io/) for the chat interface
- Powered by [Google Gemini](https://deepmind.google/technologies/gemini/), [Groq](https://groq.com/), and [OpenRouter](https://openrouter.ai/) for AI recommendations

---

**Happy Learning with Talimci! 🎓✨**
