# 🎓 Talimci - Personalized Course Recommendation Assistant

Talimci is an AI-powered course recommendation system that helps users find the most suitable learning resources based on their goals, experience, and preferences. Using advanced language models and intelligent course retrieval, Talimci provides personalized recommendations from various online learning platforms.

## 🌐 Demo

🔗 **[Try Talimci Live Demo](https://huggingface.co/spaces/emrebayir/talimci)** 


## ✨ Features

- **Intelligent Profile Building**: Analyzes user input to create detailed learning profiles
- **Multi-language Support**: Works with users in their preferred language
- **Smart Course Retrieval**: Searches and retrieves relevant courses from YouTube and Udemy
- **Personalized Recommendations**: Provides tailored course suggestions based on user goals
- **Interactive Chat Interface**: Easy-to-use conversational interface built with Gradio
- **Continuous Learning**: Adapts recommendations based on ongoing conversations

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Google Gemini API key
- YouTube Data API key (optional, for enhanced course data)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/emrebayir1/talimci
   cd talimci
   ```

2. **Install required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the root directory:
   ```bash
   touch .env
   ```
   
   Add your API keys to the `.env` file:
   ```env
   GEMINI_API=your_google_gemini_api_key_here
   YOUTUBE_API=your_youtube_data_api_key_here
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the interface**
   
   Open your browser and navigate to the URL shown in the terminal (typically `http://127.0.0.1:7860`)

## 🔑 API Keys Setup

### Google Gemini API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Create a new API key
4. Copy the key and add it to your `.env` file

### YouTube Data API Key (Optional)

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the YouTube Data API v3
4. Create credentials (API key)
5. Copy the key and add it to your `.env` file

## 💬 How to Use

1. **Start a conversation**: Launch the application and begin by sharing your learning goals
2. **Provide context**: You can also share:
   - Your current experience level
   - Career objectives
   - Preference for free or paid courses
   - Time constraints
   - Specific technologies or skills you want to learn

3. **Get recommendations**: The system will analyze your input and provide personalized course recommendations

4. **Continue the conversation**: Ask follow-up questions, request alternative courses, or refine your requirements

### Example Interactions

**Basic Learning Goal:**
```
User: "I want to learn web development. I'm a complete beginner and prefer free courses."
System: "Great! I'll help you find beginner-friendly web development courses..."
```

**Career-Focused Request:**
```
User: "I'm aiming for a data scientist position. I have Python basics but need ML knowledge."
System: "Perfect! Based on your Python background and data science goals, here are some excellent machine learning courses..."
```

**Follow-up Questions:**
```
User: "Can you recommend something more advanced in React?"
User: "I changed my mind. Can you recommend some courses about Javascript instead of Python?"
```

## 🏗️ Project Structure

```
├── app.py                               # Launches the Gradio interface
├── requirements.txt                     # Project dependencies
├── README.md                            # Project description
├── .env                                 #  # Stores environment variables (Create this)
│
├── core/                                # Application logic
│   ├── __init__.py                      # Core module package
│   ├── course_recommender.py            # Course recommendation engine
│   ├── courses_retriever.py             # Searches and recommends courses
│   ├── courses_dataframe_generator.py   # Fetches and structures course data
│   ├── course_titles_generator.py       # Generates course titles
│   ├── keywords_generator.py            # Generates search keywords
│   ├── learning_session.py              # User session and profile model
│   └── user_orienter.py                 # Builds user profile
│
└── utils/                               # Helper functions
    ├── __init__.py                      # Utils module package
    └── utilities.py                     # Async and retry helpers
```

## 🔧 Core Components

### Learning Session Management
- Tracks user interactions and preferences
- Maintains conversation context
- Stores user profile and course history

### Course Retrieval System
- Searches Udemy and YouTube
- Filters courses based on user criteria
- Retrieves detailed course information

### Recommendation Engine
- Analyzes user goals and experience
- Matches users with suitable courses
- Provides personalized explanations

### User Profiling
- Extracts learning objectives from conversations
- Analyzes resume (CV) and job postings
- Identifies Subscription type (paid/free/all)
- Builds comprehensive learner profiles for personalized course recommendations

## 🛠️ Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API` | ✅ Yes | Google Gemini API key for AI-powered recommendations |
| `YOUTUBE_API` | ✅ Yes| YouTube Data API key for enhanced course data |

### Customization Options

You can customize various aspects of the application by modifying the core modules:

- **Course Sources**: Add new learning platforms in the course retriever
- **Recommendation Logic**: Adjust the recommendation algorithms
- **UI Interface**: Customize the Gradio interface in `app.py`
- **LLM & Prompts**: Improve or replace language models and prompts for better recommendations

## 📋 Dependencies

The application requires the following Python packages:

- `pandas` - Data manipulation and analysis
- `requests` - HTTP library for API calls
- `beautifulsoup4` - Web scraping and HTML parsing
- `google-api-python-client` - Google APIs client library
- `google-generativeai` - Google's Generative AI library
- `google-genai` - Additional Google AI utilities
- `python-dotenv` - Environment variable management
- `lingua-language-detector` - Language detection
- `pydantic` - Data validation and settings management
- `gradio` - Web interface framework

## 🚨 Troubleshooting

### Common Issues

**API Key Errors:**
- Ensure your `.env` file is in the root directory
- Verify that your API keys are valid and have proper permissions
- Check that there are no extra spaces or quotes around the keys

**Module Import Errors:**
- Make sure all requirements are installed: `pip install -r requirements.txt`
- Verify you're running Python 3.8 or higher
- Check that all core modules are present in the `core/` directory

### Getting Help

If you encounter issues:
1. Check the terminal output for detailed error messages
2. Verify all dependencies are correctly installed
3. Ensure your API keys have the necessary permissions
4. Make sure your internet connection is stable for API calls

## 🤝 Contributing

I welcome contributions to improve Talimci! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Make your changes** and test thoroughly
4. **Commit your changes**: `git commit -am 'Add new feature'`
5. **Push to the branch**: `git push origin feature-name`
6. **Create a Pull Request**

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [Gradio](https://gradio.app/) for the web interface
- Powered by [Google Gemini](https://deepmind.google/technologies/gemini/) for AI recommendations
- Thanks to all the open-source libraries that made this project possible


---

**Happy Learning with Talimci! 🎓✨**
