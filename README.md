# MUL Chatbot 2 - Deployed

A conversational AI chatbot for Minhaj University Lahore (MUL) built with LangChain, providing information about university programs, fee structures, admission requirements, and faculty information.

## Features

- **Conversational AI**: Built with LangChain and OpenAI's GPT-4 model
- **Vector Search**: Uses Qdrant vector database for efficient document retrieval
- **University Information**: Provides comprehensive information about MUL including:
  - Program details and fee structures
  - Admission requirements
  - Faculty information
  - Course information

## Technology Stack

- **LangChain**: Framework for building LLM applications
- **OpenAI GPT-4**: Language model for generating responses
- **Qdrant**: Vector database for semantic search
- **HuggingFace Embeddings**: Text embeddings using sentence-transformers
- **Python**: Backend implementation

## Files

- `chain.py`: Main chain implementation with RAG (Retrieval-Augmented Generation) pipeline
- `main.py`: Application entry point
- `.env`: Environment variables (not included in repo)

## Setup

1. Clone the repository:
```bash
git clone https://github.com/Razamirxa/MuLchatbot2-deploed.git
cd MuLchatbot2-deploed
```

2. Install dependencies:
```bash
pip install langchain langchain-huggingface langchain-openai langchain-qdrant qdrant-client python-dotenv
```

3. Create a `.env` file with your API keys:
```
OPENAI_API_KEY_1=your_openai_api_key
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_URL=your_qdrant_url
```

4. Run the application:
```bash
python main.py
```

## Features Overview

- **Chat History**: Maintains conversation context
- **Document Retrieval**: Searches through university documents
- **URL Handling**: Provides relevant university website links
- **Fallback Handling**: Graceful error handling for better user experience

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is for educational purposes.
