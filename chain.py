from langchain_huggingface import HuggingFaceEmbeddings
from operator import itemgetter
from typing import List, Tuple
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
import os
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import qdrant_client

# Load environment variables
load_dotenv()

# Initialize clients and models
qdrant_api_key = os.getenv("QDRANT_API_KEY") or "your_default_qdrant_api_key"
qdrant_url = os.getenv("QDRANT_URL") or "your_default_qdrant_url"
openai_api_key = os.getenv("OPENAI_API_KEY_1")

client = qdrant_client.QdrantClient(qdrant_url, api_key=qdrant_api_key)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

vectorstore = QdrantVectorStore(
    client=client,
    collection_name="MUL_data_enhanced",
    embedding=embeddings,
)

llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# Condense chat history and follow-up questions
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
    """Given the following conversation and a follow-up question, 
    rephrase the follow-up question to be a standalone question in its original language.

    Chat History:
    {chat_history}

    Follow-Up Input: {question}

    Standalone Question:"""
)

# Updated RAG Answer Synthesis Prompt with strict URL handling
template = """You are an AI assistant for MUL University. Your role is to provide accurate and helpful information to students and faculty based on the retrieved context (delimited by ```).
When user said hi,hello, or greetings, you should respond with a friendly greeting and offer assistance Hello there! 👋 I'm Minhaj University Lahore chatbot, your guide about MUL How can I help you explore this Minhaj University today? 🚀
You have data about Minhaj University Lahore, including its programs, fee structures, admission requirements, faculty information and others.
If user ask about admission open you always reply Yes Admissions Open Fall 2025 cheak at this link "https://mul.edu.pk/en/admissions-open"
Follow these guidelines:
1. Always base your answers on the provided context documents. If context is limited, supplement your response using your general knowledge.
2. Provide details for fee structure queries, including:
   - Total Fee
   - Installment Plans
   - Admission Requirements
   - Core Program Details (semesters, courses, career paths).
3. If the context documents do not contain sufficient information, respond with:
   "I apologize, but I don't have enough information to answer that question accurately."

4. **CRITICAL URL HANDLING INSTRUCTIONS:**
   - ONLY use URLs that are explicitly provided in the context documents
   - Look for URLs in the "Available URLs" section at the end of the context
   - If specific URLs are provided, use them EXACTLY as they appear
   - DO NOT construct or modify URLs
   - DO NOT create URLs based on assumptions
   - If no relevant URL is found in the context, do not provide any URL
   - Always verify the URL matches the topic being discussed

5. Format URL references as: "For more information, visit: [EXACT_URL_FROM_CONTEXT]"

6. **URL Priority Order:**
   - First: Use the most specific URL that matches the query topic
   - Second: Use a relevant section URL if available
   - Third: Only use general URLs if no specific ones are found
   - Never: Create or construct URLs

Context:```{context}```
"""

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{question}"),
    ]
)

# Enhanced function to format retrieved documents with accurate URL extraction
def _combine_documents(docs):
    combined_content = []
    all_urls = []
    
    for doc in docs:
        # Add the page content
        combined_content.append(doc.page_content)
        
        # Extract URLs from metadata with more thorough checking
        if hasattr(doc, 'metadata') and doc.metadata:
            # Check various possible URL fields in metadata
            url_fields = ['source', 'url', 'link', 'page_url', 'document_url']
            for field in url_fields:
                if field in doc.metadata and doc.metadata[field]:
                    url = doc.metadata[field]
                    # Only add valid URLs
                    if url.startswith('http') and 'mul.edu.pk' in url:
                        all_urls.append(url)
        
        # Also check if URLs are mentioned in the page content itself
        import re
        url_pattern = r'https?://[^\s<>"\']*mul\.edu\.pk[^\s<>"\']*'
        found_urls = re.findall(url_pattern, doc.page_content)
        all_urls.extend(found_urls)
    
    # Combine content
    result = "\n\n".join(combined_content)
    
    # Add URLs section if found
    if all_urls:
        # Remove duplicates while preserving order
        unique_urls = list(dict.fromkeys(all_urls))
        
        # Categorize URLs for better organization
        specific_urls = []
        general_urls = []
        
        for url in unique_urls:
            # Check if URL contains specific page indicators
            specific_indicators = [
                '/scholarships', '/fee', '/admission', '/program/', 
                '/department/', '/faculty/', '/course/', '/about/'
            ]
            
            if any(indicator in url.lower() for indicator in specific_indicators):
                specific_urls.append(url)
            else:
                general_urls.append(url)
        
        # Add URLs to context in order of relevance
        if specific_urls or general_urls:
            result += "\n\n=== Available URLs ==="
            if specific_urls:
                result += "\nSpecific Page URLs: " + ", ".join(specific_urls)
            if general_urls:
                result += "\nGeneral URLs: " + ", ".join(general_urls)
    
    return result

# Function to format chat history
def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    if not chat_history:
        return []

    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer

# User input schema
class ChatHistory(BaseModel):
    chat_history: List[Tuple[str, str]] = Field(..., extra={"widget": {"type": "chat"}})
    question: str

# Runnable to check and handle chat history
_search_query = RunnableBranch(
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(run_name="HasChatHistoryCheck"),
        RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | llm
        | StrOutputParser(),
    ),
    RunnableLambda(itemgetter("question")),
)

# Combine input processing
_inputs = RunnableParallel(
    {
        "question": lambda x: x["question"],
        "chat_history": lambda x: _format_chat_history(x["chat_history"]),
        "context": _search_query | retriever | _combine_documents,
    }
)

# Define the final RAG chain
chain = (
    (_inputs | ANSWER_PROMPT | llm | StrOutputParser())
    .with_types(input_type=ChatHistory)
    .with_fallbacks(
        [
            RunnableLambda(
                lambda prompt: "There was an error while generating your response. Please try again."
            )
        ]
    )
)
