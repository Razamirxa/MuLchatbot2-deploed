from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Qdrant
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain.schema import Document
from qdrant_client import QdrantClient
from typing import List, Dict, Any, Set, Tuple
from operator import itemgetter
from pydantic import BaseModel, Field
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.messages import HumanMessage, AIMessage
import os
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Your Qdrant credentials

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COLLECTION_NAME = "mul-data-with-pdf-3"

# Initialize components
# Initialize components
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model="gpt-4.1-nano", api_key=OPENAI_API_KEY, temperature=0)
# Initialize Qdrant client and vector store
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings
)

# Create a base retriever with higher k value
base_retriever = vector_store.as_retriever(search_kwargs={"k": 15})

# Create a multi-query retriever
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm
)

# Simple keyword mapping for common Urdu queries
def map_urdu_to_english_keywords(query: str) -> str:
    """Map common Urdu queries to English keywords"""
    query_lower = query.lower()
    
    # Common mappings
    keyword_mappings = {
        "registart": "registrar",
        "vc": "vice chancellor",
        "vice chancellor": "vice chancellor",
        "nam": "name",
        "kya": "what is",
        "hy": "is",
        "fee": "fee",
        "structure": "structure",
        "admission": "admission",
        "program": "program",
        "course": "course",
        "computer science": "computer science",
        "mba": "mba",
        "bs": "bs",
        "m.phil": "m.phil",
        "phd": "phd"
    }
    
    # Replace Urdu words with English equivalents
    for urdu_word, english_word in keyword_mappings.items():
        if urdu_word in query_lower:
            query_lower = query_lower.replace(urdu_word, english_word)
    
    return query_lower

# Define helper functions
def format_docs(docs: List[Document]) -> str:
    """Format documents into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)

def extract_program_name(query: str) -> str:
    """Extract program name from query"""
    # Common patterns for program names
    patterns = [
        r'bs\s+([a-z\s]+)',
        r'm\.?phil\.?\s+([a-z\s]+)',
        r'phd\s+([a-z\s]+)',
        r'mba\s+([a-z\s]+)',
        r'bachelor\s+of\s+([a-z\s]+)',
        r'master\s+of\s+([a-z\s]+)',
        r'associate\s+degree\s+in\s+([a-z\s]+)',
        r'post\s+graduate\s+diploma\s+in\s+([a-z\s]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query.lower())
        if match:
            return match.group(1).strip()
    
    return ""

def url_relevance_score(url: str, title: str, query: str, program_name: str) -> int:
    """Calculate relevance score for a URL based on query and program name"""
    score = 0
    query_lower = query.lower()
    url_lower = url.lower()
    title_lower = title.lower()
    
    # Very high priority: Exact match for program name in URL
    if program_name and program_name.replace(" ", "-") in url_lower:
        score += 20
    # High priority: URL contains program name without spaces
    elif program_name and program_name.replace(" ", "") in url_lower:
        score += 15
    # Medium priority: URL contains individual words from program name
    elif program_name:
        program_words = program_name.split()
        for word in program_words:
            if word in url_lower:
                score += 5
    
    # Very high priority for fee-related queries with fee in URL
    if any(keyword in query_lower for keyword in ["fee", "structure", "cost", "tuition", "admission"]):
        if "fee" in url_lower:
            score += 20
        elif "tuition" in url_lower:
            score += 15
        elif "admission" in url_lower:
            score += 10
    
    # Medium priority: URL contains query keywords
    query_keywords = ["program", "course"]
    for keyword in query_keywords:
        if keyword in url_lower:
            score += 3
    
    # Lower priority: Title contains program name or query keywords
    if program_name and program_name in title_lower:
        score += 5
    for keyword in ["fee", "structure", "cost", "tuition", "admission"]:
        if keyword in title_lower:
            score += 3
    
    return score

def extract_and_rank_urls(query: str, docs: List[Document]) -> List[str]:
    """Extract and rank URLs based on relevance to query"""
    # Extract program name from query
    program_name = extract_program_name(query)
    
    # Collect all URLs with their metadata
    url_data = []
    for doc in docs:
        if doc.metadata.get("type") == "json" and doc.metadata.get("url"):
            url_data.append({
                "url": doc.metadata["url"],
                "title": doc.metadata.get("title", ""),
                "content": doc.page_content
            })
    
    if not url_data:
        return []
    
    # Score each URL
    scored_urls = []
    for data in url_data:
        score = url_relevance_score(
            data["url"], 
            data["title"], 
            query, 
            program_name
        )
        scored_urls.append((data["url"], score))
    
    # Sort by score descending
    scored_urls.sort(key=lambda x: x[1], reverse=True)
    
    # Extract URLs in order of relevance, but only include high-scoring ones
    # Set a minimum score threshold to filter out less relevant URLs
    min_score_threshold = 10
    ranked_urls = [url for url, score in scored_urls if score >= min_score_threshold]
    
    # Limit the number of URLs to show (max 3 for program-specific queries, max 5 for general queries)
    max_urls = 3 if program_name else 5
    ranked_urls = ranked_urls[:max_urls]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_ranked_urls = []
    for url in ranked_urls:
        if url not in seen:
            seen.add(url)
            unique_ranked_urls.append(url)
    
    return unique_ranked_urls

def format_response(answer: str, urls: List[str]) -> str:
    """Format the final response with answer and URLs"""
    # Check if the answer indicates no information was found
    no_info_phrases = [
        "I don't have information about this",
        "I don't have information about this topic",
        "I don't have information about this in my knowledge base"
    ]
    
    # Only include URLs if the answer doesn't contain any of the "no information" phrases
    include_urls = not any(phrase in answer for phrase in no_info_phrases)
    
    if include_urls and urls:
        url_section = "\n\nFor more information, you can visit:\n"
        for url in urls:
            url_section += f"- {url}\n"
        return answer + url_section
    else:
        return answer

# Condense chat history and follow-up questions
CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_template(
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
If someone ask about minhaj university tell  in detail about minhaj university lahore and also give url of main page https://www.mul.edu.pk/en/
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
        "context": _search_query | multi_query_retriever | _combine_documents,
    }
)

# Create the improved LCEL chain with URL ranking and keyword mapping
def create_improved_rag_chain():
    # Create a chain that maps keywords if needed, then retrieves documents and generates response
    rag_chain = (
        RunnablePassthrough.assign(
            mapped_query=lambda x: map_urdu_to_english_keywords(x["question"])
        )
        | RunnableParallel(
            {
                "question": lambda x: x["mapped_query"],  # Use mapped query for the rest
                "docs": lambda x: multi_query_retriever.get_relevant_documents(x["mapped_query"]),
                "chat_history": lambda x: _format_chat_history(x["chat_history"]),
            }
        )
        | RunnableParallel(
            {
                "context": lambda x: _combine_documents(x["docs"]),
                "urls": lambda x: extract_and_rank_urls(x["question"], x["docs"]),
                "question": lambda x: x["question"],  # Use original question for the prompt
                "chat_history": lambda x: x["chat_history"],
            }
        )
        | RunnableParallel(
            {
                "answer": (RunnablePassthrough() 
                          | (lambda x: {"context": x["context"], "question": x["question"], "chat_history": x["chat_history"]})
                          | ANSWER_PROMPT
                          | llm
                          | StrOutputParser()),
                "urls": lambda x: x["urls"]
            }
        )
        | (lambda x: format_response(x["answer"], x["urls"]))
    )
    
    return rag_chain

# Define the final RAG chain
chain = (
    create_improved_rag_chain()
    .with_types(input_type=ChatHistory)
    .with_fallbacks(
        [
            RunnableLambda(
                lambda prompt: "There was an error while generating your response. Please try again."
            )
        ]
    )
)
