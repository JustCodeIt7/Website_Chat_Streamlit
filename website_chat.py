import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, urlparse
import time
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os

# Set the OpenAI API key
# ============= Configuration Component =============
def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "crawled_urls" not in st.session_state:
        st.session_state.crawled_urls = set()
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = ""
    
    # Set the OpenAI API key - user key takes precedence over secrets
    api_key = st.session_state.openai_api_key
    if not api_key and "openai_api_key" in st.secrets:
        api_key = st.secrets["openai_api_key"]
    
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key


def setup_page_config():
    """Configure the Streamlit page."""
    st.set_page_config(page_title="Chat with Websites", page_icon="🌐", layout="wide")
    st.title("Chat with Websites using LangChain and Ollama")


def create_sidebar_config():
    """Create and return the configuration from the sidebar."""
    with st.sidebar:
        st.header("Configuration")
                # OpenAI API Key input
        api_key = st.text_input("OpenAI API Key (optional)", 
                               value=st.session_state.openai_api_key, 
                               type="password",
                               help="Enter your own OpenAI API key. If not provided, the app will use the default key if available.")
        
        # Update session state if the key changed
        if api_key != st.session_state.openai_api_key:
            st.session_state.openai_api_key = api_key
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
                st.success("API key updated!")
            elif "openai_api_key" in st.secrets:
                os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
                st.info("Using default API key")
            else:
                st.warning("No API key provided")
        
        config = {
            "ollama_model": st.selectbox(
                "Select Ollama Model",
                ["deepseek-r1:1.5b", "qwen2.5:0.5b", "llama3.2:1b"],
                index=0,
            ),
            "embedding_model": st.selectbox(
                "Select Embedding Model",
                ["all-minilm:33m", "snowflake-arctic-embed:latest"],
                index=0,
            ),
            "text_threshold": st.slider(
                "Text size threshold (characters) for vector DB vs. full context",
                min_value=1000,
                max_value=50000,
                value=10000,
                step=1000,
            ),
            "chunk_size": st.slider(
                "Chunk size for text splitting",
                min_value=100,
                max_value=2000,
                value=500,
                step=100,
            ),
            "chunk_overlap": st.slider(
                "Chunk overlap", min_value=0, max_value=500, value=50, step=10
            ),
            # Web crawling parameters
            "max_pages": st.slider(
                "Maximum pages to crawl", min_value=1, max_value=50, value=5, step=1
            ),
            "crawl_depth": st.slider(
                "Crawling depth", min_value=1, max_value=3, value=1, step=1
            ),
            "exclude_patterns": st.text_input(
                "URL patterns to exclude (comma-separated)",
                value="login,signup,register,cart,checkout,account",
            ).split(","),
        }
        return config


# ============= Web Crawling Component =============
def is_valid_url(url):
    """Check if URL is valid and reachable."""
    try:
        response = requests.head(url, timeout=5)
        return response.status_code < 400
    except:
        return False


def extract_links(url, soup, base_domain):
    """Extract valid links from the page that stay within the same domain."""
    links = []
    for a_tag in soup.find_all("a", href=True):
        link = a_tag["href"]
        # Convert relative URLs to absolute
        full_url = urljoin(url, link)
        # Parse the URL to get the domain
        parsed_url = urlparse(full_url)
        # Only include links from the same domain
        if parsed_url.netloc == base_domain and parsed_url.scheme in ["http", "https"]:
            links.append(full_url)
    return links

def should_exclude_url(url, exclude_patterns):
    """Check if URL should be excluded based on patterns."""
    for pattern in exclude_patterns:
        if pattern.strip() and pattern.strip() in url:
            return True
    return False


def crawl_website(start_url, config, progress_bar=None, progress_text=None):
    """
    Crawl website starting from the given URL, respecting depth and page limits.

    Args:
        start_url (str): Starting URL for the crawl
        config (dict): Configuration parameters
        progress_bar: Streamlit progress bar
        progress_text: Streamlit text element for updates

    Returns:
        str: Combined text from all crawled pages
    """
    max_pages = config["max_pages"]
    max_depth = config["crawl_depth"]
    exclude_patterns = config["exclude_patterns"]

    # Parse base domain from start URL
    base_domain = urlparse(start_url).netloc

    # Initialize tracking variables
    queue = [(start_url, 0)]  # (url, depth)
    visited = set()
    all_text = ""
    page_count = 0

    while queue and page_count < max_pages:
        # Get URL and its depth from queue
        url, depth = queue.pop(0)

        # Skip if already visited or should be excluded
        if url in visited or should_exclude_url(url, exclude_patterns):
            continue

        # Mark as visited
        visited.add(url)

        # Update progress
        if progress_text:
            progress_text.text(f"Crawling page {page_count+1}/{max_pages}: {url}")

        try:
            # Load and extract text from the page
            text = extract_text_from_webpage(url)
            if text:
                all_text += f"\n\n--- Content from: {url} ---\n\n" + text
                page_count += 1

                # Update progress bar
                if progress_bar:
                    progress_bar.progress(min(page_count / max_pages, 1.0))

                # If we haven't reached max depth, extract more links
                if depth < max_depth:
                    response = requests.get(url, timeout=10)
                    soup = BeautifulSoup(response.text, "html.parser")
                    links = extract_links(url, soup, base_domain)

                    # Add new links to the queue
                    for link in links:
                        if link not in visited:
                            queue.append((link, depth + 1))
        except Exception as e:
            st.warning(f"Error crawling {url}: {e}")

        # Short delay to be nice to servers
        time.sleep(0.5)

    # Store the crawled URLs in session state
    st.session_state.crawled_urls = visited

    return all_text


# ============= Web Processing Component =============
def extract_text_from_webpage(url):
    """
    Extract and clean text from a webpage.

    Args:
        url (str): The URL of the webpage to extract text from.

    Returns:
        str or None: The extracted text, or None if an error occurred.
    """
    try:
        loader = WebBaseLoader(url)
        data = loader.load()
        return data[0].page_content
    except Exception as e:
        st.warning(f"Error loading {url}: {e}")
        return None


def create_full_context_processor(llm, text):
    """
    Create a processor for the full context approach.

    Args:
        llm (ChatOllama): The language model.
        text (str): The full text from the webpage.

    Returns:
        function: A function that processes user queries using the full context.
    """

    def process_query(user_input):
        prompt = f"""
        You are an AI assistant that helps users understand website content.

        Website content:
        {text}

        User question: {user_input}

        Please provide a helpful, accurate, and concise answer based on the website content.
        """
        response = llm.invoke(prompt)
        return response.content

    return process_query


def create_vector_db_processor(qa_chain):
    """
    Create a processor for the vector database approach.

    Args:
        qa_chain (ConversationalRetrievalChain): The conversational retrieval chain.

    Returns:
        function: A function that processes user queries using the vector database.
    """
    def process_query(user_input):
        response = qa_chain.invoke(
            {"question": user_input, "chat_history": st.session_state.chat_history}
        )
        response_text = response["answer"]
        st.session_state.chat_history.append((user_input, response_text))
        return response_text

    return process_query


def setup_vector_approach(text, config):
    """
    Set up the vector database approach for large texts.

    Args:
        text (str): The text to process.
        config (dict): Configuration parameters.

    Returns:
        ConversationalRetrievalChain: The conversational retrieval chain.
    """
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"], chunk_overlap=config["chunk_overlap"]
    )
    chunks = text_splitter.split_text(text)

    # Create embeddings and vector store
    # embeddings = OllamaEmbeddings(model=config["embedding_model"])
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small') 
    vectorstore = FAISS.from_texts(chunks, embeddings)

    # Create a retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Set up memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Create the conversational chain
    # llm = ChatOllama(model=config["ollama_model"],temperature=0.3)
    llm = ChatOpenAI(model='gpt-4o-mini',temperature=0.3)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, memory=memory
    )

    return qa_chain


def process_website(url, config):
    """
    Process a website by crawling its pages and setting up a query processor.

    Args:
        url (str): The starting URL of the website to process.
        config (dict): Configuration parameters.

    Returns:
        tuple: A tuple containing:
            - function: The query processor function.
            - int: The length of the extracted text.
            - int: The number of pages crawled.
    """
    # Set up progress tracking
    progress_text = st.empty()
    progress_bar = st.progress(0)

    progress_text.text("Starting website crawl...")

    # Crawl the website and get the combined text
    with st.spinner("Crawling website..."):
        all_text = crawl_website(url, config, progress_bar, progress_text)

    text_length = len(all_text)
    num_pages = len(st.session_state.crawled_urls)

    progress_text.text(f"Extracted {text_length} characters from {num_pages} pages")
    st.success(f"Website crawl complete! Processed {num_pages} pages")

    # Initialize the LLM
    llm = ChatOpenAI(model='gpt-4o-mini',temperature=0.3)

    # If text is smaller than threshold, use full context approach
    if text_length < config["text_threshold"]:
        st.success("Using full context approach (text is relatively small)")
        processor = create_full_context_processor(llm, all_text)
        st.session_state.full_text = all_text
        st.session_state.vector_approach = False
    else:
        # Otherwise, use vector embeddings approach
        st.success("Using vector embeddings approach (text is relatively large)")
        with st.spinner("Creating vector database..."):
            qa_chain = setup_vector_approach(all_text, config)
            processor = create_vector_db_processor(qa_chain)
            st.session_state.full_text = None
            st.session_state.vector_approach = True

    progress_bar.empty()
    progress_text.empty()

    return processor, text_length, num_pages


# ============= Chat UI Component =============
def display_chat_messages():
    """Display all messages in the chat history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


def handle_user_input(query_processor):
    """
    Handle user input and generate responses.

    Args:
        query_processor (function): The function to process user queries.
    """
    user_input = st.chat_input("Ask a question about the website:")

    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Display user message
        with st.chat_message("user"):
            st.write(user_input)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_text = query_processor(user_input)
                st.write(response_text)

        # Add assistant response to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": response_text}
        )



# ============= Main Application =============
def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()

    # Setup page configuration
    setup_page_config()

    # Get configuration from sidebar
    config = create_sidebar_config()

    # Add button to clear chat history
    if st.sidebar.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    # Main app interface
    col1, col2 = st.columns([3, 1])
    with col1:
        url_input = st.text_input(
            "Enter a website URL:",
            "https://python.langchain.com/docs/get_started/introduction",
        )
    with col2:
        process_button = st.button("Process Website", use_container_width=True)

    if process_button:
        if url_input:
            # Store the URL in session state
            st.session_state.url = url_input

            # Process the website
            query_processor, text_length, num_pages = process_website(url_input, config)

            if query_processor is not None:
                st.session_state.query_processor = query_processor
                st.session_state.text_length = text_length
                st.session_state.num_pages = num_pages
                st.session_state.chat_history = []
                st.session_state.messages = [
                    {
                        "role": "assistant",
                        "content": f"Website processed! Crawled {num_pages} pages with total {text_length} characters. You can now ask questions about it.",
                    }
                ]
                st.rerun()
        else:
            st.warning("Please enter a valid URL")

    # If crawled URLs are available, show them in an expander
    if hasattr(st.session_state, "crawled_urls") and st.session_state.crawled_urls:
        with st.expander("Crawled Pages"):
            for url in st.session_state.crawled_urls:
                st.write(f"- {url}")

    # Display chat messages
    display_chat_messages()

    # Handle user input if a webpage has been processed
    if "query_processor" in st.session_state:
        handle_user_input(st.session_state.query_processor)


if __name__ == "__main__":
    main()
