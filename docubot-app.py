#!/usr/bin/env python3

# Fix from: https://docs.trychroma.com/troubleshooting#sqlite
__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from langchain.vectorstores import Chroma
from typing import List, TypeVar
T = TypeVar("T")

# create the length function
def tiktoken_len(text: str) -> int:
    """
    Returns the length of the tokenized version of the input text using the TikToken library.

    Args:
        text (str): The input text to tokenize.

    Returns:
        int: The length of the tokenized version of the input text.
    """
    import tiktoken

    tokenizer = tiktoken.encoding_for_model("text-embedding-ada-002")
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)

def embedding_cost(document: List[T]) -> (int, float):
    """
    Calculates the total number of tokens and embedding cost for a list of texts.

    Args:
    - document (List[T]): A list of texts to calculate the embedding cost for.

    Returns:
    - tuple: A tuple containing the total number of tokens and the embedding cost for the given list of texts.
    """
    total_tokens = sum([tiktoken_len(page.page_content) for page in document])
    return (total_tokens, (total_tokens / 1000) * 0.0001)

def chunk_data(data: str, chunk_size: int = 512, chunk_overlap: int = 20) -> List[T]:
    """
    Splits the input data into chunks of specified size using a RecursiveCharacterTextSplitter.

    Args:
        data (str): The input data to be split into chunks.
        chunk_size (int, optional): The size of each chunk. Defaults to 512.

    Returns:
        List[T]: A list of chunks, where each chunk is of type T.
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_documents(data)
    return chunks

def load_pdf_document(document_name: str, verbose: bool = False) -> List[T]:
    """
    Load a PDF document using PyPDFLoader.

    Args:
        document_name (str): The name of the PDF document to load.
        verbose (bool, optional): Whether to print a message indicating that the document is being loaded. Defaults to False.

    Returns:
        List[T]: A list of objects representing the loaded PDF document.
    """
    from langchain.document_loaders import PyPDFLoader

    if verbose:
        print(f"Loading {document_name}")
    return PyPDFLoader(document_name).load()


def load_docx_document(document_name: str, verbose: bool = False) -> List[T]:
    """
    Load a document in .docx format using the Docx2txtLoader class.

    Args:
        document_name (str): The name of the document to load.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        List[T]: A list of strings representing the document's contents.
    """
    from langchain.document_loaders import Docx2txtLoader

    if verbose:
        print(f"Loading {document_name}")
    return Docx2txtLoader(document_name).load()


def load_markdown_document(document_name: str, verbose: bool = False) -> List[T]:
    """
    Load a markdown document using the UnstructuredMarkdownLoader.

    Args:
        document_name (str): The name of the markdown document to load.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        List[T]: A list of objects loaded from the markdown document.
    """
    from langchain.document_loaders import UnstructuredMarkdownLoader

    if verbose:
        print(f"Loading {document_name}")
    return UnstructuredMarkdownLoader(document_name).load()


def load_txt_document(document_name: str, verbose: bool = False) -> List[T]:
    """
    Load a text document using the TextLoader class from langchain.document_loaders.

    Args:
        document_name (str): The name of the document to load.
        verbose (bool, optional): Whether to print a message indicating that the document is being loaded. Defaults to False.

    Returns:
        List[T]: A list of the loaded document's contents.
    """
    from langchain.document_loaders import TextLoader

    if verbose:
        print(f"Loading {document_name}")
    return TextLoader(document_name).load()


def load_from_wikipedia(
    query: str, lang: str = "en", load_max_docs: int = 2
) -> List[T]:
    """
    Load documents from Wikipedia based on the given query and language.

    Args:
        query (str): The query to search for on Wikipedia.
        lang (str, optional): The language of the Wikipedia to search. Defaults to "en".
        load_max_docs (int, optional): The maximum number of documents to load. Defaults to 2.

    Returns:
        List[T]: A list of documents loaded from Wikipedia.
    """
    from langchain.document_loaders import WikipediaLoader

    return WikipediaLoader(query=query, lang=lang, load_max_docs=load_max_docs).load()


def return_url_extension(url: str) -> str:
    """
    Return the file extension from the given URL.

    Args:
    url (str): The URL to extract the file extension from.

    Returns:
    str: The file extension from the given URL.
    """
    from urllib.parse import urlparse
    import os

    parsed = urlparse(url)
    root, ext = os.path.splitext(parsed.path)
    return ext


def load_document(document_name: str) -> List[T]:
    """
    Load a document from a file or URL.

    Args:
        document_name (str): The name of the document file or URL.

    Returns:
        List[T]: A list of the loaded document's contents.

    Raises:
        None.
    """
    import os
    import validators

    if validators.url(document_name):
        if return_url_extension(document_name) == ".pdf":
            return load_pdf_document(document_name)
        else:
            print("Unsupported file extension")
            return None
    else:
        name, extension = os.path.splitext(document_name)
        if extension == ".pdf":
            return load_pdf_document(document_name)
        elif extension == ".docx":
            return load_docx_document(document_name)
        elif extension == ".md":
            return load_markdown_document(document_name)
        elif extension == ".txt":
            return load_txt_document(document_name)
        else:
            print("Unsupported file extension")
            return None

def create_embeddings(chunks: List[T]) -> Chroma:
    from langchain.embeddings.openai import OpenAIEmbeddings
    """
    Create embeddings for the given chunks.

    Args:
        chunks (List[T]): A list of chunks to create embeddings for.

    Returns:
        PineCone: A PineCone vector store containing the embeddings for the given chunks.
    """
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store

def answer_query(query: str, vector_store: Chroma, num_neighbors: int = 5):
    """
    This function takes a query string, a vector store, and an optional number of neighbors to return as input.
    It uses the LangChain library to create a RetrievalQA object, which is used to run the query against the vector store.
    The function returns the result of the query as a string.
    """
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.7)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': num_neighbors})
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    return chain.run(query)

def init():
    """
    Initializes the application by loading API keys from a .env file.

    Args:
        None

    Returns:
        None
    """
    from dotenv import load_dotenv, find_dotenv
    # Load API keys
    load_dotenv(find_dotenv(), override=True)

def clear_history():
    import streamlit as st
    """
    Clears the chat history stored in the session state.

    If the 'history' key exists in the session state, it is deleted.
    """
    if 'history' in st.session_state:
        del st.session_state['history']

def draw_sidebar():
    import streamlit as st
    import os
    """
    Draws the sidebar for the DocuBot app, which includes options for uploading a file, adding data, and asking a question
    about the uploaded file. The function also displays the chat history and the answer to the user's question, if available.

    Returns:
        None
    """

    st.title("DocuBot")
    st.subheader("LLM Question Answering System")
    with st.sidebar:
        api_key = st.text_input("OpenAI API Key", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        
        uploaded_file = st.file_uploader('Upload one or more files', type=['pdf', 'docx', 'md', 'txt'], accept_multiple_files=True)
        chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=512, on_change=clear_history)
        num_neighbors = st.number_input('num_neighbors', min_value=1, max_value=10, value=5, on_change=clear_history)
        add_data = st.button('Add data', on_click=clear_history)

        data = []
        if (len(uploaded_file) > 0) and add_data:
            with st.spinner('Reading, chunking, and embedding files...'):
                for file in uploaded_file:
                    bytes_data = file.read()
                    file_name = os.path.join('./', file.name)
                    # Write to local file
                    with open(file_name, 'wb') as fp:
                        fp.write(bytes_data)
                    data.extend(load_document(file_name))

                chunks = chunk_data(data, chunk_size=chunk_size)
                st.write(f"Number of chunks: {len(chunks)}")
                st.write(f"Chunk size: {chunk_size}")

                tokens, cost = embedding_cost(chunks)
                st.write(f"Num tokens: {tokens}")
                st.write(f"Embedding cost: ${cost:.4f}")

                vector_store = create_embeddings(chunks)
                st.session_state.vector_store = vector_store
                st.success('File uploaded and embedded successfully!')

    query = st.text_input('Ask a question about uploaded file:')
    answer = ''
    if query and 'vector_store' in st.session_state:
        with st.spinner('Searching for answer...'):
            answer = answer_query(query, st.session_state.vector_store, num_neighbors=num_neighbors)
            st.text_area('Answer:', value=answer)

        st.divider()
        if 'history' not in st.session_state:
            st.session_state.history = ''

        value = f'Q: {query}\nA: {answer}'
        st.session_state.history = f'{value}\n{"-"*80}\n{st.session_state.history}'
        history = st.session_state.history
        st.text_area(label='Chat history', value = history, key='history', height=400)
        
def webapp():
    """
    Initializes the web application and draws the sidebar.
    """
    init()
    draw_sidebar()

if __name__ == "__main__":
    webapp()