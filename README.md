## AI-Powered-Medical-Chatbot-MediGuide

I built MediGuide, an AI-driven medical assistant that leverages Retrieval-Augmented Generation (RAG) to provide users with personalized, accurate, and up-to-date medical information. Designed to reduce misinformation and support informed health decisions, this chatbot integrates modern LLM infrastructure with intuitive user interaction.

Tools & Libraries Used:
- LangChain – for building RAG pipelines and chaining retrieval, embedding, and generation modules.
- OpenAI GPT-3.5 Turbo – core LLM for response generation.
- Chroma – vector database for fast and accurate similarity search.
- Streamlit – frontend interface for user interaction.
- BeautifulSoup & WebBaseLoader – for scraping and loading structured medical content from web sources.
- OpenAIEmbeddings – to convert text into dense vector representations.

Frameworks Used:
Backend: Python-based application using LangChain with ChromaDB for retrieval and OpenAI API for inference.
Frontend: Developed using Streamlit, enabling rapid deployment and a chat-like UX without heavy frontend code.

To ensure factual and grounded responses, I implemented RAG as follows:
- Scraped content from WebMD Drug Index using WebBaseLoader.
- Chunked documents with RecursiveCharacterTextSplitter for optimal embedding.
- Embedded content using OpenAIEmbeddings and stored it in Chroma.
- Configured a retrieval chain: relevant chunks are fetched per query → passed into a RAG prompt → processed by GPT-3.5 Turbo.
- This setup grounds the LLM’s outputs in real, trustworthy documents, significantly reducing hallucination.

This project demonstrates the power of combining RAG architecture with LLMs for domain-specific, user-centric chat applications. From real-time retrieval to chat history and seamless UX, MediGuide embodies the fusion of cutting-edge AI and practical healthcare utility.
