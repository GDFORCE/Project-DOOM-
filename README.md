Project Doom :books:
Project Doom is an AI-powered application designed to help users query and analyze the content of their PDF documents. By leveraging the power of LangChain, Google Generative AI, and FAISS, this project enables the extraction, segmentation, and embedding of text from uploaded PDFs. Users can ask questions related to their documents, and the application provides intelligent, context-aware answers based on the content of those PDFs.

Key Features:
PDF Text Extraction: Upload multiple PDFs, and the text is extracted and processed efficiently.
Text Chunking: Long documents are split into smaller, manageable chunks for optimal performance.
Vectorization & Embedding: The extracted text is transformed into embeddings using Google Generative AI, which are stored in FAISS for fast retrieval.
Conversational Interface: Interact with your documents through a conversational AI, asking questions and receiving context-based answers.
Chat History: Keep track of your conversations with the AI and download the chat history for future reference.
Easy-to-use UI: Streamlit-based interface for a smooth user experience.
Technologies Used:
Streamlit: For creating the interactive web interface.
LangChain: For document processing, text splitting, and conversational AI.
Google Generative AI: For powerful text embeddings and natural language processing.
FAISS: For efficient storage and retrieval of document embeddings.
How to Use:
Upload your PDF documents.
Click on "Process" to extract and process the text from the documents.
Once the documents are ready, ask questions related to the content.
The AI will respond with context-aware answers based on your document content.
Export the chat history for later use.
