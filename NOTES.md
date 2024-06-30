    - FastAPI endpoints to upload documents, ask questions

    - Uploaded documents are converted into vector embeddings (maybe using Ada or Langchain)
        - Considerations: Assuming the documents to be of filetypes like PDF, DOC, HTML, MD or RST.

    - Store those embeddings in a vector db (like pinecone)

    - When user makes a question, use Ada or Langchain to turn their question into embeddings

    - Use those embeddings to search the vector db (via cosine similarity)

    - Return all the relevant strings from the vector db

    - Construct a prompt gpt 3-4 to answer the original user question using info contained in the returned strings from the vector db

    - Send result to user
