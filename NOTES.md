    - FastAPI endpoints to upload documents, ask questions

    - Uploaded documents are converted into vector embeddings (maybe using Ada or Langchain)
        - Considerations: Assuming the documents to be of filetypes like PDF, DOC, HTML, MD or RST.

    - Store those embeddings in a vector db (like pinecone)

    - When user makes a question, use Ada or Langchain to turn their question into embeddings

    - Use those embeddings to search the vector db (via cosine similarity)

    - Return all the relevant strings from the vector db

    - Construct a prompt gpt 3-4 to answer the original user question using info contained in the returned strings from the vector db

    - Send result to user


The nature of the document store is not provided, hence assuming it could be anything. So has to implement multiple 


Challenges:
    - Finding opensource solutions for embedding models, llm models etc which are fast and hardware friendly.
        - [link] https://huggingface.co/spaces/mteb/leaderboard [link]
    - Finding open source vector stores for storing the generated vector embeddings














I'm planning to build a RAG system to power a ChatGPT like system for my company's internal documents. I'll be using Langchain, Langchain Community, FAISS store in Langchain document store, LLAMA3 model from Ollama used as LLM and Embedding model. It will be behind a FastAPI backend with API endpoints to upload a single file (PDF, TXT, CSV or any such files), an endpoint to setup a background task to load a directory of documents to the document store, another endpoint to ask questions about the uploaded document, another to retrieve details about the background tasks, another endpoint to check the files etc. There is also a frontend which I will be building using streamlit. 

The primary goal is to enable users to ask questions about these documents and receive short, precise, and well-sourced answers. Each answer should include a link to the specific part of the document where the information was found, allowing users to verify the information directly.

The guidelines to follow are:

- Provide a solution that meets the requirements of this challenge. The implementation should be in Python and expose a REST API for interaction. Bonus points if it comes with a simple frontend interface to interact with the

API.

- Ensure the solution is production-ready, adhering to best practices for software development and deployment.

- The solution should be containerized.

- Include a README file with detailed instructions on how to run the solution.

- Provide a small technical report explaining the approach taken, the challenges faced, and the solutions implemented.

