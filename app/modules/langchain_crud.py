import os
import uuid
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_community.document_loaders import (
    TextLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader
)
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors.rankllm_rerank import RankLLMRerank
from app.config import config_settings

class LangchainDocManager:
    def __init__(self, pg_connection_str: str, collection_name: str):
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=config_settings.EMBEDDING_MODEL,
                google_api_key=config_settings.GEMINI_API_KEY
            )
            self.vectorstore = PGVector(
                embeddings=self.embeddings,
                connection=pg_connection_str,
                collection_name=collection_name,
                async_mode=True,
                # pre_delete_collection=True # Uncomment to wipe collection on init
            )
            self.llm = ChatGoogleGenerativeAI(
                model=config_settings.LLM_MODEL,
                api_key=config_settings.GEMINI_API_KEY
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LangchainDocManager: {e}")

    async def load_and_add_doc(self, file_path: str) -> List[dict]:
        """Asynchronously load and parse document, then store in PGVector."""
        try:
            ext = os.path.splitext(file_path)[-1].lower()
            if ext == ".txt":
                loader = TextLoader(file_path)
            elif ext == ".csv":
                loader = CSVLoader(file_path)
            elif ext == ".md":
                loader = UnstructuredMarkdownLoader(file_path)
            elif ext == ".docx":
                loader = UnstructuredWordDocumentLoader(file_path)
            else:
                raise ValueError(f"Unsupported file format: {ext}")

            docs = loader.load()
            if not docs:
                raise ValueError("Document loader returned no content")

            enriched_docs = []
            for doc in docs:
                doc_id = str(uuid.uuid4())
                enriched_docs.append(Document(
                    page_content=doc.page_content,
                    metadata={"id": doc_id, "source": os.path.basename(file_path)},
                    id=doc_id
                ))

            await self.vectorstore.aadd_documents(enriched_docs)
            return [doc.metadata for doc in enriched_docs]

        except Exception as e:
            raise RuntimeError(f"Failed to load and store document: {e}")

    async def list_documents(self) -> List[dict]:
        """Asynchronously list metadata of stored documents."""
        try:
            docs = await self.vectorstore.asimilarity_search("", k=100)
            return [doc.metadata for doc in docs]
        except Exception as e:
            raise RuntimeError(f"Failed to list documents: {e}")

    async def update_document(self, doc_id: str, new_content: str):
        """Asynchronously update document content."""
        try:
            if not new_content.strip():
                raise ValueError("New content is empty")
            doc = Document(page_content=new_content, metadata={"id": doc_id})
            await self.vectorstore.aupdate_document(doc_id, doc)
        except Exception as e:
            raise RuntimeError(f"Failed to update document {doc_id}: {e}")

    async def delete_document(self, doc_id: str):
        """Asynchronously delete a document by ID."""
        try:
            await self.vectorstore.adelete([doc_id])
        except Exception as e:
            raise RuntimeError(f"Failed to delete document {doc_id}: {e}")

    async def answer_query(self, query: str) -> str:
        """Answer query using retrieved context and Gemini."""
        try:
            if not query.strip():
                raise ValueError("Query is empty")

            docs = await self.vectorstore.asimilarity_search(query)
            context = "\n\n".join([doc.page_content for doc in docs]) if docs else "No relevant documents found."

            messages = [
                ("system", "Use the following context to answer."),
                ("user", f"Context:\n{context}\n\nQuestion: {query}")
            ]

            async for chunk in self.llm.astream(messages):
                return chunk.content

            raise RuntimeError("No response received from Gemini")
        except Exception as e:
            raise RuntimeError(f"Failed to answer query: {e}")

    async def wipe_vectorstore(self):
        """Asynchronously delete all documents."""
        try:
            list_documents = await self.list_documents()
            if not list_documents:
                return {"message": "No documents to delete."}
            await self.vectorstore.adelete([doc["id"] for doc in list_documents])
            return {"message": "Vector store wiped successfully."}
        except Exception as e:
            raise RuntimeError(f"Failed to wipe vector store: {e}")

    async def rerank_with_gemini(self, query: str, documents: List[Document], top_n: int = 5) -> List[Document]:
        """
        Rerank documents using Gemini based on their relevance to the query.
        :param query: The query string.
        :param documents: List of documents to rerank.
        :param top_n: Number of top documents to return after reranking.
        :return: List of reranked documents.
        """
        try:
            if not documents:
                raise ValueError("No documents provided for reranking.")

            # Prepare the input for Gemini
            context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(documents)])
            messages = [
                ("system", "Rerank the following documents based on their relevance to the query."),
                ("user", f"Query: {query}\n\nDocuments:\n{context}")
            ]

            # Get the reranked order from Gemini
            response = ""
            async for chunk in self.llm.astream(messages):
                response += chunk.content

            # Parse the response to extract the ranking
            try:
                # Extract document rankings from the response
                ranked_docs = []
                for i, doc in enumerate(documents):
                    if f"Document {i+1}:" in response and "relevant" in response.lower():
                        ranked_docs.append((i, doc))

                # Sort documents based on their relevance
                ranked_docs = sorted(ranked_docs, key=lambda x: response.index(f"Document {x[0]+1}"))
                reranked_docs = [doc for _, doc in ranked_docs]

                print("Reranked Documents:", reranked_docs)
                return reranked_docs[:top_n]
            except Exception:
                raise ValueError(f"Failed to parse Gemini response for reranking: {response}")

        except Exception as e:
            raise RuntimeError(f"Failed to rerank documents with Gemini: {e}")
