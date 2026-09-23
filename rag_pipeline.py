
from retriever import SemanticRetriever
from answer_generator import GroqAnswerGenerator


class RAGPipeline:

    def __init__(self):

        print("Loading semantic retriever...")
        self.retriever = SemanticRetriever()

        print("Loading Groq answer generator...")
        self.generator = GroqAnswerGenerator()

        print("RAG pipeline initialized successfully!")

    def ask(
        self,
        question: str,
        top_k: int = 5,
    ):

        if not question or not question.strip():
            return "Please enter a valid question."

        question = question.strip()

        # Step 1: Retrieve relevant chunks
        retrieved_chunks = self.retriever.search(
            query=question,
            top_k=top_k,
        )

        # Step 2: Generate an answer using Groq
        answer = self.generator.generate_answer(
            question=question,
            retrieved_chunks=retrieved_chunks,
        )

        return answer
