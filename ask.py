import Initialize
from link_agent import LinkRetrieverAgent

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PDFPlumberLoader


def main():
    # Paths from Initialize.py
    chroma_path = Initialize.db_path

    # Must match the embeddings used when building the DB
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

    # Load the persisted Chroma DB
    vectordb = Chroma(
        persist_directory=chroma_path,
        embedding_function=embeddings
    )

    agent = LinkRetrieverAgent(
        linked_docs_dir=r"C:\Projects\Hakathon\linked_docs",  # optional folder for local linked PDFs
        state_dir=r"C:\Projects\Hakathon\.agent_state",       # cache folder
        max_new_docs_per_question=2
    )

    #retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    #retriever = vectordb.as_retriever(search_kwargs={"k": 2})
    retriever = vectordb.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
        "score_threshold": 0.7,
        "k": 3
    }
)

    # Local LLM via Ollama (no key, no cost per call)
    #llm = ChatOllama(model="llama3.1", temperature=0)
    llm = ChatOllama(model="gemma3:4b", temperature=0)

    prompt1 = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. Answer ONLY using the provided context. "
         "If the answer is not in the context, say 'I don't know based on the document.' "
         "Be concise and clear."),
        ("human",
         "Question: {question}\n\n"
         "Context:\n{context}\n\n"
         "Answer:")
    ])


    prompt = ChatPromptTemplate.from_messages([
        ("system",
        "You are a medical assistant. Answer ONLY using the provided context. "
        "Answer in ENGLISH. Do not summarise the document. "
        "If the answer is not in the context, say: "
        "'I don't know based on the document.'"),
        ("human",
        "Question: {question}\n\nContext:\n{context}\n\nAnswer:")
    ])

    print(f"✅ Loaded Chroma DB from: {chroma_path}")
    print("Type a question (or type 'exit')\n")

    while True:
        question = input("❓ Question: ").strip()
        if question.lower() in {"exit", "quit"}:
            break

        #docs = retriever.invoke(question)

        docs = vectordb.similarity_search_with_relevance_scores(question, k=6)

        # keep only docs above threshold
        threshold = 0.5
        filtered = [d for d, score in docs if score >= threshold]

        if not filtered:
        # fallback to top-k without threshold
            filtered = [d for d, score in docs[:3]]

        docs = filtered

        # 🔗 Expand context by following links found in retrieved chunks
        actions = agent.expand_and_ingest(docs, vectordb)
        for a in actions:
            print(a)

        # If we ingested anything, retrieve again (now DB includes linked docs)
        if any(a.startswith("✅") for a in actions):
            # re-run retrieval so the new linked docs can be used
            scored = vectordb.similarity_search_with_relevance_scores(question, k=6)

            threshold = 0.5
            filtered2 = [d for d, score in scored if score >= threshold]
            if not filtered2:
                filtered2 = [d for d, score in scored[:3]]

            docs = filtered2

        context = "\n\n---\n\n".join(
            [f"(Page: {d.metadata.get('page', 'N/A')})\n{d.page_content}" for d in docs]
            )

        chain = prompt | llm
        answer = chain.invoke({"question": question, "context": context})

        print("\n✅ Answer:\n")
        print(answer.content)


        if "don't know" not in answer.content.lower():
            print("\n📌 Sources used:")
            for i, d in enumerate(docs, start=1):
                print(f"{i}. Page: {d.metadata.get('page', 'N/A')}")

        #print("\n📌 Sources used:")
        #for i, d in enumerate(docs, start=1):
        #    print(f"  {i}. Page: {d.metadata.get('page', 'N/A')}")
        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()