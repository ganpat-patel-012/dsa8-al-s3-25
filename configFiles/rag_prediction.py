import pandas as pd
import uuid
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from configFiles.config import GroqAPI


def load_data():
    df = pd.read_csv("liar_dataset/train.tsv", sep='\t', header=None, names=[
        "id", "label", "statement", "subject", "speaker", "speaker_title",
        "speaker_state", "speaker_party", "before_true", "before_false",
        "before_barely_true", "before_half_true", "before_mostly_true",
        "before_pants_on_fire", "context"
    ])

    label_map = {
        "true": "true", "mostly-true": "true", "half-true": "false",
        "barely-true": "false", "false": "false", "pants-fire": "false"
    }
    df["binary_label"] = df["label"].map(label_map)
    df["statement_text"] = df["statement"]

    data = df[[
        "id", "label", "binary_label", "statement", "statement_text",
        "subject", "speaker", "speaker_title", "speaker_state", "speaker_party",
        "before_true", "before_false", "before_barely_true", "before_half_true",
        "before_mostly_true", "before_pants_on_fire", "context"
    ]]
    return data

def create_vector_db(data):
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    documents = [
        Document(
            page_content=row["statement"],
            metadata={
                "id": row["id"],
                "label": row["label"],
                "subject": row["subject"],
                "speaker": row["speaker"],
                "speaker_title": row["speaker_title"],
                "speaker_state": row["speaker_state"],
                "speaker_party": row["speaker_party"],
                "before_true": row["before_true"],
                "before_false": row["before_false"],
                "before_barely_true": row["before_barely_true"],
                "before_half_true": row["before_half_true"],
                "before_mostly_true": row["before_mostly_true"],
                "before_pants_on_fire": row["before_pants_on_fire"],
                "context": row["context"]
            }
        )
        for _, row in data.iterrows()
    ]
    vector_db = FAISS.from_documents(documents, embedding)
    vector_db.save_local("liar_vector_db")
    return embedding


def detect_fake_news(new_claim, embedding, vector_db_path, api_key, k=3, claim_metadata=None, df=None):
    vector_db = FAISS.load_local(
        folder_path=vector_db_path,
        embeddings=embedding,
        allow_dangerous_deserialization=True
    )

    llm = ChatGroq(api_key=api_key, model="llama3-8b-8192")

    def generate_response_from_db():
        docs = vector_db.similarity_search(new_claim, k=k)
        context = "\n\n".join([
            f"""Statement: {doc.page_content}"
Label: {doc.metadata.get('label')}
Speaker: {doc.metadata.get('speaker')} ({doc.metadata.get('speaker_party')} - {doc.metadata.get('speaker_state')})
Subject: {doc.metadata.get('subject')}
Prior Ratings: True={doc.metadata.get('before_true')}, False={doc.metadata.get('before_false')}, Half-True={doc.metadata.get('before_half_true')}, Pants-on-Fire={doc.metadata.get('before_pants_on_fire')}
Context: {doc.metadata.get('context')}"""
            for doc in docs
        ])
        claim_info = ""
        if claim_metadata:
            claim_info = f"""
                            New Claim Metadata:
                            Speaker: {claim_metadata.get('speaker')} ({claim_metadata.get('speaker_party')} - {claim_metadata.get('speaker_state')})
                            Subject: {claim_metadata.get('subject')}
                            Context: {claim_metadata.get('context')}
                            """
        prompt = PromptTemplate.from_template("""You are a fake news detector. Based on the following examples and metadata, provide a clear, concise, and fact-based justification (250 words) that directly supports or refutes the new claim. Focus on evidence relevant to the claim's truthfulness.

                Examples:
                {context}

                {claim_info}
                New Claim:
                "{query}"

                Provide a justification that directly addresses the claim's accuracy, citing specific evidence from the examples or metadata. Make sure if you mention label of the classification then its either true or false.
                """)
        
        query = prompt.format(context=context, claim_info=claim_info, query=new_claim)
        return llm.invoke(input=query).content

    response_before = generate_response_from_db()
    added_ids = []

    if df is not None:
        new_documents = []
        for _, row in df.iterrows():
            doc_id = str(uuid.uuid4())
            metadata = {
                "statement": row.get("statement", ""),
                # Do not include url or probability
                "doc_id": doc_id
            }
            new_documents.append(Document(
                page_content=row.get("content_summary", ""),
                metadata=metadata
            ))
            added_ids.append(doc_id)

        vector_db.add_documents(new_documents, ids=added_ids)
        vector_db.save_local(vector_db_path)

    response_after = generate_response_from_db()

    if added_ids:
        vector_db.delete(added_ids)
        vector_db.save_local(vector_db_path)

    return [response_before, response_after]


# Remove RoBERTa NLI scoring and replace with LLM-based True/False classification

def classify_claim_with_llm(claim, justification, llm):
    prompt = f"""Claim: {claim}\nJustification: {justification}\nBased on the justification, is the claim True or False? Answer with only 'True' or 'False'."""
    response = llm.invoke(input=prompt).content.strip()
    if response not in ["True", "False"]:
        response = "Unknown"
    return response


def build_justification_dataframe(claim, justifications, llm_labels):
    data = []
    for justification, label in zip(justifications, llm_labels):
        data.append({
            "claim": claim,
            "justification": justification,
            "llm_label": label
        })
    return pd.DataFrame(data)


def main(web_df: pd.DataFrame, CLAIM: str):
    API_KEY = GroqAPI

    # Load data
    data_df = load_data()

    # Only create vector DB if not already present
    vector_db_path = "liar_vector_db"
    embedding = create_vector_db(data_df)

    # Run fake news detection
    result = detect_fake_news(
        new_claim=CLAIM,
        embedding=embedding,
        vector_db_path=vector_db_path,
        api_key=API_KEY,
        k=3,
        claim_metadata=None,
        df=web_df
    )

    # Evaluate justifications
    justification_one, justification_two = result
    llm = ChatGroq(api_key=API_KEY, model="llama3-8b-8192")
    label_one = classify_claim_with_llm(CLAIM, justification_one, llm)
    label_two = classify_claim_with_llm(CLAIM, justification_two, llm)

    # Build output DataFrame
    output_df = build_justification_dataframe(
        CLAIM,
        [justification_one, justification_two],
        [label_one, label_two]
    )

    print(output_df)
    return output_df

# if __name__ == "__main__":
#     web_df = pd.read_csv("liar_dataset/web_df.csv")
#     CLAIM = "Democrats are cutting our school funding. Four times in the last 10 years before we came into office."
#     main(web_df,CLAIM)
