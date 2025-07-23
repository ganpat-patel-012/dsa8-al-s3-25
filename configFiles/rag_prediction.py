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


def load_data():
    

    df = pd.read_csv("liar_dataset/test.tsv", sep='\t', header=None, names=[
        "id", "label", "statement", "subject", "speaker", "speaker_title",
        "speaker_state", "speaker_party", "before_true", "before_false",
        "before_barely_true", "before_half_true", "before_mostly_true",
        "before_pants_on_fire", "context"
    ])

    label_map = {
        "true": "true", "mostly-true": "true", "half-true": "true",
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
            f"""Statement: {doc.page_content}
URL: {doc.metadata.get('url')}
Source Claim: {doc.metadata.get('statement')}
Probability Score: {doc.metadata.get('probability')}"""
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
        prompt = PromptTemplate.from_template("""You are a fake news detector. Based on the following examples and metadata, classify the new claim with a 4-line justification.

Examples:
{context}

{claim_info}
New Claim:
"{query}"

Answer only 'True' or 'False' with a brief justification in single string line.
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
                "url": row.get("url", ""),
                "probability": row.get("probability", ""),
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


def predict_claim_truth(claim, justification):
    tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
    model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")

    inputs = tokenizer.encode_plus(claim, justification, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1).squeeze()
    return float(probs[0])  # entailment


def build_justification_dataframe(claim, justifications, probabilities):
    data = []
    for justification, prob in zip(justifications, probabilities):
        data.append({
            "claim": claim,
            "justification": justification,
            "probability": prob
        })
    return pd.DataFrame(data)



def main(web_df: pd.DataFrame, CLAIM: str):
    API_KEY = "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"

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
    probs_one = predict_claim_truth(CLAIM, justification_one)
    probs_two = predict_claim_truth(CLAIM, justification_two)

    # Build output DataFrame
    output_df = build_justification_dataframe(
        CLAIM,
        [justification_one, justification_two],
        [probs_one, probs_two]
    )

    print(output_df)
    return output_df

if __name__ == "__main__":
    web_df = pd.read_csv("liar_dataset/web_df.csv")
    CLAIM = "Democrats are cutting our school funding. Four times in the last 10 years before we came into office."
    main(web_df,CLAIM)
