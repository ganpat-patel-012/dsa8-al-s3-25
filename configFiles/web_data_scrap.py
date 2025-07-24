import requests
from bs4 import BeautifulSoup
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import logging
import time
import re
from nltk.tokenize import sent_tokenize
from collections import Counter
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from configFiles.config import GroqAPI

# Set up logging for web scraping
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize BERT model and tokenizer for evidence evaluation
evidence_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
evidence_classifier = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
evidence_classifier.eval()

def summarize_web_evidence(statement, web_content, max_sentences=5, max_length=2000):
    """Summarize scraped web content relevant to the input statement using ChatGroq."""
    try:
        api_key = GroqAPI  # Replace with your actual Groq API key
        llm = ChatGroq(api_key=api_key, model="llama3-8b-8192")
        prompt_template = PromptTemplate.from_template(
            """Given the following statement: '{statement}'
        And the following web content: '{web_content}'
        Summarize the web content in up to {max_sentences} sentences, including only information directly relevant to the statement. 
        Exclude irrelevant details and ensure the summary is coherent, concise, and avoids gibberish. 
        Limit the summary to {max_length} characters. If no relevant information is found, return 'No relevant evidence summary found'."""
        )
        prompt = prompt_template.format(
            statement=statement,
            web_content=web_content[:10000],
            max_sentences=max_sentences,
            max_length=max_length
        )
        response = llm.invoke(input=prompt)
        evidence_summary = response.content.strip()

        if len(evidence_summary) > max_length:
            sentences = sent_tokenize(evidence_summary)
            truncated_summary = ""
            current_length = 0
            for sentence in sentences:
                if current_length + len(sentence) <= max_length:
                    truncated_summary += sentence + " "
                    current_length += len(sentence) + 1
                else:
                    break
            evidence_summary = truncated_summary.strip() or "No relevant evidence summary found"

        if not evidence_summary or evidence_summary == "No relevant evidence summary found":
            logger.warning("ChatGroq returned empty or irrelevant summary")
            return fallback_summarize_web_evidence(statement, web_content, max_sentences, max_length)

        logger.info(f"Generated evidence summary for statement '{statement[:60]}...': {evidence_summary[:100]}...")
        return evidence_summary

    except Exception as e:
        logger.error(f"Error summarizing web evidence: {e}")
        return fallback_summarize_web_evidence(statement, web_content, max_sentences, max_length)

def fallback_summarize_web_evidence(statement, web_content, max_sentences=5, max_length=2000):
    """Fallback heuristic-based summarization of web evidence if generative AI fails."""
    try:
        sentences = sent_tokenize(web_content)
        if not sentences:
            logger.warning("No sentences found in web content")
            return "No relevant evidence summary found"

        stop_words = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'and', 'or', 'for', 'with', 'by', 'from', 'of'}
        statement_keywords = set(re.findall(r'\b\w+\b', statement.lower())) - stop_words
        if not statement_keywords:
            logger.warning("No meaningful keywords extracted from statement")
            return "No relevant evidence summary found"

        scored_sentences = []
        for sentence in sentences:
            sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
            if not sentence_words:
                continue
            overlap = len(statement_keywords.intersection(sentence_words))
            word_counts = Counter(sentence_words)
            repetition_score = sum(count > 2 for count in word_counts.values())
            sentence_length = len(sentence_words)
            is_valid = sentence_length > 3 and repetition_score == 0
            if overlap > 0 and is_valid:
                scored_sentences.append((sentence, overlap))

        scored_sentences = sorted(scored_sentences, key=lambda x: x[1], reverse=True)[:max_sentences]
        summary_sentences = [s[0] for s in scored_sentences]

        if not summary_sentences:
            logger.warning("No relevant sentences found for evidence summary")
            return "No relevant evidence summary found"

        evidence_summary = " ".join(summary_sentences).strip()
        if len(evidence_summary) > max_length:
            truncated_summary = ""
            current_length = 0
            for sentence in summary_sentences:
                if current_length + len(sentence) <= max_length:
                    truncated_summary += sentence + " "
                    current_length += len(sentence) + 1
                else:
                    break
            evidence_summary = truncated_summary.strip()

        if not evidence_summary:
            logger.warning("Evidence summary is empty after processing")
            return "No relevant evidence summary found"

        logger.info(f"Fallback evidence summary for statement '{statement[:60]}...': {evidence_summary[:100]}...")
        return evidence_summary

    except Exception as e:
        logger.error(f"Error in fallback evidence summarization: {e}")
        return "No relevant evidence summary found"

def scrape_web_page(url):
    """Scrape text content from a web page."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        text_elements = soup.find_all(["p", "div", "article", "span"])
        scraped_content = " ".join([elem.get_text(strip=True) for elem in text_elements if elem.get_text(strip=True)])
        if not scraped_content:
            logger.warning(f"No content scraped from {url}")
            return "No content scraped"
        return scraped_content[:10000]
    except requests.exceptions.RequestException as e:
        logger.error(f"Error scraping {url}: {e}")
        return f"Error scraping content: {e}"
    except Exception as e:
        logger.error(f"Unexpected error for {url}: {e}")
        return f"Error scraping content: {e}"

def search_web_for_evidence(query, max_results=10):
    """Search the web for evidence related to the query."""
    try:
        from googlesearch import search
        urls = list(search(query, num_results=max_results * 2))
        evidence_results = []
        skipped = 0
        for url in urls:
            if len(evidence_results) >= max_results:
                break
            if not url.startswith(("http://", "https://")):
                logger.warning(f"Skipping invalid URL: {url}")
                skipped += 1
                continue
            logger.info(f"Scraping evidence from {url}")
            scraped_content = scrape_web_page(url)
            if scraped_content.startswith("Error") or scraped_content == "No content scraped":
                logger.info(f"Skipping {url} due to {'error' if scraped_content.startswith('Error') else 'empty content'}")
                skipped += 1
            else:
                evidence_results.append({"url": url, "scraped_content": scraped_content})
            time.sleep(1)
        logger.info(f"Skipped {skipped} URLs, scraped {len(evidence_results)} valid results for query: {query[:60]}...")
        return evidence_results
    except Exception as e:
        logger.error(f"Web search error for query '{query[:60]}...': {e}")
        return [{"url": None, "scraped_content": f"Search error: {e}"}]

def evaluate_evidence_relevance(statement, evidence_data):
    """Evaluate the relevance of scraped evidence using BERT."""
    evidence_summaries = [
        summarize_web_evidence(statement, item["scraped_content"])
        for item in evidence_data
        if not item["scraped_content"].startswith("Error") and item["scraped_content"] != "No content scraped"
    ]

    if not evidence_summaries:
        logger.warning(f"No valid evidence summaries for statement: {statement[:60]}...")
        return [], []

    relevance_scores = []
    valid_summaries = []
    for summary in evidence_summaries:
        if summary == "No relevant evidence summary found":
            continue
        input_text = f"[CLS] {statement} [SEP] {summary} [SEP]"
        inputs = evidence_tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = evidence_classifier(**inputs)
            scores = torch.softmax(outputs.logits, dim=1).detach().numpy()[0]
            relevance_score = float(scores[1])
            relevance_scores.append(relevance_score)
            valid_summaries.append(summary)

    logger.info(f"Evidence relevance scores: {relevance_scores}")
    return relevance_scores, valid_summaries

def scrape_web_evidence(statement, max_results=10):
    """
    Scrape web evidence for a given statement and return a DataFrame with results.
    
    Args:
        statement (str): The statement to find web evidence for
        max_results (int): Maximum number of web pages to scrape (default: 10)
    
    Returns:
        pandas.DataFrame: DataFrame containing web evidence with columns:
            - statement
            - url
            - scraped_content
            - evidence_summary
            - relevance_score
    """
    logger.info(f"Scraping web evidence for statement: {statement[:60]}...")
    
    # Search for web evidence
    web_evidence = search_web_for_evidence(statement, max_results)
    
    # Evaluate evidence relevance
    evidence_data = []
    if web_evidence:
        relevance_scores, evidence_summaries = evaluate_evidence_relevance(statement, web_evidence)
    else:
        logger.warning(f"No web evidence found for statement: {statement[:60]}...")
        relevance_scores, evidence_summaries = [], []

    # Compile evidence data
    valid_evidence_index = 0
    for idx, evidence in enumerate(web_evidence):
        original_content = evidence["scraped_content"]
        summary = evidence_summaries[idx] if idx < len(evidence_summaries) and evidence_summaries[idx] not in ["Error", "No content scraped", "No relevant evidence summary found"] else original_content
        score = relevance_scores[valid_evidence_index] if valid_evidence_index < len(relevance_scores) and summary != "No relevant evidence summary found" else None
        evidence_data.append({
            "statement": statement,
            "url": evidence["url"],
            "scraped_content": original_content,
            "evidence_summary": summary,
            "relevance_score": score
        })
        if summary != "No relevant evidence summary found" and not original_content.startswith("Error") and original_content != "No content scraped":
            valid_evidence_index += 1

    evidence_df = pd.DataFrame(evidence_data)
    return evidence_df

# if __name__ == "__main__":
#     # Example usage
#     test_statement = "One man sacrificed for his country. One man opposed a flawed war."
#     evidence_df = scrape_web_evidence(test_statement)
#     print(evidence_df)