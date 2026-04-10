import argparse
import json
import os
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
import math
from collections import Counter

from tqdm.asyncio import tqdm_asyncio
import logging

# Import our new LLM client
from llm_clients import SalesLLMClientManager
from openai import OpenAI

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="User Likeness Evaluation")

    # Target Model (User Simulator) arguments
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to evaluate")
    parser.add_argument("--api_end_point", type=str, required=True, help="API endpoint")
    parser.add_argument("--api_key", type=str, required=True, help="API key")
    parser.add_argument("--api_type", type=str, choices=["openai", "azure", "other"], default="openai")
    parser.add_argument("--api_version", type=str, help="Azure API version")

    # Execution settings
    parser.add_argument("--max_workers", type=int, default=1, help="Maximum number of concurrent workers")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.99)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--reasoning_effort", type=str, default=None)

    # Input/Output
    parser.add_argument("--input_files", nargs='+', required=True, help="List of input JSONL files")
    parser.add_argument("--output_file", type=str, default="user_likeness_results.jsonl", help="Output JSONL file")
    parser.add_argument("--max_samples", type=int, default=-1, help="Max samples per file")
    parser.add_argument("--language", type=str, default="zh", help="Language filter if needed")

    # Embedding settings
    parser.add_argument("--embed_api_base", type=str, default="http://YOUR_EMBED_API_ENDPOINT")
    parser.add_argument("--embed_api_key", type=str, default="YOUR_API_KEY")
    parser.add_argument("--embed_model", type=str, default="BAAI/bge-m3")

    args = parser.parse_args()
    return args

def _ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def _modified_precision(cand, ref, n):
    cand_ngrams = Counter(_ngrams(cand, n))
    ref_ngrams = Counter(_ngrams(ref, n))
    overlap = 0
    total = 0
    for k, v in cand_ngrams.items():
        total += v
        overlap += min(v, ref_ngrams.get(k, 0))
    if total == 0:
        if len(cand) < n:
            return 1.0
        return 0.0
    return (overlap + 1.0) / (total + 1.0)

def _brevity_penalty(c, r):
    if c == 0:
        return 0.0
    if c > r:
        return 1.0
    return math.exp(1 - float(r) / float(c))

def _bleu_4(ref, cand):
    weights = [0.25, 0.25, 0.25, 0.25]
    precisions = []
    for n in range(1, 5):
        precisions.append(_modified_precision(cand, ref, n))
    log_sum = 0.0
    for w, p in zip(weights, precisions):
        if p == 0:
            return 0.0
        log_sum += w * math.log(p)
    bp = _brevity_penalty(len(cand), len(ref))
    return bp * math.exp(log_sum)

def _rouge_n_f1(ref, cand, n):
    cand_ngrams = Counter(_ngrams(cand, n))
    ref_ngrams = Counter(_ngrams(ref, n))
    overlap = 0
    for k, v in cand_ngrams.items():
        overlap += min(v, ref_ngrams.get(k, 0))
    p = overlap / sum(cand_ngrams.values()) if sum(cand_ngrams.values()) > 0 else 0.0
    r = overlap / sum(ref_ngrams.values()) if sum(ref_ngrams.values()) > 0 else 0.0
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)

def _lcs(a, b):
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            if ai == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = dp[i - 1][j] if dp[i - 1][j] >= dp[i][j - 1] else dp[i][j - 1]
    return dp[n][m]

def _rouge_l_f1(ref, cand):
    lcs = _lcs(ref, cand)
    p = lcs / len(cand) if len(cand) > 0 else 0.0
    r = lcs / len(ref) if len(ref) > 0 else 0.0
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)

def _expand_contractions(s: str) -> str:
    mapping = {
        "i'm": "i am", "you're": "you are", "he's": "he is", "she's": "she is", "it's": "it is", "we're": "we are", "they're": "they are",
        "i've": "i have", "you've": "you have", "we've": "we have", "they've": "they have",
        "can't": "can not", "don't": "do not", "doesn't": "does not", "didn't": "did not", "won't": "will not", "wouldn't": "would not", "shouldn't": "should not", "couldn't": "could not",
        "isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
        "i'll": "i will", "you'll": "you will", "he'll": "he will", "she'll": "she will", "it'll": "it will", "we'll": "we will", "they'll": "they will",
        "i'd": "i would", "you'd": "you would", "he'd": "he would", "she'd": "she would", "it'd": "it would", "we'd": "we would", "they'd": "they would"
    }
    for k, v in mapping.items():
        s = s.replace(k, v)
    return s

def create_embedding_model_fn(api_base: str, api_key: str, model: str) -> Callable[[str], List[float]]:
    client = OpenAI(api_key=api_key, base_url=api_base)
    def embedding_model_fn(text: str) -> List[float]:
        response = client.embeddings.create(model=model, input=[text])
        return response.data[0].embedding
    return embedding_model_fn

def _cosine_vec(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    s = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        s += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    import math
    return s / (math.sqrt(na) * math.sqrt(nb))

def calculate_metrics(text1: str, text2: str, embed_fn: Optional[Callable[[str], List[float]]] = None) -> Dict[str, float]:
    """
    Calculate comprehensive similarity metrics.
    Includes:
    - Difflib Ratio: Standard string matching.
    - Jaccard Similarity: Character set overlap.
    - Cosine Similarity: Character N-Gram vector similarity.
    - BLEU-2: Bilingual Evaluation Understudy (Precision-oriented).
    - ROUGE-L: Recall-Oriented Understudy for Gisting Evaluation (Recall-oriented).
    """
    metrics = {}
    
    import re
    text1_lower = text1.lower()
    text2_lower = text2.lower()
    text1_clean = re.sub(r'[^\w\u4e00-\u9fff\s\']', '', text1_lower)
    text2_clean = re.sub(r'[^\w\u4e00-\u9fff\s\']', '', text2_lower)
    has_chinese = bool(re.search(r'[\u4e00-\u9fff]', text1_clean))
    if not has_chinese:
        text1_clean = _expand_contractions(text1_clean)
        text2_clean = _expand_contractions(text2_clean)
    text1_nospace = re.sub(r'[\s\']+', '', text1_clean)
    text2_nospace = re.sub(r'[\s\']+', '', text2_clean)

    # Tokenize (Adaptive)
    
    if has_chinese:
        # Chinese: Character-level tokenization
        tokens1 = list(text1_nospace)
        tokens2 = list(text2_nospace)
    else:
        # English: Word-level tokenization
        tokens1 = text1_clean.split()
        tokens2 = text2_clean.split()

    tokens_ref = tokens1
    tokens_cand = tokens2
    if not tokens_ref or not tokens_cand:
        return {"bleu-4": 0.0, "rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}
    metrics["bleu-4"] = _bleu_4(tokens_ref, tokens_cand)
    metrics["rouge-1"] = _rouge_n_f1(tokens_ref, tokens_cand, 1)
    metrics["rouge-2"] = _rouge_n_f1(tokens_ref, tokens_cand, 2)
    metrics["rouge-l"] = _rouge_l_f1(tokens_ref, tokens_cand)

    if embed_fn is not None:
        try:
            e1 = embed_fn(text1)
            e2 = embed_fn(text2)
            metrics["embedding"] = _cosine_vec(e1, e2)
        except Exception as e:
            logger.warning(f"Embedding calculation error: {e}")
            metrics["embedding"] = 0.0

    return metrics

async def process_conversation(
    client_manager: SalesLLMClientManager,
    conversation: Dict[str, Any],
    args: argparse.Namespace,
    embed_fn: Optional[Callable[[str], List[float]]] = None
) -> List[Dict[str, Any]]:
    """
    Process a single conversation:
    1. Iterate through turns.
    2. Identify turns where the real user (Assistant in JSONL) spoke.
    3. Generate model response for the same context.
    4. Record comparison.
    """
    messages = conversation.get("messages", [])
    if not messages:
        return []

    results = []
    
    # We build the history as we go
    history = []
    
    # Extract system prompt if present
    if messages[0]['role'] == 'system':
        history.append(messages[0])
        start_idx = 1
    else:
        start_idx = 0

    for i in range(start_idx, len(messages)):
        current_msg = messages[i]
        role = current_msg['role']
        content = current_msg['content']
        
        # In the dataset: 
        # 'system' -> Instructions
        # 'assistant' -> User (Customer) Response
        # 'user' -> Sales Agent Response
        
        # We want to predict 'assistant' turns based on history.
        if role == 'assistant':
            # FILTER: Only evaluate if the Ground Truth (real user response) length > 5
            # This filters out short/meaningless responses like "嗯" or "好的".
            if len(content) <= 5:
                # Add to history but skip evaluation
                history.append(current_msg)
                continue

            # It's the user's turn to speak. 
            # We use the current history (which ends with a 'user'/Sales Agent turn or is just System)
            # to generate a response.
            
            # Generate response
            generated_content = await client_manager.clients["target_model"].chat_completion_async(
                messages=history,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                reasoning_effort=args.reasoning_effort
            )
            
            if generated_content:
                metrics = calculate_metrics(content, generated_content, embed_fn=embed_fn)
                
                result_entry = {
                    "history_length": len(history),
                    "last_message": history[-1]['content'] if history else "",
                    "real_response": content,
                    "generated_response": generated_content,
                    "metrics": metrics,
                    # Placeholder for colloquialism (requires advanced judging)
                    "colloquialism_check": "manual_review_needed"
                }
                results.append(result_entry)
        
        # Add current message to history for next turns
        history.append(current_msg)
        
    return results

async def main():
    args = parse_args()
    
    # Initialize Client Manager
    client_manager = SalesLLMClientManager()
    client_manager.add_client(
        client_name="target_model",
        api_base=args.api_end_point,
        api_key=args.api_key,
        model_name=args.model_name,
        api_type=args.api_type,
        api_version=args.api_version
    )
    
    all_results = []
    tasks = []

    embed_fn = None
    if args.embed_api_base and args.embed_model:
        try:
            embed_fn = create_embedding_model_fn(
                api_base=args.embed_api_base,
                api_key=args.embed_api_key,
                model=args.embed_model
            )
        except Exception as e:
            logger.warning(f"Failed to initialize embedding model: {e}")
    
    # Read all input files
    conversations = []
    for file_path in args.input_files:
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            continue
            
        logger.info(f"Reading file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        conversations.append(json.loads(line))
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")

    if args.max_samples > 0:
        conversations = conversations[:args.max_samples]
    
    logger.info(f"Total conversations to process: {len(conversations)}")

    # Process conversations concurrently
    sem = asyncio.Semaphore(args.max_workers)
    
    async def sem_process(conv):
        async with sem:
            return await process_conversation(client_manager, conv, args, embed_fn)

    tasks = [sem_process(conv) for conv in conversations]
    
    # Run tasks with progress bar
    results_list = await tqdm_asyncio.gather(*tasks)
    
    # Flatten results
    for res in results_list:
        all_results.extend(res)
        
    # Calculate and print summary statistics
    if all_results:
        summary = {
            "total_samples": len(all_results),
            "bleu-4": 0.0,
            "rouge-1": 0.0,
            "rouge-2": 0.0,
            "rouge-l": 0.0,
            "embedding": 0.0
        }
        
        for item in all_results:
            metrics = item.get("metrics", {})
            summary["bleu-4"] += metrics.get("bleu-4", 0.0)
            summary["rouge-1"] += metrics.get("rouge-1", 0.0)
            summary["rouge-2"] += metrics.get("rouge-2", 0.0)
            summary["rouge-l"] += metrics.get("rouge-l", 0.0)
            summary["embedding"] += metrics.get("embedding", 0.0)
            
        # Average
        for key in ["bleu-4", "rouge-1", "rouge-2", "rouge-l", "embedding"]:
            summary[key] /= len(all_results)
            
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Total Samples: {summary['total_samples']}")
        print(f"BLEU-4:   {summary['bleu-4']:.4f}")
        print(f"ROUGE-1:  {summary['rouge-1']:.4f}")
        print(f"ROUGE-2:  {summary['rouge-2']:.4f}")
        print(f"ROUGE-L:  {summary['rouge-l']:.4f}")
        print(f"EMBED:    {summary['embedding']:.4f}")
        print("="*50 + "\n")
    else:
        logger.warning("No results generated to summarize.")

    # Save results
    logger.info(f"Saving {len(all_results)} comparison samples to {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for item in all_results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    asyncio.run(main())
