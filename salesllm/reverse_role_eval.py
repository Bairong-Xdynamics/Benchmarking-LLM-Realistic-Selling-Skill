import argparse
import json
import os
import time
import asyncio
import concurrent.futures
from datetime import datetime
from typing import List, Dict, Any
import re
import glob

from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

import logging

# Import our new LLM client
from llm_clients import SalesLLMClientManager, create_system_message, create_user_message, create_assistant_message

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reverse Role Evaluation - Detect when User model acts as Assistant")

    # Model arguments
    parser.add_argument("--model_name", type=str, required=True, help="Model to use for detection")
    parser.add_argument("--api_end_point", type=str, required=True)
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--api_type", type=str, choices=["openai", "azure", "other"], default="openai")
    parser.add_argument("--api_version", type=str, help="Azure API version")

    # Input/Output
    parser.add_argument("--input_files", type=str, nargs='+', required=True, help="Input JSONL files or directories (supports glob patterns)")
    parser.add_argument("--output_dir", type=str, default="./reverse_role_results", help="Output directory")
    
    # Execution settings
    parser.add_argument("--execution_mode", type=str, choices=["sync", "async", "concurrent"], default="concurrent",
                       help="Execution mode: sync (sequential), async (asyncio), concurrent (thread pool)")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of concurrent workers")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for processing")
    
    # Model parameters
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_tokens", type=int, default=512)
    
    # Detection sensitivity
    parser.add_argument("--min_turns", type=int, default=2, 
                       help="Minimum turns required for valid conversation")
    
    # Extra parameters
    parser.add_argument("--extra_body", type=str, default=None, help="JSON string for extra API parameters")

    args = parser.parse_args()
    return args

def expand_file_paths(paths: List[str]) -> List[str]:
    """Expand directories and glob patterns."""
    all_files = []
    for path in paths:
        if os.path.isdir(path):
            # If directory, find all .jsonl files recursively
            # Skip files starting with intermediate_results or stats_
            candidates = glob.glob(os.path.join(path, "**/*.jsonl"), recursive=True)
            for c in candidates:
                basename = os.path.basename(c)
                if not basename.startswith("intermediate_results") and not basename.startswith("stats_"):
                    all_files.append(c)
        elif '*' in path or '?' in path:
            # Glob pattern
            all_files.extend(glob.glob(path, recursive=True))
        else:
            # Single file
            if os.path.exists(path):
                all_files.append(path)
            else:
                logger.warning(f"File not found: {path}")
    
    # Deduplicate
    return sorted(list(set(all_files)))

def detect_language_from_path(file_path: str) -> str:
    """Detect language from file path or name."""
    path_lower = file_path.lower()
    
    # Check directory structure first
    if '/zh/' in path_lower or '/zh_' in path_lower:
        return 'zh'
    if '/en/' in path_lower or '/en_' in path_lower:
        return 'en'
    
    # Check filename
    filename = os.path.basename(path_lower)
    if 'zh' in filename:
        return 'zh'
    if 'en' in filename:
        return 'en'
        
    # Default to en if uncertain (can be improved by checking content)
    return 'en'

def read_conversations(input_path: str) -> List[Dict[str, Any]]:
    """Read conversations from JSONL file."""
    conversations = []
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        conv = json.loads(line)
                        conversations.append(conv)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON line {line_num} in {input_path}: {e}")
                        continue
    except Exception as e:
        logger.error(f"Error reading file {input_path}: {e}")
        return []
        
    logger.info(f"Successfully loaded {len(conversations)} conversations from {input_path}")
    return conversations

def create_role_detection_prompt(messages: List[Dict[str, Any]], language: str) -> str:
    """Create prompt for LLM-based role reversion detection."""
    
    conversation_text = ""
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        conversation_text += f"Turn {i+1} [{role.upper()}]: {content}\n"
    
    if language == "zh":
        prompt = f"""请分析以下销售对话，判断用户(user)是否错误地扮演了助理(assistant)的角色。

对话内容：
{conversation_text}

你的任务：
检查 User 的每一条发言，判断其是否出现了"角色反转"（即 User 抢了 Assistant 的台词，开始推销产品、提供服务、或者表现得像个客服/销售）。

判定标准：
1. **角色反转**：User 主动介绍产品功能、提供报价、安抚客户情绪、询问"有什么可以帮您"等明显属于销售/客服的行为。
2. **正常互动**：User 只是提问、表达需求、讨价还价、拒绝或接受，这些都属于正常 User 行为。

请输出以下格式的 JSON 结果：
{{
    "detected": true/false,
    "inverted_turn_indices": [1, 3],
    "severity": "none/low/medium/high", 
    "reason": "简要说明判断理由",
    "examples": ["User说的错误语句1", "User说的错误语句2"]
}}

注意：
- inverted_turn_indices 应该包含所有出现角色反转的 Turn 编号（对应对话内容中的 Turn X）。如果没有，请返回空列表 []。
- 只要检测到 User 有明显的销售/客服行为，detected 设为 true。
- 仅输出 JSON，不要包含 Markdown 格式标记或其他文字。"""

    else:  # English
        prompt = f"""Please analyze the following sales conversation and determine if the USER incorrectly acted as the ASSISTANT.

Conversation:
{conversation_text}

Your Task:
Check EACH USER message to see if they exhibited "Role Reversal" (i.e., the USER started pitching products, offering services, or acting like a customer service agent/salesperson).

Criteria:
1. **Role Reversal**: The USER proactively explains product features, offers quotes, appeases the customer, or asks "How can I help you?" - behaviors clearly belonging to a sales/support role.
2. **Normal Interaction**: The USER asks questions, states needs, bargains, refuses, or accepts. These are normal USER behaviors.

Please output JSON in this format:
{{
    "detected": true/false,
    "inverted_turn_indices": [1, 3],
    "severity": "none/low/medium/high",
    "reason": "Brief reason for judgment",
    "examples": ["Incorrect statement by USER 1", "Incorrect statement by USER 2"]
}}

Note:
- 'inverted_turn_indices' should include ALL Turn numbers (corresponding to Turn X in the conversation) where Role Reversal occurred. Return empty list [] if none.
- Set 'detected' to true if ANY obvious sales/support behavior is found in USER messages.
- Output ONLY JSON. Do not include Markdown formatting or other text."""

    return prompt

async def detect_role_reversion_llm(messages: List[Dict[str, Any]], language: str, 
                                   client_manager: SalesLLMClientManager, model_name: str,
                                   temperature: float = 0.1, max_tokens: int = 512,
                                   extra_body: Dict[str, Any] = None) -> Dict[str, Any]:
    """LLM-based detection of role reversion."""
    
    try:
        prompt = create_role_detection_prompt(messages, language)
        messages_llm = [{"role": "user", "content": prompt}]
        
        # Prepare kwargs
        kwargs = {}
        if extra_body:
            kwargs["extra_body"] = extra_body

        response = await client_manager.chat_with_role_async(
            "detector",
            messages_llm,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        if not response:
            return {
                "detected": False,
                "inverted_turn_indices": [],
                "severity": "none",
                "reason": "LLM detection failed - no response",
                "examples": [],
                "method": "llm_failed"
            }
        
        # Clean up response (remove markdown code blocks if present)
        response = re.sub(r'^```json\s*', '', response)
        response = re.sub(r'^```\s*', '', response)
        response = re.sub(r'\s*```$', '', response)
        
        # Parse JSON response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group())
                result["method"] = "llm_based"
                return result
            except json.JSONDecodeError:
                pass
        
        # Fallback if JSON parsing fails
        return {
            "detected": False,
            "inverted_turn_indices": [],
            "severity": "none", 
            "reason": f"LLM detection failed - invalid JSON response: {response[:100]}",
            "examples": [],
            "method": "llm_failed"
        }
        
    except Exception as e:
        logger.error(f"LLM detection error: {str(e)}")
        return {
            "detected": False,
            "severity": "none",
            "reason": f"LLM detection error: {str(e)}",
            "examples": [],
            "method": "llm_error"
        }

class ReverseRoleEvaluator:
    """Main evaluator for detecting role reversions."""
    
    def __init__(self, args):
        self.args = args
        self.client_manager = SalesLLMClientManager()
        self.setup_client()
        
        # Parse extra_body if provided
        self.extra_body = None
        if self.args.extra_body:
            try:
                self.extra_body = json.loads(self.args.extra_body)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse extra_body JSON: {e}")
        
        # Statistics separated by language
        self.stats = {
            "zh": self._init_stats(),
            "en": self._init_stats(),
            "overall": self._init_stats()
        }
    
    def _init_stats(self):
        return {
            "total_conversations": 0,
            "conversations_with_reversion": 0,
            "high_severity": 0,
            "medium_severity": 0,
            "low_severity": 0,
            "llm_based_detections": 0,
            "total_user_turns": 0,
            "inverted_user_turns": 0
        }
    
    def setup_client(self):
        """Setup LLM client for detection."""
        self.client_manager.add_client(
            "detector",
            api_base=self.args.api_end_point,
            api_key=self.args.api_key,
            model_name=self.args.model_name,
            api_type=self.args.api_type,
            api_version=self.args.api_version,
            max_retries=3,
            retry_delay=1.0
        )
    
    def is_valid_conversation(self, messages: List[Dict[str, Any]]) -> bool:
        """Check if conversation has minimum required turns."""
        if len(messages) < self.args.min_turns:
            return False
        
        # Check if there's at least one user message and one assistant message
        has_user = any(msg.get("role") == "user" for msg in messages)
        has_assistant = any(msg.get("role") == "assistant" for msg in messages)
        
        return has_user and has_assistant
    
    def update_stats(self, lang: str, result: Dict[str, Any], total_user_turns: int):
        """Update statistics for specific language and overall."""
        inverted_indices = result.get("inverted_turn_indices", [])
        inverted_count = len(inverted_indices)
        
        for key in [lang, "overall"]:
            self.stats[key]["total_conversations"] += 1
            self.stats[key]["total_user_turns"] += total_user_turns
            self.stats[key]["inverted_user_turns"] += inverted_count
            
            if result.get("detected", False) or inverted_count > 0:
                self.stats[key]["conversations_with_reversion"] += 1
                
                severity = result.get("severity", "none")
                if severity == "high":
                    self.stats[key]["high_severity"] += 1
                elif severity == "medium":
                    self.stats[key]["medium_severity"] += 1
                elif severity == "low":
                    self.stats[key]["low_severity"] += 1
                
                self.stats[key]["llm_based_detections"] += 1

    async def evaluate_single_conversation(self, conversation: Dict[str, Any], lang: str) -> Dict[str, Any]:
        """Evaluate a single conversation for role reversions."""
        try:
            messages = conversation.get("messages", [])
            
            # Skip invalid conversations
            if not self.is_valid_conversation(messages):
                return {
                    **conversation,
                    "reversion_analysis": {
                        "detected": False,
                        "severity": "none",
                        "reason": "Invalid conversation - insufficient turns or missing roles",
                        "method": "skipped"
                    },
                    "status": "skipped"
                }
            
            # Calculate total user turns
            total_user_turns = sum(1 for msg in messages if msg.get("role") == "user")

            # Perform detection using ONLY LLM
            result = await detect_role_reversion_llm(messages, lang, self.client_manager,
                                                   self.args.model_name, self.args.temperature,
                                                   self.args.max_tokens, extra_body=self.extra_body)
            
            # Update statistics
            self.update_stats(lang, result, total_user_turns)
            
            return {
                **conversation,
                "reversion_analysis": result,
                "detected_language": lang,
                "status": "analyzed",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing conversation {conversation.get('id', 'unknown')}: {str(e)}")
            return {
                **conversation,
                "reversion_analysis": {
                    "detected": False,
                    "severity": "none",
                    "reason": f"Analysis error: {str(e)}",
                    "method": "error"
                },
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def process_file(self, file_path: str, output_dir: str):
        """Process a single file and save results."""
        # 1. Detect language
        lang = detect_language_from_path(file_path)
        logger.info(f"Processing file: {file_path}")
        logger.info(f"Detected Language: {lang}")
        
        # 2. Read conversations
        conversations = read_conversations(file_path)
        if not conversations:
            return
            
        # 3. Evaluate
        tasks = []
        for conv in conversations:
             tasks.append(self.evaluate_single_conversation(conv, lang))
        
        results = []
        if self.args.batch_size > 0:
            for i in range(0, len(tasks), self.args.batch_size):
                batch = tasks[i:i + self.args.batch_size]
                batch_results = await tqdm_asyncio.gather(*batch, 
                                                         desc=f"Processing {os.path.basename(file_path)} batch {i//self.args.batch_size + 1}")
                results.extend(batch_results)
        else:
            results = await tqdm_asyncio.gather(*tasks, desc=f"Processing {os.path.basename(file_path)}")
            
        # 4. Save file-specific results
        filename = os.path.basename(file_path)
        # Use a flat structure for results in the output dir, or preserve hierarchy? 
        # User asked for "save jsonl for each case". Let's put them in output_dir/lang/
        
        lang_output_dir = os.path.join(output_dir, lang)
        os.makedirs(lang_output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Keep original filename but prepend reverse_role_
        result_filename = f"reverse_role_{os.path.splitext(filename)[0]}.jsonl"
        result_path = os.path.join(lang_output_dir, result_filename)
        
        self.save_results(results, result_path)
    
    def save_results(self, results: List[Dict[str, Any]], output_path: str):
        """Save evaluation results to JSONL file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')
        
        logger.info(f"Results saved to {output_path}")
    
    def save_statistics(self, output_path: str):
        """Save evaluation statistics to JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Calculate percentages for each category
        for key in self.stats:
            if self.stats[key]["total_conversations"] > 0:
                self.stats[key]["reversion_rate"] = (self.stats[key]["conversations_with_reversion"] / self.stats[key]["total_conversations"]) * 100
                self.stats[key]["high_severity_rate"] = (self.stats[key]["high_severity"] / self.stats[key]["total_conversations"]) * 100
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Statistics saved to {output_path}")
        logger.info(f"Final statistics summary:\n{json.dumps(self.stats, indent=2, ensure_ascii=False)}")

async def main_async():
    args = parse_args()
    input_files = expand_file_paths(args.input_files)
    
    if not input_files:
        logger.error("No valid input files found!")
        return
    
    logger.info(f"Found {len(input_files)} files to process.")

    evaluator = ReverseRoleEvaluator(args)
    
    # Process all files
    for file_path in input_files:
        await evaluator.process_file(file_path, args.output_dir)
        
    # Save overall stats
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stats_path = os.path.join(args.output_dir, f"reverse_role_summary_stats_{timestamp}.json")
    evaluator.save_statistics(stats_path)

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
