import argparse
import json
import os
import time
import asyncio
import concurrent.futures
from datetime import datetime
from typing import List, Dict, Any

from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

import logging

# Import our new LLM client
from llm_clients import SalesLLMClientManager, create_system_message, create_user_message, create_assistant_message

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SalesLLM Evaluation with Concurrent and Async Support")

    # Model arguments
    parser.add_argument("--assistant_model_name", type=str, required=True)
    parser.add_argument("--user_model_name", type=str, required=True)


    # API endpoints
    parser.add_argument("--assistant_API_end_point", type=str, required=True)
    parser.add_argument("--assistant_API_key", type=str, required=True)
    parser.add_argument("--user_API_end_point", type=str, required=True)
    parser.add_argument("--user_API_key", type=str, required=True)



    # API types
    parser.add_argument("--assistant_api_type", type=str, choices=["openai", "azure", "other"], default="openai")
    parser.add_argument("--user_api_type", type=str, choices=["openai", "azure", "other"], default="openai")

    
    # Azure specific
    parser.add_argument("--assistant_api_version", type=str, help="Azure API version for assistant")
    parser.add_argument("--user_api_version", type=str, help="Azure API version for user")


    # Local model settings
    parser.add_argument("--local_model_name_path", type=str, help="Path to local model for vLLM")
    parser.add_argument("--vllm_server_port", type=int, default=8000, help="Port for vLLM server")

    # Execution mode
    parser.add_argument("--execution_mode", type=str, choices=["sync", "async", "concurrent"], default="concurrent",
                       help="Execution mode: sync (sequential), async (asyncio), concurrent (thread pool)")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of concurrent workers")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing")

    # Model parameters
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--user_temperature", type=float, default=None, help="Temperature for user model")
    parser.add_argument("--assistant_temperature", type=float, default=None, help="Temperature for assistant model")

    parser.add_argument("--top_p", type=float, default=0.99)
    parser.add_argument("--max_tokens", type=int, default=2048)

    # Reasoning effort
    parser.add_argument("--reasoning_effort", type=str, default=None)
    parser.add_argument("--user_reasoning_effort", type=str, default=None)
    parser.add_argument("--assistant_reasoning_effort", type=str, default=None)

    # Extra body parameters (JSON string)
    parser.add_argument("--assistant_extra_body", type=str, default=None, help="JSON string for assistant extra_body")
    parser.add_argument("--user_extra_body", type=str, default=None, help="JSON string for user extra_body")

    # Evaluation settings
    parser.add_argument("--round_num", type=int, default=20, help="Number of conversation rounds")
    parser.add_argument("--input_file", type=str, default="sampled_1000_data.json", help="Input JSON file")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--language", type=str, choices=["zh", "en"], default="zh")
    parser.add_argument("--save_intermediate", action="store_true", help="Save intermediate results")
    parser.add_argument("--max_samples", type=int, default=-1, help="Maximum number of samples to process")

    # System settings
    parser.add_argument("--cuda_visible", type=str, default="0")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum retries for API calls")
    parser.add_argument("--retry_delay", type=float, default=1.0, help="Delay between retries")

    args = parser.parse_args()
    return args


def read_json_config(input_path: str) -> List[Dict[str, Any]]:
    """Read JSONL configuration file."""
    configs = []
    with open(input_path, 'r', encoding='utf-8') as f:
        # JSONL format: one JSON object per line
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    config = json.loads(line)
                    configs.append(config)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON line {line_num}: {e}")
                    continue
    
    logger.info(f"Successfully loaded {len(configs)} configurations from {input_path}")
    return configs


def generate_output_filename(args, timestamp: str) -> str:
    """Generate output filename based on parameters."""
    user_temp = args.user_temperature or args.temperature
    assistant_temp = args.assistant_temperature or args.temperature 
    
    parts = [
        f"salesllm_eval",
        f"assistant_{args.assistant_model_name.replace('/', '_')}_T{assistant_temp:.2f}",
        f"user_{args.user_model_name.replace('/', '_')}_T{user_temp:.2f}",
        f"mode_{args.execution_mode}",
        f"batch_{args.batch_size}",
        timestamp
    ]
    
    return "_".join(parts) + ".jsonl"


def is_conversation_end(signal: str) -> bool:
    """Check if conversation should end based on signal."""
    import re
    pattern = re.compile(r"(?i)\b(再见|拜拜|bye|goodbye|see\s*you|下次再说|maybe\s*next\s*time|先这样|that's\s*it\s*for\s*now|不需要|not\s*needed|不要|I\s*don't\s*want\s*it)\b")
    return bool(pattern.search(signal))


def create_conversation_messages(assistant_system_message: str,user_system_message: str, trigger_sentence: str, language: str) -> tuple:
    """Create initial conversation messages for assistant and user."""
    lang_suffix = " 请说中文。" if language == "zh" else " Please speak english."
    
    assistant_messages = [
        create_system_message(assistant_system_message+'/n' + lang_suffix),
        create_user_message(trigger_sentence)
    ]
    
    user_messages = [
        create_system_message(user_system_message+'/n' + lang_suffix),
        create_assistant_message(trigger_sentence)
    ]
    
    return assistant_messages, user_messages


class SalesLLMEvaluator:
    """Main evaluator class for SalesLLM conversations."""
    
    def __init__(self, args):
        self.args = args
        self.client_manager = SalesLLMClientManager()
        self.setup_clients()
        
        # Temperature settings
        self.user_temp = args.user_temperature or args.temperature
        self.assistant_temp = args.assistant_temperature or args.temperature

        
        # Reasoning effort settings
        self.user_reasoning = args.user_reasoning_effort or args.reasoning_effort
        self.assistant_reasoning = args.assistant_reasoning_effort or args.reasoning_effort

        # Extra body settings
        self.assistant_extra_body = None
        if args.assistant_extra_body:
            try:
                self.assistant_extra_body = json.loads(args.assistant_extra_body)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse assistant_extra_body JSON: {args.assistant_extra_body}")

        self.user_extra_body = None
        if args.user_extra_body:
            try:
                self.user_extra_body = json.loads(args.user_extra_body)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse user_extra_body JSON: {args.user_extra_body}")
        
        # Statistics
        self.stats = {
            "total_conversations": 0,
            "successful_conversations": 0,
            "failed_conversations": 0,
            "average_turns": 0,
            "total_turns": 0
        }
    
    def setup_clients(self):
        """Setup LLM clients for different roles."""
        # Assistant client
        self.client_manager.add_client(
            "assistant",
            api_base=self.args.assistant_API_end_point,
            api_key=self.args.assistant_API_key,
            model_name=self.args.assistant_model_name,
            api_type=self.args.assistant_api_type,
            api_version=self.args.assistant_api_version,
            max_retries=self.args.max_retries,
            retry_delay=self.args.retry_delay
        )
        
        # User client
        self.client_manager.add_client(
            "user",
            api_base=self.args.user_API_end_point,
            api_key=self.args.user_API_key,
            model_name=self.args.user_model_name,
            api_type=self.args.user_api_type,
            api_version=self.args.user_api_version,
            max_retries=self.args.max_retries,
            retry_delay=self.args.retry_delay
        )
        

    
    def run_single_conversation(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single conversation between assistant and user."""
        try:
            # Extract system messages
            user_system_message = config["user_system_prompt"]
            assistant_system_message = config["assistant_system_prompt"]
            # Get trigger sentence based on language
            trigger_sentence = config[f'trigger_sentence_{self.args.language}']
            
            # Create initial conversation states
            assistant_messages, user_messages = create_conversation_messages(
                assistant_system_message,user_system_message, trigger_sentence, self.args.language
            )
            
            records = []
            current_speaker = "assistant"
            turn_count = 0
            
            # Add initial trigger to records
            records.append({"role": "user", "content": trigger_sentence})
            
            for turn in range(self.args.round_num):
                role = current_speaker
                messages = assistant_messages if role == "assistant" else user_messages
                temperature = self.assistant_temp if role == "assistant" else self.user_temp
                reasoning_effort = self.assistant_reasoning if role == "assistant" else self.user_reasoning
                extra_body = self.assistant_extra_body if role == "assistant" else self.user_extra_body

                response = self.client_manager.chat_with_role(
                    role,
                    messages,
                    temperature=temperature,
                    top_p=self.args.top_p,
                    max_tokens=self.args.max_tokens,
                    reasoning_effort=reasoning_effort,
                    extra_body=extra_body if extra_body else None
                )

                if response is None:
                    if role == "user":
                        # Remove the last assistant message if user fails to respond
                        assistant_messages.pop()
                    logger.warning(f"{role.capitalize()} failed to respond at turn {turn}")
                    break

                # Update conversation histories and records
                if role == "assistant":
                    assistant_messages.append(create_assistant_message(response))
                    user_messages.append(create_user_message(response))
                    records.append({"role": "assistant", "content": response})
                    current_speaker = "user"
                else:
                    # Check if conversation should end
                    if is_conversation_end(response):
                        records.append({"role": "user", "content": response})
                        logger.info(f"Conversation ended naturally at turn {turn}")
                        break
                    user_messages.append(create_assistant_message(response))
                    assistant_messages.append(create_user_message(response))
                    records.append({"role": "user", "content": response})
                    current_speaker = "assistant"
                
                turn_count += 1
            
            # Update statistics
            self.stats["total_conversations"] += 1
            self.stats["total_turns"] += turn_count
            if turn_count > 0:
                self.stats["successful_conversations"] += 1
            else:
                self.stats["failed_conversations"] += 1
            
            # Create result
            result = config.copy()
            result["messages"] = records
            result["turn_count"] = turn_count
            result["status"] = "completed" if turn_count > 0 else "failed"
            result["timestamp"] = datetime.now().isoformat()
            result['language'] = self.args.language
            return result
            
        except Exception as e:
            logger.error(f"Error in conversation {config.get('id', 'unknown')}: {str(e)}")
            self.stats["failed_conversations"] += 1
            self.stats["total_conversations"] += 1
            
            result = config.copy()
            result["messages"] = []
            result["turn_count"] = 0
            result["status"] = "error"
            result["error"] = str(e)
            result["timestamp"] = datetime.now().isoformat()
            
            return result

    async def run_single_conversation_async(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single conversation between assistant and user asynchronously."""
        try:
            # Extract system messages
            user_system_message = config["user_system_prompt"]
            assistant_system_message = config["assistant_system_prompt"]
            # Get trigger sentence based on language
            trigger_sentence = config[f'trigger_sentence_{self.args.language}']
            
            # Create initial conversation states
            assistant_messages, user_messages = create_conversation_messages(
                assistant_system_message,user_system_message, trigger_sentence, self.args.language
            )
            
            records = []
            current_speaker = "assistant"
            turn_count = 0
            
            # Add initial trigger to records
            records.append({"role": "user", "content": trigger_sentence})
            
            for turn in range(self.args.round_num):
                role = current_speaker
                messages = assistant_messages if role == "assistant" else user_messages
                temperature = self.assistant_temp if role == "assistant" else self.user_temp
                reasoning_effort = self.assistant_reasoning if role == "assistant" else self.user_reasoning
                extra_body = self.assistant_extra_body if role == "assistant" else self.user_extra_body

                response = await self.client_manager.chat_with_role_async(
                    role,
                    messages,
                    temperature=temperature,
                    top_p=self.args.top_p,
                    max_tokens=self.args.max_tokens,
                    reasoning_effort=reasoning_effort,
                    extra_body=extra_body if extra_body else None
                )

                if response is None:
                    if role == "user":
                        # Remove the last assistant message if user fails to respond
                        assistant_messages.pop()
                    logger.warning(f"{role.capitalize()} failed to respond at turn {turn}")
                    break

                # Update conversation histories and records
                if role == "assistant":
                    assistant_messages.append(create_assistant_message(response))
                    user_messages.append(create_user_message(response))
                    records.append({"role": "assistant", "content": response})
                    current_speaker = "user"
                else:
                    # Check if conversation should end
                    if is_conversation_end(response):
                        records.append({"role": "user", "content": response})
                        logger.info(f"Conversation ended naturally at turn {turn}")
                        break
                    user_messages.append(create_assistant_message(response))
                    assistant_messages.append(create_user_message(response))
                    records.append({"role": "user", "content": response})
                    current_speaker = "assistant"
                
                turn_count += 1
            
            # Update statistics
            self.stats["total_conversations"] += 1
            self.stats["total_turns"] += turn_count
            if turn_count > 0:
                self.stats["successful_conversations"] += 1
            else:
                self.stats["failed_conversations"] += 1
            
            # Create result
            result = config.copy()
            result["messages"] = records
            result["turn_count"] = turn_count
            result["status"] = "completed" if turn_count > 0 else "failed"
            result["timestamp"] = datetime.now().isoformat()
            result['language'] = self.args.language
            return result
            
        except Exception as e:
            logger.error(f"Error in conversation {config.get('id', 'unknown')}: {str(e)}")
            self.stats["failed_conversations"] += 1
            self.stats["total_conversations"] += 1
            
            result = config.copy()
            result["messages"] = []
            result["turn_count"] = 0
            result["status"] = "error"
            result["error"] = str(e)
            result["timestamp"] = datetime.now().isoformat()
            
            return result
    
    def run_conversations_sync(self, configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run conversations synchronously (sequential)."""
        results = []
        
        for config in tqdm(configs, desc="Processing conversations"):
            result = self.run_single_conversation(config)
            self._process_conversation_result(result, results)
        
        return results
    
    def run_conversations_async(self, configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run conversations asynchronously."""
        
        async def run_all():
            # Create tasks with semaphore to limit concurrent operations
            semaphore = asyncio.Semaphore(self.args.max_workers)
            
            async def bounded_conversation(config):
                async with semaphore:
                    return await self.run_single_conversation_async(config)
            
            tasks = [bounded_conversation(config) for config in configs]
            
            # Process in batches if specified
            if self.args.batch_size > 0:
                results = []
                for i in range(0, len(tasks), self.args.batch_size):
                    batch = tasks[i:i + self.args.batch_size]
                    batch_results = await tqdm_asyncio.gather(*batch, desc=f"Processing batch {i//self.args.batch_size + 1}")
                    for res in batch_results:
                        self._process_conversation_result(res, results)
            else:
                results = await tqdm_asyncio.gather(*tasks, desc="Processing conversations")
            
            return results
        
        return asyncio.run(run_all())
    
    def run_conversations_concurrent(self, configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run conversations using concurrent thread pool."""
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.args.max_workers) as executor:
            # Submit all tasks
            future_to_config = {
                executor.submit(self.run_single_conversation, config): config 
                for config in configs
            }
            
            # Process completed tasks
            for future in tqdm(concurrent.futures.as_completed(future_to_config), 
                             total=len(configs), desc="Processing conversations"):
                try:
                    result = future.result()
                    self._process_conversation_result(result, results)
                        
                except Exception as e:
                    config = future_to_config[future]
                    logger.error(f"Conversation {config.get('id', 'unknown')} failed: {str(e)}")
                    
                    error_result = config.copy()
                    error_result["messages"] = []
                    error_result["turn_count"] = 0
                    error_result["status"] = "error"
                    error_result["error"] = str(e)
                    error_result["timestamp"] = datetime.now().isoformat()
                    results.append(error_result)
        
        return results

    def _process_conversation_result(self, result: Dict[str, Any], all_results: List[Dict[str, Any]]):
        """Helper to process a single conversation result and save intermediate results."""
        all_results.append(result)
        if self.args.save_intermediate and len(all_results) % 10 == 0:
            self.save_intermediate_results(all_results)
    
    def save_intermediate_results(self, results: List[Dict[str, Any]]):
        """Save intermediate results to file."""
        filename = f"intermediate_results.jsonl"
        
        # Ensure directory exists
        os.makedirs(self.args.output_dir, exist_ok=True)
        filepath = os.path.join(self.args.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for result in results:
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')
        
        logger.info(f"Saved intermediate results to {filepath}")
    
    def save_final_results(self, results: List[Dict[str, Any]]):
        """Save final results and statistics."""
        # Calculate final statistics
        if self.stats["successful_conversations"] > 0:
            self.stats["average_turns"] = self.stats["total_turns"] / self.stats["successful_conversations"]
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = generate_output_filename(self.args, timestamp)
        filepath = os.path.join(self.args.output_dir, filename)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.args.output_dir, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for result in results:
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')
        
        # Save statistics
        stats_filename = f"stats_{timestamp}.json"
        stats_filepath = os.path.join(self.args.output_dir, stats_filename)
        
        with open(stats_filepath, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {filepath}")
        logger.info(f"Statistics saved to {stats_filepath}")
        logger.info(f"Final statistics: {self.stats}")
    
    def run_evaluation(self, configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run the complete evaluation."""
        logger.info(f"Starting evaluation with {len(configs)} conversations")
        logger.info(f"Execution mode: {self.args.execution_mode}")
        logger.info(f"Max workers: {self.args.max_workers}")
        logger.info(f"Batch size: {self.args.batch_size}")
        
        start_time = time.time()
        
        # Run conversations based on execution mode
        if self.args.execution_mode == "sync":
            results = self.run_conversations_sync(configs)
        elif self.args.execution_mode == "async":
            results = self.run_conversations_async(configs)
        elif self.args.execution_mode == "concurrent":
            results = self.run_conversations_concurrent(configs)
        else:
            raise ValueError(f"Unknown execution mode: {self.args.execution_mode}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        logger.info(f"Evaluation completed in {total_time:.2f} seconds")
        logger.info(f"Average time per conversation: {total_time/len(configs):.2f} seconds")
        
        return results


def main():
    """Main function."""
    args = parse_args()
    
    # Read configurations
    configs = read_json_config(args.input_file)
    
    # Filter configurations if max_samples is set
    if args.max_samples != -1:
        logger.info(f"Limiting execution to first {args.max_samples} samples")
        configs = configs[:args.max_samples]
        
    logger.info(f"Loaded {len(configs)} conversation configurations")
    
    # Create evaluator
    evaluator = SalesLLMEvaluator(args)
    
    # Run evaluation
    results = evaluator.run_evaluation(configs)
    
    # Save final results
    evaluator.save_final_results(results)
    
    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()