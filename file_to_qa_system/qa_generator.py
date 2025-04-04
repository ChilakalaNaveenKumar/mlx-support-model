"""
QA Generator module for creating comprehensive Q&A pairs from files.
Uses a local LLM to generate pairs that cover all file content.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_file_description(file_extension: str) -> str:
    """
    Get a description of the file type based on its extension.
    
    Args:
        file_extension: File extension (e.g., .py, .js)
        
    Returns:
        Description of the file type
    """
    file_types = {
        ".py": "Python code",
        ".js": "JavaScript code",
        ".ts": "TypeScript code",
        ".jsx": "React JSX code",
        ".tsx": "React TSX code",
        ".html": "HTML markup",
        ".css": "CSS stylesheet",
        ".md": "Markdown document",
        ".txt": "Text document",
        ".json": "JSON data",
        ".yaml": "YAML data",
        ".yml": "YAML data",
        ".xml": "XML data",
        ".csv": "CSV data",
        ".sql": "SQL query",
        ".sh": "Shell script",
        ".bat": "Batch script",
        ".ps1": "PowerShell script",
        ".rb": "Ruby code",
        ".php": "PHP code",
        ".java": "Java code",
        ".cpp": "C++ code",
        ".c": "C code",
        ".h": "C/C++ header",
        ".cs": "C# code",
        ".go": "Go code",
        ".rs": "Rust code",
        ".swift": "Swift code",
        ".kt": "Kotlin code",
        ".scala": "Scala code",
        ".r": "R code",
    }
    
    return file_types.get(file_extension.lower(), "text file")


def create_qa_generation_prompt(
    file_content: str,
    file_path: str,
    file_extension: str,
    min_pairs: int,
    priority_keywords: List[str]
) -> str:
    """
    Create a prompt for generating Q&A pairs from a file.
    
    Args:
        file_content: Content of the file
        file_path: Path to the file
        file_extension: File extension
        min_pairs: Minimum number of Q&A pairs to generate
        priority_keywords: Keywords to prioritize in Q&A generation
        
    Returns:
        Formatted prompt for the LLM
    """
    file_type = get_file_description(file_extension)
    file_name = os.path.basename(file_path)
    
    # Format priority keywords as a comma-separated list
    keywords_str = ', '.join(f'"{kw}"' for kw in priority_keywords)
    
    prompt = f"""You are an expert Q&A generation system. Your task is to create comprehensive question-answer pairs based on a {file_type} file. These Q&A pairs will be used to fine-tune a language model to learn the file's content.

FILE NAME: {file_name}
FILE PATH: {file_path}

PRIORITY KEYWORDS: {keywords_str}

FILE CONTENT:
```{file_extension[1:] if file_extension.startswith('.') else file_extension}
{file_content}
```

INSTRUCTIONS:
1. Generate at least {min_pairs} question-answer pairs that completely cover ALL information in the file.
2. Be exhaustive - every significant piece of information should be included in at least one Q&A pair.
3. Questions with the priority keywords should receive special attention.
4. Include a mix of question types:
   - Factual questions about specific details
   - Conceptual questions about overall purpose
   - How-to questions about functionality
   - Why questions about design decisions
   - Questions that require synthesizing multiple parts of the file
5. For code files, include questions about:
   - Function/method purposes and parameters
   - Class hierarchies and relationships
   - Variable meanings and usage patterns
   - Import statements and dependencies
   - Logic flow and algorithms
6. Answers should be detailed and accurate, only using information from the file.
7. Every word and concept in the file should be covered by the Q&A pairs.

FORMAT YOUR RESPONSE AS JSONL (one JSON object per line) with each object having:
{{"question": "Your question here", "answer": "Your comprehensive answer here"}}

Begin generating the Q&A pairs now:
"""
    
    return prompt


def parse_qa_from_text(text: str) -> List[Dict[str, str]]:
    """
    Parse Q&A pairs from the LLM response text.
    
    Args:
        text: Text containing Q&A pairs in JSONL format
        
    Returns:
        List of Q&A pair dictionaries
    """
    qa_pairs = []
    
    # Clean up the text first
    lines = text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('//'):
            continue
            
        try:
            # Try to parse as JSON
            qa_pair = json.loads(line)
            
            # Validate that it has question and answer
            if 'question' in qa_pair and 'answer' in qa_pair:
                qa_pairs.append(qa_pair)
        except json.JSONDecodeError:
            # Try to find JSON-like structure in the line
            if '{' in line and '}' in line:
                try:
                    json_part = line[line.find('{'):line.rfind('}')+1]
                    qa_pair = json.loads(json_part)
                    if 'question' in qa_pair and 'answer' in qa_pair:
                        qa_pairs.append(qa_pair)
                except Exception:
                    pass
    
    # If parsing failed completely, try a more aggressive approach
    if not qa_pairs:
        try:
            # Try to find all JSON objects in the text
            text = text.replace('\n', ' ')
            start_indices = [i for i, char in enumerate(text) if char == '{']
            for start_idx in start_indices:
                # Find the matching closing brace
                open_braces = 1
                for j in range(start_idx + 1, len(text)):
                    if text[j] == '{':
                        open_braces += 1
                    elif text[j] == '}':
                        open_braces -= 1
                        if open_braces == 0:
                            try:
                                json_part = text[start_idx:j+1]
                                qa_pair = json.loads(json_part)
                                if 'question' in qa_pair and 'answer' in qa_pair:
                                    qa_pairs.append(qa_pair)
                            except Exception:
                                pass
                            break
        except Exception:
            logger.warning("Failed to parse Q&A pairs")
    
    return qa_pairs


def call_local_llm(prompt: str, model_name: Optional[str] = None) -> str:
    """
    Call the local LLM to generate a response.
    
    Args:
        prompt: Input prompt for the LLM
        model_name: Optional model name to use
        
    Returns:
        Generated response from the LLM
    """
    try:
        # Import the MLX support model framework
        try:
            # First check if numpy is properly installed
            try:
                import numpy as np
                logger.info(f"NumPy version: {np.__version__}")
            except ImportError:
                logger.error("NumPy not installed properly. Please reinstall numpy.")
                return ""
            
            # Then try to import MLX support model components
            try:
                from mlx_support_model.services.model_service import ModelService
                from mlx_support_model.services.cache_service import CacheService
                from mlx_support_model.config import DEFAULT_MODEL, DEFAULT_GENERATION_PARAMS
            except ImportError as e:
                logger.error(f"Failed to import MLX support model framework: {e}")
                logger.error("Please ensure that the MLX support model is properly installed")
                return ""
            
            # Initialize services
            cache_service = CacheService(enable_cache=True)
            model_service = ModelService(cache_service=cache_service)
            
            # Load model
            model_to_use = model_name or DEFAULT_MODEL
            logger.info(f"Loading model: {model_to_use}")
            if not model_service.load_model(model_to_use):
                logger.error(f"Failed to load model: {model_to_use}")
                return ""
            
            # Generation parameters
            generation_params = {
                "temperature": 0.7,
                "max_tokens": 6000
            }
            
            # Generate response
            logger.info("Generating Q&A pairs with the local LLM...")
            response = model_service.generate_text(prompt, generation_params)
            
            return response
            
        except Exception as e:
            logger.error(f"Error importing or using MLX framework: {e}")
            
            # For debugging purposes
            import traceback
            logger.error(traceback.format_exc())
            
            # Provide guidance
            logger.error("""
            There might be compatibility issues with your Python environment. 
            Try the following:
            1. Make sure you're using Python 3.9+ on macOS with Apple Silicon
            2. Reinstall numpy: pip install -U numpy
            3. Reinstall transformers: pip install -U transformers
            4. Ensure MLX is properly installed: pip install -U mlx mlx-lm
            """)
            return ""
            
    except Exception as e:
        logger.error(f"Error calling local LLM: {e}")
        return ""


def generate_qa_from_file(
    file_content: str,
    file_path: str,
    file_extension: str,
    min_pairs: int = 50,
    priority_keywords: Optional[List[str]] = None,
    model_name: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Generate Q&A pairs from a file using a local LLM.
    
    Args:
        file_content: Content of the file
        file_path: Path to the file
        file_extension: File extension
        min_pairs: Minimum number of Q&A pairs to generate
        priority_keywords: Keywords to prioritize in Q&A generation
        model_name: Optional model name to use
        
    Returns:
        List of Q&A pair dictionaries
    """
    # Ensure priority_keywords is a list
    if priority_keywords is None:
        priority_keywords = []
    
    # Create LLM prompt
    prompt = create_qa_generation_prompt(
        file_content=file_content,
        file_path=file_path,
        file_extension=file_extension,
        min_pairs=min_pairs,
        priority_keywords=priority_keywords
    )
    
    # Call the LLM
    response = call_local_llm(prompt, model_name)
    
    if not response:
        logger.error("Failed to generate Q&A pairs")
        return []
    
    # Parse Q&A pairs from response
    qa_pairs = parse_qa_from_text(response)
    
    # If we didn't get enough pairs, try again with a more focused prompt
    if len(qa_pairs) < min_pairs:
        logger.info(f"Generated only {len(qa_pairs)} pairs, aiming for {min_pairs}. Trying again...")
        
        # Modify prompt to emphasize the need for more pairs
        additional_prompt = f"""You previously generated {len(qa_pairs)} Q&A pairs for the file {os.path.basename(file_path)}.
We need at least {min_pairs} pairs to ensure comprehensive coverage.

Please generate {min_pairs - len(qa_pairs)} MORE unique Q&A pairs that cover different aspects of the file:

FILE CONTENT:
```{file_extension[1:] if file_extension.startswith('.') else file_extension}
{file_content}
```

Focus on parts of the file that weren't covered in previous questions. Be detailed and specific.
Format your response as JSONL with {{"question": "...", "answer": "..."}} objects.

Begin generating the additional Q&A pairs now:
"""
        
        # Call the LLM again
        additional_response = call_local_llm(additional_prompt, model_name)
        additional_pairs = parse_qa_from_text(additional_response)
        
        # Add new pairs to the existing ones
        qa_pairs.extend(additional_pairs)
    
    logger.info(f"Generated {len(qa_pairs)} Q&A pairs for {file_path}")
    return qa_pairs


def create_finetune_script(
    train_dataset_path: str,
    test_dataset_path: str,
    output_script_path: str,
    model_name: Optional[str] = None
) -> bool:
    """
    Create a Python script for fine-tuning a model on the generated Q&A pairs.
    
    Args:
        train_dataset_path: Path to the training dataset
        test_dataset_path: Path to the test dataset
        output_script_path: Path to save the script
        model_name: Optional model name to use
        
    Returns:
        Boolean indicating success
    """
    try:
        # Try to import the default model name
        try:
            from mlx_support_model.config import DEFAULT_MODEL
        except ImportError:
            DEFAULT_MODEL = "mlx-community/Llama-3-8B-Instruct-4bit"  # Fallback
    
        # Use default model if none specified
        model_to_use = model_name or DEFAULT_MODEL
        
        # Create the fine-tuning script content
        script_content = f"""#!/usr/bin/env python3
\"\"\"
Fine-tuning script for MLX language models.
Uses the generated Q&A pairs to fine-tune the model.
\"\"\"

import os
import json
import argparse
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("finetune")

# Try to import MLX libraries
try:
    import mlx.core as mx
    from mlx_lm import load, generate, save
    from mlx.optimizers import Adam
    import mlx.nn as nn
except ImportError as e:
    logger.error(f"Error importing MLX libraries: {{e}}")
    logger.error("Please make sure MLX and MLX-LM are installed")
    exit(1)


def load_dataset(file_path: str) -> List[Dict[str, str]]:
    \"\"\"
    Load a dataset from a file.
    
    Args:
        file_path: Path to the dataset file
        
    Returns:
        List of Q&A pairs
    \"\"\"
    try:
        _, ext = os.path.splitext(file_path)
        
        if ext == '.jsonl':
            # Load JSONL
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse line: {{line[:50]}}")
            return data
            
        elif ext == '.json':
            # Load JSON
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        elif ext == '.csv':
            # Load CSV
            import csv
            data = []
            with open(file_path, 'r', encoding='utf-8', newline='') as f:
                reader = csv.reader(f)
                header = next(reader)  # Skip header
                for row in reader:
                    if len(row) >= 2:
                        data.append({{"question": row[0], "answer": row[1]}})
            return data
            
        else:
            raise ValueError(f"Unsupported file format: {{ext}}")
    except Exception as e:
        logger.error(f"Error loading dataset {{file_path}}: {{e}}")
        return []


def format_qa_for_training(qa_pair: Dict[str, str], tokenizer) -> str:
    \"\"\"
    Format a Q&A pair for training.
    
    Args:
        qa_pair: Q&A pair dictionary
        tokenizer: Tokenizer for the model
        
    Returns:
        Formatted text for training
    \"\"\"
    try:
        # Format as a chat conversation
        messages = [
            {{"role": "system", "content": "You are a helpful assistant with knowledge about files and code."}},
            {{"role": "user", "content": qa_pair["question"]}},
            {{"role": "assistant", "content": qa_pair["answer"]}}
        ]
        
        # Apply chat template if available
        try:
            formatted_text = tokenizer.apply_chat_template(messages, tokenize=False)
            return formatted_text
        except Exception as e:
            logger.warning(f"Error applying chat template: {{e}}. Using fallback format.")
            # Fallback to manual formatting
            return f"Question: {{qa_pair['question']}}\\nAnswer: {{qa_pair['answer']}}"
    except Exception as e:
        logger.error(f"Error formatting QA pair: {{e}}")
        # Return a safe fallback
        return "Error formatting QA pair"


def prepare_training_data(qa_pairs: List[Dict[str, str]], tokenizer) -> mx.array:
    \"\"\"
    Prepare training data for fine-tuning.
    
    Args:
        qa_pairs: List of Q&A pairs
        tokenizer: Tokenizer for the model
        
    Returns:
        Tokenized training data
    \"\"\"
    try:
        # Format each Q&A pair
        formatted_texts = []
        for pair in qa_pairs:
            if "question" in pair and "answer" in pair:
                formatted_text = format_qa_for_training(pair, tokenizer)
                formatted_texts.append(formatted_text)
            else:
                logger.warning(f"Skipping malformed QA pair: {{pair}}")
        
        # Tokenize data
        tokenized_data = []
        for text in formatted_texts:
            try:
                tokens = tokenizer.encode(text)
                tokenized_data.append(tokens)
            except Exception as e:
                logger.warning(f"Error tokenizing text: {{e}}")
        
        if not tokenized_data:
            logger.error("No valid training data after tokenization")
            # Return a single empty sample to avoid errors
            return mx.array([[0]])
        
        # Convert to MLX array
        return mx.array(tokenized_data)
    except Exception as e:
        logger.error(f"Error preparing training data: {{e}}")
        # Return a single empty sample to avoid errors
        return mx.array([[0]])


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Fine-tune MLX model on Q&A dataset")
    parser.add_argument(
        "--train-data", 
        default="{train_dataset_path}",
        help="Path to training data"
    )
    parser.add_argument(
        "--test-data", 
        default="{test_dataset_path}",
        help="Path to test data"
    )
    parser.add_argument(
        "--model", 
        default="{model_to_use}",
        help="Model to fine-tune"
    )
    parser.add_argument(
        "--output-model", 
        default="finetuned_model",
        help="Output path for fine-tuned model"
    )
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        default=1e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=3,
        help="Number of epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for training"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        # Load model
        logger.info(f"Loading model: {{args.model}}")
        model, tokenizer = load(args.model)
        
        # Load datasets
        logger.info(f"Loading training data: {{args.train_data}}")
        train_qa_pairs = load_dataset(args.train_data)
        logger.info(f"Loaded {{len(train_qa_pairs)}} training Q&A pairs")
        
        test_qa_pairs = None
        if os.path.exists(args.test_data):
            logger.info(f"Loading test data: {{args.test_data}}")
            test_qa_pairs = load_dataset(args.test_data)
            logger.info(f"Loaded {{len(test_qa_pairs)}} test Q&A pairs")
        
        # Prepare data
        logger.info("Preparing training data...")
        train_data = prepare_training_data(train_qa_pairs, tokenizer)
        
        test_data = None
        if test_qa_pairs:
            logger.info("Preparing test data...")
            test_data = prepare_training_data(test_qa_pairs, tokenizer)
        
        # Fine-tune model
        logger.info("Starting fine-tuning...")
        
        # Set up optimizer
        optimizer = Adam(learning_rate=args.learning_rate)
        
        # Define loss function
        def loss_fn(model, inputs):
            logits = model(inputs)
            # Shift logits and targets for next-token prediction
            targets = mx.concatenate([inputs[:, 1:], mx.zeros((inputs.shape[0], 1), dtype=mx.int32)], axis=1)
            loss = nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            return loss
        
        # Training loop
        batch_size = args.batch_size
        for epoch in range(args.epochs):
            total_loss = 0.0
            num_batches = 0
            
            for i in range(0, len(train_data), batch_size):
                # Get batch
                batch = train_data[i:i+batch_size]
                
                # Compute loss and gradients
                loss, grads = nn.value_and_grad(loss_fn)(model, batch)
                
                # Update model
                optimizer.update(model, grads)
                
                total_loss += loss
                num_batches += 1
                
                if i % 10 == 0:
                    logger.info(f"Epoch {{epoch+1}}/{{args.epochs}}, Batch {{i//batch_size}}, Loss: {{loss}}")
            
            avg_loss = total_loss / max(1, num_batches)
            logger.info(f"Epoch {{epoch+1}}/{{args.epochs}} complete, Average Loss: {{avg_loss}}")
            
            # Validate if validation data is provided
            if test_data is not None:
                val_loss = 0.0
                val_batches = 0
                
                for i in range(0, len(test_data), batch_size):
                    # Get batch
                    batch = test_data[i:i+batch_size]
                    
                    # Compute validation loss
                    val_loss += loss_fn(model, batch)
                    val_batches += 1
                
                avg_val_loss = val_loss / max(1, val_batches)
                logger.info(f"Validation Loss: {{avg_val_loss}}")
        
        # Save fine-tuned model
        logger.info(f"Saving fine-tuned model to {{args.output_model}}")
        save(model, tokenizer, args.output_model)
        logger.info("Fine-tuning complete!")
        
        # Test on a few examples
        logger.info("\\nTesting fine-tuned model:")
        examples = test_qa_pairs[:3] if test_qa_pairs else train_qa_pairs[:3]
        
        for i, qa_pair in enumerate(examples):
            question = qa_pair["question"]
            logger.info(f"\\nQuestion {{i+1}}: {{question}}")
            
            response = generate(model, tokenizer, question, max_tokens=200)
            logger.info(f"Model response: {{response}}")
            logger.info(f"Expected answer: {{qa_pair['answer']}}")
            
    except Exception as e:
        logger.error(f"Error during fine-tuning: {{e}}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
"""
        
        # Write the script to file
        with open(output_script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        # Make the script executable
        os.chmod(output_script_path, 0o755)
        
        logger.info(f"Created fine-tuning script: {output_script_path}")
        return True
    except Exception as e:
        logger.error(f"Error creating fine-tuning script: {e}")
        return False