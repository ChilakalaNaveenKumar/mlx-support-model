#!/usr/bin/env python3
"""
Fine-tuning script for MLX language models.
Uses the generated Q&A pairs to fine-tune the model.
"""

import os
import json
import argparse
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("finetune")

# Add a custom save function
def save_model(model, tokenizer, output_path):
    """Save the model and tokenizer to the specified path"""
    import os
    os.makedirs(output_path, exist_ok=True)
    
    # Save model weights
    model.save_weights(os.path.join(output_path, "weights.safetensors"))
    
    # Save tokenizer
    tokenizer.save_pretrained(output_path)
    
    # Save model config if available
    try:
        if hasattr(model, 'config'):
            import json
            with open(os.path.join(output_path, "config.json"), "w") as f:
                json.dump(model.config, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save model config: {e}")
    
    logger.info(f"Model saved to {output_path}")

# Try to import MLX libraries
try:
    import mlx.core as mx
    from mlx_lm import load, generate
    from mlx.optimizers import Adam
    import mlx.nn as nn
except ImportError as e:
    logger.error(f"Error importing MLX libraries: {e}")
    logger.error("Please make sure MLX and MLX-LM are installed")
    exit(1)


def load_dataset(file_path: str) -> List[Dict[str, str]]:
    """
    Load a dataset from a file.
    
    Args:
        file_path: Path to the dataset file
        
    Returns:
        List of Q&A pairs
    """
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
                        logger.warning(f"Could not parse line: {line[:50]}")
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
                        data.append({"question": row[0], "answer": row[1]})
            return data
            
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    except Exception as e:
        logger.error(f"Error loading dataset {file_path}: {e}")
        return []


def format_qa_for_training(qa_pair: Dict[str, str], tokenizer) -> str:
    """
    Format a Q&A pair for training.
    
    Args:
        qa_pair: Q&A pair dictionary
        tokenizer: Tokenizer for the model
        
    Returns:
        Formatted text for training
    """
    try:
        # Format as a chat conversation
        messages = [
            {"role": "system", "content": "You are a helpful assistant with knowledge about files and code."},
            {"role": "user", "content": qa_pair["question"]},
            {"role": "assistant", "content": qa_pair["answer"]}
        ]
        
        # Apply chat template if available
        try:
            formatted_text = tokenizer.apply_chat_template(messages, tokenize=False)
            return formatted_text
        except Exception as e:
            logger.warning(f"Error applying chat template: {e}. Using fallback format.")
            # Fallback to manual formatting
            return f"Question: {qa_pair['question']}\nAnswer: {qa_pair['answer']}"
    except Exception as e:
        logger.error(f"Error formatting QA pair: {e}")
        # Return a safe fallback
        return "Error formatting QA pair"


def prepare_training_data(qa_pairs: List[Dict[str, str]], tokenizer) -> mx.array:
    """
    Prepare training data for fine-tuning.
    
    Args:
        qa_pairs: List of Q&A pairs
        tokenizer: Tokenizer for the model
        
    Returns:
        Tokenized training data
    """
    try:
        # Format each Q&A pair
        formatted_texts = []
        for pair in qa_pairs:
            if "question" in pair and "answer" in pair:
                formatted_text = format_qa_for_training(pair, tokenizer)
                formatted_texts.append(formatted_text)
            else:
                logger.warning(f"Skipping malformed QA pair: {pair}")
        
        # Tokenize data
        tokenized_data = []
        for text in formatted_texts:
            try:
                tokens = tokenizer.encode(text)
                tokenized_data.append(tokens)
            except Exception as e:
                logger.warning(f"Error tokenizing text: {e}")
        
        if not tokenized_data:
            logger.error("No valid training data after tokenization")
            # Return a single empty sample to avoid errors
            return mx.array([[0]])
        
        # Convert to MLX array
        return mx.array(tokenized_data)
    except Exception as e:
        logger.error(f"Error preparing training data: {e}")
        # Return a single empty sample to avoid errors
        return mx.array([[0]])


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Fine-tune MLX model on Q&A dataset")
    parser.add_argument(
        "--train-data", 
        default="qa_output/train.jsonl",
        help="Path to training data"
    )
    parser.add_argument(
        "--test-data", 
        default="qa_output/test.jsonl",
        help="Path to test data"
    )
    parser.add_argument(
        "--model", 
        default="mlx-community/Qwen2.5-Coder-32B-Instruct-6bit",
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
        logger.info(f"Loading model: {args.model}")
        model, tokenizer = load(args.model)
        
        # Load datasets
        logger.info(f"Loading training data: {args.train_data}")
        train_qa_pairs = load_dataset(args.train_data)
        logger.info(f"Loaded {len(train_qa_pairs)} training Q&A pairs")
        
        test_qa_pairs = None
        if os.path.exists(args.test_data):
            logger.info(f"Loading test data: {args.test_data}")
            test_qa_pairs = load_dataset(args.test_data)
            logger.info(f"Loaded {len(test_qa_pairs)} test Q&A pairs")
        
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
                    logger.info(f"Epoch {epoch+1}/{args.epochs}, Batch {i//batch_size}, Loss: {loss}")
            
            avg_loss = total_loss / max(1, num_batches)
            logger.info(f"Epoch {epoch+1}/{args.epochs} complete, Average Loss: {avg_loss}")
            
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
                logger.info(f"Validation Loss: {avg_val_loss}")
        
        # Save fine-tuned model
        logger.info(f"Saving fine-tuned model to {args.output_model}")
        save_model(model, tokenizer, args.output_model)
        logger.info("Fine-tuning complete!")
        
        # Test on a few examples
        logger.info("\nTesting fine-tuned model:")
        examples = test_qa_pairs[:3] if test_qa_pairs else train_qa_pairs[:3]
        
        for i, qa_pair in enumerate(examples):
            question = qa_pair["question"]
            logger.info(f"\nQuestion {i+1}: {question}")
            
            response = generate(model, tokenizer, question, max_tokens=200)
            logger.info(f"Model response: {response}")
            logger.info(f"Expected answer: {qa_pair['answer']}")
            
    except Exception as e:
        logger.error(f"Error during fine-tuning: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
