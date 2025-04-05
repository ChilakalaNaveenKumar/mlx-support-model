#!/usr/bin/env python3
"""
Working MLX fine-tuning script.
Fixed gradient clipping implementation.
"""

import os
import json
import argparse
import logging
from typing import List, Dict
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("finetune")

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

logger.info(f"MLX imports successful")


def load_dataset(file_path: str) -> List[Dict[str, str]]:
    """
    Load Q&A pairs from a dataset file.
    
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
                        logger.warning(f"Could not parse line: {line[:50]}...")
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
            # Fallback to simple format
            return f"Question: {qa_pair['question']}\nAnswer: {qa_pair['answer']}"
    except Exception as e:
        logger.error(f"Error formatting QA pair: {e}")
        return "Error formatting QA pair"


def prepare_training_data(qa_pairs: List[Dict[str, str]], tokenizer) -> List[mx.array]:
    """
    Prepare training data for fine-tuning.
    
    Args:
        qa_pairs: List of Q&A pairs
        tokenizer: Tokenizer for the model
        
    Returns:
        List of tokenized examples
    """
    tokenized_examples = []
    
    for pair in qa_pairs:
        try:
            if "question" in pair and "answer" in pair:
                # Format the Q&A pair
                text = format_qa_for_training(pair, tokenizer)
                
                # Tokenize
                tokens = tokenizer.encode(text)
                if tokens:
                    tokenized_examples.append(mx.array(tokens))
        except Exception as e:
            logger.warning(f"Error preparing example: {e}")
    
    logger.info(f"Prepared {len(tokenized_examples)} examples for training")
    return tokenized_examples


def train_model(
    model,
    tokenized_examples: List[mx.array],
    epochs: int = 3,
    learning_rate: float = 5e-7,
    batch_size: int = 1,
    grad_clip: float = 0.5
):
    """
    Train model on tokenized examples.
    
    Args:
        model: The model to train
        tokenized_examples: List of tokenized examples
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training
        grad_clip: Value for gradient clipping
    """
    # Create optimizer
    optimizer = Adam(learning_rate=learning_rate)
    logger.info(f"Created optimizer with learning rate: {learning_rate}")
    
    # Group examples of similar length to reduce padding
    examples_by_length = {}
    for example in tokenized_examples:
        # Round to nearest 64 tokens
        length_bucket = ((len(example) + 63) // 64) * 64
        if length_bucket not in examples_by_length:
            examples_by_length[length_bucket] = []
        examples_by_length[length_bucket].append(example)
    
    logger.info(f"Grouped examples into {len(examples_by_length)} length buckets")
    
    # Define simple loss function in the format MLX expects
    def simple_loss_fn(parameters, x):
        """
        Compute loss for language modeling.
        
        Args:
            parameters: Model parameters
            x: Input token IDs
            
        Returns:
            Loss value
        """
        # Update model parameters
        model.update(parameters)
        
        # Forward pass
        y = model(x)
        
        # Create targets (shift inputs right by one)
        targets = mx.concatenate([
            x[:, 1:], 
            mx.zeros((x.shape[0], 1), dtype=mx.int32)
        ], axis=1)
        
        # Create padding mask
        mask = (targets != 0).astype(mx.float32)
        
        # Check if y is an array or an object with a logits attribute
        if hasattr(y, 'logits'):
            logits = y.logits
        else:
            logits = y  # Assume y is directly the logits
        
        # Reshape for cross entropy
        batch_size, seq_len, vocab_size = logits.shape
        logits = logits.reshape(-1, vocab_size)
        targets = targets.reshape(-1)
        mask = mask.reshape(-1)
        
        # Compute cross entropy
        losses = nn.losses.cross_entropy(logits, targets, reduction='none')
        
        # Apply mask
        masked_losses = losses * mask
        
        # Average over non-padding tokens
        token_count = mx.maximum(mask.sum(), 1.0)  # Avoid div by zero
        return masked_losses.sum() / token_count
    
    # Create gradient function
    value_grad_fn = nn.value_and_grad(model, simple_loss_fn)
    
    # Training loop
    for epoch in range(epochs):
        logger.info(f"Starting epoch {epoch+1}/{epochs}")
        total_loss = 0.0
        examples_seen = 0
        epoch_start_time = time.time()
        
        # Process each length bucket
        for length, examples in examples_by_length.items():
            logger.info(f"Processing {len(examples)} examples of length ~{length}")
            
            # Create batches
            for i in range(0, len(examples), batch_size):
                batch = examples[i:i+batch_size]
                
                # Skip incomplete batches if batch_size > 1
                if len(batch) < batch_size and batch_size > 1:
                    continue
                
                try:
                    # Pad sequences to the same length
                    max_len = max(len(ex) for ex in batch)
                    padded_batch = []
                    for ex in batch:
                        # Pad with zeros
                        padding = mx.zeros((max_len - len(ex),), dtype=mx.int32)
                        padded = mx.concatenate([ex, padding])
                        padded_batch.append(padded)
                    
                    # Stack batch
                    inputs = mx.stack(padded_batch)
                    
                    # Get trainable parameters
                    # Try both trainable_parameters and parameters methods
                    try:
                        params = model.trainable_parameters()
                    except:
                        params = model.parameters()
                    
                    # Compute loss and gradients
                    # Pass parameters and inputs to the value_grad_fn
                    loss, grads = value_grad_fn(params, inputs)
                    
                    # Skip invalid losses
                    if mx.isnan(loss) or mx.isinf(loss):
                        logger.warning("Invalid loss value, skipping batch")
                        continue
                    
                    # Clip gradients properly - only clip arrays, not dictionaries
                    clipped_grads = {}
                    for k, g in grads.items():
                        if g is not None:
                            # Check if g is a dictionary (nested parameters)
                            if isinstance(g, dict):
                                # Handle nested dictionary of gradients
                                clipped_grads[k] = {}
                                for sub_k, sub_g in g.items():
                                    if sub_g is not None and hasattr(sub_g, 'shape'):
                                        clipped_grads[k][sub_k] = mx.clip(sub_g, -grad_clip, grad_clip)
                                    else:
                                        clipped_grads[k][sub_k] = sub_g
                            # Check if g is an array that can be clipped
                            elif hasattr(g, 'shape'):
                                clipped_grads[k] = mx.clip(g, -grad_clip, grad_clip)
                            else:
                                # Skip non-clipable objects
                                clipped_grads[k] = g
                    
                    # Update parameters
                    optimizer.update(model, clipped_grads)
                    
                    # Update statistics
                    loss_value = float(loss)
                    total_loss += loss_value
                    examples_seen += 1
                    
                    # Log progress
                    if examples_seen % 5 == 0 or examples_seen == 1:
                        logger.info(f"Epoch {epoch+1}, Examples: {examples_seen}, Loss: {loss_value:.4f}")
                        
                except Exception as e:
                    logger.error(f"Error in training step: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    continue
        
        # Compute average loss
        avg_loss = total_loss / max(1, examples_seen)
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch+1} complete in {epoch_time:.2f} seconds. Average loss: {avg_loss:.4f}")
        
        # Save checkpoint
        checkpoint_dir = f"checkpoint_epoch_{epoch+1}"
        save_model(model, checkpoint_dir)
        logger.info(f"Saved checkpoint to {checkpoint_dir}")


def save_model(model, output_dir: str):
    """
    Save model to disk.
    
    Args:
        model: The model to save
        output_dir: Output directory
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model weights
        weights_path = os.path.join(output_dir, "weights.safetensors")
        model.save_weights(weights_path)
        logger.info(f"Saved model weights to {weights_path}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")


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
        default=1e-6,  # Slightly higher learning rate
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
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=0.5,  # Smaller gradient clipping
        help="Gradient clipping value"
    )
    
    args = parser.parse_args()
    
    try:
        # Load model
        logger.info(f"Loading model: {args.model}")
        model, tokenizer = load(args.model)
        
        # Print model information
        logger.info(f"Model loaded successfully")
        logger.info(f"Model type: {type(model).__name__}")
        
        # Load training data
        logger.info(f"Loading training data: {args.train_data}")
        train_qa_pairs = load_dataset(args.train_data)
        logger.info(f"Loaded {len(train_qa_pairs)} training Q&A pairs")
        
        # Prepare training data
        logger.info("Preparing training data")
        train_examples = prepare_training_data(train_qa_pairs, tokenizer)
        
        # Load and prepare test data if available
        test_qa_pairs = None
        if os.path.exists(args.test_data):
            logger.info(f"Loading test data: {args.test_data}")
            test_qa_pairs = load_dataset(args.test_data)
            logger.info(f"Loaded {len(test_qa_pairs)} test Q&A pairs")
        
        # Train model
        logger.info("Starting training")
        train_model(
            model=model,
            tokenized_examples=train_examples,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            grad_clip=args.grad_clip
        )
        
        # Save final model
        logger.info(f"Saving final model to {args.output_model}")
        save_model(model, args.output_model)
        
        # Save tokenizer
        tokenizer.save_pretrained(args.output_model)
        
        # Test the model on example questions
        logger.info("Testing the fine-tuned model:")
        examples = test_qa_pairs[:3] if test_qa_pairs else train_qa_pairs[:3]
        
        for i, qa_pair in enumerate(examples):
            question = qa_pair["question"]
            logger.info(f"\nQuestion {i+1}: {question}")
            
            try:
                # Generate response with very low temperature for deterministic output
                response = generate(
                    model,
                    tokenizer,
                    question,
                    max_tokens=200,
                    temperature=0.1
                )
                logger.info(f"Model response: {response}")
                logger.info(f"Expected answer: {qa_pair['answer']}")
            except Exception as e:
                logger.error(f"Error generating response: {e}")
        
        logger.info("Fine-tuning complete!")
        
    except Exception as e:
        logger.error(f"Error during fine-tuning: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()