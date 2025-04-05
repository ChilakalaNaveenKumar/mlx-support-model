#!/usr/bin/env python3
"""
Debug Frozen Parameters MLX Fine-tuning Script
With extensive logging to diagnose training issues
"""

import os
import json
import logging
import argparse
import time
from typing import List, Dict, Any, Optional
import random

# Configure logging with more detail
logging.basicConfig(
    level=logging.DEBUG,  # Use DEBUG for maximum verbosity
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("debug-frozen-finetune")

try:
    import numpy as np
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.optimizers import Adam
    from mlx_lm import load, generate
except ImportError as e:
    logger.error(f"Error importing required libraries: {e}")
    logger.error("Please make sure MLX and MLX-LM are properly installed")
    exit(1)


def log_tensor_stats(name, tensor):
    """Log statistics about a tensor."""
    try:
        if hasattr(tensor, 'numpy'):
            arr = tensor.numpy()
            logger.debug(f"{name} stats: shape={arr.shape}, "
                         f"min={np.min(arr)}, max={np.max(arr)}, "
                         f"mean={np.mean(arr)}, std={np.std(arr)}, "
                         f"has_nan={np.isnan(arr).any()}, has_inf={np.isinf(arr).any()}")
        else:
            logger.debug(f"{name}: Not a tensor or cannot convert to numpy")
    except Exception as e:
        logger.debug(f"Error getting stats for {name}: {e}")


def load_jsonl(file_path: str) -> List[Dict[str, str]]:
    """Load data from a JSONL file."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    # Log first few examples for debugging
                    if i < 3:
                        q_preview = item.get('question', '')[:50] + '...' if len(item.get('question', '')) > 50 else item.get('question', '')
                        a_preview = item.get('answer', '')[:50] + '...' if len(item.get('answer', '')) > 50 else item.get('answer', '')
                        logger.debug(f"Sample entry {i+1}: Q: {q_preview}, A: {a_preview}")
                    data.append(item)
                except json.JSONDecodeError:
                    logger.warning(f"Line {i+1} could not be parsed as JSON, skipping")
    except Exception as e:
        logger.error(f"Error loading JSONL file {file_path}: {e}")
    return data


def format_qa_example(qa_pair: Dict[str, str], tokenizer) -> mx.array:
    """Format a Q&A pair into a tokenized example."""
    question = qa_pair.get("question", "")
    answer = qa_pair.get("answer", "")
    
    text = f"Question: {question}\nAnswer: {answer}"
    
    if len(text) < 200:
        logger.debug(f"Raw text: {text}")
    else:
        logger.debug(f"Raw text (truncated): {text[:200]}...")
    
    # Tokenize returns list
    tokens = tokenizer.encode(text)
    
    logger.debug(f"Tokenized example: length={len(tokens)}, first_few={tokens[:10]}")
    
    return mx.array(tokens)



def prepare_training_data(qa_pairs: List[Dict[str, str]], tokenizer, max_length=512) -> List[mx.array]:
    """Prepare training data for fine-tuning."""
    examples = []
    
    logger.info("Starting to prepare training data...")
    for i, pair in enumerate(qa_pairs):
        try:
            logger.debug(f"Processing example {i+1}/{len(qa_pairs)}")
            
            # Format and tokenize
            tokens = format_qa_example(pair, tokenizer)
            
            # Truncate if too long
            if len(tokens) > max_length:
                logger.debug(f"Example {i+1} truncated from {len(tokens)} to {max_length} tokens")
                tokens = tokens[:max_length]
                
            examples.append(tokens)
        except Exception as e:
            logger.warning(f"Error preparing example {i+1}: {e}")
    
    logger.info(f"Prepared {len(examples)} examples for training")
    
    # Log distribution of example lengths
    if examples:
        lengths = [len(ex) for ex in examples]
        logger.info(f"Example length stats: min={min(lengths)}, max={max(lengths)}, "
                   f"mean={np.mean(lengths):.2f}, median={np.median(lengths)}")
    
    return examples


def print_model_structure(model, level=0, max_level=2):
    """Print the structure of the model."""
    if level > max_level:
        return
    
    indent = "  " * level
    if hasattr(model, '__dict__'):
        for name, value in model.__dict__.items():
            if name.startswith('_'):  # Skip private attributes
                continue
            
            if isinstance(value, nn.Module):
                logger.debug(f"{indent}{name}: {type(value).__name__}")
                print_model_structure(value, level+1, max_level)
            elif hasattr(value, 'shape'):
                logger.debug(f"{indent}{name}: {type(value).__name__}, shape={value.shape}")
            else:
                logger.debug(f"{indent}{name}: {type(value).__name__}")


def extract_trainable_layers(model, trainable_pattern="lm_head"):
    """
    Extract trainable layers based on pattern matching.
    Only layers with names matching the pattern will be trainable.
    
    Args:
        model: The model
        trainable_pattern: Pattern to match for trainable layers
        
    Returns:
        Dictionary of trainable parameters
    """
    logger.info(f"Extracting trainable layers with pattern: {trainable_pattern}")
    
    trainable_params = {}
    frozen_params = {}
    
    # Log model structure to understand available layers
    logger.debug("Model structure:")
    print_model_structure(model)
    
    # Extract all parameters for reference
    all_params = list(model.parameters().items())
    logger.info(f"Model has {len(all_params)} parameter groups")
    
    # Log a few examples of parameter names
    for i, (name, _) in enumerate(all_params[:5]):  # Log first 5
        logger.debug(f"Example parameter name {i+1}: {name}")
    
    # Extract trainable parameters
    for name, param in model.parameters().items():
        if trainable_pattern in name:
            trainable_params[name] = param
            log_tensor_stats(f"Trainable param {name}", param)
        else:
            frozen_params[name] = None  # Just to count
    
    # If no parameters match the pattern, try to find the output layer
    if not trainable_params:
        logger.warning(f"No parameters matched pattern '{trainable_pattern}'. Trying to find output layer...")
        
        # Look for other common output layer names
        output_patterns = ["output", "head", "decoder", "classifier", "projection"]
        for pattern in output_patterns:
            for name, param in model.parameters().items():
                if pattern in name.lower() and name not in trainable_params:
                    trainable_params[name] = param
                    logger.info(f"Found potential output layer: {name}")
                    log_tensor_stats(f"Trainable param {name}", param)
    
    # If still no parameters, take the last few layers as a fallback
    if not trainable_params:
        logger.warning("No output layers found. Using last few layers as fallback.")
        for name, param in all_params[-3:]:  # Use last 3 layers
            trainable_params[name] = param
            logger.info(f"Using fallback layer: {name}")
            log_tensor_stats(f"Trainable param {name}", param)
    
    total_params = sum(np.prod(param.shape) for param in trainable_params.values() if hasattr(param, 'shape'))
    logger.info(f"Training {len(trainable_params)} parameter groups with {total_params} total parameters")
    logger.info(f"Freezing {len(frozen_params)} parameter groups")
    
    return trainable_params


def basic_loss_fn(model, inputs):
    """
    Ultra-simple loss function without any boolean indexing.
    With extensive logging.
    
    Args:
        model: The model to train
        inputs: The input tensor
        
    Returns:
        Scalar loss value
    """
    logger.debug(f"Loss calculation: input shape={inputs.shape}")
    
    # Forward pass
    logger.debug("Starting forward pass")
    start_time = time.time()
    logits = model(inputs)
    forward_time = time.time() - start_time
    logger.debug(f"Forward pass completed in {forward_time:.2f}s, logits shape={logits.shape}")
    
    # Create targets (shifted right by 1)
    targets = mx.concatenate([
        inputs[:, 1:], 
        mx.zeros((inputs.shape[0], 1), dtype=mx.int32)
    ], axis=1)
    
    logger.debug(f"Targets shape={targets.shape}")
    
    # Log a sample of logits and targets
    try:
        logger.debug(f"Sample logits: {logits[0, 0, :5].tolist()}")
        logger.debug(f"Sample targets: {targets[0, :5].tolist()}")
    except Exception as e:
        logger.debug(f"Error logging samples: {e}")
    
    # Reshape logits and targets
    vocab_size = logits.shape[-1]
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)
    
    logger.debug(f"Flattened shapes: logits={logits_flat.shape}, targets={targets_flat.shape}")
    
    # Use cross_entropy with explicit reduction='mean'
    logger.debug("Computing cross entropy loss")
    loss = nn.losses.cross_entropy(logits_flat, targets_flat, reduction='mean')
    
    # Debug: convert to python float if possible
    try:
        float_loss = float(loss)
        logger.debug(f"Loss value={float_loss}, is_nan={np.isnan(float_loss)}, is_inf={np.isinf(float_loss)}")
    except Exception as e:
        logger.debug(f"Error converting loss to float: {e}")
    
    return loss


def frozen_params_fine_tune(
    model,
    tokenizer,
    train_examples,
    trainable_pattern="lm_head",
    batch_size=1,
    epochs=1,
    learning_rate=1e-5,
    checkpoint_dir="checkpoints"
):
    """Fine-tuning with most parameters frozen."""
    # Extract trainable parameters
    trainable_params = extract_trainable_layers(model, trainable_pattern)
    
    if not trainable_params:
        logger.error("No trainable parameters found")
        return model
    
    # Create optimizer with higher learning rate (safe with frozen params)
    optimizer = Adam(learning_rate=learning_rate)
    logger.info(f"Using Adam optimizer with learning rate {learning_rate}")
    
    # Log initial state of trainable parameters
    logger.info("Initial state of trainable parameters:")
    for name, param in trainable_params.items():
        log_tensor_stats(f"Initial {name}", param)
    
    # Create custom loss and gradient function
    def custom_grad_fn(inputs):
        """
        Custom gradient function that only computes gradients for trainable params.
        """
        # Forward pass and compute loss
        loss = basic_loss_fn(model, inputs)
        
        # Initialize gradients dictionary
        grads = {}
        
        # Compute gradients only for trainable parameters
        logger.debug("Computing gradients for trainable parameters")
        for name in trainable_params:
            logger.debug(f"Computing gradient for {name}")
            try:
                # Create a function that computes gradients for this parameter
                def param_loss_fn(param_value):
                    # Replace the parameter temporarily
                    original_value = model.parameters()[name]
                    model.update({name: param_value})
                    
                    # Compute loss
                    result = basic_loss_fn(model, inputs)
                    
                    # Restore original parameter
                    model.update({name: original_value})
                    
                    return result
                
                # Compute gradient for this parameter
                param_grad = mx.grad(param_loss_fn)(model.parameters()[name])
                grads[name] = param_grad
                
                # Log gradient stats
                log_tensor_stats(f"Gradient for {name}", param_grad)
                
            except Exception as e:
                logger.error(f"Error computing gradient for {name}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                grads[name] = None
        
        return loss, grads
    
    # Create checkpoints directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training loop
    logger.info("Starting training")
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")
        
        # Shuffle examples
        random.shuffle(train_examples)
        logger.info(f"Shuffled {len(train_examples)} examples")
        
        # Training metrics
        total_loss = 0.0
        num_batches = 0
        successful_batches = 0
        
        # Process examples
        for i in range(0, len(train_examples), batch_size):
            batch = train_examples[i:i+batch_size]
            
            # Skip incomplete batches
            if len(batch) < batch_size:
                continue
                
            logger.info(f"Processing batch {num_batches+1}")
            
            try:
                # Pad sequences to same length
                max_len = max(len(ex) for ex in batch)
                logger.debug(f"Max sequence length in batch: {max_len}")
                
                padded_batch = []
                for ex in batch:
                    padding = mx.zeros((max_len - len(ex),), dtype=mx.int32)
                    padded = mx.concatenate([ex, padding])
                    padded_batch.append(padded)
                
                # Stack batch
                inputs = mx.stack(padded_batch)
                logger.debug(f"Input batch shape: {inputs.shape}")
                
                # Compute loss and gradients
                logger.debug("Computing loss and gradients")
                start_time = time.time()
                loss, grads = custom_grad_fn(inputs)
                compute_time = time.time() - start_time
                logger.debug(f"Loss and gradient computation completed in {compute_time:.2f}s")
                
                # Skip invalid losses
                if mx.isnan(loss) or mx.isinf(loss):
                    logger.warning(f"Invalid loss value: {float(loss)}, skipping batch")
                    num_batches += 1
                    continue
                
                # Update parameters (only trainable ones)
                logger.debug("Updating trainable parameters")
                model_params = model.parameters()
                updates = {}
                
                for name, grad in grads.items():
                    if grad is not None:
                        # Get current parameter value
                        param = model_params[name]
                        
                        # Simple update rule (SGD-like)
                        updates[name] = param - learning_rate * grad
                        
                        # Log update stats
                        try:
                            param_array = param.numpy()
                            update_array = updates[name].numpy()
                            param_norm = np.linalg.norm(param_array)
                            update_norm = np.linalg.norm(update_array - param_array)
                            rel_change = update_norm / (param_norm + 1e-10)
                            logger.debug(f"Parameter {name}: relative change={rel_change:.8f}")
                        except Exception as e:
                            logger.debug(f"Error computing update stats: {e}")
                
                # Apply updates
                logger.debug("Applying parameter updates")
                model.update(updates)
                
                # Log updated parameters
                logger.debug("Parameters after update:")
                for name in trainable_params:
                    log_tensor_stats(f"Updated {name}", model.parameters()[name])
                
                # Log progress
                loss_value = float(loss)
                total_loss += loss_value
                num_batches += 1
                successful_batches += 1
                
                logger.info(f"Batch {successful_batches}, Loss: {loss_value:.4f}")
                
                # Save checkpoint every successful batch
                # Create checkpoint directory
                interim_path = os.path.join(checkpoint_dir, f"step_{successful_batches}")
                os.makedirs(interim_path, exist_ok=True)
                
                # Convert to NumPy first to avoid serialization issues
                params_dict = {}
                for k, v in model.parameters().items():
                    if hasattr(v, 'numpy'):
                        params_dict[k] = v.numpy()
                
                # Save
                logger.debug(f"Saving checkpoint to {interim_path}")
                np.savez(os.path.join(interim_path, "weights.npz"), **params_dict)
                logger.info(f"Saved checkpoint at step {successful_batches}")
                    
            except Exception as e:
                logger.error(f"Error in batch {num_batches+1}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                num_batches += 1
                continue
        
        # End of epoch
        if successful_batches > 0:
            avg_loss = total_loss / successful_batches
            logger.info(f"Epoch {epoch+1} complete. Average loss: {avg_loss:.4f}")
            logger.info(f"Successful batches: {successful_batches} out of {num_batches} total")
            
            # Save checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}")
            os.makedirs(checkpoint_path, exist_ok=True)
            
            # Convert to NumPy first to avoid serialization issues
            params_dict = {}
            for k, v in model.parameters().items():
                if hasattr(v, 'numpy'):
                    params_dict[k] = v.numpy()
            
            # Save
            np.savez(os.path.join(checkpoint_path, "weights.npz"), **params_dict)
            logger.info(f"Saved checkpoint for epoch {epoch+1}")
        else:
            logger.warning("No successful batches in this epoch, not saving checkpoint")
    
    return model


def test_generation(model, tokenizer, test_examples, num_examples=2):
    """Test the model by generating responses to questions."""
    logger.info("Testing generation...")
    
    if len(test_examples) == 0:
        logger.warning("No test examples provided")
        return
    
    # Select a few examples to test
    if len(test_examples) > num_examples:
        test_subset = random.sample(test_examples, num_examples)
    else:
        test_subset = test_examples
    
    for i, example in enumerate(test_subset):
        question = example.get("question", "")
        expected = example.get("answer", "")
        
        prompt = f"Question: {question}\nAnswer:"
        
        try:
            # Generate response
            logger.info(f"Generating response for example {i+1}...")
            response = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=100,
                temperature=0.1
            )
            
            # Log results
            logger.info(f"Example {i+1}:")
            logger.info(f"Question: {question}")
            logger.info(f"Generated: {response}")
            logger.info(f"Expected: {expected[:100]}...")
        except Exception as e:
            logger.error(f"Error generating response: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Debug Frozen Parameters MLX Fine-tuning")
    
    # Data arguments
    parser.add_argument("--train-data", default="qa_output/train.jsonl", help="Training data path")
    parser.add_argument("--test-data", default="qa_output/test.jsonl", help="Test data path")
    
    # Model arguments
    parser.add_argument("--model", default="mlx-community/Qwen2.5-Coder-32B-Instruct-4bit", help="Base model name")
    parser.add_argument("--output-dir", default="finetuned_model", help="Output directory")
    parser.add_argument("--trainable-pattern", default="lm_head", help="Pattern to match for trainable layers")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-seq-length", type=int, default=512, help="Maximum sequence length")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    mx.random.seed(args.seed)
    
    # Log all arguments
    logger.info("Arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    try:
        # Log MLX version if available
        try:
            import mlx
            logger.info(f"MLX version: {mlx.__version__ if hasattr(mlx, '__version__') else 'unknown'}")
        except Exception as e:
            logger.warning(f"Could not get MLX version: {e}")
        
        # Log system info
        import platform
        logger.info(f"Python version: {platform.python_version()}")
        logger.info(f"System: {platform.platform()}")
        
        # Load data
        logger.info(f"Loading training data from {args.train_data}")
        train_examples = load_jsonl(args.train_data)
        logger.info(f"Loaded {len(train_examples)} training examples")
        
        test_examples = []
        if os.path.exists(args.test_data):
            logger.info(f"Loading test data from {args.test_data}")
            test_examples = load_jsonl(args.test_data)
            logger.info(f"Loaded {len(test_examples)} test examples")
        
        # Load model
        logger.info(f"Loading model: {args.model}")
        model, tokenizer = load(args.model)
        
        # Log model and tokenizer info
        vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else "unknown"
        logger.info(f"Tokenizer vocab size: {vocab_size}")
        
        # Prepare training data
        logger.info("Preparing training data")
        train_data = prepare_training_data(
            train_examples, 
            tokenizer, 
            max_length=args.max_seq_length
        )
        
        # Fine-tune model with frozen parameters
        logger.info("Starting fine-tuning with frozen parameters")
        model = frozen_params_fine_tune(
            model=model,
            tokenizer=tokenizer,
            train_examples=train_data,
            trainable_pattern=args.trainable_pattern,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            checkpoint_dir=os.path.join(args.output_dir, "checkpoints")
        )
        
        # Save final model
        logger.info(f"Saving final model to {args.output_dir}")
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Convert to NumPy first to avoid serialization issues
        params_dict = {}
        for k, v in model.parameters().items():
            if hasattr(v, 'numpy'):
                params_dict[k] = v.numpy()
        
        # Save
        np.savez(os.path.join(args.output_dir, "weights.npz"), **params_dict)
        
        # Save tokenizer
        tokenizer.save_pretrained(args.output_dir)
        
        # Test model
        if test_examples:
            test_generation(model, tokenizer, test_examples)
        
        logger.info("Fine-tuning complete!")
    
    except Exception as e:
        logger.error(f"Error during fine-tuning: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())