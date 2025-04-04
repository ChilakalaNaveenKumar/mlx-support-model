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
import math
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

# -----------------------------
# Global model reference
# -----------------------------
model = None

def loss_fn(inputs):
    global model
    # Add input validation
    if inputs is None or inputs.shape[0] == 0:
        return mx.array(0.0)
    
    try:
        # Forward pass with gradient scaling
        logits = model(inputs)
        
        # Add numerical stability to logits
        logits = mx.clip(logits, -10, 10)  # Reduced clipping range
        
        # Create targets with proper handling
        targets = mx.concatenate(
            [inputs[:, 1:], mx.zeros((inputs.shape[0], 1), dtype=mx.int32)], axis=1
        )

        # Create mask with numerical stability
        pad_token_id = 0
        mask = (targets != pad_token_id).astype(mx.float32)
        mask = mx.maximum(mask, 1e-6)  # Prevent division by zero
        
        # Compute loss with numerical stability
        loss = nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1)
        )
        
        # Add numerical stability to loss
        loss = mx.clip(loss, -10, 10)  # Reduced clipping range
        
        # Compute masked loss with numerical stability
        masked_loss = loss * mask.reshape(-1)
        mask_sum = mx.maximum(mask.sum(), 1.0)  # Prevent division by zero
        final_loss = masked_loss.sum() / mask_sum
        
        # Add final numerical stability check
        if math.isnan(float(final_loss)) or math.isinf(float(final_loss)):
            logger.warning("Loss became NaN or Inf, returning zero loss")
            return mx.array(0.0)
        
        logger.info(f"Mask sum: {mask.sum()}, Loss mean: {final_loss.mean()}")
        return final_loss
    except Exception as e:
        logger.error(f"Error in loss computation: {e}")
        return mx.array(0.0)



def save_model(model, tokenizer, output_path):
    os.makedirs(output_path, exist_ok=True)
    model.save_weights(os.path.join(output_path, "weights.safetensors"))
    tokenizer.save_pretrained(output_path)
    if hasattr(model, 'config'):
        try:
            with open(os.path.join(output_path, "config.json"), "w") as f:
                json.dump(model.config, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save model config: {e}")
    logger.info(f"Model saved to {output_path}")


def load_dataset(file_path: str) -> List[Dict[str, str]]:
    try:
        _, ext = os.path.splitext(file_path)
        if ext == '.jsonl':
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
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif ext == '.csv':
            import csv
            data = []
            with open(file_path, 'r', encoding='utf-8', newline='') as f:
                reader = csv.reader(f)
                header = next(reader)
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
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant with knowledge about files and code."},
            {"role": "user", "content": qa_pair["question"]},
            {"role": "assistant", "content": qa_pair["answer"]}
        ]
        try:
            formatted_text = tokenizer.apply_chat_template(messages, tokenize=False)
            return formatted_text
        except Exception as e:
            logger.warning(f"Error applying chat template: {e}. Using fallback format.")
            return f"Question: {qa_pair['question']}\nAnswer: {qa_pair['answer']}"
    except Exception as e:
        logger.error(f"Error formatting QA pair: {e}")
        return "Error formatting QA pair"


def prepare_training_data(qa_pairs: List[Dict[str, str]], tokenizer) -> mx.array:
    try:
        formatted_texts = []
        for pair in qa_pairs:
            if "question" in pair and "answer" in pair:
                formatted_text = format_qa_for_training(pair, tokenizer)
                formatted_texts.append(formatted_text)
            else:
                logger.warning(f"Skipping malformed QA pair: {pair}")
        
        tokenized_data = []
        max_length = 0
        for text in formatted_texts:
            try:
                tokens = tokenizer.encode(text)
                max_length = max(max_length, len(tokens))
                tokenized_data.append(tokens)
            except Exception as e:
                logger.warning(f"Error tokenizing text: {e}")
        
        if not tokenized_data:
            logger.error("No valid training data after tokenization")
            return mx.array([[0]])
        
        # Pad all sequences to the same length
        pad_token = getattr(tokenizer, "pad_token_id", 0)
        padded = [
            tokens + [pad_token] * (max_length - len(tokens))
            for tokens in tokenized_data
        ]
        return mx.array(padded)

    except Exception as e:
        logger.error(f"Error preparing training data: {e}")
        return mx.array([[0]])


def main():
    global model

    parser = argparse.ArgumentParser(description="Fine-tune MLX model on Q&A dataset")
    parser.add_argument("--train-data", default="qa_output/train.jsonl", help="Path to training data")
    parser.add_argument("--test-data", default="qa_output/test.jsonl", help="Path to test data")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-Coder-32B-Instruct-6bit", help="Model to fine-tune")
    parser.add_argument("--output-model", default="finetuned_model", help="Output path for fine-tuned model")
    parser.add_argument("--learning-rate", type=float, default=1e-7, help="Learning rate")  # Further reduced learning rate
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--grad-clip", type=float, default=0.01, help="Gradient clipping value")  # Further reduced gradient clipping
    args = parser.parse_args()

    try:
        logger.info(f"Loading model: {args.model}")
        model, tokenizer = load(args.model)

        logger.info(f"Loading training data: {args.train_data}")
        train_qa_pairs = load_dataset(args.train_data)
        logger.info(f"Loaded {len(train_qa_pairs)} training Q&A pairs")

        test_qa_pairs = None
        if os.path.exists(args.test_data):
            logger.info(f"Loading test data: {args.test_data}")
            test_qa_pairs = load_dataset(args.test_data)
            logger.info(f"Loaded {len(test_qa_pairs)} test Q&A pairs")

        logger.info("Preparing training data...")
        train_data = prepare_training_data(train_qa_pairs, tokenizer)

        test_data = None
        if test_qa_pairs:
            logger.info("Preparing test data...")
            test_data = prepare_training_data(test_qa_pairs, tokenizer)

        logger.info("Starting fine-tuning...")
        optimizer = Adam(learning_rate=args.learning_rate)
        batch_size = args.batch_size

        for epoch in range(args.epochs):
            total_loss = 0.0
            num_batches = 0
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i+batch_size]
                loss_fn_with_grads = nn.value_and_grad(model, loss_fn)
                loss, grads = loss_fn_with_grads(batch)

                # Skip NaN batches
                if math.isnan(float(loss)):
                    logger.warning(f"Epoch {epoch+1}, Batch {i//batch_size}: Loss is NaN — skipping.")
                    continue

                # Debug logging for gradient values
                def log_gradient_stats(grad_dict, prefix=""):
                    if not isinstance(grad_dict, dict):
                        if hasattr(grad_dict, 'shape'):
                            logger.info(f"{prefix}Gradient shape: {grad_dict.shape}, mean: {float(grad_dict.mean())}, max: {float(grad_dict.max())}, min: {float(grad_dict.min())}")
                        return
                    
                    for key, value in grad_dict.items():
                        if value is not None:
                            log_gradient_stats(value, prefix + f"{key}.")

                logger.info("Gradient statistics before clipping:")
                log_gradient_stats(grads)

                # Apply gradient clipping
                def clip_gradients(grad_dict):
                    if not isinstance(grad_dict, dict):
                        if hasattr(grad_dict, 'shape'):
                            return mx.clip(grad_dict, -args.grad_clip, args.grad_clip)
                        return grad_dict
                    
                    clipped = {}
                    for key, value in grad_dict.items():
                        if value is not None:
                            clipped[key] = clip_gradients(value)
                        else:
                            clipped[key] = None
                    return clipped

                # Apply clipping to all gradients
                grads = clip_gradients(grads)

                # Log gradient statistics after clipping
                logger.info("Gradient statistics after clipping:")
                log_gradient_stats(grads)

                # Check for NaN in gradients before update
                def check_gradients_for_nan(grad_dict):
                    if not isinstance(grad_dict, dict):
                        if hasattr(grad_dict, 'shape'):
                            return mx.any(mx.isnan(grad_dict)).item()
                        return False
                    
                    for value in grad_dict.values():
                        if value is not None and check_gradients_for_nan(value):
                            return True
                    return False

                if check_gradients_for_nan(grads):
                    logger.warning("NaN detected in gradients after clipping, skipping update")
                    continue

                optimizer.update(model, grads)
                total_loss += float(loss)
                num_batches += 1

                if i % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{args.epochs}, Batch {i//batch_size}, Loss: {float(loss)}")

            avg_loss = total_loss / max(1, num_batches)
            logger.info(f"Epoch {epoch+1}/{args.epochs} complete, Average Loss: {avg_loss}")

            if test_data is not None:
                val_loss = 0.0
                val_batches = 0
                for i in range(0, len(test_data), batch_size):
                    batch = test_data[i:i+batch_size]
                    val_loss += loss_fn(batch)
                    val_batches += 1
                avg_val_loss = val_loss / max(1, val_batches)
                logger.info(f"Validation Loss: {avg_val_loss}")

        logger.info(f"Saving fine-tuned model to {args.output_model}")
        save_model(model, tokenizer, args.output_model)
        logger.info("Fine-tuning complete!")

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
