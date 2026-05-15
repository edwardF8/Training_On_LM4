"""Training loop for Capo bioS pretraining.

Wraps HuggingFace `Trainer` so we get free logging, checkpointing, and
a cosine LR schedule with minimal boilerplate. Swap to a manual loop later
if you want finer-grained mechinterp hooks during training.
"""
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments, default_data_collator


def train(model, dataset, config, output_dir="runs/exp1"):
    """Train `model` on `dataset` according to `config`.

    Args:
        model: HF causal-LM model (LlamaForCausalLM or GPT2LMHeadModel).
        dataset: PackedTokenDataset returning {"input_ids", "labels"}.
        config: Config object with BATCH_SIZE, LR, WEIGHT_DECAY, WARMUP_STEPS,
                MAX_STEPS, GRAD_CLIP, SEED.
        output_dir: where checkpoints + final model go.

    Returns:
        the trained Trainer (so you can grab metrics, the model, etc.).
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # bf16 only on real CUDA GPUs that support it; fp32 on CPU/MPS.
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    # Run name: trailing path component of output_dir (e.g. "2L-192D")
    run_name = Path(output_dir).name

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.BATCH_SIZE,
        learning_rate=config.LR,
        weight_decay=config.WEIGHT_DECAY,
        warmup_steps=config.WARMUP_STEPS,
        max_steps=config.MAX_STEPS,
        max_grad_norm=config.GRAD_CLIP,
        lr_scheduler_type="cosine",
        logging_steps=50,
        save_steps=2000,
        save_total_limit=5,
        report_to="wandb",
        run_name=run_name,
        seed=config.SEED,
        bf16=use_bf16,
        dataloader_num_workers=2,
        remove_unused_columns=False,   # keep the `labels` key
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        # Default collator just stacks dicts of equal-length tensors — no
        # padding logic, no MLM masking. Perfect for our pre-packed data.
        data_collator=default_data_collator,
    )

    trainer.train()
    trainer.save_model(f"{output_dir}/final")
    return trainer
