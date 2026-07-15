import gc
import json
import time
from functools import partial
import numpy as np
from pathlib import Path
from typing import Dict, Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from mlx.utils import tree_flatten, tree_map
from sklearn.metrics import precision_recall_fscore_support
from datasets import Dataset as HFDataset

from .args import TrainingArgs
from .collator import TextClassificationCollator


GB = 1024 ** 3


def _build_schedule(args: TrainingArgs, total_steps: int):
    warmup_steps = int(total_steps * args.warmup_ratio)
    decay_steps = total_steps - warmup_steps

    if args.lr_scheduler_type == "cosine_decay" and decay_steps > 0:
        schedule_fn = opt.cosine_decay(args.learning_rate, decay_steps)
    elif args.lr_scheduler_type == "linear_schedule" and decay_steps > 0:
        schedule_fn = opt.linear_schedule(args.learning_rate, 0.0, decay_steps)
    else:
        # Constant LR — return mx.array to avoid .astype errors in optimizer
        schedule_fn = lambda _: mx.array(args.learning_rate)

    if warmup_steps > 0:
        warmup_fn = opt.linear_schedule(0.0, args.learning_rate, warmup_steps)
        return opt.join_schedules([warmup_fn, schedule_fn], [warmup_steps])

    return schedule_fn


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        training_args: TrainingArgs,
        train_dataset: HFDataset,
        eval_dataset: Optional[HFDataset] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer._tokenizer if hasattr(tokenizer, "_tokenizer") else tokenizer
        self.args = training_args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.collator = TextClassificationCollator(
            tokenizer=self.tokenizer,
            max_length=training_args.max_length,
        )

        # Output dir
        self.output_dir = Path(training_args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Steps calc
        steps_per_epoch = len(train_dataset) // training_args.batch_size
        if len(train_dataset) % training_args.batch_size != 0:
            steps_per_epoch += 1
        steps_per_epoch = max(steps_per_epoch // training_args.gradient_accumulation_steps, 1)
        total_steps = steps_per_epoch * training_args.num_train_epochs

        # Optimizer
        schedule = _build_schedule(training_args, total_steps)
        self.optimizer = opt.AdamW(
            learning_rate=schedule,
            weight_decay=training_args.weight_decay,
        )

        # Resume from checkpoint if specified
        if training_args.resume_from_checkpoint:
            ckpt_path = Path(training_args.resume_from_checkpoint)
            # Load model weights
            model_weights = mx.load(str(ckpt_path / "model.safetensors"))
            self.model.load_weights(list(model_weights.items()))
            mx.eval(self.model.parameters())
            # Load optimizer state
            opt_path = ckpt_path / "optimizer.safetensors"
            if opt_path.exists():
                opt_state_dict = mx.load(str(opt_path))
                opt_state_list = list(opt_state_dict.items())
                self.optimizer.state.update(dict(opt_state_list))
                mx.eval(self.optimizer.state)
            print(f"Resumed from {ckpt_path}")

        # State
        self.global_step = 0
        self.logging_steps = training_args.logging_steps
        self.save_steps = training_args.save_steps
        self.next_log_step = self.logging_steps
        self.next_save_step = self.save_steps
        self.best_eval_loss = float("inf")
        self.epochs_without_improvement = 0

        self.state = [self.model.state, self.optimizer.state, mx.random.state]
        
        if self.args.grad_checkpoint:
            self._apply_grad_checkpointing()

        # Compile loss + grad (loss in fp32 for numerical stability)
        def loss_fn(model, batch):
            out = model(**batch)
            return mx.mean(out["loss"]).astype(mx.float32)

        grad_fn = nn.value_and_grad(self.model, loss_fn)

        step_state = [self.model.state, mx.random.state]
        @partial(mx.compile, inputs=step_state, outputs=step_state)
        def step_fn(batch):
            loss, grads = grad_fn(self.model, batch)
            return loss, grads

        # update_fn muta el modelo, el RNG y ADEMÁS el estado del optimizador
        update_state = [self.model.state, self.optimizer.state, mx.random.state]
        @partial(mx.compile, inputs=update_state, outputs=update_state)
        def update_fn(accumulated_grads, scale):
            scaled = tree_map(lambda g: (g * scale).astype(mx.float32), accumulated_grads)
            clipped, _ = opt.clip_grad_norm(scaled, self.args.max_grad_norm)
            self.optimizer.update(self.model, clipped)
            return None

        eval_state = [self.model.state]
        @partial(mx.compile, inputs=eval_state, outputs=eval_state)
        def eval_fn(batch):
            out = self.model(**batch)
            loss = mx.mean(out["loss"]).astype(mx.float32)
            preds = mx.argmax(out["probabilities"], axis=-1)
            return loss, preds
        
        
        self.update_fn = update_fn
        self.step_fn = step_fn
        self.eval_fn = eval_fn

        print(f"Trainer: {model.__class__.__name__} | "
              f"Train: {len(train_dataset)} | "
              f"Steps/epoch: {steps_per_epoch} | "
              f"Total steps: {total_steps}")

    def _apply_grad_checkpointing(self):
        """
        Aplica Gradient Checkpointing a las capas del modelo para reducir drásticamente
        el uso de RAM, recalculando las activaciones durante el backward pass.
        """
        def checkpoint_fn(module):
            original_call = module.__call__

            def checkpointed_call(self, **kwargs):
                # MLX maneja la caché bajo el capó, solo necesitamos envolver la llamada
                return mx.checkpoint(original_call)(self, **kwargs)

            module.__call__ = checkpointed_call
        
        layers = None
        
        # ModernBERT y arquitecturas HuggingFace típicas en MLX
        if hasattr(self.model, "layers"): 
            layers = self.model.layers
        elif hasattr(self.model, "model"):
            if hasattr(self.model.model, "layers"): 
                layers = self.model.model.layers
            elif hasattr(self.model.model, "encoder"): 
                if hasattr(self.model.model.encoder, "layers"):
                    layers = self.model.model.encoder.layers

        if layers is None:
            print("  [Warning] No se encontraron capas para hacer checkpoint. El ahorro de memoria no se aplicará.")
            return

        print(f"  [Opt] Aplicando Gradient Checkpointing a {len(layers)} capas.")
        for layer in layers:
            checkpoint_fn(layer)
            

    def _batches(self, dataset, batch_size, shuffle=False, seed=42):
        if shuffle:
            dataset = dataset.shuffle(seed=seed)
        for i in range(0, len(dataset), batch_size):
            yield dataset[i : min(i + batch_size, len(dataset))]

    def train(self):
        print("Starting training...\n")
        for epoch in range(self.args.num_train_epochs):
            print(f"Epoch {epoch + 1}/{self.args.num_train_epochs}")
            self._train_epoch()

            if self.eval_dataset is not None:
                metrics = self.evaluate()
                self._save_checkpoint(metrics)

                # Early stopping check
                if self.args.early_stopping_patience is not None:
                    current_loss = metrics["eval_loss"]
                    if current_loss < self.best_eval_loss:
                        self.best_eval_loss = current_loss
                        self.epochs_without_improvement = 0
                        print(f"  ✓ New best eval_loss: {current_loss:.4f}")
                    else:
                        self.epochs_without_improvement += 1
                        print(f"  No improvement for {self.epochs_without_improvement}/{self.args.early_stopping_patience} epochs")
                        if self.epochs_without_improvement >= self.args.early_stopping_patience:
                            print(f"\n  ⚠ Early stopping triggered after {epoch + 1} epochs")
                            break

        print("\nTraining complete.")

    def _train_epoch(self):
        self.model.train()
        n_steps = 0
        start = time.time()

        accum = None
        accum_steps = self.args.gradient_accumulation_steps
        scale = 1.0 / accum_steps if accum_steps > 1 else 1.0
        running_loss = mx.array(0.0)

        for raw_batch in self._batches(
            self.train_dataset, self.args.batch_size,
            shuffle=True, seed=self.args.seed + self.global_step
        ):
            self.global_step += 1
            batch = self.collator(raw_batch)

            loss, grads = self.step_fn(batch)

            if accum is None:
                accum = grads
            else:
                accum = tree_map(lambda a, b: a + b, accum, grads)

            mx.eval(loss, accum)
            running_loss = running_loss + loss
            n_steps += 1

            if n_steps % accum_steps == 0:
                self.update_fn(accum, scale)
                accum = None
                mx.eval(self.model.state, self.optimizer.state)
                
                mx.clear_cache()

                if self.global_step >= self.next_log_step:
                    avg_loss = running_loss.item() / n_steps
                    lr = self.optimizer.learning_rate
                    if callable(lr):
                        lr = lr(self.optimizer.step)
                    if isinstance(lr, mx.array):
                        lr = lr.item()

                    elapsed = time.time() - start
                    speed = n_steps / elapsed
                    
                    mem_active = mx.get_active_memory() / GB
                    mem_peak = mx.get_peak_memory() / GB
                    mem_cache = mx.get_cache_memory() / GB

                    print(
                        f"  Step {self.global_step:>5d} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {lr:.2e} | "
                        f"Speed: {speed:.1f} steps/s\n"
                        f"  └─> RAM - Active: {mem_active:.1f}GB | Peak: {mem_peak:.1f}GB | Cache: {mem_cache:.1f}GB"
                    )
                    
                    mx.reset_peak_memory()

                    if self.args.eval_steps and self.global_step % self.args.eval_steps == 0:
                        if self.eval_dataset is not None:
                            self.evaluate()
                            self.model.train()

                    running_loss = mx.array(0.0)
                    n_steps = 0
                    start = time.time()
                    self.next_log_step += self.logging_steps

                    if self.global_step >= self.next_save_step:
                        self._save_checkpoint({})
                        self.next_save_step += self.save_steps

    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        total_correct = 0
        total_samples = 0
        batch_preds = []
        batch_labels = []

        # 1. Limpieza preventiva antes de empezar a evaluar para evitar
        # que se junte la memoria de entrenamiento con la de evaluación
        gc.collect()
        mx.clear_cache()

        for raw_batch in self._batches(self.eval_dataset, self.args.eval_batch_size):
            batch = self.collator(raw_batch)
            labels = batch["labels"]
            
            # Usamos la función compilada: 10x más rápida y no destroza la RAM
            loss, preds = self.eval_fn(batch)
            
            # Forzamos a MLX a calcular el grafo compilado de inmediato
            mx.eval(loss, preds)

            total_loss += loss.item()
            n_batches += 1
            
            total_correct += int(mx.sum(preds == labels).item())
            total_samples += len(labels)
            
            # Movemos a NumPy para que no moleste a la caché de MLX
            batch_preds.append(np.array(preds))
            batch_labels.append(np.array(labels))

            # 2. CRÍTICO: Eliminar referencias locales y vaciar la caché en cada paso
            del loss, preds, batch
            mx.clear_cache()

        # 4. Concatenamos en NumPy (mucho más ligero y no estresa el backend de MLX)
        all_preds_np = np.concatenate(batch_preds)
        all_labels_np = np.concatenate(batch_labels)
        
        # ... (aquí sigue el resto de tu código calculando precisión, recall, etc) ...

        eval_loss = total_loss / max(n_batches, 1)
        accuracy = total_correct / max(total_samples, 1)

        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels_np, all_preds_np, average="weighted", zero_division=0
        )


        metrics = {
            "eval_loss": eval_loss,
            "accuracy": accuracy,
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
        }
        print(f"  Eval — loss: {metrics['eval_loss']:.4f} | "
              f"acc: {metrics['accuracy']:.4f} | "
              f"f1: {metrics['f1']:.4f}")
        return metrics

    def _save_checkpoint(self, metrics: Dict):
        step = self.global_step
        path = self.output_dir / f"checkpoint-{step}"
        path.mkdir(exist_ok=True)

        # Save weights
        weights = dict(tree_flatten(self.model.parameters()))
        mx.save_safetensors(str(path / "model.safetensors"), weights)

        # Save optimizer state
        opt_state = dict(tree_flatten(self.optimizer.state))
        mx.save_safetensors(str(path / "optimizer.safetensors"), opt_state)

        # Save tokenizer
        self.tokenizer.save_pretrained(str(path))

        # Save config if available
        if hasattr(self.model, "config"):
            with open(path / "config.json", "w") as f:
                json.dump(self.model.config.__dict__, f, indent=2)

        with open(path / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"  Saved checkpoint: {path}")

        # Rotate old checkpoints
        self._rotate_checkpoints()

    def _rotate_checkpoints(self):
        limit = self.args.save_total_limit
        if limit is None:
            return
        checkpoints = sorted(
            [d for d in self.output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda d: int(d.name.split("-")[1]),
        )
        while len(checkpoints) > limit:
            old = checkpoints.pop(0)
            for f in old.iterdir():
                f.unlink()
            old.rmdir()
            print(f"  Removed old checkpoint: {old.name}")
