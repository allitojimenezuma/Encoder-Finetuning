from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingArgs:
    output_dir: str = "outputs"
    num_train_epochs: int = 3
    batch_size: int = 8
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine_decay"
    max_length: int = 512
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: Optional[int] = None
    save_total_limit: Optional[int] = 3
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None
    early_stopping_patience: Optional[int] = None
    grad_checkpoint: bool = True
