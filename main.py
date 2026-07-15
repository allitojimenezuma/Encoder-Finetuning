import os
import pandas as pd
from datasets import Dataset, load_dataset
import mlx_raclate.tuner.model_card_utils as model_card_utils
from mlx_raclate.utils.utils import load
from mlx_raclate.tuner.trainer import Trainer, TrainingArgs
from sklearn.model_selection import train_test_split

# Parche para evitar el error de importación en la generación de model cards de mlx-raclate
model_card_utils.get_inference_code = lambda *args, **kwargs: "# Snippet omitido\n"
os.environ["HF_TOKEN"] = "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"

def load_and_curate_dataset(seed: int = 3407, test_size: float = 0.15):
    print("\n[1/4] Descargando y curando datasets de alta calidad...")
    
    # 1. Cargar WildJailbreak (Para balancear Ataques complejos vs Hard Benigns)
    print("  -> Cargando allenai/wildjailbreak...")
    wild_dataset = load_dataset(
        "allenai/wildjailbreak", 
        "train", 
        split="train", 
        delimiter="\t", 
        keep_default_na=False
    ).to_pandas()
    
    # Mapeamos a nuestro formato binario usando la columna correcta: 'data_type'
    # Columnas: 'vanilla' para prompts vanilla, 'adversarial' para prompts adversariales
    df_safe_easy = (
        wild_dataset[wild_dataset['data_type'] == 'vanilla_benign']
        .sample(5000, random_state=seed)
        .assign(text=lambda x: x['vanilla'])
        .assign(label=0)
    )
    df_vanilla_harmful = (
        wild_dataset[wild_dataset['data_type'] == 'vanilla_harmful']
        .sample(8000, random_state=seed)
        .assign(text=lambda x: x['vanilla'])
        .assign(label=1)
    )
    
    # 2. Cargar deepset/prompt-injections (todo, 546 muestras)
    print("  -> Cargando deepset/prompt-injections...")
    deepset = load_dataset("deepset/prompt-injections", split="train").to_pandas()
    
    # 3. Cargar un dataset de conversación normal limpia (Fondo / Background noise)
    print("  -> Cargando HuggingFaceH4/no_robots (Conversaciones benignas)...")
    norobots = load_dataset("HuggingFaceH4/no_robots", split="train").to_pandas()
    # Tomamos solo el primer mensaje (el del usuario)
    norobots["text"] = norobots["messages"].apply(lambda x: x[0]["content"] if len(x) > 0 else None)
    norobots.dropna(subset=["text"], inplace=True)
    df_safe_conversational = norobots.sample(5000, random_state=seed)[["text"]].copy()
    df_safe_conversational["label"] = 0

    # 3. Fusionar todo en un único DataFrame equilibrado de ~28,000 muestras
    df = pd.concat([
        # df_inj[["text", "label"]], 
        # df_safe_hard[["text", "label"]], 
        df_safe_easy[["text", "label"]],
        df_vanilla_harmful[["text", "label"]],
        df_safe_conversational,
        deepset[["text", "label"]]
    ], ignore_index=True)
    
    # 4. Limpieza estricta de duplicados y nulos
    df.drop_duplicates(subset="text", inplace=True)
    df.dropna(subset=["text", "label"], inplace=True)
    df["label"] = df["label"].astype(int)
    
    print(f"  -> Total curado y balanceado: {len(df)} muestras")
    print(f"     * SAFE (Benignos normales + Hard Benignos): {len(df[df['label'] == 0])}")
    print(f"     * INJECTION (Adversariales / Jailbreaks):   {len(df[df['label'] == 1])}")
    
    # Guardar dataset completo para inspección
    df.to_csv("dataset_curated.csv", index=False)
    print(f"  -> Dataset guardado en 'dataset_curated.csv'")

    # 5. Split estratificado para mantener proporciones en train/val
    train_df, eval_df = train_test_split(
        df, 
        test_size=test_size, 
        stratify=df["label"], 
        random_state=seed
    )
    
    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    eval_dataset = Dataset.from_pandas(eval_df, preserve_index=False)
    
    return train_dataset, eval_dataset

def main():
    print("=" * 70)
    print("Fine-Tuning ModernBERT-large — Prompt Injection Guard")
    print("=" * 70)

    train_dataset, eval_dataset = load_and_curate_dataset()

    # Mapeo de etiquetas
    id2label = {0: "SAFE", 1: "INJECTION"}
    label2id = {"SAFE": 0, "INJECTION": 1}

    print(f"\nDistribución de etiquetas (train):")
    label_counts = train_dataset.to_pandas()["label"].value_counts().sort_index()
    for lbl, count in label_counts.items():
        print(f"  {id2label[lbl]}: {count}")

    # 2. Cargar el modelo base
    model_id = "answerdotai/ModernBERT-large"
    print(f"\n[2/4] Cargando modelo base: {model_id}...")
    
    model, tokenizer = load(
        model_id,
        pipeline="text-classification",
        train=True,
        model_config={
            "num_labels": 2,
            "id2label": id2label,
            "label2id": label2id,
            "torch_dtype": "float16"
        }
    )

    # 3. Configurar TrainingArgs
    print("\n[3/4] Configurando TrainingArgs...")
    args = TrainingArgs(
        output_dir="outputs/modernbert_guardrail",
        learning_rate=1e-5,          # LR óptimo para ModernBERT
        num_train_epochs=1,           # 3 épocas suelen bastar para convergencia
        batch_size=4,                 # Se adapta cómodamente a tu RAM
        gradient_accumulation_steps=8, # Batch efectivo = 16
        logging_steps=40,           # Muestra logs más a menudo para vigilar la loss,
        warmup_ratio=0.1,        # Dedica el primer 10% del entrenamiento a subir el LR progresivamente
        weight_decay=0.1,        # Penaliza pesos excesivamente grandes para regularizar el modelo        
        eval_steps=480
    )

    # 4. Entrenar
    print("\n[4/4] Iniciando entrenamiento en Metal GPU...")
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        task_type="text-classification",
        training_args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    print("\n✓ ¡Entrenamiento completado! Modelo guardado en 'outputs/modernbert_guardrail'.")

if __name__ == "__main__":
    main()