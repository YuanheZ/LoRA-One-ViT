import torch
import torch.nn.functional as F
import os
import hydra
import json
import math
import wandb
from typing import List
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from custom_peft.tuners.lora.layer import Linear as LoraLinear
from typing import Tuple, List, Dict
from transformers import CLIPModel, CLIPProcessor, TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from custom_peft import PeftModel, LoraGAConfig, get_peft_model
from custom_peft.utils.lora_ga_utils import estimate_gradient, LoraGAContext, save_loraga_model_init, save_loraga_model_final
import logging
from lora_plus import LoraPlusTrainingArguments, LoraPlusTrainer
from prec_logvitTrainer import PrecTrainingArguments, LogTrainer

log = logging.getLogger(__name__)

# Load CLIP model and processor
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Define datasets
dataset_names = ["cifar10", "cifar100", "stanford_cars", "svhn", "dtd"]

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def find_all_linear_modules(model) -> List[str]:
    r"""
    Finds all available modules to apply lora.
    """
    linear_cls = torch.nn.Linear

    output_layer_names = ["lm_head", "embed_tokens"]

    module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, linear_cls) and not any(
            [output_layer in name for output_layer in output_layer_names]
        ):
            module_names.add(name.split(".")[-1])
    return list(module_names)

def train_vit_model(
    run_name: str,
    train_dataset: Dataset,
    valid_dataset: Dataset,
    model: torch.nn.Module,
    model_type: str,
    per_device_batch_size: int = 1,
    real_batch_size: int = 32,
    **kwargs,
) -> torch.nn.Module:
    # Preprocess the dataset
    train_dataset = preprocess_dataset(train_dataset)
    valid_dataset = preprocess_dataset(valid_dataset)

    assert (
        real_batch_size % per_device_batch_size == 0
    ), "real_batch_size must be divisible by per_device_batch_size"
    accu_step = real_batch_size // per_device_batch_size

    eval_steps = (
        int(len(train_dataset) * kwargs.get("eval_epochs", 1)) // real_batch_size
    )
    # Special for lorqplus
    use_loraplus = kwargs.get("use_loraplus", False)
    TrainingArgumentsClass = (
        LoraPlusTrainingArguments if use_loraplus else PrecTrainingArguments
    )
    TrainerClass = LoraPlusTrainer if use_loraplus else LogTrainer
    if use_loraplus:
        additional_kwargs = {
            "loraplus_lr_ratio": kwargs.get("loraplus_lr_ratio", 1.0),
        }
        log.info(
            f"Begin training using LoraPlusTrainer with additional kwargs: {additional_kwargs}"
        )
    else:
        additional_kwargs = {
            "prec_reg": kwargs.get("prec_reg", 1e-6),
        }
        log.info("Begin training using ViT trainer")

    # Training arguments
    output_dir = f"./results/{run_name}/{kwargs.get('seed')}"
    training_args = TrainingArgumentsClass(
        output_dir=output_dir,  # output directory
        num_train_epochs=kwargs.get(
            "num_train_epochs", 3
        ),  # total number of training epochs
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=accu_step,
        logging_dir="./logs",  # directory for storing logs
        logging_steps=kwargs.get("logging_steps", 10),  # when to print log
        bf16=kwargs.get("bf16", False),
        gradient_checkpointing=kwargs.get("gradient_checkpointing", False),
        optim=kwargs.get("optim", "adamw_torch"),
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_steps=eval_steps,
        save_strategy="steps",
        save_total_limit=1,  # No need for saving
        load_best_model_at_end=kwargs.get("load_best_model_at_end", True),
        metric_for_best_model=kwargs.get("metric_for_best_model", "eval_loss"),
        greater_is_better=kwargs.get("greater_is_better", False),
        do_eval=True,
        learning_rate=kwargs.get("learning_rate", 5e-5),
        remove_unused_columns=False,  # We tokenize the dataset on the fly
        eval_accumulation_steps=kwargs.get("eval_accumulation_steps", real_batch_size),
        label_names=[
            "labels"
        ],  # Peft are not compatible with HF's default label names yet
        # Ref: https://discuss.huggingface.co/t/eval-with-trainer-not-running-with-peft-lora-model/53286
        weight_decay = kwargs.get("weight_decay", 0), # No weight decay
        warmup_ratio = 0.03, # 0.03
        lr_scheduler_type = "cosine", # constant
        seed = kwargs.get("seed", 42),
        **additional_kwargs,
    )

    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        data_collator=collate_fn_for_training,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=kwargs.get("early_stopping_patience", 1)
            ),
        ],
    )

    trainer.train()

    return model

# Function to prepare text embeddings for a dataset
def get_text_embeddings(dataset_name, processor, model):
    dataset = load_dataset(dataset_name)
    class_names = dataset["train"].features["label"].names
    texts = [f"a photo of a {class_name}" for class_name in class_names]
    text_inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        text_embeddings = model.get_text_features(**text_inputs)
    return text_embeddings, class_names

# Custom model for classification
class CLIPForImageClassification(torch.nn.Module):
    def __init__(self, peft_model, text_embeddings):
        super().__init__()
        self.peft_model = peft_model
        self.text_embeddings = text_embeddings

    def forward(self, pixel_values, labels=None):
        image_embeddings = self.peft_model.get_image_features(pixel_values)
        logits = (image_embeddings @ self.text_embeddings.T) * self.peft_model.logit_scale.exp()
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return loss
        return logits

# Collate function for gradient estimation (contrastive loss)
def collate_fn_for_gradient(batch):
    images = [item["img"] if "img" in item else item["image"] for item in batch]
    labels = [item["label"] for item in batch]
    texts = [f"a photo of a {class_names[label]}" for label in labels]
    inputs = processor(images=images, text=texts, return_tensors="pt", padding=True)
    return inputs

# Collate function for training (classification)
def collate_fn_for_training(batch):
    images = [item["img"] if "img" in item else item["image"] for item in batch]
    labels = [item["label"] for item in batch]
    pixel_values = processor(images=images, return_tensors="pt", padding=True)["pixel_values"]
    return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}

# Compute metrics for accuracy
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(-1)
    accuracy = load_metric("accuracy")
    return accuracy.compute(predictions=predictions, references=labels)

@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def run_exp(cfg: DictConfig):
    log.info(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed)
    model_name = cfg.model.name
    model_type = cfg.model.type
    dataset_name = cfg.dataset_name
    use_peft = cfg.peft.use_peft
    if_use_rslora = cfg.peft.use_rslora
    lora_r = cfg.peft.lora_r
    lora_relative_r = cfg.peft.lora_relative_r
    lora_target_modules = cfg.peft.lora_target_modules
    train_embeddings = cfg.peft.train_embeddings
    if cfg.dry_run:
        return
    if use_peft:
        assert (lora_r is not None) ^ (
            lora_relative_r is not None
        ), "Please specify lora_r or lora_relative_r"
        assert lora_target_modules is not None, "Please specify lora_target_modules"
    else:
        lora_r = None
        lora_target_modules = None
        lora_relative_r = None
        train_embeddings = True
    config = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "use_peft": use_peft,
        "lora_r": lora_r,
        "lora_target_modules": str(lora_target_modules),
        "lora_relative_r": lora_relative_r,
        "train_embeddings": train_embeddings,
    }
    if cfg.wandb.name:
        name = cfg.wandb.name
    else:
        name = "_".join([f"{k}={v}" for k, v in config.items()])
    cfg.wandb.project += "_" + cfg.dataset_name
    wandb.init(
        entity="yuanhezhang6-university-of-warwick",
        project=cfg.wandb.project,
        name=name,
        config=config,
    )
    
    print(f"Processing {dataset_name}...")
    dataset = load_dataset(dataset_name)
    train_set = dataset["train"]
    val_set = dataset["test"] if "test" in dataset else dataset["validation"]

    model = CLIPModel.from_pretrained(cfg.model.name)
    processor = CLIPProcessor.from_pretrained(cfg.model.name)
    text_embeddings, class_names = get_text_embeddings(dataset_name, processor, model)

    if use_peft and cfg.init.mode == "gradient":

        # Prepare dataloader for gradient estimation
        train_dataloader_for_gradient = DataLoader(
            dataset["train"],
            batch_size=32,
            shuffle=True,
            collate_fn=collate_fn_for_gradient
        )

        # Estimate gradients using contrastive loss
        named_grad = estimate_gradient(
            model=model,
            dataloader=train_dataloader_for_gradient,
            accelerator=None,
            quant_flag=False,
        )
    
    if lora_target_modules == "all":
        lora_target_modules = find_all_linear_modules(model)
    else:
        lora_target_modules = list(lora_target_modules) if lora_target_modules else []
    if lora_relative_r is not None:
        hidden_size = find_hidden_state_size(model)
        lora_r = int(hidden_size * lora_relative_r)
        log.info(f"lora_r is set to {hidden_size} * {lora_relative_r} = {lora_r}")
    
    if use_peft and cfg.peft.get("dora", False):
        log.info("Using Dora")
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=cfg.peft.lora_alpha,
            target_modules=lora_target_modules,
            use_rslora=if_use_rslora,
            use_dora=True,
        )
        orig_model_params = sum(p.numel() for p in model.parameters())
        model = get_peft_model(model, peft_config)
        trainable_params, all_param = model.get_nb_trainable_parameters()
        rate = {
            "trainable_params": trainable_params,
            "orig_params": orig_model_params,
            "all_params": all_param,
            "trainable_ratio": trainable_params / all_param,
            "param_ratio": trainable_params / orig_model_params,
        }
    elif use_peft and cfg.peft.get("adalora", False):
        log.info("Using AdaLora")
        peft_config = AdaLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_r=lora_r,
            lora_alpha=cfg.peft.lora_alpha,
            target_modules=lora_target_modules,
            total_step=int(len(train_set)/cfg.model.real_batch_size)*cfg.model.epochs,
        )
        orig_model_params = sum(p.numel() for p in model.parameters())
        model = get_peft_model(model, peft_config)
        trainable_params, all_param = model.get_nb_trainable_parameters()
        rate = {
            "trainable_params": trainable_params,
            "orig_params": orig_model_params,
            "all_params": all_param,
            "trainable_ratio": trainable_params / all_param,
            "param_ratio": trainable_params / orig_model_params,
        }
    elif use_peft: # Reinit LoRA here
        if cfg.init.mode == "gradient":
           peft_config = LoraGAConfig(
                   r=lora_r,
                   lora_alpha=cfg.peft.lora_alpha,
                   target_modules=lora_target_modules,
                   use_rslora=if_use_rslora,
                   bsz=cfg.init.bsz,
                   iters=cfg.init.iters,
                   direction=cfg.init.direction,
                   scale=cfg.init.scale,
                   stable_gamma=cfg.init.stable_gamma,
           )
           # Attach gradients and get PEFT model
           with LoraGAContext(model=model, named_grad=named_grad):
                model = get_peft_model(model=model, peft_config=peft_config)

        else:
           peft_config = LoraConfig(
               r=lora_r,
               lora_alpha=cfg.peft.lora_alpha,
               target_modules=lora_target_modules,
               use_rslora=if_use_rslora,
           )
           model = get_peft_model(model=model, peft_config=peft_config)

        orig_model_params = sum(p.numel() for p in model.parameters())
        ########## We need to determine scaling parameter here
        if train_embeddings:
            model.lm_head.weight.requires_grad = True
        trainable_params, all_param = model.get_nb_trainable_parameters()
        rate = {
            "trainable_params": trainable_params,
            "orig_params": orig_model_params,
            "all_params": all_param,
            "trainable_ratio": trainable_params / all_param,
            "param_ratio": trainable_params / orig_model_params,
        }
    else:
        # full finetune
        all_param = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        rate = {
            "trainable_params": trainable_params,
            "orig_params": all_param,
            "all_params": all_param,
            "trainable_ratio": trainable_params / all_param,
            "param_ratio": 1,
        }

    # Initialize classification model
    classification_model = CLIPForImageClassification(model, text_embeddings)

    log.info(rate)
    # log rate into wandb summary
    wandb.summary.update(rate)
    training_loop = train_vit_model
    model = training_loop(
        f"{cfg.wandb.project}/{name}",
        train_set,
        val_set,
        classification_model,
        processor,
        model_type,
        optimizer=None, # using custom_optimizer
        num_train_epochs=cfg.model.epochs,
        per_device_batch_size=cfg.model.per_device_batch_size,
        real_batch_size=cfg.model.real_batch_size,
        bf16=cfg.model.bf16,
        eval_epochs=cfg.model.eval_epochs,
        early_stopping_patience=cfg.model.early_stopping_patience,
        logging_steps=cfg.model.logging_steps,
        use_loraplus=cfg.peft.use_loraplus,
        loraplus_lr_ratio=cfg.peft.loraplus_lr_ratio,
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        prec_reg=cfg.model.prec_reg,
        gradient_checkpointing=cfg.get("gradient_checkpointing", False),
        seed=cfg.seed,
    )
    wandb.finish()

if __name__ == "__main__":
    run_exp()