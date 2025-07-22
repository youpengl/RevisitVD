# How to run it?

```bash
cd finetune
conda env create -f Finetune_SLM.yml # or conda env create -f Finetune_LLM.yml
conda activate Finetune_SLM # or conda activate Finetune_LLM 

// Run in the background (e.g., screen, nohup...)

bash run.sh

# For SLMs: we fully fine-tune them.
# For LLMs: we use LoRA to fine-tune them with DeepSpeed.
```