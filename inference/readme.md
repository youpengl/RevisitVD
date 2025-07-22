# How to run it?

```bash
cd inference
conda env create -f inference.yml
conda activate inference

// In inference.py，please set your own API key by modifying the following line:

os.environ["OPENAI_API_KEY"] = ""

// Run in the background (e.g., screen, nohup...)

bash gpt.sh

```