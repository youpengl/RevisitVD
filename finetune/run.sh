
$train_data_file="../dataset/reconstructed_train.jsonl"
$valid_data_file="../dataset/reconstructed_valid.jsonl"
$test_data_file="../dataset/self_collected_benchmark.jsonl"

# Small Language Model Training

project="CodeBERT" 
model_name_or_path="microsoft/codebert-base"  # microsoft/codebert-base, ./pretrain/pdbert-base, microsoft/unixcoder-base-nine, microsoft/graphcodebert-base, Salesforce/codet5-base
# for pdbert, you need to download the pdbert-base first at https://zenodo.org/records/10140638/preview/PDBERT_data.zip?include_deleted=0#tree_item6
localtime=$(TZ="America/Chicago" date "+%Y-%m-%d-%H:%M:%S")
python Finetune_SLMs.py --project $project --output_dir "./output/" --model_type roberta --model_dir $model_name_or_path --tokenizer_name $model_name_or_path --model_name_or_path $model_name_or_path --do_train --do_test --train_data_file $train_data_file --eval_data_file $valid_data_file --test_data_file $test_data_file --epoch 10 --block_size 512 --train_batch_size 32 --eval_batch_size 64 --learning_rate 2e-5 --evaluate_during_training --seed 123456 --warmup_steps -1 --localtime "$localtime"

# Large Language Model Training

train_batch_size_per_gpu=1
gradient_accumulation_steps=8

project="CodeLlama"
model_name_or_path="codellama/CodeLlama-34b-Instruct-hf" # codellama/CodeLlama-7b-Instruct-hf, deepseek-ai/deepseek-coder-6.7b-instruct, mistralai/Mistral-7B-Instruct-v0.1, codellama/CodeLlama-13b-Instruct-hf, WizardLMTeam/WizardCoder-15B-V1.0, HuggingFaceH4/starchat-beta, codellama/CodeLlama-34b-Instruct-hf,  deepseek-ai/deepseek-coder-33b-instruct, WizardLMTeam/WizardCoder-33B-V1.1
localtime=$(TZ="America/Chicago" date "+%Y-%m-%d-%H:%M:%S")
accelerate launch --config_file deepspeed.yml Finetune_LLMs.py --project $project --output_dir "./output/" --model_name_or_path $model_name_or_path --do_train --do_test --train_data_file $train_data_file --eval_data_file $eval_data_file --test_data_file $test_data_file --epoch 10 --train_batch_size $train_batch_size_per_gpu --eval_batch_size $train_batch_size_per_gpu --test_batch_size $train_batch_size_per_gpu --learning_rate 2e-5 --evaluate_during_training --seed 123456 --warmup_steps -1 --gradient_accumulation_steps $gradient_accumulation_steps --lora --localtime "$localtime"