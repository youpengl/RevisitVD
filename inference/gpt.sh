test_data_file="../dataset/self_collected_benchmark.jsonl"
few_shots_samples_path="few_shots_samples.jsonl"
graph_prompt_types=(None data_flow flatten_AST api_call)
model_gpt=(gpt-3.5-turbo-0125)

for graph_prompt_type in "${graph_prompt_types[@]}"; do
    for model_name_or_path in "${model_gpt[@]}"; do
        python inference.py  --prompt_type zero_shot --model_name_or_path $model_name_or_path --test_data_file $test_data_file --few_shots_samples_path $few_shots_samples_path  --graph_prompt_type $graph_prompt_type
        python inference.py  --prompt_type few_shots --num_pairs 2 --random_shots --model_name_or_path $model_name_or_path --test_data_file $test_data_file --few_shots_samples_path $few_shots_samples_path  --graph_prompt_type $graph_prompt_type
    done
done