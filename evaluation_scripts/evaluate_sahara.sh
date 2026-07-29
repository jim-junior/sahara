
model_id=Sunbird/Sunflower-Qwen3.5-9B
provider=vllm
cache_dir=/workspace/.cache

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="0,1,2,4" \
            python run_sahara_eval_colab_v1.py --provider $provider --model_id $model_id --tasks \
                                        news title summary \
                                        sentiment topic xlni lid \
                                        paraphrase mt_eng2xx mt_fra2xx mt_xx2xx \
                                        mmlu mgsm belebele squad_qa \
                                        phrase pos ner \
                                        --batch_size 1000 --cache_dir $cache_dir
