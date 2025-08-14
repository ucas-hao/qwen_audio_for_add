## Environment:
Install dependencies:
```shell
conda create -n qwen_audio_sft python=3.10
conda activate qwen_audio_sft
pip install -r requiremet.txt
pip install deepspeed==0.10.3
```

## Dataset: 

You should first prepare the dataset as in data/asv_fake_audio.jsonl:
```shell
{"messages": [{"role": "user", "audio": "/data3/renyong/xule/ASVspoof2019/LA/train/wav/LA_T_4132181.wav",
            "content": "Is this audio real or fake?"}, {"role": "assistant", "content": "Fake."}]}
```

## Train
```shell
bash audio_8.sh
```

## Infer eer
```shell
python infer_eer_8.py
```

## If you want to train the audio encoder
```shell
finetune.py ---> frozen the audio encoder
finetune_audio.py ---> train the audio encoder
```

after you train the audio encoder, check the infer_eer_8.py
```shell
# non_lora_trainables = torch.load(os.path.join(peft_model_id, 'non_lora_trainables.bin'), map_location='cpu')
# non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
# non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
# merged_model.load_state_dict(non_lora_trainables, strict=False)
```
to
```shell
non_lora_trainables = torch.load(os.path.join(peft_model_id, 'non_lora_trainables.bin'), map_location='cpu')
non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
merged_model.load_state_dict(non_lora_trainables, strict=False)
```
Then you can inference the speech llm that audio encoder is trainable
