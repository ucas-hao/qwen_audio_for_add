[![arXiv](https://img.shields.io/badge/Arxiv-2505.11079-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2505.11079) 
<p align="center">
  <img src="audiollm.png" alt="ALLM4ADD main figure" width="800">
</p>

<h1 align="center">ALLM4ADD: Unlocking the Capabilities of Audio Large Language Models for Audio Deepfake Detection</h1>

<p align="center">
  <em>[ACMMM2025] Official code for ALLM4ADD</em>
</p>

<p align="center">
  Hao Gu<sup>1</sup>, Jiangyan Yi<sup>2†</sup>, Chenglong Wang<sup>3</sup>, Jianhua Tao<sup>2</sup>,<br>
  Zheng Lian<sup>1</sup>, Jiayi He<sup>1</sup>, Yong Ren<sup>1</sup>, Yujie Chen<sup>4</sup>, Zhengqi Wen<sup>5</sup>
</p>

<p align="center">
<sup>1</sup>Institute of Automation, Chinese Academy of Sciences<br>
<sup>2</sup>Department of Automation, Tsinghua University<br>
<sup>3</sup>Taizhou University<br>
<sup>4</sup>Anhui University<br>
<sup>5</sup>Beijing National Research Center for Information Science and Technology<br>
†Corresponding author.
</p>



## Good News
🔥[2025/7/5] Our work has been finally accepted by ACMMM 2025. This is the first work that fake audio detection with Audio LLM


## 🎯 Environment:
Install dependencies:
```shell
conda create -n qwen_audio_sft python=3.10
conda activate qwen_audio_sft
pip install -r requiremet.txt
pip install deepspeed==0.10.3
```

## 🎯 Dataset: 

You should first prepare the dataset as in data/asv_fake_audio.jsonl:
```shell
{"messages": [{"role": "user", "audio": "/data3/renyong/xule/ASVspoof2019/LA/train/wav/LA_T_4132181.wav",
            "content": "Is this audio real or fake?"}, {"role": "assistant", "content": "Fake."}]}
```

## 🎯 Train
```shell
bash audio_8.sh
```

## 🎯 Infer eer
```shell
python infer_eer_8.py
```

## 🎯 If you want to train the audio encoder
```shell
finetune.py ---> frozen the audio encoder
finetune_audio.py ---> train the audio encoder
```

after you train the audio encoder, change the infer_eer_8.py
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

## 🙏 Acknowledgement
We are thankful to LLaVA, Qwen-audio for releasing their models and code as open-source contributions.

## 🤖 Friends
If you have any problem, please tell me:
email: guhao2022@ia.ac.cn
wechat: 18862041656

## Cite
```shell
@inproceedings{10.1145/3746027.3755851,
author = {Gu, Hao and Yi, Jiangyan and Wang, Chenglong and Tao, Jianhua and Lian, Zheng and He, Jiayi and Ren, Yong and Chen, Yujie and Wen, Zhengqi},
title = {ALLM4ADD: Unlocking the Capabilities of Audio Large Language Models for Audio Deepfake Detection},
year = {2025},
isbn = {9798400720352},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3746027.3755851},
doi = {10.1145/3746027.3755851},
abstract = {Audio deepfake detection (ADD) has grown increasingly important due to the rise of high-fidelity audio generative models and their potential for misuse. Given that audio large language models (ALLMs) have made significant progress in various audio processing tasks, a heuristic question arises: Can ALLMs be leveraged to solve ADD?. In this paper, we first conduct a comprehensive zero-shot evaluation of ALLMs on ADD, revealing their ineffectiveness. To this end, we propose ALLM4ADD, an ALLM-driven framework for ADD. Specifically, we reformulate ADD task as an audio question answering problem, prompting the model with the question: ''Is this audio fake or real?''. We then perform supervised fine-tuning to enable the ALLM to assess the authenticity of query audio. Extensive experiments are conducted to demonstrate that our ALLM-based method can achieve superior performance in fake audio detection, particularly in data-scarce scenarios. As a pioneering study, we anticipate that this work will inspire the research community to leverage ALLMs to develop more effective ADD systems. Code is available at https://github.com/ucas-hao/qwen_audio_for_add.git.},
booktitle = {Proceedings of the 33rd ACM International Conference on Multimedia},
pages = {11736–11745},
numpages = {10},
keywords = {audio deepfake detection, audio large language model},
location = {Dublin, Ireland},
series = {MM '25}
}
```
