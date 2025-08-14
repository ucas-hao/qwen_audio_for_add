from dataclasses import dataclass, field
import json
import os
import sys
# new_path = "/share/ad/guhao/qwen_audio_sft/qwen_audio_sft"
# sys.path.append(new_path)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pathlib
from typing import Dict, Optional, List, Any, Tuple
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import transformers
from batch_modeling_qwen import QWenLMHeadModel
from tokenization_qwen import QWenTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedTokenizer
from transformers.generation import GenerationConfig
import torch
from qwen_generation_utils import get_stop_words_ids
torch.manual_seed(1234)
import json
from tqdm import tqdm
from peft import PeftModel
# Note: The default behavior now has injection attack prevention off.
tokenizer = QWenTokenizer.from_pretrained("/data3/renyong/guhao/speech_llm/qwen_audio_chat", trust_remote_code=True)
tokenizer.pad_token_id = tokenizer.eod_id
tokenizer.padding_side = 'left'

def from_list_format(list_format: List[Dict]):
    text = ''
    num_audios = 0
    audio_start_tag, audio_end_tag = '<audio>', '</audio>'
    for ele in list_format:
        if 'audio' in ele:
            num_audios += 1
            text += f'Audio {num_audios}:'
            text += audio_start_tag + ele['audio'] + audio_end_tag
            text += '\n'
        elif 'text' in ele:
            text += ele['text']
        else:
            raise ValueError("Unsupport element: " + str(ele))
    return text

def make_context(
    tokenizer: PreTrainedTokenizer,
    query: str, 
    system: str = "",
):
    audio_info = None
    
    im_start, im_end = "<|im_start|>", "<|im_end|>"
    im_start_tokens = [tokenizer.im_start_id]
    im_end_tokens = [tokenizer.im_end_id]
    nl_tokens = tokenizer.encode("\n")

    def _tokenize_str(role, content):
        audio_info = tokenizer.process_audio(content)
        return f"{role}\n{content}", tokenizer.encode(
            role, allowed_special=set(tokenizer.AUDIO_ST), audio_info=audio_info
        ) + nl_tokens + tokenizer.encode(content, allowed_special=set(tokenizer.AUDIO_ST), audio_info=audio_info),audio_info

    system_text, system_tokens_part, audio_info = _tokenize_str("system", system)
    system_tokens = im_start_tokens + system_tokens_part + im_end_tokens 

    context_tokens = system_tokens
    raw_text = f"{im_start}{system_text}{im_end}"
    
    context_tokens += nl_tokens + im_start_tokens + _tokenize_str("user", query)[1] + im_end_tokens + nl_tokens
    context_tokens += im_start_tokens + tokenizer.encode("assistant") + nl_tokens
    
    raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n" 
    audio_info = tokenizer.process_audio(raw_text)
    
    return raw_text, context_tokens, audio_info

def preprocess(
    source,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    prompt_message: str = "Can you determine if this audio is fake or real?",
    system_message: str = "You are a helpful assistant."
) -> Dict:
    audio_path = source[0]['audio']
    query_message = source[0]['content'] if source[0]['content'] else prompt_message
    query = tokenizer.from_list_format([{'audio': audio_path}, {'text': query_message},])
    raw_text, context_tokens, audio_info = make_context(
            tokenizer,
            query,
            system=system_message,
        )
    input_ids = torch.tensor(context_tokens,dtype=torch.int)
    return dict(
        input_ids = input_ids,
        attention_mask = input_ids.ne(tokenizer.pad_token_id),
        audio_info=audio_info
    )


class LazySupervisedDataset(Dataset):
    """ Dataset with cache """

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int = 2000): 
        # super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.raw_data = raw_data 
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess(self.raw_data[i]['messages'], self.tokenizer, self.max_len)
        ret = dict(
            input_ids=ret["input_ids"],
            attention_mask=ret["attention_mask"],
            audio_info=ret["audio_info"],
        )
        self.cached_data_dict[i] = ret

        return ret
    
class CustomDataCollator(transformers.DataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]: 
        audio_infos = [feature.pop('audio_info', None) for feature in features] 
        batch = super().__call__(features) 
        if any(audio_infos):
            audio_span_tokens = []
            for x in audio_infos:
                audio_span_tokens.extend(x['audio_span_tokens'])
            audio_name_files = []
            for x in audio_infos:
                audio_name_files.extend(x['audio_urls'])
            batch['audio_info'] = {
                "input_audios": torch.concat([info['input_audios'] for info in audio_infos if info]), 
                "audio_span_tokens": audio_span_tokens,
                "input_audio_lengths": torch.concat([info['input_audio_lengths'] for info in audio_infos if info]),
                "input_audio_names":audio_name_files
            }

        return batch

def process_name(path_name): 
    parts = path_name.split("/")
    sample = parts[-2]
    checkpoint = parts[-1]
    result = f"{sample}_{checkpoint}"
    return result

root_dir = f"/data3/renyong/guhao/speech_llm_loss/output"
checkpoint_folders = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    for dirname in dirnames:
        if dirname.startswith('bs_8_3e_5'):
            absolute_path = os.path.join(dirpath, dirname)
            for checkpoint_dir in os.listdir(absolute_path):
                checkpoint_dir_path = os.path.join(absolute_path, checkpoint_dir)
                if os.path.isdir(checkpoint_dir_path) and checkpoint_dir.startswith("checkpoint"):
                    checkpoint_folders.append(checkpoint_dir_path)
print(checkpoint_folders)
print(len(checkpoint_folders))

data_path = "/data3/renyong/guhao/speech_llm/data/asv_test_fake_audio.jsonl"
test_data = []
with open(data_path, "r") as f:
    for line in f:
        test_data.append(json.loads(line))
test_dataset = LazySupervisedDataset(test_data,tokenizer=tokenizer)
data_collator = CustomDataCollator(tokenizer=tokenizer)
dataloader = DataLoader(test_dataset, batch_size=4, collate_fn=data_collator)

for peft_path in checkpoint_folders:
    peft_model_id = peft_path
    output_file_name = process_name(peft_model_id)
    if os.path.exists(f'/data3/renyong/guhao/speech_llm_loss/infer_eer/{output_file_name}.json'):
        print(f"文件 {output_file_name} 已存在，跳过操作。")
    else:
        print(f"文件 {output_file_name} 不存在，需要操作。")
        model = QWenLMHeadModel.from_pretrained("/data3/renyong/guhao/speech_llm/qwen_audio_chat", device_map="cuda", trust_remote_code=True).eval()
        peft_model_id = peft_path
        model = PeftModel.from_pretrained(model, peft_model_id)
        merged_model = model.merge_and_unload()
        # print(os.path.join(peft_model_id, 'non_lora_trainables.bin'))
        # non_lora_trainables = torch.load(os.path.join(peft_model_id, 'non_lora_trainables.bin'), map_location='cpu')
        # non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
        # non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
        # merged_model.load_state_dict(non_lora_trainables, strict=False)
        answers = {}
        for batch in tqdm(dataloader): 
            logits = merged_model(
                input_ids=batch['input_ids'].cuda(),
                attention_mask=batch['attention_mask'].cuda(),
                audio_info=batch['audio_info'],
            ).logits
            word_probs = torch.stack([logits[i, -1] for i in range(len(logits))], dim=0)
            names = batch.data['audio_info']['input_audio_names']
            word_probs_target = word_probs.clone().detach()
            # tokenizer.encode('Real') = 12768
            # tokenizer.encode('Fake) = 52317
            logits_target = word_probs_target[:, [52317, 12768]].to(torch.float32)
            for index, name in enumerate(names):
                answers[name] = logits_target[index].cpu().tolist()
        output_file_name = process_name(peft_model_id)
        with open(f'/data3/renyong/guhao/speech_llm_loss/infer_eer/{output_file_name}.json','w',encoding='utf-8') as f:
            json.dump(answers,f,indent=4,ensure_ascii=False)
        del model
        del merged_model
        torch.cuda.empty_cache()