# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass
import logging
import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import pathlib


from PIL import Image
from torch.utils.data import Dataset
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from open_r1.trainer import VLMGRPOTrainer, GRPOConfig
from open_r1.vlm_modules import *
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from transformers import TrainingArguments
import yaml
import json
import random
import math

import whisper
import librosa
from decord import VideoReader, cpu, AudioReader
import numpy as np

# ----------------------- Fix the flash attention bug in the current version of transformers -----------------------
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionFlashAttention2, apply_rotary_pos_emb_flashatt, flash_attn_varlen_func
import torch
from typing import Tuple
import copy
from qwen_omni_utils import process_mm_info
import av

def check_if_video_has_audio(video_path):
    try:
        container = av.open(video_path)
        audio_streams = [stream for stream in container.streams if stream.type == "audio"]
        if not audio_streams:
            return False
        return True
    except:
        return False



logger = logging.getLogger(__name__)



def custom_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        # print(111, 222, 333, 444, 555, 666, 777, 888, 999)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos().float()
            sin = emb.sin().float()
        else:
            cos, sin = position_embeddings
            # Add this
            cos = cos.to(torch.float)
            sin = sin.to(torch.float)
        q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        q = q.squeeze(0)
        k = k.squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        attn_output = self.proj(attn_output)
        return attn_output

Qwen2_5_VLVisionFlashAttention2.forward = custom_forward


# ----------------------- Main Script -----------------------
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["format", "accuracy", "context", "reasoning"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image (for QwenVL)"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image (for QwenVL)"},
    )
    max_anyres_num: Optional[int] = field(
        default=12,
        metadata={"help": "Maximum number of anyres blocks for the image (for InternVL)"},
    )
    image_root: Optional[str] = field(
        default=None,
        metadata={"help": "Root directory of the image"},
    )
    use_audio_in_video: Optional[bool] = field(
        default=False,
        metadata={"help": "Maximum number of anyres blocks for the image (for InternVL)"},
    )

@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False




SYSTEM_PROMPT = """You are a helpful assistant. Your primary goal is to deeply analyze and interpret information from available various modalities (image, video, audio, text context) to answer questions with human-like depth and a clear, traceable thought process.

Begin by thoroughly understanding the image, video, audio or other available context information, and then proceed with an in-depth analysis related to the question. 

In reasoning, It is encouraged to incorporate self-reflection and verification into your reasoning process. You are encouraged to review the image, video, audio, or other context information to ensure the answer accuracy.

Provide your understanding of the image, video, and audio between the <context> </context> tags, detail the reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags.
"""
class LazySupervisedDataset(Dataset):

    TYPE_TEMPLATE = {
        "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
        "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
        "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
        "free-form": " Please provide your text answer within the <answer> </answer> tags.",
        "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
        "emer_ov": " Please provide the words to describe emotions within the  <answer> </answer> tags.",
        "emer_ov_mc": " Please provide only the single or multiple option letter (e.g., A for single option or A,E for multi option, etc.) within the <answer> </answer> tags.",
        "judge": " Please answer Yes or No within the <answer> </answer> tags.",


    }

    def __init__(self, data_path: str, script_args: GRPOScriptArguments, question_template: str):
        super(LazySupervisedDataset, self).__init__()
        self.script_args = script_args
        self.list_data_dict = []
        self.question_template = question_template
        self.use_audio_in_video = script_args.use_audio_in_video

        if data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                # file should be in the format of:
                # datasets:
                #   - json_path: xxxx1.json
                #     sampling_strategy: first:1000
                #   - json_path: xxxx2.json
                #     sampling_strategy: end:3000
                #   - json_path: xxxx3.json
                #     sampling_strategy: random:999
                #     data_root: xxxx/xx

                for data in datasets:
                    json_path = data.get("json_path")
                    sampling_strategy = data.get("sampling_strategy", "all")
                    sampling_number = None

                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path, "r") as json_file:
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    elif json_path.endswith(".json"):
                        with open(json_path, "r") as json_file:
                            cur_data_dict = json.load(json_file)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")

                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                        else:
                            sampling_number = int(sampling_number)

    

                    # Apply the sampling strategy
                    if sampling_strategy == "first" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "end" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    elif sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]

                    if data.get("data_root", None):
                        for each in cur_data_dict:
                            if "path" in each:
                                if isinstance(each["path"], str):
                                    each["path"] = os.path.join(data["data_root"], each["path"])
                                elif isinstance(each["path"], dict):
                                    for k in each["path"].keys():
                                        each["path"][k] = os.path.join(data["data_root"], each["path"][k])
                    print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    self.list_data_dict.extend(cur_data_dict)
        else:
            if data_path.endswith(".jsonl"):
                cur_data_dict = []
                with open(data_path, "r") as json_file:
                    for line in json_file:
                        cur_data_dict.append(json.loads(line.strip()))
            elif data_path.endswith(".json"):
                with open(data_path, "r") as json_file:
                    cur_data_dict = json.load(json_file)
            self.list_data_dict = cur_data_dict

        self.mel_size = 128
        self.frames_upbound = 16

    def __len__(self):
        return len(self.list_data_dict)


  

    def _make_conversation_image_and_video(self, example, use_audio_in_video=False):
        if example["problem_type"] == 'multiple choice' or  example["problem_type"] == 'emer_ov_mc':
            question = example['problem'] + " Options:\n"
            for op in example["options"]:
                question += op + "\n"
        else:
            question = example['problem']

        text_prompt =  f"{question}\n" + self.TYPE_TEMPLATE[example['problem_type']]

        if use_audio_in_video:
            if isinstance(example['path'], str):
                video_audio_avaliable = check_if_video_has_audio(example['path']) and example['data_type'] == "video"
                if video_audio_avaliable:
                    msg =[{
                            "role": "user",
                            "content": [
                                {
                                    "type": example['data_type'],
                                    example['data_type']: example['path']
                                },
                                {
                                "type": "audio",
                                "audio": example['path']
                                },
                                {
                                    "type": "text",
                                    "text": f"Here is a {example['data_type']}, with the audio from the video.\n" + text_prompt
                                }
                                ]
                        }]
                    
                else:
                    msg =[{
                            "role": "user",
                            "content": [
                                {
                                    "type": example['data_type'],
                                    example['data_type']: example['path']
                                },
                
                                {
                                    "type": "text",
                                    "text": f"Here is the {example['data_type']}, and there is no audio information, you don't need to process the audio.\n" + text_prompt
                                }
                                ]
                        }]
            else:
                msg =[{
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": example['path']["image"]
                                },
                                {
                                    "type": "audio",
                                    "audio": example['path']["audio"]
                                },
                                {
                                    "type": "text",
                                    "text": f"Here is the image, with the coresponding audio.\n" + text_prompt
                                }
                                ]
                        }]
        else:
            msg =[{
                        "role": "user",
                        "content": [
                            {
                                "type": example['data_type'],
                                example['data_type']: example['path']
                            },
                            {
                                "type": "text",
                                "text": text_prompt
                            }
                            ]
                    }]

    
            
        msg.insert(0, {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": SYSTEM_PROMPT
                            }
                            ]
                    })

        
        return msg

    def __getitem__(self, i):
        # Format into conversation
        num_base_retries = 3
        import traceback

        try:
            return self._get_item(i)
        except Exception as e:
            print(i)
            traceback.print_exc()


        for attempt_idx in range(num_base_retries):
            try:
                sample_idx = random.choice(range(len(self)))
                sample = self._get_item(sample_idx)
                return sample
            except Exception as e:
                # no need to sleep
                traceback.print_exc()
                print(f'[try other #{attempt_idx}] Failed to fetch sample {sample_idx}. Exception:', e)
                pass

        
        

    def _get_item(self, i):
        source = self.list_data_dict[i]


        # has_speech = ('audio' in source or 'audio_q' in source)
        # has_image = ('image' in source) or ('video' in source) or ('video_long' in source)
        # print(self.use_audio_in_video)
        if "path" in source:
            conversation  = self._make_conversation_image_and_video(source, use_audio_in_video=self.use_audio_in_video)
            problem_type = source["problem_type"]
            audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
       
        solution = source["solution"]
        # print(conversation, solution)
        # delay tokenizer
        return {
            'images': images,
            'audios': audios,
            'videos': videos,
            'conversation': conversation,
            'prompt': conversation,
            'solution': solution,
            "problem_type": problem_type
        #  
        }


def get_vlm_module(model_name_or_path):
    if "qwen" in model_name_or_path.lower() and "omni" in model_name_or_path.lower():
        return QwenOmniModule
    elif "internvl" in model_name_or_path.lower():
        return InvernVLModule
    elif "ola" in model_name_or_path.lower():
        return QwenOlaModule
    elif "qwen" in model_name_or_path.lower() and "vl" in model_name_or_path.lower():
        return Qwen2VLModule
    else:
        raise ValueError(f"Unsupported model: {model_name_or_path}")

def main(script_args, training_args, model_args):
    # Load the VLM module
    vlm_module_cls = get_vlm_module(model_args.model_name_or_path)
    print("using vlm module:", vlm_module_cls.__name__)

    # Load the reward functions
    reward_funcs_registry = {
        "accuracy": vlm_module_cls.accuracy_reward,
        "format": vlm_module_cls.format_reward,
        "reasoning": vlm_module_cls.patial_reasoning_reward,
        "context": vlm_module_cls.patial_context_reward
    }
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print(script_args.reward_funcs)
    
    print("reward_funcs:", reward_funcs)
    # import ipdb;ipdb.set_trace()

    # Load the dataset
    dataset = LazySupervisedDataset(script_args.dataset_name, script_args, question_template=vlm_module_cls.get_question_template(task_type="rec"))


    # Initialize the GRPO trainer
    trainer = VLMGRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        vlm_module=vlm_module_cls(),
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        max_anyres_num=script_args.max_anyres_num,
        torch_dtype=model_args.torch_dtype,
    )

    # Train and push the model to the Hub
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    # if training_args.push_to_hub:
    #     trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
