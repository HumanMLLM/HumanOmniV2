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

"""
Supervised fine-tuning script for decoder language models.

Usage:

# One 1 node of 8 x H100s
accelerate launch --config_file=configs/zero3.yaml src/open_r1/sft.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name HuggingFaceH4/Bespoke-Stratos-17k \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --max_seq_length 4096 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --bf16 \
    --logging_steps 5 \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir data/Qwen2.5-1.5B-Open-R1-Distill
"""

import logging
import os
import sys
import re

import datasets
import torch
from torch.utils.data import Dataset
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed, AutoProcessor
from transformers import Qwen2_5OmniThinkerForConditionalGeneration
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
from transformers.trainer_utils import get_last_checkpoint
from open_r1.configs import SFTConfig
from open_r1.utils.callbacks import get_callbacks
import yaml
import json
import math
import random
from PIL import Image

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from dataclasses import field
from qwen_vl_utils import process_vision_info
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
from dataclasses import dataclass

@dataclass
class SFTScriptArguments(ScriptArguments):
    image_root: str = field(default=None, metadata={"help": "The root directory of the image."})
    use_audio_in_video: bool = field(default=True)

@dataclass
class SFTModelConfig(ModelConfig):
    freeze_vision_modules: bool = False

processor = None

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


    }
    def __init__(self, data_path: str, script_args: ScriptArguments):
        super(LazySupervisedDataset, self).__init__()
        self.script_args = script_args
        self.list_data_dict = []

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
                                each["path"] = os.path.join(data["data_root"], each["path"])
                    print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    self.list_data_dict.extend(cur_data_dict)
        else:
            raise ValueError(f"Unsupported file type: {data_path}")

        self.mel_size = 128
        self.frames_upbound = 16

    def __len__(self):
        return len(self.list_data_dict)


    

     

    def _make_conversation_image_and_video(self, example, use_audio_in_video=False):
        if example["problem_type"] == 'multiple choice' or  example["problem_type"] == 'emer_ov_mc':
            question = example['problem'] + "Options:\n"
            for op in example["options"]:
                question += op + "\n"
        else:
            question = example['problem']

        assert "<think>" in  example['solution']
        
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
        msg.append({
            "role": "assitant",
            "content": [
                {
                    "type": "text",
                    "text": example['solution']  #example['answer']  #
                }
                ]
        })

        msg.insert(0, {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT
                    }
                    ]
            })
        # print(msg)
        
        return msg

    def __getitem__(self, i):
        # Format into conversation
        num_base_retries = 10
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

       
        messages  = self._make_conversation_image_and_video(source, use_audio_in_video=self.script_args.use_audio_in_video)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=False)

   
           
        return {
            'images': images,
            'audios': audios,
            'videos': videos,

            'messages': messages,
        #  
        }



def collate_fn(examples):
    # texts = [
    #     processor.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=True)[0]
    #     for example in examples
    # ]
    # print(texts)

    # images = [item for sublist in examples  for item in sublist["images"]] if examples[0]['images'] is not None else None
    # audios = [item for sublist in examples  for item in sublist["audios"]] if examples[0]['audios'] is not None else None
    # videos = [item for sublist in examples  for item in sublist["videos"]] if examples[0]['videos'] is not None else None

    images, videos, audios, prompts = [], [], [], []
    for each in examples:
        prompts.append(each["messages"])
        if each["images"] is not None:
            images.extend(each["images"])
        if each["audios"] is not None:
            audios.extend(each["audios"])
        if each["videos"] is not None:
            videos.extend(each["videos"])
    if len(images) == 0: images = None
    if len(audios) == 0: audios = None
    if len(videos) == 0: videos = None
    texts = processor.apply_chat_template(
                prompts,
                tokenize=False,
                add_generation_prompt=False,
            )

    batch = processor(
        text=texts,
        images=images,
        audio=audios,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False
    )
    # print(batch.keys())
    labels = batch["input_ids"].clone()

    # print(labels)
    labels[labels == processor.tokenizer.pad_token_id] = -100
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    video_token_id = processor.tokenizer.convert_tokens_to_ids(processor.video_token)
    audio_token_id = processor.tokenizer.convert_tokens_to_ids(processor.audio_token)
    labels[labels == image_token_id] = -100
    labels[labels == video_token_id] = -100
    labels[labels == audio_token_id] = -100

    batch["labels"] = labels
    # batch["images"] = 

    return batch


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    print(log_level, training_args)
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Data parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ################
    # Load datasets
    ################

    dataset = LazySupervisedDataset(script_args.dataset_name, script_args)

    ################
    # Load tokenizer
    ################
    global processor
    if "vl" in model_args.model_name_or_path.lower() or "omni" in model_args.model_name_or_path.lower():

        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
        )
        logger.info("Using AutoProcessor for vision-language model.")
    else:
        processor = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
        )
        logger.info("Using AutoTokenizer for text-only model.")
    if hasattr(processor, "pad_token") and processor.pad_token is None:
        processor.pad_token = processor.eos_token
    elif hasattr(processor.tokenizer, "pad_token") and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    ###################
    # Model init kwargs
    ###################
    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        # use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    # training_args.model_init_kwargs = model_kwargs

    if "Qwen2-VL" in model_args.model_name_or_path:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path, **model_kwargs
        )
    elif "Qwen2.5-VL" in model_args.model_name_or_path:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path, **model_kwargs
        )
    elif "qwen" in model_args.model_name_or_path.lower() and "omni" in model_args.model_name_or_path.lower():
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(model_args.model_name_or_path, **model_kwargs)
        model.config.vocab_size = 152064

        vision_modules_keywords = ['visual','audio_tower']
    else:
        raise ValueError(f"Unsupported model: {model_args.model_name_or_path}")

    if model_args.freeze_vision_modules:
        logger.info("Freezing vision modules...")
        for n, p in model.named_parameters():
            if any(keyword in n for keyword in vision_modules_keywords):
                p.requires_grad = False
    total_trainable_params = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            logger.info(f'train param: {name}')
            total_trainable_params += p.numel()
    logger.info(total_trainable_params)

    ############################
    # Initialize the SFT Trainer
    ############################
    training_args.dataset_kwargs = {
        "skip_prepare_dataset": True,
    }
    training_args.remove_unused_columns = False
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        processing_class=processor.tokenizer,
        data_collator=collate_fn,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": list(script_args.dataset_name),
        "dataset_tags": list(script_args.dataset_name),
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        # trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)
    #############
    # push to hub
    #############

    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)




if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, SFTModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    logger.info(script_args, training_args, model_args)
    main(script_args, training_args, model_args)