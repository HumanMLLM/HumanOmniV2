import os
import json
import re
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import torch
import itertools

import argparse
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn.functional as F
import copy
import av
import yaml
import math
import random

def check_if_video_has_audio(video_path):
    try:
        container = av.open(video_path)
        audio_streams = [stream for stream in container.streams if stream.type == "audio"]
        if not audio_streams:
            return False
        return True
    except:
        return False


def extract_think(output_str):
    pattern = r'<think>\s*(.*?)\s*</think>'
    match = re.search(pattern, output_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def extract_answer(text):
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def normalize_number(num_str):
    try:
        num_str = num_str.replace(',', '')
        return float(num_str)
    except Exception as e:
        return None
    
def mean_relative_accuracy(pred, target, start=0.5, end=0.95, interval=0.05):

    if not torch.is_tensor(pred):
        pred = torch.tensor(pred, dtype=torch.float32)
    if not torch.is_tensor(target):
        target = torch.tensor(target, dtype=torch.float32)
    
    epsilon = 1e-8
    rel_error = torch.abs(pred - target) / (torch.abs(target) + epsilon)
    
    thresholds = torch.arange(start, end + interval/2, interval, dtype=torch.float32)
    
    conditions = rel_error < (1 - thresholds)  
    mra = conditions.float().mean()  
    return mra.item()


def emer_ov_mc(reference, hypothesis):
    list_a = reference.split(",")
    list_b = hypothesis.split(",")
    true_positive = len(set(list_a) & set(list_b))
    precision = true_positive / len(list_a) if list_a else 0
    recall = true_positive / len(list_b) if list_b else 0
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0
    
    return f1_score

def reward_fn(output_ans, gt_ans, question_type):
    try:

        if question_type == "multiple choice":
            return 1.0 if output_ans.strip() == gt_ans.strip() else 0.0
        elif question_type == "numerical":
            gt_has_decimal = ("." in gt_ans) or ("," in gt_ans)
            out_has_decimal = ("." in output_ans) or ("," in output_ans)
            if gt_has_decimal != out_has_decimal:
                return 0.0
            gt_number = normalize_number(gt_ans)
            out_number = normalize_number(output_ans)
            if gt_number is None or out_number is None:
                return 0.0
            return 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0
        elif question_type == "regression":
            gt_number = normalize_number(gt_ans)
            out_number = normalize_number(output_ans)
            if gt_number is None or out_number is None:
                return 0.0
            mra = mean_relative_accuracy(out_number, gt_number)
            return mra
        elif question_type == "emer_ov_mc":
            return emer_ov_mc(output_ans, gt_ans)

        else:
            return 0.0
    except Exception as e:
        return 0.0

SYSTEM_PROMPT = """You are a helpful assistant. Your primary goal is to deeply analyze and interpret information from available various modalities (image, video, audio, text context) to answer questions with human-like depth and a clear, traceable thought process.

Begin by thoroughly understanding the image, video, audio or other available context information, and then proceed with an in-depth analysis related to the question. 

In reasoning, It is encouraged to incorporate self-reflection and verification into your reasoning process. You are encouraged to review the image, video, audio, or other context information to ensure the answer accuracy.

Provide your understanding of the image, video, and audio between the <context> </context> tags, detail the reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags.
"""

class MyDataset(Dataset):
    def __init__(self, data_path, processor):
        super(MyDataset, self).__init__()
        self.list_data_dict = []

        self.use_audio_in_video = True


        self.processor = processor

     
        self.TYPE_TEMPLATE = {
            "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
            "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
            "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
            "free-form": " Please provide your text answer within the <answer> </answer> tags.",
            "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
            "emer_ov_mc": " Please provide only the single or multiple option letter (e.g., A for single option or A,E for multi option, etc.) within the <answer> </answer> tags.",

        }

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


       
        self.data = self.list_data_dict

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

    def _get_item(self, index):
            data = self.data[index]
            
            if data["problem_type"] == 'multiple choice' or  data["problem_type"] == 'emer_ov_mc' :
                question = data['problem'] + " Options:\n"
                for op in data["options"]:
                    question += op + "\n"
            else:
                question = data['problem']

   
            video_path = data['video'] if "video" in data else data["path"]
            video_audio_avaliable = check_if_video_has_audio(video_path)


            # text_prompt = self.QUESTION_TEMPLATE.format(Question=question) + self.TYPE_TEMPLATE[data['problem_type']]
            text_prompt =  question + self.TYPE_TEMPLATE[data['problem_type']]
            if video_audio_avaliable:
                message = [{
                    "role": "user",
                    "content": [
                        {
                            "type": data['data_type'],
                            data['data_type']: video_path
                        },
                        {
                            "type": "audio",
                            "audio": video_path
                        },
                        
                        {
                            "type": "text",
                            "text": f"Here is a {data['data_type']}, with the audio from the video.\n" + text_prompt
                        }
                    ]
                }]
            else:
                 message = [{
                    "role": "user",
                    "content": [
                        {
                            "type": data['data_type'],
                            data['data_type']: video_path
                        },
                        
                        {
                            "type": "text",
                            "text": text_prompt
                        }
                    ]
                }]

            message.insert(0, {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT
                    }
                    ]
            })

            
            audios, images, videos = process_mm_info(message, use_audio_in_video=False)
            data_dict = {
                'images': images,
                'audios': audios,
                'videos': videos,
                'prompt': message,
                'solution': data["solution"],
                "problem_type": data["problem_type"],
                "raw_data": data
            }
            return data_dict
          
    def __len__(self):
        return len(self.data)
        
def collate_fn(examples):
    

    return examples

class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        all_indices = list(range(total_size))
        
        interleaved_indices = [
            idx for idx in all_indices if idx % world_size == rank
        ]
    
        return interleaved_indices

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def main(args):
 

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))




    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="cuda",attn_implementation="flash_attention_2",)
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)

   
    model_name = args.model_path.split("/")[-1]
      
    dataset = MyDataset("data_config/stage2.yaml", processor)

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=1,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    retained_correct_samples = []
    for inputs in tqdm(dataloader, desc=f"{torch.distributed.get_rank()} Processing batches"):

        images, videos, audios, prompts = [], [], [], []
        for each in inputs:
            prompts.append(each["prompt"])
            if each["images"] is not None:
                images.extend(each["images"])
            if each["audios"] is not None:
                audios.extend(each["audios"])
            if each["videos"] is not None:
                videos.extend(each["videos"])
        if len(images) == 0: images = None
        if len(audios) == 0: audios = None
        if len(videos) == 0: videos = None
        
        text = processor.apply_chat_template(
            prompts,
            tokenize=False,
            add_generation_prompt=True,
        )
        # print(text)
        model_inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)

        
        model_inputs = model_inputs.to(model.device).to(model.dtype)
        NUM_SAMPLES_PER_INSTANCE = 8
        text_ids = []
        with torch.inference_mode():
            for j in range(NUM_SAMPLES_PER_INSTANCE//4):
                text_ids.extend(model.generate(**model_inputs, use_audio_in_video=False, max_new_tokens=1024, num_return_sequences=4,
                do_sample=True, 
                temperature=0.9, 
                top_p=1.0,  
                ))
                # print(text_ids[0])
     

            for i, original_sample in enumerate(inputs):
                correct_predictions_count = 0
                correct_outputs_for_this_sample = [] 

             
                input_ids_length_for_this_sample = model_inputs['input_ids'][i].size(0)
                generated_ids_for_this_sample = text_ids[
                        i * NUM_SAMPLES_PER_INSTANCE : (i + 1) * NUM_SAMPLES_PER_INSTANCE
                    ]

                for j in range(NUM_SAMPLES_PER_INSTANCE):
                    try:
                        response = processor.decode(
                            generated_ids_for_this_sample[j][input_ids_length_for_this_sample:],
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False
                        )
                    except Exception as e:
                        
                        print(generated_ids_for_this_sample)
                        print(len(text_ids))
                        print(e)
                    # print(response)
                    gt = inputs[i]["solution"]

                  
                    final_ans = extract_answer(response)# response #extract_answer(response)
                    gt_ans = extract_answer(gt)
                    problem_type =  inputs[i]["raw_data"]["problem_type"]
                    if final_ans == "":
                        final_ans = response

                    reward = reward_fn(final_ans, gt_ans, problem_type)

                    if reward>0.2:
                        correct_predictions_count += 1

                   
                    sample_accuracy = correct_predictions_count / NUM_SAMPLES_PER_INSTANCE

                if 0 < sample_accuracy < 0.75 and correct_predictions_count > 0:
      
                    retained_correct_samples.append(inputs[i]["raw_data"])
                else:
                    print(inputs[i]["raw_data"])

        

    torch.distributed.barrier()

    world_size = torch.distributed.get_world_size()
   
    merged_sources = [None for _ in range(world_size)]

    torch.distributed.all_gather_object(merged_sources, retained_correct_samples)


    merged_sources = [_ for _ in itertools.chain.from_iterable(merged_sources)]


    if torch.distributed.get_rank() == 0:
      
        with open(f"data_config/{model_name}_r8.json", "w", encoding="utf-8") as f:
            json.dump(merged_sources, f, indent=2, ensure_ascii=False)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation benchmark")
    parser.add_argument('--model-path', type=str, required=False, help="Path to the model")
    args = parser.parse_args()

    main(args)

