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

def judge(reference, hypothesis):
    if "yes" in reference.lower()  and "yes" in hypothesis.lower():
        return 1
    elif "no" in reference.lower()  and "no" in hypothesis.lower():
        return 1
    else:
        return 0

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
        elif  question_type == "judge":
            return judge(output_ans, gt_ans)

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
    def __init__(self, dataset_name, data_path, processor, video_root):
        super(MyDataset, self).__init__()
        self.dataset_name = dataset_name
        self.video_root = video_root
        
        if data_path.endswith('.jsonl'):
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    data.append(json.loads(line))
        elif data_path.endswith('.json'):
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            raise ValueError("Input file must be .json or .jsonl")

        self.processor = processor

      

        self.TYPE_TEMPLATE = {
            "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
            "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
            "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
            "free-form": " Please provide your text answer within the <answer> </answer> tags.",
            "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
            "emer_ov_mc": " Please provide only the single or multiple option letter (e.g., A for single option or A,E for multi option, etc.) within the <answer> </answer> tags.",
            "judge": " Please answer Yes or No within the <answer> </answer> tags.",


        }


       
        self.data = data

    def __getitem__(self, index):
            data = self.data[index]

            if data["problem_type"] == 'multiple choice' or  data["problem_type"] == 'emer_ov_mc' :
                question = data['problem'] + " Options:\n"
                for op in data["options"]:
                    question += op + "\n"
            else:
                question = data['problem']

            if "socialiq_trans" == self.dataset_name:
                question = question.split("The question is:\n")[-1]

            video_path = data['video'] if "video" in data else data["path"]
            video_path = os.path.join(self.video_root, video_path)
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
            # inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=True)
            data_dict = {
                'images': images,
                'audios': audios,
                'videos': videos,
                'prompt': message,
                'solution': data["solution"],
                "problem_type": data["problem_type"],
                "raw": data
            }
            # import ipdb; ipdb.set_trace()
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
        # shard_size = total_size // world_size
        # left = total_size % world_size
        # shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        # begin = sum(shard_sizes[:rank])
        # end = min(sum(shard_sizes[:rank + 1]), total_size)
        # return range(begin, end)
        indices = []
        for i in range(total_size):
            if i % world_size == rank:
                indices.append(i)
        return indices

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


datasets = {
    "daily": {"gt_path":"Daily-Omni/qa_think.json","video_root":"Daily-Omni/videos"},
    "world": {"gt_path":"WorldSense/qa_think.json","video_root":"WorldSense/videos"},
    "ib": {"gt_path":"IntentBench/qa.json","video_root":"IntentBench/videos"}
}



def main(args):
 
    file_name = args.file_name

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))




    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="cuda",attn_implementation="flash_attention_2",)
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)



    for dataset_name in args.dataset:
        # model_name = args.model_path.split("/")[-1]
        print(file_name)
        OUTPUT_PATH = f"./eval_results/{dataset_name}_{file_name}.json"
        gt_path = datasets[dataset_name]["gt_path"]
        video_root =  datasets[dataset_name]["video_root"]
        dataset = MyDataset(dataset_name, gt_path, processor, video_root)
        # problem_type = dataset[0]["problem_type"]
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=InferenceSampler(len(dataset)),
            batch_size=1,
            num_workers=8,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )
        
        # import pdb; pdb.set_trace()

        final_output = []
        mean_acc = []
        mean_mra = []

        gts = []
        sources = []
        rets = []
    
        # if os.path.exists(OUTPUT_PATH):
        #     continue 

  
        for inputs in tqdm(dataloader, desc="Processing batches"):
           
            # import ipdb; ipdb.set_trace()
           
            
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
            with torch.inference_mode():
                text_ids = model.generate(**model_inputs, use_audio_in_video=False, max_new_tokens=2048,
                # do_sample=True, 
                # temperature=0.8, 
                # top_p=0.9,  
                )
          

            for j, (sample, model_output, input_text) in enumerate(zip(inputs, text_ids, model_inputs.input_ids)):
                model_output = processor.decode(model_output[input_text.size(0):], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                # model_output = model_output.replace(input_text, "")
                print(model_output, sample["solution"])
                sample.pop("images")
                sample.pop("videos")
                sample.pop("audios")

                # print(model_output)
                # print(sample)
                rets.append(model_output)
                gts.append(sample["solution"])
                sources.append(sample)
                
                



        torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        merged_gts = [None for _ in range(world_size)]
        merged_sources = [None for _ in range(world_size)]
        merged_responses = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_gts, gts)
        torch.distributed.all_gather_object(merged_sources, sources)
        torch.distributed.all_gather_object(merged_responses, rets)

        merged_gts = [_ for _ in itertools.chain.from_iterable(merged_gts)]
        merged_sources = [_ for _ in itertools.chain.from_iterable(merged_sources)]
        merged_responses = [
            _ for _ in itertools.chain.from_iterable(merged_responses)
        ]


        final_output = []
        reward_sum = 0
        if torch.distributed.get_rank() == 0:
            print(f"Evaluating {dataset_name} ...")
            for gt, response, sample in zip(merged_gts, merged_responses, merged_sources):
               
                # think_chain = extract_think(response)
                final_ans = extract_answer(response)# response #extract_answer(response)
                gt_ans = extract_answer(gt)
                if final_ans == "":
                    final_ans = model_output
               
                sample["output"] = response
                sample["prediction"] = final_ans
                problem_type = sample['raw']["problem_type"]
                reward = reward_fn(final_ans, gt_ans, problem_type)

               
                sample['reward'] = reward
                reward_sum += reward
               
                final_output.append(sample)
                

       
            final_acc={'mean_acc': 0.0}
            final_acc['mean_acc'] = float(reward_sum)/len(final_output)

            print(final_acc)
            try:
                os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
                with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                    json.dump({"results": final_output, "final_acc": [final_acc]}, f, indent=2, ensure_ascii=False)
                print(f"Final accuracy saved to {OUTPUT_PATH}")
            except Exception as e:
                print(f"Error writing final accuracy to output file: {e}")
            
            print(f"Results saved to {OUTPUT_PATH}")
        # torch.distributed.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation benchmark")
    parser.add_argument('--model-path', type=str, required=False, help="Path to the model")
    parser.add_argument('--file-name', type=str, required=False, help="Name of the file", default="debug")
    parser.add_argument('--dataset', default=['ib', 'daily', 'world'], nargs='+', type=str)
    args = parser.parse_args()

    main(args)

"""
export PYTHONPATH=./

python -m torch.distributed.launch --use_env  --nproc_per_node 8 --master-port 29502 --nnodes 1  eval/eval_humanomniv2.py \
    --model-path output/qwenomni-stage3  \
    --file-name humanomniv2

"""
