from transformers import Qwen2_5OmniThinkerForConditionalGeneration, AutoProcessor, Qwen2_5OmniProcessor
from typing import Dict, Any, Union
from trl.data_utils import maybe_apply_chat_template
import torch
from qwen_omni_utils import process_mm_info

from datetime import datetime
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import os
import re
import ast
import time
from open_r1.vlm_modules.vlm_module import VLMBaseModule
import requests
import re


url = os.environ["API"]
token = os.environ["API_KEY"]

def gpt_api(prompt, model_name):
    messages = [
                {
                    "role": "user",
                    "content": prompt

                }
            ]
    success = False
    max_try = 20
    tries = 0
    response_message = ""
    while (not success and tries <max_try):
        try:

            data = {

                    "model": "qwen2.5-72b-instruct",
                    "messages":messages,
                    # "n": 1
                }

            headers = {
                    "Content-Type": "application/json",
                        "Authorization": 'Bearer ' + token}
            response = requests.post(url, json=data, headers=headers, timeout=15)
            # print(response.json())
            response = response.json()
            response_message = response['choices'][0]['message']['content']


            success = True
        except Exception as e:
            print(f'{response},{e}')
            time.sleep(1)
            tries +=1


    return response_message

class QwenOmniModule(VLMBaseModule):
    def __init__(self):
        super().__init__()

    def get_vlm_key(self):
        return "qwen"

    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        
        return Qwen2_5OmniThinkerForConditionalGeneration
    
    def post_model_init(self, model, processing_class):
        pass
    
    def get_processing_class(self):
        return Qwen2_5OmniProcessor
    
    def get_vision_modules_keywords(self):  
        return ['visual','audio_tower']
    
    def get_custom_multimodal_keywords(self):
        return ['pixel_values', 'pixel_values_videos', 'image_grid_thw', 'video_grid_thw', 'video_second_per_grid', 'feature_attention_mask', 'input_features', 'audio_feature_lengths', 'use_audio_in_video', 'rope_deltas']

    def get_non_generate_params(self):
        return []
    
    def get_custom_processing_keywords(self):
        return ['max_pixels', 'min_pixels']
    
    def prepare_prompt(self, processing_class, inputs: dict[str, Union[torch.Tensor, Any]]):
        prompts_text = [maybe_apply_chat_template(example, processing_class)["prompt"] for example in inputs]
        return prompts_text
    
    def prepare_model_inputs(self, processing_class, prompts_text, images, audios, videos, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False, use_audio_in_video=False):

        # print(audios)
        prompt_inputs = processing_class(
            text=prompts_text,
            images=images,
            audio=audios,
            videos=videos,
            return_tensors=return_tensors,
            padding=padding,
            padding_side=padding_side,
            add_special_tokens=add_special_tokens,
            use_audio_in_video=use_audio_in_video)
       
        return prompt_inputs
    
    @staticmethod
    def get_question_template(task_type: str):
        match task_type:
            case "rec":
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
            case _:
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."
            
            
    @staticmethod
    def format_reward(completions, **kwargs):
        """Check if the Qwen model output matches a specific format."""

        pattern = r"^<context>.*?</context>\s*<think>.*?</think>\s*<answer>.*?</answer>"
        # pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>"
        
        completion_contents = [completion[0]["content"] for completion in completions]
        # print(completion_contents)
        matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]
        rewards = []
        for content in completion_contents:
            reward = 0.0
            matches = re.search(pattern, content, re.DOTALL) 
            if matches is not None:

                reward += 1
            
            rewards.append(reward)

        # print(rewards)
        return  rewards
    


    @staticmethod
    def precision_reward(completions, solution, **kwargs):

        completions = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion, sol in zip(completions, solution):
            reward = 0.0
            # print(completion, sol)
            answer_tag_pattern = r'<answer>(.*?)</answer>'
            # Try symbolic verification first
            # try:
            content_answer_match = re.search(answer_tag_pattern, completion, re.DOTALL)
            if content_answer_match:
                content_answer = content_answer_match.group(1).strip()
                words = content_answer.split(",")
                count = 0
                for each in sol:
                    if each.lower() in content_answer or each in content_answer:
                        count +=1

                reward = float(count)/len(sol)
                # bbox_match = re.search(bbox_pattern, content_answer)
            rewards.append(reward)
            # except Exception as e :
            #     pass  # Continue to next verification method if this fails
        # print(rewards)
        return rewards
      
        
    @staticmethod
    def recall_reward(completions, solution, **kwargs):
        import re
        completions = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion, sol in zip(completions, solution):
            reward = 0.0
            # print(completion, sol)
            answer_tag_pattern = r'<answer>(.*?)</answer>'
            # Try symbolic verification first
            # try:
            content_answer_match = re.search(answer_tag_pattern, completion, re.DOTALL)
            if content_answer_match:
                content_answer = content_answer_match.group(1).strip()
                words = content_answer.split(",")
                count = 0
                for each in sol:
                    if each.lower() in content_answer or each in content_answer:
                        count +=1

                reward = float(count)/len(sol)
                # bbox_match = re.search(bbox_pattern, content_answer)
            rewards.append(reward)
            # except Exception as e :
            #     pass  # Continue to next verification method if this fails
        # print(rewards)
        return rewards

    @staticmethod
    def accuracy_reward(completions, solution, **kwargs):
    
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
                print(f"Error converting '{num_str}' to float: {e}")
                return None

        def wer(reference, hypothesis):
            ref_words = reference.split()
            hyp_words = hypothesis.split()
            m = len(ref_words)
            n = len(hyp_words)
            d = [[0]*(n+1) for _ in range(m+1)]
            for i in range(m+1):
                d[i][0] = i
            for j in range(n+1):
                d[0][j] = j
            for i in range(1, m+1):
                for j in range(1, n+1):
                    if ref_words[i-1] == hyp_words[j-1]:
                        d[i][j] = d[i-1][j-1]
                    else:
                        d[i][j] = 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
            return d[m][n] / max(1, m)


        def compute_rouge_score(reference, hypothesis, use_stemmer=True):
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=use_stemmer)
            scores = scorer.score(reference, hypothesis)
            average_fmeasure = (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3
            return average_fmeasure

        def similarity(reference, hypothesis):
            prompt = f"""
            Analyze the consistency between the content of the two compared texts and assign a score based on the following criteria:

            Grading criteria description (content consistency):

            5 points: The core facts, details, and logical relationships in the two texts are entirely consistent, with no differences.
            3-4 points: The core content is consistent, but there are differences in non-critical details (such as expression, supplementary information, examples, etc.).
            1-2 points: Some content is consistent, but there are contradictions or differences in key information.
            0 points: The core content is inconsistent or completely irrelevant.

            Example analysis process:

            Extract the core information from both texts (time, place, people, events, data, conclusions, etc.).
            Compare whether key facts align (e.g., whether the times are the same, whether the data matches).
            Analyze the consistency of logical relationships (causal relationships, sequence, etc.).
            Determine whether the differences are merely expressive (such as synonym replacement, sentence adjustment) or substantive content differences.

            reference: {reference}
            hypothesis: {hypothesis}

            only return the score number:
            """
    
            try:
                reward = gpt_api(prompt=prompt, model_name="qwen-plus")
                reward = ast.literal_eval(reward) / 5.0
            except:
                return 0

            return reward
        
 
        
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


        question_type = kwargs['problem_type'][0]
        
        contents = [completion[0]["content"] for completion in completions]
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        rewards = []

        for content, sol in zip(contents, solution):
        
            try:
                output_ans = extract_answer(content)
                gt_ans = extract_answer(sol)
                if question_type == "multiple choice":
                    reward = 1.0 if output_ans.strip() == gt_ans.strip() else 0.0
                elif question_type == "numerical":
                    gt_has_decimal = ("." in gt_ans) or ("," in gt_ans)
                    out_has_decimal = ("." in output_ans) or ("," in output_ans)
                    if gt_has_decimal != out_has_decimal:
                        reward = 0.0
                    else:
                        gt_number = normalize_number(gt_ans)
                        out_number = normalize_number(output_ans)
                        if gt_number is None or out_number is None:
                            reward = 0.0
                        else:
                            reward = 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0
                elif question_type == "OCR":
                    error_rate = wer(gt_ans, output_ans)
                    reward = 1 - error_rate
                    reward = max(0.0, min(1.0, reward))
                elif question_type == "free-form":
                    reward = similarity(gt_ans, output_ans)
                    # score = compute_rouge_score(gt_ans, output_ans)
                    # reward = max(0.0, min(1.0, score))
                elif question_type == "regression":
                    gt_number = normalize_number(gt_ans)
                    out_number = normalize_number(output_ans)
                    if gt_number is None or out_number is None:
                        reward = 0.0
                    rel_diff = (abs(out_number - gt_number) + 1e-9) / (abs(gt_number) + 1e-9)
                    rel_diff = min(1.0, max(0.0, rel_diff))
                    reward = 1 - rel_diff
                elif question_type == "emer_ov":
                    reward = emer_ov_gpt(gt_ans, output_ans)
                elif question_type == "emer_ov_mc":
                    reward = emer_ov_mc(gt_ans, output_ans)
                elif  question_type == "judge":
                    reward = judge(output_ans, gt_ans)
                else:
                    reward = 0.0
            except Exception as e:
                print(f"Error in reward_fn for question_type '{question_type}': {e}")
                reward = 0.0
        
            rewards.append(reward)
            
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                # local_rank = int(os.getenv("LOCAL_RANK", 0))
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n")
                
        return rewards

    @staticmethod
    def patial_context_reward(completions, solution, **kwargs):
    
        def extract_parts(text, pattern):

            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
            return ""

        def similarity(reference, hypothesis):

            prompt = \
f"""You are assessing how well the 'hypothesis' text covers the key information from the 'reference' text. Differences in wording or extra details in the 'hypothesis' are fine if the 'reference's' main points are included.:

Score based on this coverage:

5 points : Hypothesis clearly and accurately reflects significant core themes or key aspects of the reference. It demonstrates a good understanding of a substantial part of the reference material.
4 points : Hypothesis reflects some important themes or aspects of the reference. The connection is evident, though perhaps not as comprehensive or central as a 5.
2 points : Hypothesis shows a recognizable connection to themes or aspects of the reference, but it might be more superficial, focus on less central points, or only partially grasp a key aspect.
1 points : Hypothesis has a tenuous or very limited connection to the reference. It might touch on a peripheral detail or a heavily reinterpreted aspect, but largely misses the main substance.
0 points : Hypothesis does not reflect any significant themes or key aspects of the reference, or is on a completely different topic.

Example analysis process:

Identify main themes and key aspects in 'reference'.
Determine if 'hypothesis' connects to or discusses any of these themes/aspects from 'reference'.
Judge the strength and relevance of this connection. Is a core part of the 'reference' reflected?
Differences are expected; evaluate if the 'hypothesis' still meaningfully reflects some key part of the 'reference'.
Assign score based on how well a significant aspect is reflected.

reference: {reference}
hypothesis: {hypothesis}

only return the score number:"""
    
            try:
                reward = gpt_api(prompt=prompt, model_name="qwen-plus")
                reward = ast.literal_eval(reward) / 5.0
            except:
                return 0

            return reward
           


        question_type = kwargs['problem_type'][0]
        
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
        
            try:
                output_evidence = extract_parts(content, pattern=r'<context>\s*(.*?)\s*</context>')
                
                gt_evidence = extract_parts(sol, pattern=r'<context>\s*(.*?)\s*</context>')
                if len(gt_evidence)==0:
                    reward = 0.0
          
                else:
           
                    reward = similarity(gt_evidence, output_evidence)

            except Exception as e:
                print(f"Error in reward_fn for question_type '{question_type}': {e}")
                reward = 0.0
        
            rewards.append(reward)
            
     
        return rewards

    @staticmethod
    def patial_reasoning_reward(completions, solution, **kwargs):
    
        def extract_parts(text, pattern):
         
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
            return ""

        def rationality(reference, hypothesis):
            
           
            prompt = \
f"""Please analyze whether the reasoning text is derived from the evidence and context text based on the following criteria and give a score of 0-5:
Grading criteria description (relevance and rationality):

Integration of Clues (1 point): During the reasoning process, there is incorporation of clues from the video, image, or audio.

Reflection and Confirmation (1 point): The reasoning involves reflection or second confirmation of choices or answers, including revisiting video, image, or audio evidence.

Logical Reasoning (1 point): The thought process is clear, deriving conclusions through rigorous logical reasoning, analysis, or extension without additional assumptions or contradictions.

Problem Analysis (1 point): The reasoning process includes thorough analysis in conjunction with the problem at hand.

Overall Consistency (1 point): The reasoning text is based on visual or audio evidence and context information, presenting no extra assumptions or contradictions.

Assign one point for each criterion that is met, for a total possible score of five points. Verify that each criterion is addressed and reflect this in your scoring.

context: {reference}
reasoning path: {hypothesis}

only return the score number:
            """
            try:
                reward = gpt_api(prompt=prompt, model_name="qwen-plus")
                reward = ast.literal_eval(reward) / 5.0
            except:
                return 0

            return reward


        question_type = kwargs['problem_type'][0]
        
        contents = [completion[0]["content"] for completion in completions]

        rewards = []

        for content, sol in zip(contents, solution):
        
            try:
                evidence =  extract_parts(content, pattern=r'<context>\s*(.*?)\s*</context>')
                think_path = extract_parts(content, pattern=r'<think>\s*(.*?)\s*</think>')
                answer = extract_parts(content, pattern=r'<answer>\s*(.*?)\s*</answer>')
                
                if len(evidence)==0 or len(think_path)==0:
                    reward = 0.0
                    
                else:
                    # output_think = extract_parts(content, pattern=r'<think>\s*(.*?)\s*</think>')


                    reward = rationality(evidence, think_path)

            except Exception as e:
                print(f"Error in reward_fn for question_type '{question_type}': {e}")
                reward = 0.0
        
            rewards.append(reward)
            
     
        return rewards

        