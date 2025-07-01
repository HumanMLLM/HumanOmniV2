
from typing import Dict, Any, Union
from trl.data_utils import maybe_apply_chat_template
from transformers import AutoModel, AutoProcessor, AutoConfig
import torch
import re
from open_r1.vlm_modules.vlm_module import VLMBaseModule

from ola.model import OlaQwenForCausalLM
from ola.model.speech_encoder.builder import build_speech_encoder
from ola.conversation import conv_templates, SeparatorStyle
from ola.model.builder import load_pretrained_model
from ola.datasets.preprocess import tokenizer_image_token, tokenizer_speech_image_token, tokenizer_speech_question_image_token, tokenizer_speech_token
from ola.mm_utils import KeywordsStoppingCriteria, process_anyres_video, process_anyres_highres_image
from ola.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_SPEECH_TOKEN, SPEECH_TOKEN_INDEX
from ola.train.train import preprocess_qwen
import re
import os
from datetime import datetime
class QwenOlaModule(VLMBaseModule):
    def __init__(self):
        super().__init__()

    def get_vlm_key(self):
        return "qwen"

    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        if "Ola" in model_id:
            model_cls = OlaQwenForCausalLM
        
        return model_cls

    def is_embeds_input(self):
        return True
    
    def post_model_init(self, model, processing_class):
       
        model.get_model().speech_encoder = build_speech_encoder(model.config)
        model.get_model().speech_encoder.to(device=model.device, dtype=torch.bfloat16)

        vision_tower = model.get_vision_tower()
        print("Loading vision tower...")
        # if not vision_tower.is_loaded:
        vision_tower.load_model()
        # if device != "auto":
            # vision_tower.to(device=model.device, dtype=torch.bfloat16)
        # else:
        vision_tower.to(device=model.device, dtype=torch.bfloat16)
        self.image_processor = vision_tower.image_processor
        self.tokenizer = processing_class

    def get_processing_class(self):
        # vision_tower = self.model.get_vision_tower()
        # self.image_processor = vision_tower.image_processor
        return AutoProcessor
    
    def get_vision_modules_keywords(self):  
        return ['visual', 'speech_projector', 'speech_encoder', 'mm_projector', 'vision_tower', 'vision_resampler', 'whisper_model']
    
    def get_custom_multimodal_keywords(self):
        return ['images', 'speech', 'speech_chunks', 'speech_wav', 'speech_lengths', 'image_sizes', 'modalities', 'images_highres']

    def get_non_generate_params(self):
        return []
    
    def get_custom_processing_keywords(self):
        return ['max_pixels', 'min_pixels']
    
    def prepare_prompt(self, processing_class, inputs: dict[str, Union[torch.Tensor, Any]]):
        prompts_text = [maybe_apply_chat_template(example, processing_class)["prompt"] for example in inputs]
        return prompts_text
    
    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids] 
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=batch_first,
            padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def prepare_model_inputs(self, inputs, processing_class, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False, speech=None):
        # FIXME
        # This could only process pure-multimodal or pure-text inputs
        # prompt_inputs = []
        instances = []
        for each in inputs:
            data_dict = {}
            video = each['image']
            data_dict['speech'] = each['speech']
            data_dict['speech_lengths'] = each['speech_lengths']
            data_dict['speech_chunks'] = each['speech_chunks']
            data_dict['speech_wav'] = each['speech_wav']
            video_processed = []
            for idx, frame in enumerate(video):
                frame = process_anyres_video(frame, self.image_processor)

                video_processed.append(frame.unsqueeze(0))

            video_processed = torch.cat(video_processed, dim=0)
            video_processed = (video_processed, video_processed)
            video = (video_processed, (384, 384), "video")

            data_dict['image'] = [video]

            process_data_dict= preprocess_qwen(
                [each['prompt']],
                self.tokenizer,
                has_speech=True,
                has_image=True)

            data_dict.update(dict(input_ids=process_data_dict["input_ids"][0],
                                labels=process_data_dict["labels"][0]))
            instances.append(data_dict)

        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = [_input_ids[:self.tokenizer.model_max_length] for _input_ids in input_ids]
        labels = [_labels[:self.tokenizer.model_max_length] for _labels in labels]
        if self.tokenizer.pad_token_id is None:
            if "qwen" in self.tokenizer.name_or_path.lower() or "oryx" in self.tokenizer.name_or_path.lower():
                print("Setting pad token to bos token for qwen model.")
                self.tokenizer.pad_token_id = 151643
            else:
                raise NotImplementedError
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # FIXME: this could only be triggered for llama3 model.
        input_ids = self.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = self.pad_sequence(labels,
                                   batch_first=True,
                                   padding_value=IGNORE_INDEX)
        
        prompt_inputs = dict(
            inputs=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
        )
        if 'speech' in instances[0]:
            speeches = [instance['speech'] for instance in instances]
            speeches_lengths = [instance['speech_lengths'] for instance in instances]
            speeches_chunks = [instance['speech_chunks'] for instance in instances]
            speeches_wav = [instance['speech_wav'] for instance in instances]

            prompt_inputs['speech_chunks'] = [au for audio_list in speeches_chunks for au in audio_list]
            prompt_inputs['speech_chunks'] = torch.stack(prompt_inputs['speech_chunks'])
            
            prompt_inputs['speech'] = [au for audio_list in speeches for au in audio_list]
            
            prompt_inputs['speech_lengths'] = [au for audio_list in speeches_lengths for au in audio_list]
            prompt_inputs['speech_lengths'] = torch.stack(prompt_inputs['speech_lengths'])

            prompt_inputs['speech_wav'] = [au for audio_list in speeches_wav for au in audio_list]
            prompt_inputs['speech_wav'] = torch.stack(prompt_inputs['speech_wav'])


            if all(x is not None and x.shape == speeches[0][0].shape for x in prompt_inputs['speech']):
                prompt_inputs['speech'] = torch.stack(prompt_inputs['speech'])

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            prompt_inputs['image_sizes'] = [im[1] for im_list in images for im in im_list]
            prompt_inputs['modalities'] = [im[2] for im_list in images for im in im_list]
            images_lowres = [im[0][0] for im_list in images for im in im_list]
            images_highres = [im[0][1] for im_list in images for im in im_list]
            prompt_inputs['images_highres'] = images_highres
            if all(x is not None and x.shape == images_lowres[0].shape for x in images_lowres):
                prompt_inputs['images'] = torch.stack(images_lowres)
            else:
                prompt_inputs['images'] = images_lowres
                
        return prompt_inputs

        # return prompt_inputs
    
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
        import re
        pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
        # print(completions)
        completion_contents = completions
        matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]
        rewards = []
        for content in completion_contents:
            reward = 0.0
            matches = re.search(pattern, content, re.DOTALL) 
            if matches is not None:
                reward += 0.5
                think_tag_pattern = r'<think>(.*?)</think>'
          
                think_content_match = re.findall(think_tag_pattern, content, re.DOTALL)

                think_pattern = r"<visual>.*?</visual>\s*<auditory>.*?</auditory>"
                print(think_content_match[0])
                if len(think_content_match)==1 and  re.search(think_pattern, think_content_match[0], re.DOTALL) is not None:
                    print(think_content_match[0])
                    reward += 0.5
            
            rewards.append(reward)

        print(rewards)
        return  rewards
    


    @staticmethod
    def precision_reward(completions, solution, **kwargs):
        pass
        
    @staticmethod
    def recall_reward(completions, solution, **kwargs):
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
        print(rewards)
        return rewards
      