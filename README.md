# HumanOmniV2: From Understanding to Omni-Modal Reasoning with Context
- Paper: [Arxiv](https://arxiv.org/abs/2506.21277)
- [IntentBench](https://huggingface.co/datasets/PhilipC/IntentBench)
- [Huggingface](https://huggingface.co/PhilipC/HumanOmniV2)
- [ModelScope](https://modelscope.cn/models/iic/humanomniv2)
## 👀 HumanOmniV2 Overview

<p align="center">
    <img src="./assets/case1.png" width="100%" height="100%">
</p>


With the rapid evolution of multimodal large language models, the capacity to deeply understand and interpret human intentions has emerged as a critical capability, which demands detailed and thoughtful reasoning. In recent studies, Reinforcement Learning (RL) has demonstrated potential in enhancing the reasoning capabilities of Large Language Models (LLMs). Nonetheless, the challenges associated with adapting RL to multimodal data and formats remain largely unaddressed. In this paper, we identify two issues in existing multimodal reasoning models: insufficient global context understanding and shortcut problems. To tackle these issues, we emphasize the necessity for the model to reason with a clear understanding of the global context within multimodal inputs. This global context understanding can effectively prevent the model from overlooking key multimodal cues and ensure a thorough reasoning process. To ensure the accurate interpretation of multimodal context information and improve complex reasoning capability, we implement context reward and logical reward judged by a large language model, alongside format and accuracy rewards. Our proposed method demonstrates advanced performance across multiple omni-modal benchmarks compared to other open-source omni-modal models.

#### 🌟 Contributions in HumanOmniV2

1. We propose that models should summarize the context of multimodal inputs before engaging in the reasoning process. This approach aims to mitigate issues such as skipping crucial multimodal information and context understanding on multimodal inputs.

2. We curate a human-centric benchmark, IntentBench, for omni-modal evaluation, which requires simultaneously understanding video and audio, the global context, complex social relationships, and careful observation.

3. Our proposed HumanOmniV2 achieves the best performance across multiple omni-modal benchmarks compared to existing open-source omni-modal methods.

<p align="center">
    <img src="./assets/model.png" width="100%" height="100%">
</p>

## 🔥 News
We are seeking self-motivated Research Interns and conducting Campus Recruitment in China. Please feel free to contact Tech Manager via WeChat (id: weixihan1)

[2025/07/01] We release training and evaluation codes, model weights, IntentBench, and training data in huggingface🤗 and modelscope🤖

[2025/06/27] We release our paper and part of codes

## 📈 Experimental Results

#### 📍 Results

<p align="center">
    <img src="./assets/daily.png" width="100%" height="100%">
</p>

<p align="center">
    <img src="./assets/world.png" width="100%" height="100%">
</p>

<p align="center">
    <img src="./assets/intent.png" width="100%" height="100%">
</p>



## ⭐ Training detail and evaluation

### Evaluation
Download [IntentBench](https://huggingface.co/datasets/PhilipC/IntentBench), [Daily-Omni](https://huggingface.co/datasets/liarliar/Daily-Omni) and [WorldSense](https://huggingface.co/datasets/honglyhly/WorldSense), and modify the video path in the code. 

```
export PYTHONPATH=./

python -m torch.distributed.launch --use_env  --nproc_per_node 8 --master-port 29502 --nnodes 1  eval/eval_humanomniv2.py \
    --model-path output/humanomniv2  \
    --file-name humanomniv2 \
    --dataset ib
```



### Training

#### Prepare

Download our long CoT data from [huggingface](https://huggingface.co/datasets/PhilipC/IntentTrain), [Video-R1](https://github.com/tulerfeng/Video-R1), [AVQA-R1-6K](https://huggingface.co/datasets/harryhsing/AVQA-R1-6K) or [OmniInstruct](https://huggingface.co/datasets/m-a-p/OmniInstruct)

Modify the config files .yaml for different stages in `data_config`, specify the json and video file paths


#### Training


```bash
# cold start
bash run_scripts/run_sft_qwenomni.sh

# Stage 1 RL
bash run_scripts/run_grpo_qwenomni_stage2.sh

# Stage 2 RL
bash run_scripts/run_grpo_qwenomni_stage3.sh
```

### Examples
https://private-user-images.githubusercontent.com/81946943/461066941-6129fb84-0b5a-4ba3-8b77-9a60380df474
```
Question: What is the relationship between these two people?

A. They want to call attention to the product.
B. The relationship between the two individuals are business partners, who are trying to pitch their product.
C. They are promoting a different product.
D. It seems like they just met so that he can show his product on her show.


HumanOmniV2: 
<context>The video shows two people sitting on opposite sides of a table, with a laptop and a flower vase in the middle. The background features a stone wall and greenery, creating a relaxed and formal atmosphere. The man is wearing a light blue shirt, and the woman is wearing a black dress, with jewelry. They are frequently smiling and nodding, displaying a friendly and cooperative relationship. The logo "The Balancing ACT" is clearly displayed in the background.
The man in the light blue shirt speaks in a rapid and excited tone, emphasizing the importance of the topic; the woman in the black dress responds with a calm and gentle tone, deliberately slowing her speech to ease the tension. The conversation revolves around the introduction of a new feature (Facebook function) of the "Family iBoard" product, with the woman in the black dress expressing high interest and gratitude multiple times.</context><think>Okay, I'm trying to understand the relationship between these two people. I'm looking at the video and listening to their conversation.

First, I'm noticing the setting. They're in what looks like a studio, with a "The Balancing ACT" logo in the background. This makes me think it's some kind of show or segment. The woman is dressed formally, and the man is in a light blue shirt. They seem comfortable with each other, smiling and nodding, which suggests a professional but friendly dynamic.

Now, listening to what they're saying, the man is explaining a new feature of a product called "Family iBoard." He mentions something about a "Facebook function" and how it can "post to Facebook automatically." The woman, on the other hand, sounds really interested. She says things like, "That is so great," and thanks him, even calling it one of her "favorite segments." This tells me she's not just a customer; she seems to be the host or a key figure here, and she's clearly promoting the product.

So, if I consider the options:

One idea is that they're trying to "attract attention to the product." That's definitely happening, but it's a bit too general. I need something more specific.
Another thought is that they're "business partners pitching a product." I don't really see any clear evidence of that. There's no talk about profits or business strategies between them; it's all about the product itself.

What if they're "promoting a different product"? No, that doesn't fit either. The conversation is entirely focused on the "Family iBoard." I don't hear any mention of other products.
The idea that they're "collaborating to showcase the product on a show" seems to fit best. The woman is the host, and the man is presenting the product. Their interaction, like when he says, "Thank you for having me," and she responds, "I can't thank you enough," really points to this. It looks like he's demonstrating the product for her show, and she's the one promoting it.
Let me just re-check the visual cues. Yes, the "The Balancing ACT" logo, the professional attire, and their friendly yet professional interaction all support the idea that this is a collaborative effort on a show. The conversation is clearly about introducing this new feature of the "Family iBoard" to her audience.</think><answer>D</answer>

```

https://private-user-images.githubusercontent.com/81946943/461067390-789d13c6-926d-4845-8c11-84cef7a5b72a
```
Question: What are the dominant emotional states of the man in the video?
A. Questioning
B. Helpless
C. Curiosity
D. Sentimentality
E. Angry
F. Friendly

HumanOmniV2:
<context>The video shows a male character inside a car. Initially, his facial expression is serious, with slightly furrowed brows and a determined look. As the video progresses, his expression becomes more intense, with his brows furrowing deeper and his eyes squinting slightly. His mouth is slightly open, and his head is slightly tilted forward. Towards the end, his expression softens, and he appears to be sighing. The audio features the character speaking in a low, gentle tone, with a slow pace. The subtitles display the text: "I really can't stand her sitting opposite me, looking pitiful and trying to please me. This is not a normal mother-son relationship at all."</context><think>Okay, I'm watching this video. At the start, the man in the car looks quite serious. His brows are a bit furrowed, and he has this determined expression. He seems focused on something, maybe a conversation.

As I continue to watch, his face gets more intense. His brows are furrowing deeper, and his eyes are squinting a little. He looks like he's really trying to get his point across, or perhaps he's feeling some pressure. His mouth is slightly open, and he's leaning forward a bit, which makes me think he's quite engaged and possibly a bit agitated. Now, towards the end of this sequence, his expression changes. He seems to be sighing, and his face has softened. This makes me think he's reached a conclusion or is feeling something different. 
Now, I'm listening to the audio. He's speaking, and his voice is low and gentle. He's not shouting or raising his voice at all; it's quite calm. 

The subtitles show what he's saying: 'I really can't stand her sitting opposite me, looking pitiful and trying to please me. This is not a normal mother-son relationship at all.' So, he's expressing strong dissatisfaction with how someone (presumably a woman, his mother, given the context of the subtitle) is behaving towards him. He finds her pleading look unpleasant and feels this isn't a healthy way for them to interact.

This statement, combined with his earlier intense expression and now softer, sighing demeanor, suggests he's feeling a strong sense of frustration and perhaps a bit of helplessness about this situation. He's not just angry about the behavior itself, but he feels stuck or unable to change it, which makes me think he's feeling helpless.

The initial intensity and the sighing at the end, along with the content of his speech, point to a deep-seated feeling of being unable to accept or deal with the current state of affairs. This leads me to believe he's feeling **Helpless** and also quite **Angry** about the situation.</think><answer>B,E</answer>
```

### ✒️ Citation

If you find our work helpful for your research, please consider citing our work.   

```bibtex
@article{yang2025humanomniv2,
  title={HumanOmniV2: From Understanding to Omni-Modal Reasoning with Context},
  author={Yang, Qize and Yao, Shimin and Chen, Weixuan and Fu, Shenghao and Bai, Detao and Zhao, Jiaxing and Sun, Boyuan and Yin, Bowen and Wei, Xihan and Zhou, Jingren},
  journal={arXiv preprint arXiv:2506.21277},
  year={2025}
}
```



## 👍 Acknowledgement
HumanOmniV2 is built with reference to the following outstanding works: [Video-R1](https://github.com/tulerfeng/Video-R1), [Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni), and many other awesome models and datasets! Thanks！


## 📜 License

- Our models and code are under the Apache License 2.0.
- But our self-collected videos are under [**CC BY-NC-SA 4.0**](https://creativecommons.org/licenses/by-nc-nd/4.0/) license.
