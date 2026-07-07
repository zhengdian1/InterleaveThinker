
<h1 align="left">
  <img src="assets/logo.png" height="55" align="absmiddle"> InterleaveThinker: Reinforcing Agentic Interleaved Generation
</h1>

Official repository for the paper "[InterleaveThinker: Reinforcing Agentic Interleaved Generation]()".

<!-- [[🌍 Project Page](https://zhengdian1.github.io/InterleaveThinker-proj/)] [[📖 Paper](assets/paper.pdf)] [[🤗 Train-Data](https://huggingface.co/InterleaveThinker/Train-Data)] [[🤗 InterleaveThinker-Planner-8B](https://huggingface.co/InterleaveThinker/InterleaveThinker-Planner-8B)] [[🤗 Critic-SFT-8B](https://huggingface.co/InterleaveThinker/Critic-SFT-8B)] [[🤗 InterleaveThinker-Critic-8B](https://huggingface.co/InterleaveThinker/InterleaveThinker-Critic-8B)]  -->

[[🌍 Project Page](https://zhengdian1.github.io/InterleaveThinker-proj/)] [[📖 Paper](https://arxiv.org/pdf/2606.13679)] [[🤗 Demo](https://huggingface.co/spaces/zhengli1013/interleavethinker)] [[🤗 InterleaveThinker-Planner-8B](https://huggingface.co/InterleaveThinker/InterleaveThinker-Planner-8B)] [[🤗 Critic-SFT-8B](https://huggingface.co/InterleaveThinker/Critic-SFT-8B)] [[🤗 InterleaveThinker-Critic-8B](https://huggingface.co/InterleaveThinker/InterleaveThinker-Critic-8B)] 

## 💥 News
- **[2026.07.07]** Release HF demo, enjoy it! 🚀
- **[2026.06.12]** Release paper, models, training, inference. 🚀

## 💭 Introduction

<p align="center">
  <img src="assets/teaser.jpg" width="90%">
</p>

We introduce **InterleaveThinker**, as the first multi-agent pipeline designed to **endow any existing image generator with interleaved generation capabilities**. InterleaveThinker can organize the image-text input sequence via a planner agent, evaluate generator outputs, identify deviations, and refine instructions via a critic agent, **enabling complex interleaved text-image sequence generation for visual narratives, guidance, embodied manipulation and long-horizon sub-task annotation.**

We build three dedicated training datasets—Interleave-Planner-SFT-80k, Interleave-Critic-SFT-112k, and Interleave-Critic-RL-13k—for interleaved generation and step-wise instruction correction using GRPO with proposed accuracy and step-wise rewards.

InterleaveThinker achieves **performance comparable to Nano Banana and GPT-5 on interleaved generation benchmarks**, delivering substantial gains on reasoning-based benchmarks (e.g., boosting WISE from 0.47 to 0.73 and RISE from 13.3 to 28.9 on 4-step FLUX.2-klein). It also demonstrates strong transferability, improving performance across various existing image generators.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/zhengdian1/InterleaveThinker.git
cd InterleaveThinker

# build inference environment
conda create -n interleavethinker python=3.11
conda activate interleavethinker

pip install torch==2.6.0 torchvision --extra-index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation

# build SFT environment
conda create -n llamafactory python=3.11 
conda activate llamafactory
cd train/LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation

# build RL environment
conda create -n easyr1 python=3.11 
conda activate easyr1
cd train/EasyR1
pip install -e .

```

For more details for the SFT and RL environment installation, please refer to [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory),  [EasyR1](https://github.com/hiyouga/EasyR1)

Our final folder architecture is as follows and you need to set the ROOT correctly in each file.

```text
ROOT/
├── cache/
├── code/
│   └── InterleaveThinker/
├── ckpt/
│   ├── planner_sft/
│   ├── critic_sft/
│   └── critic_rl/
└── envs/
    └── interleavethinker
```

<!-- For more details for the SFT and RL environment installation, please refer to [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory),  [EasyR1](https://github.com/hiyouga/EasyR1)

Then, download the training datasets [[🤗 Train-Data](https://huggingface.co/datasets/InterleaveThinker/Train-Data)] and unzip all the data.

The `planner_sft.json`, `critic_sft.json` is for SFT cold start while `critic_rl.jsonl` file is for RL training. For SFT data, you need to modify the image path to abs path and place it into `ROOT/data`.

Our final folder architecture is as follows and you need to set the ROOT correctly in each file.

```text
ROOT/
├── cache/
├── data/
│   └── InterleaveThinker/
│       └── Train-Data
├── code/
│   └── InterleaveThinker/
├── ckpt/
│   ├── planner_sft/
│   ├── critic_sft/
│   └── critic_rl/
└── envs/
    └── interleavethinker
``` -->

### 🎓 Training


#### Supervised Fine-Tuning (SFT)

```bash
cd InterleaveThinker
bash train/LLaMA-Factory/local_scripts/sft_planner.sh
bash train/LLaMA-Factory/local_scripts/sft_critic.sh
```

#### Setup Image Generator API Service

Before RL training, you need to start the image generator API service. We support multiple editing models including **FLUX.2-klein**

```bash
# Navigate to the server directory
cd inference/server

# Activate the inference environment 
conda activate interleavethinker

# Start the FLUX.2-klein service (Service 11, Port 8011)
bash start_service_with_ip.sh 11 ip.txt

# The service will automatically save its IP to ip.txt
# Service will be available at http://<your-ip>:<port>
```

**Service Options:**
- `1` - Qwen-Image Generation (Port: 8001)
- `2` - Qwen-Image Lightning Generation (Port: 8002)
- `3` - FLUX.1-Krea-dev Generation (Port: 8003)
- `4` - Qwen-Image-Edit (Port: 8004)
- `5` - Qwen-Image-Edit Lightning (Port: 8005)
- `6` - FLUX.1-Kontext-dev Edit (Port: 8006)
- `7` - FLUX.1-Fill-dev Fill (Port: 8007)
- `8` - LongCat-Image-Edit (Port: 8008)
- `9` - OmniGen2-Image-Edit (Port: 8009)
- `10` - Qwen-Image-Edit-Plus (Port: 8010)"
- `11` - FLUX.2-klein (Port: 8011)

**Multi-Node Deployment with Load Balancing**

For production deployment with high throughput requirements, you can deploy multiple service nodes and use Nginx for load balancing.

**Step 1: Start Multiple Service Nodes**

```bash
cd inference/server

# On Node 1 (e.g., 192.168.1.101)
conda activate interleavethinker
bash start_service_with_ip.sh 11 ip.txt

# On Node 2 (e.g., 192.168.1.102)
conda activate interleavethinker
bash start_service_with_ip.sh 11 ip.txt

# On Node 3 (e.g., 192.168.1.103)
conda activate interleavethinker
bash start_service_with_ip.sh 11 ip.txt

# All nodes will automatically save their IPs to ip.txt
```

**Step 2: Setup Nginx Load Balancer**

Use the provided script to automatically configure and start a user-level Nginx load balancer (no sudo required):

```bash
cd inference/server

# Setup Nginx load balancer
# Parameters:
#   -i: IP file path (contains all backend server IPs)
#   -d: Base directory for Nginx logs/config
#   -b: Backend port (the port your services are running on)
#   -p: Proxy port (the port Nginx will listen on)

bash setup_user_nginx.sh \
  -i ip.txt \
  -d /tmp/nginx_edit_service \
  -b 8011 \
  -p 8080
```

The script will automatically:
- ✅ Read all server IPs from `ip.txt`
- ✅ Generate Nginx configuration with load balancing
- ✅ Start user-level Nginx instance (no sudo required)
- ✅ Create management scripts (start/stop)
- ✅ Display access URL and management commands

**Step 3: Verify and Use**

```bash
# Check backend health
curl http://<nginx-server-ip>:8080/health

# Use the load balancer endpoint in your inference
EDIT_API_ENDPOINT="http://<nginx-server-ip>:8080"
```

#### Reinforcement Learning (RL)

**Step 1: Configure Reward Function**

Edit the configuration in `train/EasyR1/verl/reward_function/interleave_thinker_reward.py`:

```python
# Configure your GPT-4.1 API for evaluation
GEMINI_API_KEYS: List[str] = ["your_api_key_here"]
EDIT_API_ENDPOINT: Optional[str] = "http://your-nginx-ip:8080"  # Use FLUX.1-Kontext service
EDIT_MODEL_NAME: str = "klein"  
```

Edit the configuration in `train/EasyR1/verl/reward_function/gemini.py`:

```python
GEMINI_AZURE_ENDPOINT: str = "https://your-endpoint.com/v1/openai/native"
```

**Step 2: Run RL Training**

Configure and run the training script:

```bash
bash train/EasyR1/local_scripts/run_interleave_thinker_rl.sh
```

### 📊 Inference

We provide two inference scripts: **Klein** (no api, relative low quality) and **Nano-Pro** (more powerful).

Note that our critic is extremely strict to handle hard cases, if you only want a simple, quick interleaved generation, using our Critic-SFT model instead Critic-RL.

### 📊 Evaluation on UEval

We provide evluation code of InterleaveThinker + FLUX.2-klein-9B on UEval

```bash
cd UEval
bash eval.sh
python merge.py
```
`merge.py` is used to merge the sample on the rank jsons into one. Then evaluate using the official code of UEval.

```bash
python ueval_eval.py \
  --model_output_path result/final.json \
  --output_path result/score.json \
```

For the details, please refer to [UEval](https://github.com/zlab-princeton/UEval)

### 📊 Interleaved Data Construction Pipeline

We provide our raw data construction pipeline and how to use the data to construct interleaved sequence for UMM training. Please refer to [DATA](https://github.com/zhengdian1/InterleaveThinker/blob/main/data_gen/README.md)

## Acknowledgements

We sincerely appreciate the contributions of the open-source community. The related projects are as follows: [EasyR1](https://github.com/hiyouga/EasyR1), [verl](https://github.com/volcengine/verl),  [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory),  [EditThinker](https://github.com/appletea233/EditThinker)


## ✔️ Citation

Please cite us if you find this project helpful:

```bibtex
@article{zheng2026interleavethinker,
  title={InterleaveThinker: Reinforcing Agentic Interleaved Generation},
  author={Zheng, Dian and Lee, Harry and Zhang, Manyuan and Feng, Kaituo and Guo, Zoey and Zhang, Ray and Li, Hongsheng},
  journal={arXiv preprint arXiv:2606.13679},
  year={2026}
}
```
