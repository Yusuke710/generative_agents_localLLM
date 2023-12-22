# Copy and paste your OpenAI API Key
openai_api_key = "" # keep this empty as we use LocalLLM
# Put your name
key_owner = "" # keep this empty

# huggingface key to load localLLM
from huggingface_hub import login
hf_hey = ""
login(hf_hey)
checkpoint = "meta-llama/Llama-2-7b-chat-hf" #"TheBloke/Llama-2-70B-Chat-fp16"  #"bigscience/bloom-560m"
embedding_checkpoint = "jinaai/jina-embeddings-v2-base-en"

# declare model here so function do not have to call this part everytime
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device) # device_map="auto" distributes LLM accross multiple GPUs (DON'T SET DEVICE MAP FOR TRAINING; ONLY FOR INFERENCING)


maze_assets_loc = "../../environment/frontend_server/static_dirs/assets"
env_matrix = f"{maze_assets_loc}/the_ville/matrix"
env_visuals = f"{maze_assets_loc}/the_ville/visuals"

fs_storage = "../../environment/frontend_server/storage"
fs_temp_storage = "../../environment/frontend_server/temp_storage"

collision_block_id = "32125"

# Verbose 
debug = True
