
import datasets
import torch
from smolagents import CodeAgent, DuckDuckGoSearchTool, TransformersModel

# 1. Load Data
gaia_data = datasets.load_dataset("gaia-benchmark/GAIA", "2023_all", split="validation")

# 2. Setup Local Model (Qwen 2.5 Coder 7B)
print("⏳ Loading local model...")
model = TransformersModel(
    model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    flatten_messages_to_text=False
)

# 3. Configure Agent
agent = CodeAgent(
    tools=[DuckDuckGoSearchTool()], 
    model=model,
    add_base_tools=True, 
    max_steps=20, 
    verbosity_level=1
)

# 4. Run Evaluation
print("🚀 Starting Local Evaluation...")
for i, task in enumerate(gaia_data):
    if i >= 5: break
    print(f"[{i+1}] {task['Question'][:100]}...")
    try:
        instruction = "Use Python for math. Search for facts."
        answer = agent.run(instruction + "\nQUESTION: " + task['Question'])
        print(f"✅ Answer: {answer}\n")
    except Exception as e:
        print(f"❌ Failed: {e}\n")
