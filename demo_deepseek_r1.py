from vlmeval.config import supported_VLM
model = supported_VLM['VolcEngine_DeepSeekR1']()
struct = [
    {"type": "text", "value": "What is the capital of France?"}
]
answer, reasoning = model.generate(struct)
print("Answer: ", answer)
print("Reasoning: ", reasoning)
