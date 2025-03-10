from vlmeval.config import supported_VLM
model = supported_VLM['VolcEngine_DeepSeekR1'](model='ep-20250216235228-69vhs', has_reasoning=True, temperature=0, retry=3, verbose=False)
struct = [
    {"type": "text", "value": "What is the capital of Australia?"}
]
answer, reasoning = model.generate(struct)
print("Answer: ", answer)
print("Reasoning: ", reasoning)
