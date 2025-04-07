from vlmeval.config import supported_VLM
# model = supported_VLM['VolcEngine_DeepSeekR1'](model='', has_reasoning=True, temperature=0, retry=3, verbose=False)
model = supported_VLM['VolcEngine_Doubao-1.5-pro-32k'](model='doubao-1-5-pro-32k-250115', has_reasoning=False, temperature=0, retry=3, verbose=False)
struct = [
    {"type": "text", "value": "What is the capital of Australia?"}
]
# answer, reasoning = model.generate(struct)
# print("Answer: ", answer)
# print("Reasoning: ", reasoning)
answer = model.generate(struct)
print("Answer: ", answer)
