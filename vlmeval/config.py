from vlmeval.api import *
from functools import partial


api_models = {
    # GPT
    'GPT4V': partial(GPT4V, model='gpt-4-1106-vision-preview', temperature=0, img_size=512, img_detail='low', retry=10, verbose=False),
    'GPT4V_HIGH': partial(GPT4V, model='gpt-4-1106-vision-preview', temperature=0, img_size=-1, img_detail='high', retry=10, verbose=False),
    'GPT4V_20240409': partial(GPT4V, model='gpt-4-turbo-2024-04-09', temperature=0, img_size=512, img_detail='low', retry=10, verbose=False),
    'GPT4V_20240409_HIGH': partial(GPT4V, model='gpt-4-turbo-2024-04-09', temperature=0, img_size=-1, img_detail='high', retry=10, verbose=False),
    'GPT4o': partial(GPT4V, model='gpt-4o-2024-05-13', temperature=0, img_size=512, img_detail='low', retry=10, verbose=False),
    'GPT4o_HIGH': partial(GPT4V, model='gpt-4o-2024-05-13', temperature=0, img_size=-1, img_detail='high', retry=10, verbose=False),
    'GPT4o_20240806': partial(GPT4V, model='gpt-4o-2024-08-06', temperature=0, img_size=-1, img_detail='high', retry=10, verbose=False),
    'GPT4o_MINI': partial(GPT4V, model='gpt-4o-mini-2024-07-18', temperature=0, img_size=-1, img_detail='high', retry=10, verbose=False),
    # Gemini
    'GeminiPro1-0': partial(GeminiProVision, model='gemini-1.0-pro', temperature=0, retry=10),  # now GeminiPro1-0 is only supported by vertex backend
    'GeminiPro1-5': partial(GeminiProVision, model='gemini-1.5-pro', temperature=0, retry=10),
    'GeminiFlash1-5': partial(GeminiProVision, model='gemini-1.5-flash', temperature=0, retry=10),
    'GeminiFlash2-0': partial(GeminiProVision, model='gemini-2.0-flash', temperature=0, retry=10, min_interval=6.0), # limit 10 calls per minute
    'GeminiFlash2-0-EXP': partial(GeminiProVision, model='gemini-2.0-flash-exp', temperature=0, retry=10, min_interval=6.0), # limit 10 calls per minute
    # Claude
    'Claude3V_Opus': partial(Claude3V, model='claude-3-opus-20240229', temperature=0, retry=10, verbose=False),
    'Claude3V_Sonnet': partial(Claude3V, model='claude-3-sonnet-20240229', temperature=0, retry=10, verbose=False),
    'Claude3V_Haiku': partial(Claude3V, model='claude-3-haiku-20240307', temperature=0, retry=10, verbose=False),
    'Claude3-5V_Sonnet': partial(Claude3V, model='claude-3-5-sonnet-20240620', temperature=0, retry=10, verbose=False),
    'Claude3-5V_Sonnet_20241022': partial(Claude3V, model='claude-3-5-sonnet-20241022', temperature=0, retry=10, verbose=False),
    # VolcEngine
    'VolcEngine_DeepSeekR1': partial(VolcDeepSeekR1, model='ep-20250216235228-69vhs', has_reasoning=True, temperature=0, retry=10, verbose=False),
}

supported_VLM = {}

model_groups = [
    api_models,
]

for grp in model_groups:
    supported_VLM.update(grp)

