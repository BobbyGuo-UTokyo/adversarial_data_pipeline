from ..smp import *
import os
import sys
from .base import BaseAPI

APIBASES = {
    'OFFICIAL': 'https://api.deepseek.com/chat/completions',
}


def GPT_context_window(model):
    length_map = {
        'deepseek-chat': 64000,
        'deepseek-reasoner': 64000
    }
    if model in length_map:
        return length_map[model]
    else:
        return 128000


class DeepSeekWrapper(BaseAPI):

    is_api: bool = True

    def __init__(self,
                 model: str = 'deepseek-reasoner',
                 retry: int = 5,
                 wait: int = 5,
                 key: str = None,
                 verbose: bool = False,
                 system_prompt: str = None,
                 temperature: float = 1.0,
                 timeout: int = 60,
                 api_base: str = None,
                 max_tokens: int = 8192,
                 has_reasoning: bool = False,
                 **kwargs):

        self.model = model
        self.cur_idx = 0
        self.fail_msg = 'Failed to obtain answer via API. '
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.has_reasoning = has_reasoning

        env_key = os.environ.get('DEEPSEEK_API_KEY', '')
        if key is None:
            key = env_key
        assert isinstance(key, str) and key.startswith('sk-'), (
            f'Illegal DEEPSEEK_API_KEY {key}. '
            'Please set the environment variable DEEPSEEK_API_KEY to your deepseek api key. '
        )

        self.key = key
        self.timeout = timeout

        super().__init__(wait=wait, retry=retry, system_prompt=system_prompt, verbose=verbose, has_reasoning=has_reasoning, **kwargs)

        if api_base is None:
            if 'DEEPSEEK_API_BASE' in os.environ and os.environ['DEEPSEEK_API_BASE'] != '':
                self.logger.info('Environment variable DEEPSEEK_API_BASE is set. Will use it as api_base. ')
                api_base = os.environ['DEEPSEEK_API_BASE']
            else:
                api_base = 'OFFICIAL'

        assert api_base is not None

        if api_base in APIBASES:
            self.api_base = APIBASES[api_base]
        elif api_base.startswith('http'):
            self.api_base = api_base
        else:
            self.logger.error('Unknown API Base. ')
            raise NotImplementedError

        self.logger.info(f'Using API Base: {self.api_base}; API Key: {self.key}')

    def prepare_inputs(self, inputs):
        input_msgs = []
        if self.system_prompt is not None:
            input_msgs.append(dict(role='system', content=self.system_prompt))
        assert isinstance(inputs, list) and isinstance(inputs[0], dict) and len(inputs) == 1
        assert np.all(['type' in x for x in inputs]) or np.all(['role' in x for x in inputs]), inputs
        if 'role' in inputs[0]:
            assert inputs[0]['role'] == 'user', inputs[0]
            input_msgs.append(dict(role=inputs[0]['role'], content=inputs[0]['content']))
        else:
            input_msgs.append(dict(role='user', content=inputs[0]['value']))
        return input_msgs

    def generate_inner(self, inputs, **kwargs) -> str:
        input_msgs = self.prepare_inputs(inputs)
        temperature = kwargs.pop('temperature', self.temperature)
        max_tokens = kwargs.pop('max_tokens', self.max_tokens)

        headers = {'Content-Type': 'application/json', 'Accept': 'application/json', 'Authorization': f'Bearer {self.key}'}
        payload = dict(
            model=self.model,
            messages=input_msgs,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs)
        response = requests.post(
            self.api_base,
            headers=headers, data=json.dumps(payload), timeout=self.timeout * 1.1)
        ret_code = response.status_code
        ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code
        answer = self.fail_msg
        try:
            resp_struct = json.loads(response.text)
            answer = resp_struct['choices'][0]['message']['content'].strip()
            if self.has_reasoning:
                reasoning = resp_struct['choices'][0]['message']['reasoning_content'].strip()
        except Exception as err:
            if self.verbose:
                self.logger.error(f'{type(err)}: {err}')
                self.logger.error(response.text if hasattr(response, 'text') else response)

        if not self.has_reasoning:
            return ret_code, answer, response
        else:
            return ret_code, answer, reasoning, response

    def get_image_token_len(self, img_path, detail='low'):
        import math
        if detail == 'low':
            return 85

        im = Image.open(img_path)
        height, width = im.size
        if width > 1024 or height > 1024:
            if width > height:
                height = int(height * 1024 / width)
                width = 1024
            else:
                width = int(width * 1024 / height)
                height = 1024

        h = math.ceil(height / 512)
        w = math.ceil(width / 512)
        total = 85 + 170 * h * w
        return total

    def get_token_len(self, inputs) -> int:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(self.model)
        except Exception as err:
            if 'gpt' in self.model.lower():
                if self.verbose:
                    self.logger.warning(f'{type(err)}: {err}')
                enc = tiktoken.encoding_for_model('gpt-4')
            else:
                return 0
        assert isinstance(inputs, list)
        tot = 0
        for item in inputs:
            if 'role' in item:
                tot += self.get_token_len(item['content'])
            elif item['type'] == 'text':
                tot += len(enc.encode(item['value']))
            elif item['type'] == 'image':
                tot += self.get_image_token_len(item['value'], detail=self.img_detail)
        return tot


class DeepSeekR1(DeepSeekWrapper):
    def generate(self, message):
        result = super(DeepSeekR1, self).generate(message)
        answer, reasoning = result
        return answer, reasoning

class DeepSeekV3(DeepSeekWrapper):
    def generate(self, message):
        result = super(DeepSeekV3, self).generate(message)
        answer = result
        return answer
