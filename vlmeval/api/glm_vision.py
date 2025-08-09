import requests
requests.packages.urllib3.disable_warnings()

from vlmeval.smp import *
from vlmeval.api.base import BaseAPI
from vlmeval.smp.vlm import encode_image_file_to_base64


class GLMVisionWrapper(BaseAPI):

    is_api: bool = True

    def __init__(self,
                 model: str,
                 retry: int = 5,
                 key: str = None,
                 verbose: bool = True,
                 system_prompt: str = None,
                 temperature: float = 1.0,
                 timeout: int = 60,
                 max_tokens: int = 4096,
                 has_reasoning: bool = False,
                 proxy: str = None,
                 **kwargs):

        from zhipuai import ZhipuAI
        self.model = model
        self.temperature = temperature
        self.has_reasoning = has_reasoning
        self.fail_msg = 'Failed to obtain answer via API. '
        if key is None:
            key = os.environ.get('GLMV_API_KEY', None)
        assert key is not None, (
            'Please set the API Key (obtain it here: '
            'https://bigmodel.cn)'
        )
        self.client = ZhipuAI(api_key=key, timeout=timeout)
        super().__init__(retry=retry, system_prompt=system_prompt, verbose=verbose, has_reasoning=has_reasoning, **kwargs)

    def build_msgs(self, msgs_raw, system_prompt=None, dataset=None):
        msgs = cp.deepcopy(msgs_raw)
        content = []
        for i, msg in enumerate(msgs):
            if msg['type'] == 'text':
                content.append(dict(type='text', text=msg['value']))
            elif msg['type'] == 'image':
                content.append(dict(type='image_url', image_url=dict(url=encode_image_file_to_base64(msg['value']))))
        if dataset in {'HallusionBench', 'POPE'}:
            content.append(dict(type="text", text="Please answer yes or no."))
        ret = [dict(role='user', content=content)]
        return ret

    def generate_inner(self, inputs, **kwargs) -> str:
        assert isinstance(inputs, str) or isinstance(inputs, list)
        inputs = [inputs] if isinstance(inputs, str) else inputs

        messages = self.build_msgs(msgs_raw=inputs, dataset=kwargs.get('dataset', None))

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            do_sample=False,
            max_tokens=2048,
            temperature=self.temperature,
            thinking={
                "type": "enabled" if self.has_reasoning else "disabled",
            }
        )
        try:
            answer = response.choices[0].message.content.strip()
            if self.verbose:
                self.logger.info(f'inputs: {inputs}\nanswer: {answer}')
            if self.has_reasoning:
                if isinstance(response.choices[0].message.reasoning_content, str):
                    reasoning = response.choices[0].message.reasoning_content.strip()
                else:
                    reasoning = ''
                return 0, answer, reasoning, 'Succeeded!'
            else:
                return 0, answer, 'Succeeded!'
        except Exception as err:
            if self.verbose:
                self.logger.error(f'{type(err)}: {err}')
                self.logger.error(f'The input messages are {inputs}.')
            if self.has_reasoning:
                return -1, self.fail_msg, '', ''
            else:
                return -1, self.fail_msg, ''


class GLMVisionAPI(GLMVisionWrapper):
    def generate(self, message, dataset=None):
        return super(GLMVisionAPI, self).generate(message, dataset=dataset)
