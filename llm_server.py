# llm_server.py
import os
import requests

class ModelAPI:
    def __init__(self, MODEL_URL, provider="local", api_key=None, timeout=60):
        self.url = MODEL_URL
        self.provider = provider
        self.api_key = api_key
        self.timeout = timeout

        self.headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

        print(f"[LLM] provider={self.provider}, url={self.url}")
        print(f"[DEBUG] api_key is None? {self.api_key is None}")
        print(f"[DEBUG] headers={self.headers}")


    def generate(self, prompt, max_tokens=1500, temperature=0.0):
        if self.provider == "deepseek":
            payload = {
                "model": "deepseek-chat",   
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            resp = requests.post(
                self.url,
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        else:
            payload = {
                "prompt": prompt,
                "max_length": max_tokens,
                "temperature": temperature
            }
            resp = requests.post(
                self.url,
                json=payload,
                timeout=self.timeout
            )
            resp.raise_for_status()
            return resp.json().get("text", "")
    def chat(self, query, history=None, **kwargs):
        """
        兼容原项目的 chat 接口
        :param query: 当前输入
        :param history: 对话历史（原项目需要，但 DeepSeek/OpenAI 不强制）
        :return: (answer, new_history)
        """
        if history is None:
            history = []

        # 简单做法：把 history 拼进 prompt（可选）
        prompt = ""
        for h in history:
            prompt += f"用户：{h[0]}\n助手：{h[1]}\n"
        prompt += f"用户：{query}\n助手："

        answer = self.generate(prompt, **kwargs)

        # 按原项目约定，history 是 [(q, a), ...]
        new_history = history + [(query, answer)]
        # 此处暂时不返回历史
        return answer
