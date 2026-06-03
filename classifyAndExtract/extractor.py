import json
import re
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


_PROMPT_TEMPLATE = """\
你是一个专业的合同条款提取助手。请从以下合同段落中，提取与"{clause_type}"直接相关的原文内容。

要求：
1. 只提取与该条款类型直接相关的句子，不得修改或概括任何文字。
2. 如果段落中不包含该类型条款，返回空列表。
3. 严格以 JSON 格式输出，键名为 "clauses"，值为原文字符串列表。

合同段落：
{segment}

输出（仅 JSON，不要包含其他内容）：\
"""


class ClauseExtractor:
    """
    Wraps a local causal LM (e.g. Qwen2.5-72B-Instruct) to extract verbatim
    contract clauses from pre-classified segments.
    """

    def __init__(self, model_name: str, device_map: str = 'auto'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.model.eval()

    def extract(
        self,
        segment: str,
        clause_type: str,
        max_new_tokens: int = 512,
    ) -> List[str]:
        """
        Extract verbatim clauses of `clause_type` from `segment`.

        Returns:
            List of verbatim clause strings found in the segment.
            Empty list if none are found or the model output cannot be parsed.
        """
        prompt = _PROMPT_TEMPLATE.format(clause_type=clause_type, segment=segment)
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = self.tokenizer.decode(
            output_ids[0][inputs['input_ids'].shape[-1]:],
            skip_special_tokens=True,
        )
        return self._parse(generated)

    @staticmethod
    def _parse(text: str) -> List[str]:
        """Extract the 'clauses' list from the model's JSON output."""
        try:
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                data = json.loads(match.group())
                clauses = data.get('clauses', [])
                if isinstance(clauses, list):
                    return [c for c in clauses if isinstance(c, str) and c.strip()]
        except (json.JSONDecodeError, AttributeError):
            pass
        return []
