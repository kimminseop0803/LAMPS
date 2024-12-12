import pandas as pd
import inspect
import os
import json
import pickle
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import ChatPromptTemplate
import numpy as np
from openai import OpenAI

import os
from config import *
# Assuming you have imported your config file like this
# from config import OPENAI_API_KEY

# Set the API key as an environment variable
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

new_prompt = """
### 판례

{text}
---

주어진 판례에서 **법적 판단**, **주장**, **판결 이유** 등을 제외하고 **Keyfact**를 추출하세요. Keyfact는 다음 정의를 따릅니다.

---

### Keyfact 정의 및 기준

1. **Keyfact**는 단일 핵심 정보를 담은 간결한 문장입니다.
2. **각 문장은 최대 1개의 사건과 2개의 관련 엔티티(인물, 사건, 날짜 등)**만 포함해야 합니다.
   - 하나의 문장에 여러 사건이나 복잡한 상황을 포함하지 마십시오.
3. **복잡한 사건은 Keyfact를 여러 문장으로 나누어 작성**합니다.
4. 사건의 **사실 관계**를 기반으로 명확하게 서술합니다.
5. 사건의 시간 순서에 따라 Keyfact를 나열합니다.
---

### 제외 항목

- 법적 판단, 판결 이유, 법률 조항.
- 피고인이나 피해자 측의 법적 주장 및 의견.
- 법적 해석이나 주관적 표현.

---

### 작성 형식 및 주의사항

- 각 문장은 단순하고 명확해야 하며, 여러 정보를 한 문장에 포함하지 마십시오.
- 각 Keyfact는 독립적이어야 하며, 논리적으로 사건의 흐름을 나타내야 합니다.
- 필요하다면 동일 사건을 잘게 나누어 2~3개의 Keyfact로 분리하십시오.
- 긴 문장을 방지하기 위해 Keyfact를 최대한 단순하게 나누세요.

Key Fact만 아래와 같은 JSON 형식으로 작성하시오:

"Key Fact": 
    "0":"Key Fact 1",
    "1":"Key Fact 2",
    "2":"Key Fact 3",
    ...
---

JSON 형식으로 #RESPONSE FORMAT#에 맞게 답하시오:

{format_instructions}
"""


class KeyfactLLM:
    def __init__(self, model_name):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            max_tokens=2000
        )
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    def extract_keyfact(self, text):
        response_schemas = [
            ResponseSchema(
                name="Keyfact",
                description=(
                    "Keyfact는 단일한 핵심 정보를 담은 간결한 문장입니다. "
                    "각 문장은 하나의 사건과 2~3개의 관련 엔티티(인물, 사건, 날짜 등)로 구성됩니다. "
                    "모든 Keyfact는 리스트로 반환되어야 하며, 각 항목은 개별 문장입니다. "
                    "긴 문장은 반드시 나누어 단일 핵심 정보를 담는 여러 문장으로 작성해야 합니다. "
                    "사건의 시간 순서에 따라 Keyfact를 나열하십시오."
                ),
            )
        ]

        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()

        custom_format_instructions = (
            format_instructions +
            "\n\n**주의사항**: Keyfact는 dict 형태로 반환되어야 하며, 인덱스가 key, 각 문장이 dict의 value 개별 항목이 되어야 합니다. "
            "하나의 문장에 여러 사건을 포함하지 마십시오."
        )

        prompt = ChatPromptTemplate.from_template(
            new_prompt,
            partial_variables={"format_instructions": custom_format_instructions},
        )

        chain = prompt | self.llm | output_parser

        chain = chain.with_config(
            {
                "run_name": inspect.currentframe().f_code.co_name,
                "tags": [self.llm.model_name],
            }
        )

        result = chain.invoke({"text": text})

        return result

    

keyfact_extracter = KeyfactLLM(model_name="gpt-4o")

import os
import json

input_folder = "법전원판례"
output_folder = "keyfact"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    input_path = os.path.join(input_folder, filename)

    if os.path.isfile(input_path):
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()

        if "【이유】" in text:
            text_after_reason = text.split("【이유】", 1)[1]
        else:
            text_after_reason = text 

        result = keyfact_extracter.extract_keyfact(text_after_reason)
        output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_keyfacts.json")
        with open(output_path, "w", encoding="utf-8") as json_file:
            json.dump(result, json_file, ensure_ascii=False, indent=4)
        
        print(f"Processed: {filename} -> Saved to: {output_path}")

print("All files processed and saved.")



