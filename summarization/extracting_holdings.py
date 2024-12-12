import pandas as pd
import inspect
import os
import pickle
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import ChatPromptTemplate
import numpy as np
from openai import OpenAI
import json
import os
import re
from config import *
# Assuming you have imported your config file like this
# from config import OPENAI_API_KEY

# Set the API key as an environment variable
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

issue_prompt = """
판결문에서 전문을 읽고 판시 사항을 뽑고자 합니다.

# 판시 사항이란??
'판시 사항'은 법원이 특정 사건을 판결하면서 판단한 핵심적인 법적 쟁점이나 문제를 의미합니다. 
이는 판결문에서 다루어진 주요 논점을 간략하게 정리한 것으로, 해당 판결의 법리적 판단을 이해하는 데 도움을 줍니다. 
예를 들어, 판결문에서 "간접점유자의 주민등록이 주택임대차의 유효한 공시방법이 되는지 여부"와 같이 표현되며, 이는 해당 사건에서 다루어진 주요 쟁점을 나타냅니다.

# 판결문이 짧으면 판시 사항을 적게 뽑아주세요.

###
판시사항은 대법원 판결에서 해당 사건의 핵심 쟁점이나 법적 판단의 요지를 간략하게 정리한 부분입니다. 
이는 판례의 이해를 돕기 위해 작성되며, 판결의 주요 내용을 한눈에 파악할 수 있도록 돕습니다. 
판시사항은 일반적으로 판결문 상단에 위치하며, 해당 사건에서 다루어진 법률적 쟁점과 그에 대한 법원의 판단을 요약합니다.

판시사항을 작성할 때는 다음과 같은 요소를 고려합니다:

1. 핵심 쟁점 파악: 사건에서 가장 중요한 법적 쟁점이 무엇인지 식별합니다.
2. 간결한 표현: 복잡한 법률 용어나 긴 설명을 피하고, 핵심 내용을 간략하게 정리합니다.
3. 명확한 구조: 쟁점과 그에 대한 법원의 판단을 명확하게 구분하여 서술합니다.
4. 일관성 유지: 판시사항의 작성 방식은 다른 판례들과 일관성을 유지하여 독자가 쉽게 이해할 수 있도록 합니다.


### 예를 들어 판시사항은 다음과 같이 작성될 수 있습니다:
{ref_text1}
{ref_text2}
{ref_text3}

## 주의 사항
# 판시 사항에서는 환송 사유 같은 건 없습니다.
# 판시 사항에서는 피고인, 피해자와 같은 용어는 사용하지 않습니다. 갑, 을 등의 용어로 대체하시기 바랍니다.
# 판시 사항은 평서문으로 작성하지 않습니다.

---
아래의 판결문에 대해 판시 사항을 작성하시오.
{input_text}

Provide your answer in JSON format.
{format_instructions}
"""

class IssueLLM:
    def __init__(self, model_name):
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    ## Completeness, Conciseness
    def extract_issue(self, text, ref_text1, ref_text2, ref_text3):
        
        response_schemas = [
            ResponseSchema(
                name="판시 사항",
                description="판시 사항을 깔끔하게 정리하시오",
            )
        ]
        
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()

        prompt = ChatPromptTemplate.from_template(
            issue_prompt,
            partial_variables={"format_instructions": format_instructions},
        )

        chain = prompt | self.llm | output_parser

        chain = chain.with_config(
            {
                "run_name": inspect.currentframe().f_code.co_name,
                "tags": [self.llm.model_name],
            }
        )

        result = chain.invoke({"input_text": text, "ref_text1": ref_text1, "ref_text2": ref_text2, "ref_text3": ref_text3})

        return result

issue_extracter = IssueLLM(model_name="gpt-4o")

# 주문 텍스트 추출 함수
def extract_sections(text):

    # 판시 사항 추출
    issue_match = re.search(r'(【판시사항】.*?)【판결요지】|(【판시사항】.*?)【참조조문】|(【판시사항】.*?)【참조판례】|(【판시사항】.*?)【전문】', text, re.DOTALL)
    issue = next((m for m in issue_match.groups() if m), "No Issue Found") if issue_match else "No Issue Found"

    # 주문 이후 텍스트 추출
    order_match = re.search(r'(【주문】.*)', text, re.DOTALL)
    order_text = order_match.group(1).strip() if order_match else "No Order Text Found"

    return issue.strip(), order_text

from datasets import load_dataset


input_text_folder = "법전원판례" #원본 파일【주문】
result_folder = "extract_issue"  # 결과 저장 폴더

ref_text1_path = "대법원 2016. 8. 17. 선고 2014다235080 판결.txt"
ref_text2_path = "대법원 2015. 11. 27. 선고 2015두48136 판결.txt"
ref_text3_path = "대법원 2011. 10. 27. 선고 2011다42324 판결.txt"

lbox_result_folder = "lbox_extract_issue"
os.makedirs(lbox_result_folder, exist_ok=True)

# 결과 저장 폴더 생성 (없으면 생성)
os.makedirs(result_folder, exist_ok=True)

ds = load_dataset("lbox/kbl", "kbl_reasoning_case_relevance_qa_q")

for data in ds['test']:

    input_text = data['retrieved_case_judicial_opinion']
    precedent_name = data['doc-id']

    with open(ref_text1_path, "r") as f:
        ref_text1 = f.read()

    with open(ref_text2_path, "r") as f:
        ref_text2 = f.read()

    with open(ref_text3_path, "r") as f:
        ref_text3 = f.read()

    result = issue_extracter.extract_issue(input_text, ref_text1, ref_text2, ref_text3)

    result_file = os.path.join(lbox_result_folder, f"{precedent_name}_issues.json")

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump({"issues": result}, f, ensure_ascii=False, indent = 4)

    print(f"Generated Response for {precedent_name}: \n Result: {result}")
