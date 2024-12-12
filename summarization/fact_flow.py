import pandas as pd
import inspect
import os
import time
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import ChatPromptTemplate
import numpy as np
from openai import OpenAI
import json
import os
from config import OPENAI_API_KEY
import re

# Assuming you have imported your config file like this
# from config import OPENAI_API_KEY

# Set the API key as an environment variable
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

new_prompt = """
### 판례

{text}
---

주어진 판례에서 **법적 주장**, **판결 이유**, **법적 근거** 등을 제외하고 **사실 관계**만 간결하게 추출하세요. 다양한 유형의 판례에 보편적으로 적용할 수 있도록 작성합니다.

---

### 추출 기준

1. **사건 개요**:
   - 사건 발생 배경과 주요 인물(피고인, 피해자, 관계자 등)의 정보.
   - 사건의 발생 원인, 상황, 배경에 대한 상세한 설명.

2. **진행 과정**:
   - 사건이 시간 순서에 따라 어떻게 전개되었는지 세부적으로 작성.
   - 사건의 전개 과정에서 이루어진 행위, 결정, 조치 등을 구체적으로 서술.
   - 관련된 사실적 세부 사항(일시, 장소, 상황 등 포함).
   - 필요한 경우 주요 인물의 상태 변화나 상황에 따라 추가된 행위를 기술.

3. **결과**:
   - 사건의 최종 결과(사망, 손해, 피해, 혐의 인정 등).
   - 주요 인물의 상태 변화 및 사건으로 인해 발생한 결과.
   - 사건으로 발생한 직접적, 간접적 영향을 포함하여 상세히 기술

---

### 제외 항목

- 법적 판단, 판결 이유, 주장의 타당성.
- 법률 조항, 판례 인용, 법적 해석.
- 양측(피고인 및 피해자)의 법적 주장 및 의견.

---

### 작성 형식
- **항목별로 나누어 작성**하며, 각 항목은 구체적이고 명확하게 기술.
- 사건의 흐름을 **시간 순서**에 따라 서술하며, 가능한 세부 사항을 포함.

#### 1. 사건 개요
   - 사건의 배경과 주요 인물의 신원 및 관계
   - 사건의 발생 원인 및 계기

#### 2. 진행 과정
   - 사건이 시간 순으로 어떻게 전개되었는지 세부적으로 작성
   - 구체적인 행위와 그로 인한 상황 변화

#### 3. 결과
   - 사건의 최종 결과 및 주요 인물의 상태 변화
   - 사건으로 인한 추가적인 여파

---

### 주의사항
- 작성할 때 각 사실을 **명확하고 구체적**으로 기술합니다.
- 중요 세부 사항을 빠짐없이 서술하되, 법적 판단은 배제합니다.
- 독자가 사건의 모든 흐름을 이해할 수 있도록 충분히 자세히 작성합니다.

JSON 형식으로 #RESPONSE FORMAT#에 맞게 답하시오:

{format_instructions}
"""

class PrecedentSummarizeLLM:
    def __init__(self, model_name):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0,  
            max_tokens=2000  
        )

    def generate_summary(self, text):
        response_schemas = [
            ResponseSchema(
                name="사건 개요",
                description=(
                    "사건의 배경과 주요 인물(피고인, 피해자 등)의 신원 및 관계를 기술합니다. "
                    "사건 발생의 원인과 계기, 사건의 시간적/공간적 맥락, 사건의 사회적/경제적 배경 등을 포함합니다."
                ),
            ),
            ResponseSchema(
                name="진행 과정",
                description=(
                    "사건이 시간 순서대로 어떻게 전개되었는지를 구체적으로 서술합니다. "
                    "관련된 모든 행위, 결정, 조치, 장소, 날짜 등을 상세히 작성합니다. "
                    "사건 중 발생한 상황 변화와 주요 인물의 상태 변화를 서술하며, 가능한 모든 세부 사항을 포함합니다."
                ),
            ),
            ResponseSchema(
                name="결과",
                description=(
                    "사건의 최종 결과를 상세히 서술합니다. "
                    "사건의 결과로 인한 법적/사회적/경제적 여파를 포함하고, 주요 인물의 상태 변화와 사건의 여파를 분석적으로 작성합니다."
                ),
            )
        ]

        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()

        custom_format_instructions = (
            format_instructions +
            "\n\n**주의사항**: 각 항목은 구체적이고 세부적으로 서술하시오. 가능한 모든 세부 정보를 포함하시오."
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

precedent_summarizer = PrecedentSummarizeLLM(model_name="gpt-4o")
batch_size = 10
max_retries = 3

def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return sentences

def batch_inference(precedent_summarizer, texts, start_idx, df):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_results = []
        
        for text in batch:
            retries = 0
            success = False
            while retries < max_retries:
                try:
                    # GPT 호출
                    result = precedent_summarizer.generate_summary(text)
                    batch_results.append(result)
                    success = True
                    break
                except TimeoutError:
                    retries += 1
                    print(f"Timeout error. Retrying {retries}/{max_retries}...")
                except Exception as e:
                    print(f"Error: {e}")
                    break
            if not success:
                batch_results.append("empty")
        
        for j, result in enumerate(batch_results):
            df.at[start_idx + i + j, 'Fact_flow'] = result
        
        df.to_csv("only_fact.csv", index=False)
        print(batch_results)
        results.extend(batch_results)
        time.sleep(1) 
    
    return results

df = pd.read_csv("lbox_with_issues.csv")

if 'Fact_flow' not in df.columns:
    df['Fact_flow'] = None  # 또는 "" (빈 문자열)
    
texts = []
for idx, row in df.iterrows():
    full_content = row['retrieved_case_judicial_opinion']
    if "【이유】" in full_content:
        text_after_reason = full_content.split("【이유】", 1)[1]
    else:
        text_after_reason = full_content  # "【이유】"가 없을 경우 전체 텍스트 사용
    texts.append(text_after_reason)

precedent_summarizer = PrecedentSummarizeLLM(model_name="gpt-4o")

results = batch_inference(precedent_summarizer, texts, start_idx=0, df=df)
