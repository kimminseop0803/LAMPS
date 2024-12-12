import pandas as pd
import inspect
import os
import pickle
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import ChatPromptTemplate
import numpy as np
from openai import OpenAI
import time
import os
from config import OPENAI_API_KEY
from langsmith import Client
from datasets import load_dataset
import re

ds = load_dataset("lbox/kbl", "kbl_reasoning_case_relevance_qa_q")
# Assuming you have imported your config file like this
# from config import OPENAI_API_KEY

# Set the API key as an environment variable
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
# os.environ["LANGCHAIN_TRACING_V2"] = "true"

new_prompt = """
다음 판결문이 의뢰인의 주장을 뒷받침하나요? 

의뢰인의 주장 : {arguments}
---

**판결문**

상고인 : {client} 
사실관계 : {fact} 
당사자들의 주장 : {client_opinion} 
판사의 의견 : {judge_opinoin}

다음 판결문이 의뢰인의 주장을 뒷받침하나요? 

A:{A}, B:{B} 중 하나를 선택하여 '답변: A'과 같이 단답식으로 답하시요.

{format_instructions}
"""

class PrecedentSummarizeLLM:
    def __init__(self, model_name):
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    def generate_summary(self, client_opinion, client, fact, arguments, judge_opinoin, A, B):

        response_schemas = [
            ResponseSchema(
                name="Answer",
                description="Answer as 'A' or 'B' according to upper request.",
            )
        ]
        
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()

        prompt = ChatPromptTemplate.from_template(
            new_prompt,
            partial_variables={"format_instructions": format_instructions},
        )

        chain = prompt | self.llm | output_parser

        chain = chain.with_config(
            {
                "run_name": inspect.currentframe().f_code.co_name,
                "tags": [self.llm.model_name],
                "verbose": True,
            }
        )
        # print("client_opinion:", client_opinion)
        # print("client:", client)
        # print("fact:", fact)
        # print("arguments:", arguments)
        # print("judge_opinoin:", judge_opinoin)
        result = chain.invoke({"client_opinion": client_opinion, "client": client, "fact": fact, "arguments": arguments, "judge_opinoin": judge_opinoin, "A" : A, "B" : B})

        return result
    

precedent_summarizer = PrecedentSummarizeLLM(model_name="gpt-4o")

df = pd.read_csv("final_with_fact.csv")

answer_list = []

for idx, row in df.iterrows():

    client_opinion = row['retrieved_case_claim']
    client = row['retrieved_case_appellant']
    fact = row['retrieved_case_fact']
    arguments = row['query']

    legal_opinion, fact_opinion = "", ""

    tmp_legal_dict = eval(row['legal_ground'])
    tmp_fact_dict = eval(row['Fact_flow'])


    for key in list(tmp_legal_dict.keys()):
        legal_opinion += (tmp_legal_dict[key]['Legal Grounds'])

    for key in list(tmp_fact_dict.keys()):
        fact_opinion += tmp_fact_dict[key]

    judge_opinoin = fact_opinion + legal_opinion

    retries = 3
    result = None

    for attempt in range(retries):
        try:
            result = precedent_summarizer.generate_summary(client_opinion, client, fact, arguments, judge_opinoin, row['A'], row['B'])
            break  # 성공하면 루프 종료
        except TimeoutError:
            print(f"Timeout occurred on attempt {attempt + 1} for df.iloc[idx] {idx}. Retrying after 30 seconds...")
            time.sleep(30)  # 30초 대기
        except Exception as e:
            print(f"{e}")
            break  # 다른 예외는 즉시 중단
    answer_list.append(str(result))
    print(idx)
    print(result)

    
df['answer'] = answer_list

df.to_csv("lbox_with_ansewr.csv", index=False)    