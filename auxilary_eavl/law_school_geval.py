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

# Assuming you have imported your config file like this
# from config import OPENAI_API_KEY

# Set the API key as an environment variable
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
# os.environ["LANGCHAIN_TRACING_V2"] = "true"

new_prompt = """
## Situation
{situation}


## Question
{question}
######

**변호사 조언**
{golden_label}
######

주어진 상황 하에서 사용자 질문인 Question에 대한 변호사의 실제 조언은 **변호사 조언**이다.
사용자 질문에 대한 답변 후보인 A와 B를 확인하고 **변호사 조언**과 내용, 근거 및 결론이 일치하는 정도를 판단하고, A와 B 중 어떤 답변이 더 좋은 답변인지 판단하여 출력하라.

답변은 단순히 A 혹은 B로 더 좋은 답변을 출력하여야 한다. 답변은 #Example foramt#을 따라서 작성하여라. 이때, 반드시 A 혹은 B 중 하나의 답변을 선택하여라.

## A
{answer_with_sum}

## B
{answer}


#Example format#
{{"Answer" : ""}}

{format_instructions}
"""

class PrecedentSummarizeLLM:
    def __init__(self, model_name):
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    def generate_summary(self, situation, question, golden_label, answer_with_sum, answer):

        response_schemas = [
            ResponseSchema(
                name="Answer",
                description="Choose A or B which is closer to the **변호사 조언** you think. And you have to answer with JSON FORMAT",
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
        # print("situation:", situation)
        # print("question:", question)
        # print("precedents:", precedents)
        result = chain.invoke({"situation": situation, "question": question, "golden_label": golden_label, "answer_with_sum" : answer_with_sum, "answer": answer})

        return result
    
df = pd.read_csv("legal_qa_data_with_ansewr.csv")
df_result = pd.read_csv("legal_qa_data_with_original.csv")
precedent_summarizer = PrecedentSummarizeLLM(model_name="gpt-4o")


answer_list = []

for i, row in df.iterrows():
    situation = row['input']
    question = row['instruction']
    golden_label = row['output']
    answer_with_sum = row['answer']
    answer = row['answer_판결요지']

    retries = 3
    result = None

    for attempt in range(retries):
        try:
            result = precedent_summarizer.generate_summary(situation, question, golden_label, answer, answer_with_sum)
            break
        except TimeoutError:
            print(f"Timeout occurred on attempt {attempt + 1} for df.iloc[idx] {i}. Retrying after 30 seconds...")
            time.sleep(30)
        except Exception as e:
            print(f"{e}")
            break 
    answer_list.append((str(result)))
    print(i)
    print(result)
    print(pd.Series(answer_list).value_counts())


df_result['gpt_win_result'] = answer_list

df_result.to_csv("./lawschool_g_eval.csv", index=False)
    
        

    