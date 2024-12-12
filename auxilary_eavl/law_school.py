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

new_prompt = """{situation}

{question}
######

**관련 판례**
{precedents}
######

{question}

위 판례를 근거로 하여, 주어진 질문에 대해 답하시오. 답변 시,답변에서 해당 판례의 내용을 근거로 충분히 활용하시오. 판례를 인용할 시에는, 판례가 인용된 부분에 판례의 이름을 작성하시오.

{format_instructions}
"""

class PrecedentSummarizeLLM:
    def __init__(self, model_name):
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    def generate_summary(self, situation, question, precedents):

        response_schemas = [
            ResponseSchema(
                name="Answer",
                description="Answer about user's question. When You answer You must answer in Korean. Also you have to generate specific answer not about probability",
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
        result = chain.invoke({"situation": situation, "question": question, "precedents": precedents})

        return result
    

precedent_summarizer = PrecedentSummarizeLLM(model_name="gpt-4o")


df = pd.read_csv("legal_qa_dataset.csv")

pattern = r'\b\d{2,4}[가-힣]\d+\b'
ds_result = []

idx_list = []
cnt = 0
answer_list= []

for idx, row in df.iterrows():
    situation = row["input"]
    question = row['instruction']
    precedents_list = eval(row['legal_grounds'])
    precedents_list2 = eval(row['Fact_Sum'])

    precedent = ""
    for i in precedents_list:
        precedent += i + "\n\n"
    retries = 3
    result = None

    for attempt in range(retries):
        try:
            result = precedent_summarizer.generate_summary(situation, question, precedent)
            break  
        except TimeoutError:
            print(f"Timeout occurred on attempt {attempt + 1} for df.iloc[idx] {idx}. Retrying after 30 seconds...")
            time.sleep(30)  
        except Exception as e:
            print(f"{e}")
            break  
    answer_list.append(str(result))
    print(idx)
    print(result)

    
df['answer'] = answer_list

df.to_csv("./legal_qa_data_with_ansewr.csv", index=False)    