import pandas as pd
import inspect
import os
import pickle
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import ChatPromptTemplate
import numpy as np
from openai import OpenAI
import re
import json
from tqdm import tqdm

import os
from config import *
# Assuming you have imported your config file like this
# from config import OPENAI_API_KEY

# Set the API key as an environment variable
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

keylogic_prompt = '''
Please extract as many key reasoning points in **KOREAN** as possible from the following KOREA Supreme Court summary judgment in propositional form. 
Break down each logical component into separate statements to capture all reasoning details. 

Each propositional statement should include:

1.Main reasoning and legal grounds
2.Key issues in dispute
3.Influence of relevant precedents or statutes on the decision

Present each logical component as an independent proposition. 
The proposition should be short and compact.

## Caution
The propositions must be in **KOREAN**
The proposition must clearly demonstrate its logical nature.

---
Below is the Supreme Court summary judgment:

{text}
---
Present **only** the propositions in a JSON format as follows:

{{"Key Logic": 
    {{"0":"<Proposition 1>"}},
    {{"1":"<Proposition 2>"}},
    {{"2":"<Proposition 3>"}},
    ...
}}

If the summary is "nan", return {{"Key Logic": "nan"}}

Respond in this JSON format:

{format_instructions}

'''



class KeylogicLLM:
    def __init__(self, model_name):
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    def extract_keylogic(self, text):
        response_schemas = [
            ResponseSchema(
                name="Key Logic",
                description="The answer to the user's question.",
            )
        ]
        
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()

        prompt = ChatPromptTemplate.from_template(
            keylogic_prompt,
            partial_variables={"format_instructions": format_instructions},
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
    

keylogic_extracter = KeylogicLLM(model_name="gpt-4o")

import pandas as pd
import ast

# 파일 경로 설정
input_path ="factsumAdd2.csv"
output_path = "output_with_keylogic.csv"

keylogic_cache = {}

def process_case_summaries(case_summaries):
    try:
        case_list = ast.literal_eval(case_summaries)
        unique_cases = list(set(case_list))
        
        keylogic_results = []
        for case_summary in unique_cases:
            precedent_name = re.match(r'^[^:]+', case_summary).group()
            if case_summary in keylogic_cache:
                keylogic_results.append(keylogic_cache[case_summary])
            else:
                retries = 3
                for attempt in range(retries):
                    try:
                        result = keylogic_extracter.extract_keylogic(case_summary)
                        break 
                    except Exception as e:
                        print(f"An error occurred on attempt {attempt + 1} for file {case_summary}: {e}")
                        break
                
                print("precedent_name: ", precedent_name, "result: ", result)
                keylogic_cache[case_summary] = {precedent_name:result}
                keylogic_results.append({precedent_name:result})
        
        return keylogic_results
    except Exception as e:
        print(f"Error processing: {case_summaries}, Error: {e}")
        return []

data = pd.read_csv(input_path)

data["keylogic"] = data["판결 요지"].apply(process_case_summaries)

data.to_csv(output_path, index=False)
print(f"처리가 완료된 파일이 {output_path}에 저장되었습니다.")
