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
# Assuming you have imported your config file like this
# from config import OPENAI_API_KEY

# Set the API key as an environment variable
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
# os.environ["LANGCHAIN_TRACING_V2"] = "true"

tmp = """

**Key Issues (Holding)**
{issues}
######
"""
# Logic Summarization Prompt
# **Key Issue (Holding):** 에 판시 사항 입력. 자동화 필요!
new_prompt = """
**Supreme Court Precedent**

{text}
######

You are tasked with analyzing the given legal case and extracting the following information. You must seperate Used Laws and Cited Precedents exactly.
You MUST answer in Korean. You must select precedents exclusively from those that start with '(대법원)' when providing an answer.

1. **Used Laws**:
   - Identify all laws or statutes 'ONLY' mentioned in the case.
   - For each law, provide:
     - **Law Name**: The name of the statute or legal code.
     - **Provision**: The specific provision or article used in the case.

2. **Cited Precedents**:
   - Identify all Supreme Court precedents 'ONLY' referenced in the case.
   - For each precedent, provide:
     - **Precedent Name**: The name or identifier of the cited precedent case.

Respond in this JSON format:

{format_instructions}
"""

class PrecedentSummarizeLLM:
    def __init__(self, model_name):
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    def generate_summary(self, text):

        response_schemas = [
            ResponseSchema(
                name="Case Analysis",
                description="Extracts details about the laws and precedents cited in the given case. If there is not exist 'Cited Laws' or 'Cited Precedent', you MUST answer as 'blank list' with respect to each fields. Else you MUST answer as 'JSON FORMAT' with respect to each fields.",
                fields=[
                    {
                        "name": "Cited Laws",
                        "description": "A list of laws used in the given case, including their names and specific provisions.",
                        "fields": [
                            {
                                "name": "Law Name and Provision",
                                "description": "Name of the statute or legal code and specific provision or article cited in the case."
                            },
                        ]
                    },
                    {
                        "name": "Cited Precedents",
                        "description": "A list of precedents cited in the given case. You must select precedents exclusively from those that start with '(대법원)' when providing an answer.",
                        "fields": [
                            {
                                "name": "Precedent Name",
                                "description": "Name of the cited precedent case."
                            }
                        ]
                    }
                ]
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

        result = chain.invoke({"text": text})

        return result
    

precedent_summarizer = PrecedentSummarizeLLM(model_name="gpt-4o")

"""
파일명, 판례 전문, 판시사항, 판결요지 여부, 참조조문, 참조판례, 정리된 텍스트
"""

df = pd.read_csv("lbox_with_issues.csv")

LLM_cited_laws = []
LLM_cited_precedent = []

for idx, row in df.iterrows():
    cleaned_content = row['retrieved_case_judicial_opinion']
    filename = row['doc-id']
    retries = 3
    result = None

    for attempt in range(retries):
        try:
            result = precedent_summarizer.generate_summary(cleaned_content)
            break  
        except TimeoutError:
            print(f"Timeout occurred on attempt {attempt + 1} for row {idx} (filename: {filename}). Retrying after 30 seconds...")
            time.sleep(30) 
        except Exception as e:
            print(f"An error occurred on attempt {attempt + 1} for row {idx} (filename: {filename}): {e}")
            break  

    if result is None:
        LLM_cited_laws.append([])
        LLM_cited_precedent.append([])
        continue

    if 'Case Analysis' in result:
        case_analysis = result['Case Analysis']
        if 'Used Laws' in case_analysis:
            LLM_cited_laws.append(case_analysis['Used Laws'])
        else:
            print(f"'Used Laws' key is missing in the result for row {idx}.")
            LLM_cited_laws.append([])

        if 'Cited Precedents' in case_analysis:
            LLM_cited_precedent.append(case_analysis['Cited Precedents'])
        else:
            print(f"'Cited Precedents' key is missing in the result for row {idx}.")
            LLM_cited_precedent.append([])
    else:
        print(f"'Case Analysis' key is missing in the result for row {idx}.")
        LLM_cited_laws.append([])
        LLM_cited_precedent.append([])
    
    print(idx)
    print(result)
    
    
df['LLM_cited_laws'] = LLM_cited_laws
df['LLM_cited_precedent'] = LLM_cited_precedent

df.to_csv("./middle_result.csv", index=False)