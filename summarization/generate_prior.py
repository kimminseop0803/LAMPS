import pandas as pd
import inspect
import os
import time
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import ChatPromptTemplate
import numpy as np
from openai import OpenAI
import re
from config import OPENAI_API_KEY


# from config import OPENAI_API_KEY

# Set the API key as an environment variable
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
# os.environ["LANGCHAIN_TRACING_V2"] = "true"

new_prompt = """
**Supreme Court Precedent**
{text}
######

**Candidate Laws**
{laws}

**Candidate Precedents**
{precedents}

**Key Issues (Holding)**
{issues}
######

You are tasked with thoroughly analyzing the given **Korean Supreme Court Precedent** and extracting #Cited Precedents# and #Cited Laws#. The provided **Key issues** represent the core aspects addressed in the case, and you must interpret the given case for each key issue individually. 
Key issues are presented in numbered or lettered format when addressing multiple aspects, while a single aspect is expressed in a simple sentence format.
For each **Key issue**, you must extract #Cited Precedents# and #Cited Laws# and respond accordingly. You must select your answers regarding **Cited Laws** and **Cited Precedents** exclusively from **Candidate Laws** and **Candidate Precedents**, respectively. 
You must answer in Korean.

1. Cited Laws
    - List the #laws# referenced and utilized in the analyzed case for each key issue.
    - Describe the interpretive approach applied to each cited law.

2. Cited Precedents
    - List the #precedents# referenced and utilized in the analyzed case for each key issue.
    - Describe the interpretive approach applied to each cited precedent.

Respond in this JSON format:

{format_instructions}
"""

class PrecedentSummarizeLLM:
    def __init__(self, model_name):
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    def generate_summary(self, text, issues, laws, precedents):

        response_schemas = [
            ResponseSchema(
                name="Key Issues",
                description="Each key issue's index with its associated cited laws and precedents.",
                fields=[
                    {
                        "name": "Key Issue",
                        "description": "Description of the specific key issue"
                    },
                    {
                        "name": "Cited Laws",
                        "description": "A list of cited laws or legal status, with details on each and its relevance to the issue",
                        "fields": [
                            {
                                "name": "Law",
                                "description": "Name of the statute or legal code, including the relevant provisions."
                            },
                            {
                                "name": "Relevance",
                                "description": "Brief explanation of how it supports the issue when explaining the Supreme Court case."
                            }
                        ]
                    },
                    {
                        "name": "Cited Precedents",
                        "description": "A list of cited cases, with details on each case and its relevance to the issue.",
                        "fields": [
                            {
                                "name": "Precedent",
                                "description": "Name of the cited case precedent."
                            },
                            {
                                "name": "Relevance",
                                "description": "Brief explanation of how it supports the issue when explaining the Supreme Court case"
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
                "tags": ["generate_prior"],
                "verbose": True,
            }
        )

        result = chain.invoke({"text": text, "issues" : issues, "laws" : laws, "precedents" : precedents})

        return result

precedent_summarizer = PrecedentSummarizeLLM(model_name="gpt-4o")

df = pd.read_csv("middle_result.csv")

prior_information_list = []


def split_issues_by_index(text):
    pattern1 = r"\[\d{1,2}\]"
    pattern2 = r"\n[가-힣]\."
    pattern3 = r"\［\d+\］"
    pattern4 = r"\n\d{1,2}\."

    split_case1 = re.split(pattern1, text)
    split_case2 = re.split(pattern2, text)
    split_case3 = re.split(pattern3, text)
    split_case4 = re.split(pattern4, text)
    if len(split_case1) != 1:
        return [text.strip() for text in split_case1][1:]
    elif len(split_case2) != 1:
        return [text.strip() for text in split_case2][1:]
    elif len(split_case3) != 1:
        return [text.strip() for text in split_case3][1:]
    elif len(split_case4) != 1:
        return [text.strip() for text in split_case4][1:]
    else:
        return [text] 

for idx, row in df.iterrows():
    filename = row['doc-id']
    cleaned_content = row['retrieved_case_judicial_opinion']
    issues = str(row['issues'])
    LLM_cited_laws = eval(row['LLM_cited_laws'])
    LLM_cited_precedents = eval(row['LLM_cited_precedent'])
    input_laws = ""
    result = None
    empty_flag = False
    number_of_issues = len(split_issues_by_index(issues))
    if not len(LLM_cited_laws) == 0:
        for i in range(len(LLM_cited_laws)):
            input_laws += LLM_cited_laws[i]['Law Name'] + LLM_cited_laws[i]['Provision'] + "\n"
    if len(LLM_cited_laws) == 0 and len(LLM_cited_precedents) == 0:
        issues_list = split_issues_by_index(issues)
        tmp_dict = {}
        for idx2, tmp_issue in enumerate(issues_list):
            tmp_dict[str(idx2)] = {'Cited Laws': [], 'Cited Precedents': []}
        result = {"Key Issues": tmp_dict}
        empty_flag = True

    if not empty_flag:
        retry_count = 0
        while retry_count < 3:
            try:
                result = precedent_summarizer.generate_summary(
                    cleaned_content, issues, input_laws, LLM_cited_precedents
                )
                break  
            except Exception as e:
                retry_count += 1
                print(f"Error at index {idx}, attempt {retry_count}: {e}")
                time.sleep(2) 
                if retry_count == 3:
                    print(f"Failed to process index {idx} after 3 attempts. Skipping.")
                    result = {"Error": str(e)}

    prior_information_list.append(result)
    print(idx)
    print(result)
df['prior_information'] = prior_information_list

df.to_csv('lbox_with_prior.csv', index = False)