import pandas as pd
import inspect
import os
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
#Supreme Court Precedent#

{text}

---

The above text is a precedent from the Supreme Court of KOREA. 
Please carefully review and extract the **specific legal reasoning and supporting grounds** that address the issue below. When drafting the Legal Grounds, interpret the case from the perspective of the key issues outlined in the judgment. 
Ensure that your analysis incorporates and references the relevant laws and cited precedents associated with these key issues. Your response should align with the interpretative approach used in the case. You MUST answer in Korean.

**Key Issue (Holding)** : {issues}

**Cited Laws** : {laws}

**Cited Precedent** : {precedents}

In your response, please include the following:
**Detailed Legal Grounds**:
   - Summarize the court’s primary reasoning and supporting arguments, focusing on specific points relevant to the issue.
   - Include any interpretive approaches (e.g., literal, purposive) used by the court to address this legal question.

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
                name="Legal Grounds",
                description="Detailed summary of the court's reasoning and arguments specific to the issue",
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
                "tags": ["logic_flow"],
                "verbose": True,
            }
        )

        result = chain.invoke({"text": text, "issues" : issues, "laws" : laws, "precedents" : precedents})

        return result
    

precedent_summarizer = PrecedentSummarizeLLM(model_name="gpt-4o")

df = pd.read_csv("lbox_with_prior.csv")

def split_issues_by_index(text):
    text = text.replace("[판시사항]", "").strip()

    pattern = re.compile(r"[［\[]\d+[］\]]") 

    split_items = pattern.split(text) 
    indices = pattern.findall(text) 
    

    results = []
    for i, item in enumerate(split_items):
        if item.strip():
            if i > 0: 
                results.append(f"{indices[i-1]} {item.strip()}")
            else:
                results.append(item.strip())
    
    return results

legal_graound_list = []

for idx, row in df.iterrows():
    filename = row['doc-id']
    cleaned_content = row['retrieved_case_judicial_opinion']
    whole_issues = str(row['issues'])

    prior_information = eval(row['prior_information'])

    issues_list = split_issues_by_index(whole_issues)
    cnt = 0
    summary_results = {} 
    if isinstance(prior_information, dict):
        try:
            for key in list(prior_information['Key Issues']):
                cited_law = prior_information['Key Issues'][key]['Cited Laws']
                cited_precedent = prior_information['Key Issues'][key]['Cited Precedents']
                try:
                    result = precedent_summarizer.generate_summary(
                        cleaned_content, issues_list[cnt], cited_law, cited_precedent
                    )
                    summary_results[issues_list[cnt]] = result
                except Exception as e:
                    print(f"Error at idx: {idx}, issue: {issues_list[cnt]}, error: {e}")
                    summary_results[issues_list[cnt]] = "empty"

                cnt += 1
        except Exception as e:
            print(f"Critical error at idx: {idx}, skipping this row. Error: {e}")
            summary_results = "empty"

    legal_graound_list.append(summary_results)
    print(idx)
    print(summary_results)
    
df['legal_ground'] = legal_graound_list
df.to_csv("final.csv", index=False)

