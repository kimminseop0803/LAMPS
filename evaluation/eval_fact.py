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

## Completeness, Conciseness 프롬프트
eval_prompt = """
Given the summary text and a list of key fact statements, please evaluate whether each key fact is represented in the summary. 

For each key fact, respond with:
1.'yes' if the key fact is present in the summary, and 'no' if it is not. The format should be [present: yes or no]
2.If 'yes', list the Summary Sentence index where the key fact is included. The format should be [Summary Sentences: [<Summary Sentence index if present, otherwise empty>]]

Therefore the overall respond should be like this:
"Eval":
"<keyfact index>" : "present": <yes or no>, "Summary Sentences": [<Summary Sentence index if present, otherwise empty>]
...continue

Analyze each Summary Sentence individually and provide specific responses. 

---
Here is the summary text:

{summary}
---
And here are the key fact statements to check against the summary:

{keyfact}
---
Respond in this JSON format:

{format_instructions}

"""
## Faithfulness 프롬프트
eval_factual_prompt = """You will receive a transcript followed by a corresponding summary.

Your task is to assess the factuality of each summary sentence across nine categories:

* no error: the statement aligns explicitly with the content of the transcript and is factually consistent with it.
* out-of-context error: the statement contains information not present in the transcript.
* entity error: the primary arguments (or their attributes) of the predicate are wrong.
* predicate error: the predicate in the summary statement is inconsistent with the transcript.
* circumstantial error: the additional information (like location or time) specifying the circumstance around a predicate is wrong.
* grammatical error: the grammar of the sentence is so wrong that it becomes meaningless.
* coreference error: a pronoun or reference with wrong or nonexisting antecedent.
* linking error: error in how multiple statements are linkedtogether in the discourse (for example temporal ordering or causal link).
* other error: the statement contains any factuality error which is not defined here.

Instruction:
First, compare each summary sentence with the transcript.
Second, provide a single sentence explaining which factuality error the sentence has.
Third, answer the classified error category for each sentence in the summary.
---
Transcript:

{input_text}
---

Here is the summary text:

{summary}

---
Provide your answer in JSON format. The answer should be a list of dictionaries whose keys are "sentence", "reason", and "category":

{{"Eval":
[{{"sentence": "<Sentence Index>", "reason": "<your reason>", "category": "no error"}}, 
{{"sentence": "<Sentence Index>", "reason": "<your reason>", "category": "out-of-context error"}}, 
{{"sentence":"<Sentence Index>", "reason": "<your reason>", "category": "entityerror"}}]
}}
{format_instructions}
"""
    
class SummarizeEvalLLM:
    def __init__(self, model_name):
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    ## Completeness, Conciseness
    def eval_summary(self, summary, keyfact):
        
        response_schemas = [
            ResponseSchema(
                name="Eval",
                description="The answer to the user's question.",
            )
        ]
        
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()

        prompt = ChatPromptTemplate.from_template(
            eval_prompt,
            partial_variables={"format_instructions": format_instructions},
        )

        chain = prompt | self.llm | output_parser

        chain = chain.with_config(
            {
                "run_name": inspect.currentframe().f_code.co_name,
                "tags": [self.llm.model_name],
            }
        )

        result = chain.invoke({"summary": summary, "keyfact" : keyfact})

        return result
    
    ## Faithfulness
    def eval_fact_summary(self, summary, input_text):
        
        response_schemas = [
            ResponseSchema(
                name="Eval",
                description="The answer to the user's question.",
            )
        ]
        
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()

        prompt = ChatPromptTemplate.from_template(
            eval_factual_prompt,
            partial_variables={"format_instructions": format_instructions},
        )

        chain = prompt | self.llm | output_parser

        chain = chain.with_config(
            {
                "run_name": inspect.currentframe().f_code.co_name,
                "tags": [self.llm.model_name],
            }
        )

        result = chain.invoke({"summary": summary, "input_text" : input_text})

        return result
    

eval_summarizer = SummarizeEvalLLM(model_name="gpt-4o")

# 주문 텍스트 추출 함수
def extract_sections(text):
    order_match = re.search(r'(【주문】.*)', text, re.DOTALL)
    order_text = order_match.group(1).strip() if order_match else "No Order Text Found"

    return order_text

summary_folder = "fact_flow"
keyfact_folder = "keyfact"
result_folder = "eval_fact_results"  # 결과 저장 폴더
input_text_folder = "법령조문별판례목록_txt" #원본 파일【주문】

os.makedirs(result_folder, exist_ok=True)

keyfact_files = [f for f in os.listdir(keyfact_folder) if f.endswith("_keyfacts.json")]

for keyfact_file in keyfact_files:
    keyfact_path = os.path.join(keyfact_folder, keyfact_file)
    
    precedent_name = keyfact_file.split("_keyfact")[0]

    summary_file = f"{precedent_name}_output.json"
    summary_path = os.path.join(summary_folder, summary_file)

    input_text_file = f"{precedent_name}.txt"
    input_text_path = os.path.join(input_text_folder, input_text_file)

    if not os.path.exists(summary_path):
        print(f"Summary file not found for keyfact file: {keyfact_file}")
        continue

    if not os.path.exists(input_text_path):
        print(f"Input Text file not found for keyfact file: {keyfact_file}")
        continue

    with open(keyfact_path, "r") as f:
        keyfact = json.load(f)
    
    with open(summary_path, "r") as f:
        summary = json.load(f)


    with open(input_text_path, "r") as f:
        input_text = f.read()

    result_Completeness_Conciseness = eval_summarizer.eval_summary(summary, keyfact)
    result_Faithfulness = eval_summarizer.eval_fact_summary(summary, input_text)

    result_file = os.path.join(result_folder, f"{precedent_name}_result.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump({"keyfact_file": keyfact_file,"summary": summary, "keyfact": keyfact,
                    "result_Completeness_Conciseness": result_Completeness_Conciseness,
                    "result_Faithfulness": result_Faithfulness}, f, ensure_ascii=False, indent=4)

    print(f"Generated Response for {precedent_name}: {result_Completeness_Conciseness}")
    print(f"Generated Response for {precedent_name}: {result_Faithfulness}")
