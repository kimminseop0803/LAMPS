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
from config import *
# Assuming you have imported your config file like this
# from config import OPENAI_API_KEY

# Set the API key as an environment variable
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

eval_prompt = """
Given the summary text and a list of key logic statements, please evaluate whether each key logic is represented in the summary. 

For each key logic, respond with:
1.'yes' if the key logic is present in the summary, and 'no' if it is not. The format should be [present: yes or no]
2.If 'yes', list the legal ground index where the key logic is included. The format should be [legal grounds: [<legal ground numbers if present, otherwise empty>]]

Therefore the overall respond should be like this:
"Eval":
"<keylogic index>" : "present": <yes or no>, "legal grounds": [<legal ground index if present, otherwise empty>]
...continue

Analyze each legal ground individually and provide specific responses. 
----
Here is the summary text:

{summary}
----
And here are the key logic statements to check against the summary:

{keylogic}

---

Respond in this JSON format:

{format_instructions}

"""

class SummarizeEvalLLM:
    def __init__(self, model_name):
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    def eval_summary(self, summary, keylogic):
        
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

        result = chain.invoke({"summary": summary, "keylogic" : keylogic})

        return result
    

eval_summarizer = SummarizeEvalLLM(model_name="gpt-4o")

summary_folder = "logic_flow_aggregation"
keylogic_folder = "법전원판례_keylogic"
result_folder = "eval_logic_results" 

os.makedirs(result_folder, exist_ok=True)

keylogic_files = [f for f in os.listdir(keylogic_folder) if f.endswith("_keylogic.json")]

for keylogic_file in keylogic_files:
    keylogic_path = os.path.join(keylogic_folder, keylogic_file)
    precedent_name = keylogic_file.split("_keylogic")[0]

    summary_file = f"{precedent_name}.json"
    summary_path = os.path.join(summary_folder, summary_file)

    if not os.path.exists(summary_path):
        print(f"Summary file not found for keylogic file: {keylogic_file}")
        continue

    with open(keylogic_path, "r") as f:
        keylogic = json.load(f)

    with open(summary_path, "r") as f:
        summary = json.load(f)

    # 함수 실행
    result = eval_summarizer.eval_summary(summary, keylogic)
    
    # 결과 JSON 파일로 저장
    result_file = os.path.join(result_folder, f"{precedent_name}_result.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump({"keylogic_file": keylogic_file,"summary": summary, "keylogic": keylogic, "result": result}, f, ensure_ascii=False, indent=4)

    print(f"Generated Response for {keylogic_file}: {result}")
