# from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
# import yfinance
# import json
# import openai as client

import streamlit as st

st.set_page_config(
    page_title="AssistantsAPI",
    page_icon="🔧",
)

st.markdown(
    """
# AssistantsAPI  

** To be continued... **
"""
)

# # send message -> get_messages -> run -> get_runt().status
# # TODO: code challenge: implement this notebook to streamlit
# # TODO: hint keep refresh run assistants api
# # TODO: pull runs over and over


# # * type of inputs is object
# def get_ticker(inputs):
#     ddg = DuckDuckGoSearchAPIWrapper(backend="api")
#     company_name = inputs["company_name"]
#     result = ddg.run(f"Ticker symbol of {company_name}")
#     return json.dumps({"ticker": result})


# def get_income_statement(inputs):
#     ticker = inputs["ticker"]
#     stock = yfinance.Ticker(ticker)
#     data = stock.income_stmt.to_json()
#     return json.dumps({"data": data})


# def get_balance_sheet(inputs):
#     ticker = inputs["ticker"]
#     stock = yfinance.Ticker(ticker)
#     data = stock.balance_sheet.to_json()
#     return json.dumps({"data": data})


# def get_daily_stock_performance(inputs):
#     ticker = inputs["ticker"]
#     stock = yfinance.Ticker(ticker)
#     data = stock.history(period="3mo").to_json()
#     return json.dumps({"data": data})


# # print(get_ticker({"company_name": "Apple"}))
# functions_map = {
#     "get_ticker": get_ticker,
#     "get_income_statement": get_income_statement,
#     "get_balance_sheet": get_balance_sheet,
#     "get_daily_stock_performance": get_daily_stock_performance,
# }


# functions = [
#     {
#         "type": "function",
#         "function": {
#             "name": "get_ticker",
#             "description": "Given the name of a company returns its ticker symbols",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "company_name": {
#                         "type": "string",
#                         "description": "The name of the company",
#                     }
#                 },
#                 "required": ["company_name"],
#             },
#         },
#     },
#     {
#         "type": "function",
#         "function": {
#             "name": "get_income_statement",
#             "description": "Given a ticker symbol (i.e AAPL) returns the company's income statement.",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "ticker": {
#                         "type": "string",
#                         "description": "Ticker symbol of the company",
#                     },
#                 },
#                 "required": ["ticker"],
#             },
#         },
#     },
#     {
#         "type": "function",
#         "function": {
#             "name": "get_balance_sheet",
#             "description": "Given a ticker symbol (i.e AAPL) returns the company's balance sheet.",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "ticker": {
#                         "type": "string",
#                         "description": "Ticker symbol of the company",
#                     },
#                 },
#                 "required": ["ticker"],
#             },
#         },
#     },
#     {
#         "type": "function",
#         "function": {
#             "name": "get_daily_stock_performance",
#             "description": "Given a ticker symbol (i.e AAPL) returns the performance of the stock for the last 100 days.",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "ticker": {
#                         "type": "string",
#                         "description": "Ticker symbol of the company",
#                     },
#                 },
#                 "required": ["ticker"],
#             },
#         },
#     },
# ]


# def create_assistant():
#     return client.beta.assistants.create(
#         name="Investor Assistant",
#         instructions="You help users do research on publicly traded companies and you help users decide if they should buy the stock or not.",
#         model="gpt-4.1-nano",
#         tools=functions,  # type: ignore
#     )


# # assistant = create_assistant()
# # print(assistant)
# assistant_id = "asst_E0GIp1cIf6nx68JeRYesxYtD"


# def create_thread(content):
#     return client.beta.threads.create(
#         messages=[
#             {
#                 "role": "role",  # type: ignore
#                 "content": content,
#             },
#         ]
#     )


# # thread = create_thread("put your content here")
# # thread_id = thread.id
# thread_id = "thread_jCP9ahIO5uTLWWt5ao0wkXBL"


# def create_run(assistant_id, thread_id):
#     return client.beta.threads.runs.create(
#         thread_id=thread_id,
#         assistant_id=assistant_id,
#     )


# run = create_run(assistant_id, thread_id)


# def get_run(thread_id, run_id):
#     return client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)


# def send_message(thread_id, content):
#     return client.beta.threads.messages.create(
#         thread_id=thread_id,
#         role="user",
#         content=content,
#     )


# def get_messages(thread_id):
#     messages = client.beta.threads.messages.list(
#         thread_id=thread_id,
#     )
#     messages = list(messages)
#     # the oldest one first
#     messages.reverse()
#     for message in messages:
#         print(f"{message.role}: {message.content[0].text.value}")  # type: ignore
#         # print(message)


# def get_tool_outputs(thread_id, run_id):
#     run = get_run(thread_id, run_id)
#     outputs = []
#     for action in run.required_action.submit_tool_outputs.tool_calls:  # type: ignore
#         # print(action)
#         action_id = action.id
#         function = action.function
#         print(f"Function calling {function.name} with arg {function.arguments}")
#         # function.arguments: '{"company_name":"Amazon"}' -> need to convert to json
#         outputs.append(
#             {
#                 "tool_call_id": action_id,
#                 "output": functions_map[function.name](json.loads(function.arguments)),
#             }
#         )
#         # {"tool_call_id": "call_8Tr9udsGdHMBhmyv17hJEAwL",
#         #  "output": output,}
#     return outputs


# def submit_tool_outputs(thread_id, run_id):
#     outputs = get_tool_outputs(thread_id, run_id)
#     return client.beta.threads.runs.submit_tool_outputs(
#         run_id=run_id,
#         thread_id=thread_id,
#         tool_outputs=outputs,
#     )


# # print(f"thread_id: {thread_id}")
# # get_messages(thread_id)

# # get_run(thread_id, run.id)

# # get_run(thread_id, run.id).status

# # send_message(thread_id, "yes")

# # get_tool_outputs(thread_id, run.id)

# # submit_tool_outputs(thread_id, run.id)
