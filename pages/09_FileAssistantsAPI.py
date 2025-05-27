# import openai as client

import streamlit as st

st.set_page_config(
    page_title="FileAssistantsAPI",
    page_icon="📁",
)

st.markdown(
    """
# FileAssistantsAPI  

**To be continued...**

OpenAI Assistants API implementation for file handling.
"""
)

#
# TODO: code challenge
# TODO: Modify get_message func to show source and annotations of the answers
# change "【12:1†preface.txt】" to actual text
# use start_index and end_index to highlith the text
# TODO: Then implement to streamlit

# * function for Investor assistants api
# def create_assistant():
#     return client.beta.assistants.create(
#         name="Investor Assistant",
#         instructions="You help users do research on publicly traded companies and you help users decide if they should buy the stock or not.",
#         model="gpt-4.1-nano",
#         tools=functions,  # type: ignore
#     )


# * function for Book assistants api
# def create_assistant():
#     assistant = client.beta.assistants.create(
#         name="Book Assistant",
#         instructions="You help users with their question on the files they upload",
#         model="gpt-4.1-nano",
#         tools=[{"type": "file_search"}],  # type: ignore
#     )
#     print(assistant)
#     return assistant


# def create_thread(content):
#     thread = client.beta.threads.create(
#         messages=[
#             {
#                 "role": "user",  # type: ignore
#                 "content": content,
#             }
#         ],
#     )
#     print(thread)
#     return thread


# def create_file(file_path):
#     created_file = client.files.create(
#         file=client.file_from_path(file_path),
#         purpose="assistants",
#     )
#     print(created_file)
#     return created_file


# def create_file_thread(content, file_id):
#     file_thread = client.beta.threads.create(
#         messages=[
#             {
#                 "role": "user",  # type: ignore
#                 "content": content,
#                 "attachments": [
#                     {"file_id": file_id, "tools": [{"type": "file_search"}]}
#                 ],
#             },
#         ]
#     )
#     print(file_thread)
#     return file_thread


# def create_run(assistant_id, thread_id):
#     run = client.beta.threads.runs.create(
#         thread_id=thread_id,
#         assistant_id=assistant_id,
#     )
#     print(run)
#     return run

# # assistant = create_assistant()
# assistant_id = "asst_uvhM7JHUryJf6sWa8wPQXFcc"
# # thread_id = "thread_jCP9ahIO5uTLWWt5ao0wkXBL"


# # thread = create_thread("I want to know if the Tesla stock is a good buy")
# # thread_id = thread.id
# thread_id = "thread_tToVc2RWXlxMTscufoHN6pyI"

# # file = create_file("../files/preface.txt")
# file_id = "file-VxGVSfMj2qBATAodSm9Qcw"

# # create_file_thread("I want you to help me with this file", file_id)

# thread_id = "thread_Hxuonwg0t19GkooDNzwtf6NZ"

# # run = create_run(assistant_id, thread_id)

# def get_run(thread_id, run_id):
#     return client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)


# def send_message(thread_id, content):
#     return client.beta.threads.messages.create(
#         thread_id=thread_id,
#         role="user",
#         content=content,
#     )


# def send_file_message(thread_id, content, file_id):
#     return client.beta.threads.messages.create(
#         thread_id=thread_id,
#         role="user",
#         content=[
#             {"type": "text", "text": content},
#             {"type": "file", "file_id": file_id},
#         ],
#     )


# def get_messages(thread_id):
#     messages = client.beta.threads.messages.list(
#         thread_id=thread_id,
#     )
#     messages = list(messages)
#     # the oldest one first
#     messages.reverse()
#     for message in messages:
#         print("-----------------------------")
#         print(f"{message.role}: {message.content[0].text.value}")  # type: ignore
#         for annotation in message.content[0].text.annotations:
#             print(f"\nSource: {annotation.file_citation}")
#         # print("-----------------------------")
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

# send_message(
#     thread_id,
#     "Who is Jane Austen?",
# )
# #! after send message, u must excute <<run>>

# run = create_run(assistant_id, thread_id)

# print(f"thread_id: {thread_id}")
# get_messages(thread_id)

# # get_tool_outputs(thread_id, run.id)
# # submit_tool_outputs(thread_id, run.id)
