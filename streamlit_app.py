import numpy as np
import streamlit as st
import tiktoken
import openai 
import logging
from redis import Redis
from redis.commands.search.query import Query
import os
from datetime import date

# Define a function to get the session id and the remote ip 
# Caution: this function is implemented in a hacky way and may break in the future
from streamlit import runtime
from streamlit.runtime.scriptrunner import get_script_run_ctx

GPT_MODEL = "gpt-3.5-turbo-16k"
VECTOR_DIM = 1536 
DISTANCE_METRIC = "COSINE"  
INDEX_NAME = "IsraelHamasNewsOnline"

def get_session_info():
    """Get the session id and the remote ip."""
    try:
        ctx = get_script_run_ctx()
        if ctx is None:
            return {'session_id': 'unknown', 'remote_ip': 'unknown'}

        session_info = runtime.get_instance().get_client(ctx.session_id)
        if session_info is None:
            return {'session_id': 'unknown', 'remote_ip': 'unknown'}
    except Exception as e:
        return {'session_id': 'unknown', 'remote_ip': 'unknown'}
    return {'session_id': ctx.session_id, 'remote_ip': session_info.request.remote_ip}

# Define a custom logger adapter to add session info to log messages
class SessionInfoAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return f"{self.extra['remote_ip']} - {self.extra['session_id']} - {msg}", kwargs

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
session_info = get_session_info()
logger = SessionInfoAdapter(logging.getLogger(), session_info)

# The streamlit script starts here
logger.info("Starting Streamlit script ...")
st.set_page_config(
    page_title="💥一个关注巴以局势的AI",
    page_icon="",
)
# st.subheader("💥巴以局势动态")
st.write("""
    我是一个关注巴以局势的AI，我的信息来源是**微软Bing**新闻搜索。
    我会每三小时根据互联网上的消息分析巴以冲突的最新形势，欢迎向我提问或和我讨论。
    
    我还在学习中，如果你觉得我的回答有任何错误或不妥，请联系我的主人：mingyu.li.cn@gmail.com
""")
    
# Prepare to connect to Redis
redis_host = os.getenv('REDIS_HOST', 'localhost')  # default to 'localhost' if not set
redis_port = os.getenv('REDIS_PORT', '6379')  # default to '6379' if not set
redis_db = os.getenv('REDIS_DB', '0')  # default to '1' if not set. RediSearch only operates on the default (0) db
 # Instantiates a Redis client. decode_responses=False to avoid decoding the returned embedding vectors
r = Redis(host=redis_host, port=redis_port, db=redis_db, decode_responses=False)

if "messages" not in st.session_state.keys():
    # Initialize the session_state.messages
    st.session_state.messages = [{"role": "system", "content": "You are an AI assistant with knowledge about Israel-Palestine situation."}]
else: # Since streamlit script is executed every time a widget is changed, this "else" is not necessary, but improves readability
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user" or message["role"] == "assistant":
            with st.chat_message(message["role"]):
                st.write(message["content"])

# Prepare the api_key to call OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.error("KEY is not set.")
    st.error('KEY未设置，请联系我的主人。')
    st.stop() 
else:
    openai.api_key = openai_api_key

# User-provided prompt
if user_prompt := st.chat_input('在此输入您的问题'):
    logger.info(f"User's prompt: {user_prompt}")
    with st.chat_message("user"):
        st.write(user_prompt)
        st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("assistant"):
        with st.spinner("人工智能正在思考..."):
            QUERY_GEN_PROMPT = f"""
Generate a brief query based on the chat history. This query will be used to search the answer to the user's question.
Today is {date.today().strftime("%A, %B %d, %Y")}. You can decide whether to include the date in the query.
"""
            st.session_state.messages.append({"role": "system", "content": QUERY_GEN_PROMPT})
            try:
                r.incr("IsraelHamasNewsOnline:n_asked") # Count how many questions are asked
                gpt_response = openai.ChatCompletion.create(
                        model=GPT_MODEL,
                        messages=st.session_state.messages,
                        temperature=0.5,
                        max_tokens=100
                    )
            except Exception as e:
                logger.error(f"Error generating response from OpenAI: {e}")
                st.error('AI没有响应，可能是因为我们正在经历访问高峰，请稍后刷新页面重试。如果问题仍然存在，请联系我的主人。')
                st.stop() 
            generated_query = gpt_response.choices[0].message.content
            logger.info(f"Generated query: {generated_query}")
            logger.info(f"Token usage: {gpt_response.usage.prompt_tokens} -> {gpt_response.usage.completion_tokens}")
            try:
                query_embedding = openai.Embedding.create(input=generated_query, model="text-embedding-ada-002")["data"][0]["embedding"]
                query_vec = np.array(query_embedding).astype(np.float32).tobytes()
                # Prepare the query
                query_base = (Query("*=>[KNN 20 @embedding $vec as score]").sort_by("timeStamp", asc=False).paging(0, 20).return_fields("score", "url", "datePublished", "description").dialect(2))
                query_param = {"vec": query_vec}
                query_results = r.ft(INDEX_NAME).search(query_base, query_param).docs
                formatted_result = ""
                for query_result in query_results:
                    formatted_result += f"URL: {query_result['url']}\nDate: {query_result['datePublished']}\nContent: {query_result['description']}\n\n"
            except Exception as e:
                logger.error(f"Error querying Reids with embedding: {e}")
                st.error("无法搜索答案，这很可能是系统故障导致，请联系我的主人。")
                st.stop()
            logger.info(f"Search results: ({query_results[0].score}){query_results[0].description[:20]}".replace("\n", "") + 
                        "..." + f"({query_results[1].score}){query_results[1].description[:20]}".replace("\n", "") + "...")
            st.session_state.messages[-1] = {"role": "system", 
                "content": f"""
    Write an answer in Chinese to the user's question based on the given search results. 
    SEARCH_RESULTS: {formatted_result}

    Today is {date.today().strftime("%A, %B %d, %Y")}. You can use this date to filter the search results according to the user's question.
    Include as much information as possible in the answer. List the reference search result URLs at the end of your answer.
    """}
            try:
                gpt_response = openai.ChatCompletion.create(
                        model=GPT_MODEL,
                        messages=st.session_state.messages,
                        temperature=0.5,
                        stream=True)
                resp_display = st.empty()
                collected_resp_content = ""
                for chunk in gpt_response:
                    if not chunk['choices'][0]['finish_reason']:
                        collected_resp_content += chunk['choices'][0]['delta']['content']
                        resp_display.write(collected_resp_content)
                # Count how many answeres are generated
                r.incr("n_answered")
            except Exception as e:
                logger.error(f"Error generating response from OpenAI: {e}")
                st.error('AI没有响应，可能是因为我们正在经历访问高峰，请稍后刷新页面重试。如果问题仍然存在，请联系我的主人。')
                st.stop() 
            logger.info(f"AI's response: {collected_resp_content[:50]}".replace("\n", "") + "..." + f"{collected_resp_content[-50:]}".replace("\n", ""))
    # Count the number of tokens used
    tokenizer = tiktoken.encoding_for_model(GPT_MODEL)
    # Because of ChatGPT's message format, there are 4 extra tokens for each message
    # Ref: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    n_token_input = 0
    for message in st.session_state.messages:
        n_token_input += len(tokenizer.encode(message["content"])) + 4 
    n_token_input += 3 # every reply is primed with <|start|>assistant<|message|>
    n_token_output = len(tokenizer.encode(collected_resp_content))
    logger.info(f"Token usage: {n_token_input} -> {n_token_output}")

    # Add the generated msg to session state
    st.session_state.messages[-1] = {"role": "assistant", "content": collected_resp_content}

st.write("**注意：人工智能生成内容仅供参考。**")
logger.info("Streamlit script ended.")