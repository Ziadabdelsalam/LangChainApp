import os
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

os.environ['OPENAI_API_KEY'] = 'Your OPENAI API key can be found here: https://openai.com/blog/openai-api'
#app framework
st.title('GeideaGPT')
prompt = st.text_input('Plug in your prompt here')
#prompt templates
title_template = PromptTemplate(
    input_variables=['topic'],
    template='write me a small definition of {topic}'
)
script_template = PromptTemplate(
    input_variables=['title', 'wikipedia_research'],
    template='write me a detailed definition of {title} while leveraging this wikipedia researc: {wikipedia_research}'
)

#Memory
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')


#LLMs
llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm=llm,prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm,prompt=script_template, verbose=True,output_key='script', memory=script_memory)

wiki = WikipediaAPIWrapper()

#show stuff to the screen if there is a prompt
if prompt:
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    script = script_chain.run(title=title, wiki_research=wiki_research)

    st.write(title)
    st.write(script)
    
    with st.expander('Title History'):
        st.info(title_memory.buffer)
    with st.expander('Script History'):
        st.info(script_memory.buffer)
    with st.expander('Wiki History'):
        st.info(wiki_research)

