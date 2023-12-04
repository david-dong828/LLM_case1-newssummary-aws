# Name: Dong Han
# Student ID: 202111878
# Mail: dongh@mun.ca
import os

# os.environ['HTTP_PROXY'] = 'http://localhost:7890'
# os.environ['HTTPS_PROXY'] = 'http://localhost:7890'
import random

os.environ['OPENAI_API_KEY'] = 'xxxxxx'

os.environ['SERPAPI_API_KEY'] = 'xxxxx'
os.environ['ACTIVELOOP_TOKEN'] = 'xxxx'

os.environ['HUGGINGFACEHUB_API_TOKEN'] = 'xxxxx'

import yt_dlp
import whisper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain import OpenAI,LLMChain,PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
import textwrap

nOfVideo=0

def download_youtube_video(url,filename):
    # Set the options for the download
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
        'outtmpl': filename,
        'quiet': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(url,download=True)
        title = result.get('title','')
        author = result.get('uploader','')

def extrac_words_n_save2file(vFileName,tFileName):
    model = whisper.load_model('base')
    result = model.transcribe(vFileName)
    text = result['text']
    # print('Below are Video texts: ')
    # print()
    # print(text)

    # with open(tFileName,'w') as f:
    #     f.write(text)

    return text

def text_summary_directly(text):
    llm = OpenAI(temperature=0)
    chain = load_summarize_chain(llm=llm,chain_type='map_reduce')

    output_summary = chain.run(text)
    wrapped_text = textwrap.fill(output_summary,width=100)
    print('The direct summary from the video text: ')
    print(wrapped_text)
    print('..........')

    # print(chain.llm_chain.prompt.template) # it can be used to summarize the text as well
    # and the output be like: Write a concise summary of the following:\n\n\n"{text}"\n\n\n CONCISE SUMMARY:


def split_doc(docName):

    text_splitter = RecursiveCharacterTextSplitter(chunk_size= 1000, chunk_overlap=0,separators=['\\n\\n','\\n',',',', '])
    with open(docName,'r') as f:
        text = f.read()

    texts = text_splitter.split_text(text)
    docs = [Document(page_content=t) for t in texts[:4]]

    return docs


def text_summary_w_prompt(docs):
    llm = OpenAI(temperature=0)
    prompt_template = '''
    Write a concise bullet point summary of the following:

    {text}

    CONSISE SUMMARY IN BULLET POINTS:
    '''

    Bullet_prompt = PromptTemplate(input_variables=['text'], template=prompt_template)

    chain = load_summarize_chain(llm=llm, chain_type='stuff', prompt=Bullet_prompt)

    output_summary = chain.run(docs)

    wrapped_text = textwrap.fill(output_summary,width=1000,break_long_words=False,replace_whitespace=False)

    print('Summary from prompt and stuff chain in bullet: ')
    print(wrapped_text)
    print('..........................')

    ####################refine chain below### basically more accurate#######
    chain = load_summarize_chain(llm=llm, chain_type='refine')

    output_summary = chain.run(docs)

    wrapped_text = textwrap.fill(output_summary, width=1000, break_long_words=False, replace_whitespace=False)
    print('Summary from prompt and stuff chain in bullet: ')
    print(wrapped_text)
    print('..........................')

def upload_database(docs):
    my_activeloop_org_id = 'daviddong828'
    my_activeloop_dataset_name = "langchain_course_youtube_summarizer"
    dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"

    embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')
    db = DeepLake(dataset_path=dataset_path,embedding_function=embeddings)

    db.add_documents(docs)

    return db

def database_retriver(db):
    retriever = db.as_retriever()
    retriever.search_kwargs['distance_metric'] = 'cos'
    retriever.search_kwargs['k'] = 4

    return retriever

def prompt_preparation():
    prompt_template = '''
    Use the following pieces of transcripts from a video to answer the question in bullet points and summarized. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}
    
    Question: {question}
    Summarized answer in bullter points:
    '''

    prompt = PromptTemplate(input_variables = ['question','context'],template=prompt_template)

    return prompt

def answer(prompt,retriever):
    from langchain.chains import RetrievalQA

    chain_type_kwargs = {'prompt':prompt}

    llm = OpenAI(temperature=0)

    qa = RetrievalQA.from_chain_type(llm=llm,chain_type='stuff',chain_type_kwargs=chain_type_kwargs,retriever=retriever)
    question = 'summary what is the main point?'
    print(question)

    res = qa.run(question)
    print(res)

def youtube_extract_words(url):
    vfileName = str(random.randint(0,99999))+'.mp4'
    download_youtube_video(url,vfileName)

    tfileNme = vfileName[:-4]+'.text'
    text = extrac_words_n_save2file(vfileName, tfileNme)

    return text

def main():
    # ============================download youtube video==================
    # url = 'https://www.youtube.com/watch?v=mBjPyte2ZZo'
    # vfilename = 'Introduction to large language models.mp4'
    # tfilename = 'test1.txt'
    # download_youtube_video(url, vfilename)

    #============================vedio words extract Test==================
    # use a downloaded video for the below test, save to 'tfilename' file
    vfilename = 'C:\\Users\\David\\Downloads\\zhongyao-baba.mp4'
    tfilename = 'test1.txt'
    text = extrac_words_n_save2file(vfilename, tfilename)

    # =======================Directly SUMMARY the extracted words n PRINT============
    # Using OpenAI api
    text_summary_directly(text)

    #=======================Split n upload to database==============================

    docs = split_doc(tfilename)
    # text_summary_w_prompt(docs)

    db = upload_database(docs)
    retriever = database_retriver(db)

    # =======================Summary from DATABASE==============================
    promt = prompt_preparation()
    answer(promt,retriever)



if __name__ == '__main__':
    main()