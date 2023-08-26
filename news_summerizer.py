# Name: Dong Han
# Student ID: 202111878
# Mail: dongh@mun.ca

import requests
from langchain.chat_models import ChatOpenAI
# from langchain.prompts.chat import (HumanMessagePromptTemplate,SystemMessagePromptTemplate,ChatPromptTemplate)
from newspaper import Article
from langchain.schema import HumanMessage

import os


os.environ['OPENAI_API_KEY'] = 'xxxxxxxxxxxxxxxxxxxxxxxxxx'

def scrape_news_article_and_parse(url):
    headers = {
        'User-Agent': 'Mozilla / 5.0(Windows NT 10.0; Win64; x64) AppleWebKit / 537.36(KHTML, like Gecko)'
                      'Chrome / 100.0.4896.127Safari / 537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    }

    try:
        content = requests.get(url, headers=headers)
        if content.status_code == 200:
            article =Article(url)
            article.download()
            article.parse()
            # print(f"articel title: {article.title}")
            # print(f'article text: {article.text}')
            return article.title,article.text
        else:
            print('link connection wrong')
            return -1,-1
    except requests.RequestException as e:
        print(e)
        return -2,-2

def prompt_preparation(article_title,article_text,language):
    template='''You are an expert of summarizing the online articles.
    Here is the article you need to summarize.
    =========
    Title:{article_title}
    {article_text}
    ==========
    write a summary of the previous article in a bulleted list format, in {language}.
    Also put the Title on the head.
    '''
    if article_title == -1:
        return -1
    if article_title == -2:
        return -2
    prompt = template.format(article_title=article_title,article_text=article_text,language=language)
    return prompt

def summarizer_news(prompt):
    if prompt.isdigit() and int(prompt) < 0:
        return -1 if prompt== 1 else -2
    messages = [HumanMessage(content=prompt)]
    chat = ChatOpenAI(temperature=0,model_name='gpt-3.5-turbo-16k')
    summary = chat(messages)
    return summary.content
    # print(summary.content)

def summarize_for_aws(url,lang):
    
    language = lang
    article_title, article_text = scrape_news_article_and_parse(url.strip())
    chat_prompt = prompt_preparation(article_title, article_text, language)
    final_summary = summarizer_news(chat_prompt)
    if final_summary.isdigit() and final_summary < 0:
        return -1 if final_summary == -1 else -2
    return final_summary

def main():
    # url='https://edition.cnn.com/2023/07/04/politics/american-political-divisions-july-fourth/index.html'
    url=input('type the url: ')
    language='Chinese'
    article_title, article_text = scrape_news_article_and_parse(url.strip())
    chat_prompt = prompt_preparation(article_title,article_text,language)
    print(summarizer_news(chat_prompt))




if __name__ == '__main__':
    main()
