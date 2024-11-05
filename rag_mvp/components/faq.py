import streamlit as st


def faq():
    st.markdown(
        """
# FAQ
## Как работает умный поиск?
Умный поиск разбивает исходный текст на небольшие фрагменты и при помощи 
специальной модели ИИ ищет наиболее подходящие фрагменты для ответа на вопрос.
Затем генеративная модель ИИ генерирует ответ на основе этих фрагментов.

## В безопасности ли мои данные?
Да, ваша конфиденциальность очень важна для нас. 
Мы храним документы только на время текущей сессии и не используем их для других целей, кроме поиска ответов на ваши вопросы.
Все загруженные документы удаляются после закрытия вкладки браузера.

## Почему индексация занимает так много времени?
Индексация документов - это процесс, при котором AimateDocs обрабатывает весь текст документа,
для быстрого поиска. Если в ваших документах очень много текстовой информации,
то индексация может занять некоторое время. Но не волнуйтесь, это происходит только один раз при загрузке документов.

## What do the numbers mean under each source?
For a PDF document, you will see a citation number like this: 3-12. 
The first number is the page number and the second number is 
the chunk number on that page. For DOCS and TXT documents, 
the first number is set to 1 and the second number is the chunk number.

## Are the answers 100% accurate?
No, the answers are not 100% accurate. KnowledgeGPT uses GPT-3 to generate
answers. GPT-3 is a powerful language model, but it sometimes makes mistakes 
and is prone to hallucinations. Also, KnowledgeGPT uses semantic search
to find the most relevant chunks and does not see the entire document,
which means that it may not be able to find all the relevant information and
may not be able to answer all questions (especially summary-type questions
or questions that require a lot of context from the document).

But for most use cases, KnowledgeGPT is very accurate and can answer
most questions. Always check with the sources to make sure that the answers
are correct.
"""
    )
