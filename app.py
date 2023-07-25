import os
import openai
import nltk
import spacy
import textstat
import language_tool_python
import numpy as np
import requests
import pprint
import uuid
import json
import plotly.express as px
import streamlit_authenticator as stauth
import jdk

import pandas as pd

import streamlit as st

from tqdm import tqdm

from lexical_diversity import lex_div as ld
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer


from dotenv import find_dotenv, load_dotenv
import yaml
from yaml.loader import SafeLoader

with open('config.yml') as file:
    config = yaml.load(file, Loader=SafeLoader)
    
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

name, authentication_status, username = authenticator.login('Login', 'main')

load_dotenv(find_dotenv())

nltk.download('punkt')
nltk.downloader.download('vader_lexicon')

API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_TOKEN')}"}

openai.api_key = os.getenv('OPENAI_API_KEY')

pp = pprint.PrettyPrinter(indent=4)


@st.cache_resource
def init():
    spacy.cli.download("en_core_web_sm")


@st.cache_data
def load_skill_description():
    with open("skill_explanation.md", 'r') as f:
        data = f.read()
    return data


def radar_chart(scores):
    df = pd.Series(scores).to_frame().reset_index().rename(columns={
        "index": "skill",
        0: "score"
    })
    fig = px.line_polar(df, r="score", theta="skill", line_close=True, range_r=[0, 5])
    st.write(fig)
    st.dataframe(df)
    
def feature_extraction(scoring):
    df = pd.Series(scoring).to_frame().reset_index().rename(columns={
        "index": "feature",
        0: "score"
    })
    df["score"] = np.round(df["score"])
    st.dataframe(df)


def calculate_word_uniqueness(prompts):
    print("Calculating word uniqueness...")
    text = ' '.join(prompts)

    # Tokenize the text into individual words
    words = word_tokenize(text)
    
    # Remove punctuation and convert to lower case
    words = [word.lower() for word in words if word.isalpha()]
    
    # Calculate the number of unique words and the total number of words
    unique_words = len(set(words))
    total_words = len(words)
    
    # Calculate and return the ratio of unique words to total words
    return (unique_words / total_words) * 100


def calculate_word_diversity(prompts):
    print("Calculating word diversity...")
    text = ' '.join(prompts)
    
    # Tokenize the text into individual words
    words = word_tokenize(text)
    
    # Remove punctuation and convert to lower case
    words = [word.lower() for word in words if word.isalpha()]
    
    # Calculate and return MTLD
    return ld.mtld(words)


def calculate_readibility_score(prompts):
    print("Calculating readibility score...")
    text = ' '.join(prompts)
    return 100 - textstat.flesch_reading_ease(text)


def is_question(sentence):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)
    
    # Check if the sentence contains a typical question word.
    if any(token.text.lower() in ["who", "what", "where", "when", "why", "how", "is", "are", "can", "does", "do"] for token in doc):
        return True
    
    # Check if the sentence has a structure typical of questions (e.g., "Verb + Noun/Pronoun + ...").
    if len(doc) > 1 and "VERB" in [token.pos_ for token in doc] and "NOUN" in [token.pos_ for token in doc]:
        return True

    # If the sentence ends with a question mark, it's a question.
    if sentence.strip().endswith('?'):
        return True

    return False


def get_proportion_of_questions(prompts):
    print("Calculating proportions of questions...")
    is_questions = [is_question(sentence) for sentence in prompts]
    return sum(is_questions) / len(prompts) * 100


def grammar_check(prompts):
    text = ' '.join(prompts)
    tool = language_tool_python.LanguageTool('en-US')
    errors = tool.check(text)
    return (1 - len(errors) / len(text)) * 100


def get_average_sentence_length(prompts):
    text = ' '.join(prompts)
    sentences = sent_tokenize(text)
    avg_sent_length = sum(len(s.split()) for s in sentences) / len(sentences)
    return avg_sent_length / 12 * 100


def sentiment_calculator(prompts, attribute='compound', agg='mean', multiplier=100):
    print("Calculating sentiment: attribute = {}, aggregation = {}...".format(attribute, agg))
    sia = SentimentIntensityAnalyzer()
    sent = []
    for prompt in prompts:
        sentiment = sia.polarity_scores(prompt)
        sent.append(sentiment[attribute])
        
    sent = np.array(sent)
    if agg == 'mean':
        return np.mean(sent) * multiplier
    elif agg == 'var':
        return np.var(sent) * multiplier
    elif agg == 'max':
        return np.max(sent) * multiplier
    elif agg == 'min':
        return np.min(sent) * multiplier


def calculate_sentence_similarity(sentence1, sentence2):
    payload = {
        "inputs": {
            "source_sentence": sentence1,
            "sentences": [sentence2]
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    print(response)
    return response.json()[0]


def engagement_analysis(conversation):
    print("Calculating engagement score...")
    engagement_pts_list = []
    for i in tqdm(range(2, len(conversation), 2)):
        user_prompt = conversation[i]["content"]
        assistant_prompt = conversation[i-1]["content"]
        engagement_pts = calculate_sentence_similarity(user_prompt, assistant_prompt)
        engagement_pts_list.append(engagement_pts)
    return np.mean(np.array(engagement_pts_list)) * 100


def calculate_personal_connection_score(prompts):
    text = ' '.join(prompts)
    
    # Tokenize the text into words
    words = nltk.word_tokenize(text.lower())
    sentences = sent_tokenize(text)
    
    # Count the occurrences of personal pronouns
    personal_pronouns = ["i", "my", "me", "mine", "myself"]
    personal_pronoun_count = sum(1 for word in words if word in personal_pronouns)
    
    # Calculate the personal connection proportion
    personal_connection_proportion = personal_pronoun_count / len(sentences)
    
    return personal_connection_proportion * 200


def load_chat_history(session_id):
    print(f"chat_history_{session_id}.json")
    try:
        with open(f"chat_history_{session_id}.json", "r") as file:
            chat_history = json.load(file)
    except FileNotFoundError:
        print("File not found")
        return []
    return chat_history


def save_chat_history(session_id, chat_history):
    with open(f"chat_history_{session_id}.json", "w") as file:
        json.dump(chat_history, file)
        
        
def get_openai_response(message):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are Elonie acting as Elon Musk who is talking to a 8-year-old child"}] + message[-10:],
        temperature=0.5,
        top_p=1,
    )
    return {"role": "assistant", "content": response["choices"][0]["message"]["content"]}

def clear_text():
    st.session_state["text"] = st.session_state["prompt"]
    st.session_state["prompt"] = ""


if authentication_status:
    init()
    authenticator.logout('Logout', 'main')
    
    st.title("Testing the Progress Tracking method - Version 0.1.0")
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = uuid.uuid4()
    if "text" not in st.session_state:
        st.session_state["text"] = ""

    session_id = st.session_state.get("session_id")

    chat_history = []
    with st.form("myform"):
        user_input = st.text_input("Input your chat:", key="prompt")
        send = st.form_submit_button(label="Send", on_click=clear_text)
        
    if st.session_state["text"]:
        chat_history = load_chat_history(session_id)
        print(chat_history)

        user_obj = {"role": "user", "content": st.session_state["text"]}
        chat_history.append(user_obj)
        
        assistant_obj = get_openai_response(chat_history)
        chat_history.append(assistant_obj)
        
        save_chat_history(session_id, chat_history)

    if st.button("End Conversation"):
        conversation = chat_history
        prompts = [obj["content"] for obj in conversation if obj["role"] == "user"]

        progress_text = "Calculating progress tracking score. Please wait..."
        bar = st.progress(0.0, text=progress_text)
        steps = 10
        scoring = {}
        
        bar.progress(1/steps, text="Calculating word uniqueness...")
        scoring["word_uniqueness"] = calculate_word_uniqueness(prompts)
        
        bar.progress(2/steps, text="Calculating lexical diversity...")
        scoring["lexical_diversity"] = calculate_word_diversity(prompts)
        
        bar.progress(3/steps, text="Calculating readibility score...")
        scoring["readibility_difficulty"] = max(calculate_readibility_score(prompts), 0)
        
        bar.progress(4/steps, text="Calculating the amount of questions...")
        scoring["proportion_of_questions"] = get_proportion_of_questions(prompts)
        
        bar.progress(5/steps, text="Calculating proportion of grammar errors...")
        scoring["proportion_of_grammar_error"] = grammar_check(prompts)
        
        bar.progress(6/steps, text="Calculating average sentence length...")
        scoring["average_sentence_length_score"] = min(get_average_sentence_length(prompts), 100)
        
        bar.progress(7/steps, text="Analyzing sentiment...")
        scoring["compound_sentiment_var"] = min(sentiment_calculator(prompts, attribute='compound', agg='var', multiplier=1000), 100)
        scoring["pos_sentiment_avg"] = min(sentiment_calculator(prompts, attribute="pos", agg='mean', multiplier=150), 100)
        
        bar.progress(8/steps, text="Analyzing engagement...")
        scoring["engagement_analysis"] = engagement_analysis(conversation)
        
        bar.progress(9/steps, text="Analyzing personal connection...")
        scoring["personal_connection"] = min(calculate_personal_connection_score(prompts), 100)
        
        # scoring = {
        #     "word_uniqueness": calculate_word_uniqueness(prompts),
        #     "lexical_diversity": calculate_word_diversity(prompts),
        #     "readibility_difficulty": calculate_readibility_score(prompts),
        #     "proportion_of_questions": get_proportion_of_questions(prompts),
        #     "proportion_of_grammar_error": grammar_check(prompts),
        #     "average_sentence_length_score": min(get_average_sentence_length(prompts), 100),
        #     "compound_sentiment_var": min(sentiment_calculator(prompts, attribute='compound', agg='var', multiplier=1000), 100),
        #     "pos_sentiment_avg": min(sentiment_calculator(prompts, attribute="pos", agg='mean', multiplier=150), 100),
        #     # "neg_sentiment_avg": sentiment_calculator(prompts, attribute="neg", agg='mean'),
        #     "engagement_analysis": engagement_analysis(conversation),
        #     "personal_connection": min(calculate_personal_connection_score(prompts), 100)
        # }
        # st.write("Evaluation")
        # st.json(scoring)
        
        bar.progress(10/steps, text="Generating radar...")
        skill_evaluation = {
            "creative_thinking": np.round((scoring["word_uniqueness"] + scoring["lexical_diversity"] + scoring["readibility_difficulty"]) / 300 * 5, 2),
            "problem_solving": np.round((scoring["proportion_of_questions"]) / 100 * 5, 2),
            "communication": np.round((scoring["proportion_of_grammar_error"] + scoring["pos_sentiment_avg"] + scoring["average_sentence_length_score"] + min(130 - scoring["readibility_difficulty"], 100)) / 400 * 5, 2),
            "emotional_awareness": np.round((scoring["compound_sentiment_var"] + scoring["average_sentence_length_score"] + scoring["personal_connection"]) / 300 * 5, 2),
            "critical_thinking": np.round((scoring["lexical_diversity"] + scoring["engagement_analysis"]) / 200 * 5, 2)
        }
        st.write("**Skill scoring details**")
        radar_chart(skill_evaluation)
        
        st.write("**Feature details (min 0, max 100)**")
        feature_extraction(scoring)
        
        # Change the session_id
        st.session_state["session_id"] = uuid.uuid4()
        
    st.subheader("Conversation:")
    if len(chat_history) == 0:
        st.markdown("*Conversation is empty*")
    for prompt in chat_history:
        st.markdown(f"**{prompt['role'].capitalize()}**: {prompt['content']}")

    skill_description = load_skill_description()
    with st.expander("About the app"):
        st.markdown(skill_description)

    # st.json(chat_history)

    # conversation = [
    #     {"role": "user", "content": "Hey Elonie, do you know about soccer?"},
    #     {"role": "assistant", "content": "Yes, I do. Soccer is a popular sport played between two teams of eleven players with a spherical ball."},
    #     {"role": "user", "content": "Cool! Who's the best soccer player?"},
    #     {"role": "assistant", "content": "There have been many great soccer players over the years, but a few that are often mentioned are Lionel Messi, Cristiano Ronaldo, and Pelé."},
    #     {"role": "user", "content": "I like Messi!"},
    #     {"role": "assistant", "content": "Lionel Messi is indeed an incredible player. He's known for his skill, precision, and creativity on the field."},
    #     {"role": "user", "content": "I want to play like him one day."},
    #     {"role": "assistant", "content": "That's a great aspiration! With hard work, practice, and dedication, you could certainly improve your soccer skills."}
    # ]
    # prompts = [obj["content"] for obj in conversation if obj["role"] == "user"]

    # scoring = {
    #     "word_uniqueness": calculate_word_uniqueness(prompts),
    #     "lexical_diversity": calculate_word_diversity(prompts),
    #     "readibility_difficulty": calculate_readibility_score(prompts),
    #     "proportion_of_questions": get_proportion_of_questions(prompts),
    #     "proportion_of_grammar_error": grammar_check(prompts),
    #     "average_sentence_length_score": min(get_average_sentence_length(prompts), 100),
    #     "compound_sentiment_var": min(sentiment_calculator(prompts, attribute='compound', agg='var', multiplier=1000), 100),
    #     "pos_sentiment_avg": min(sentiment_calculator(prompts, attribute="pos", agg='mean', multiplier=150), 100),
    #     # "neg_sentiment_avg": sentiment_calculator(prompts, attribute="neg", agg='mean'),
    #     "engagement_analysis": engagement_analysis(conversation),
    #     "personal_connection": min(calculate_personal_connection_score(prompts), 100)
    # }
    # print("Evaluation")
    # pp.pprint(scoring)


    # skill_evaluation = {
    #     "creative_thinking": np.round((scoring["word_uniqueness"] + scoring["lexical_diversity"] + scoring["readibility_difficulty"]) / 300 * 5, 2),
    #     "problem_solving": np.round((scoring["proportion_of_questions"]) / 100 * 5, 2),
    #     "communication": np.round((scoring["proportion_of_grammar_error"] + scoring["pos_sentiment_avg"] + scoring["average_sentence_length_score"] + min(130 - scoring["readibility_difficulty"], 100)) / 400 * 5, 2),
    #     "emotional_awareness": np.round((scoring["compound_sentiment_var"] + scoring["average_sentence_length_score"] + scoring["personal_connection"]) / 300 * 5, 2),
    #     "critical_thinking": np.round((scoring["lexical_diversity"] + scoring["engagement_analysis"]) / 200 * 5, 2)
    # }
    # print("Skill inferred")
    # pp.pprint(skill_evaluation)
    
elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')
 

