from textblob import TextBlob
import pandas as pd
import streamlit as st
import cleantext
import nltk
import numpy as np
#import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string
import tensorflow as tf

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
import torch

#
import re

tokenizer=AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
st.header('Sentiment Analysis And Chatbot')
with st.expander('Analyze Text'):
    st.write("Enter review to analyse and get sentiment")
    text=st.text_input('Enter Text: ')

    if text:


        text = cleantext.clean(text , lowercase=True,extra_spaces=True,stopwords=True,numbers=True,punct=True)
        st.write(text)
        blob=TextBlob(text)
        st.write('Polarity (emotional sentiment expressed in a text): ',round(blob.sentiment.polarity,2))
        st.write('Subjectivity (extent to which a text contains personal opinions): ', round(blob.sentiment.subjectivity,2))

        tokens = tokenizer.encode(text, return_tensors='pt')
        result = model(tokens)


        sentiment=int(torch.argmax(result.logits))+1
        if sentiment==3:
            st.write("Neutral Sentiment")
        elif sentiment>3:
            st.write("Postive Feedback")
        elif sentiment<3:
            st.write("Negative Feedback")

        st.write("Feedback Score: ", sentiment )

responses = {
    "laptop": "Yes, we have laptops available.",
    "phone": "Yes, we have phones available.",

    "order": "Your order is currently in transit.",
    "password": "To reset your password, go to the login page and click on the 'Forgot Password' link.",
    "restaurant": "I recommend trying out 'Delicious Bites' restaurant. They have excellent food and great ambiance.",
    "support": "Our customer support team is available from Monday to Friday, 9:00 AM to 6:00 PM, and on weekends from 10:00 AM to 4:00 PM.",
    "return": "To initiate a product return, log in to your account, go to the 'Orders' section, and click on 'Return Request.'",
    "payment methods": "We accept various payment methods, including credit/debit cards, PayPal, and bank transfers.",
    "warranty": "Yes, all our products come with a standard one-year warranty.",
    "track order": "To track your order, log in to your account and go to the 'Orders' section.",
    "free shipping": "Yes, we offer free standard shipping on all orders over $50.",
    "discounts": "Yes, we currently have a 20% discount on selected items. Check out our 'Sale' section for more deals.",
    "shipping address": "Unfortunately, we cannot modify the shipping address once the order has been placed. Please contact our customer support team for assistance.",
    "return policy": "We have a 30-day return policy. If you are not satisfied with your purchase, you can return the item within 30 days of delivery for a full refund.",
    "international shipping": "Yes, we provide international shipping to most countries. Shipping rates and delivery times may vary based on the destination.",
    "cancel order": "To cancel your order, go to the 'Orders' section in your account and select 'Cancel Order.' You can cancel the order if it has not been shipped yet.",
    "brands": "We offer a wide range of popular brands, including XYZ, ABC, and DEF. You can explore our product catalog to find your favorite brands.",
    "contact support": "You can reach our customer support team via email at support@ecommerce.com or call our toll-free number 1-800-123-4567.",
    "payment security": "Absolutely! We use industry-standard encryption to protect your payment information, and we do not store your credit card details.",
    "loyalty rewards program": "Yes, we have a loyalty rewards program. You earn points for every purchase, which can be redeemed for discounts on future orders."
}


def get_response(text):
    for keyword, response in responses.items():
        if keyword in text:
            return response
    return "I'm sorry, I couldn't understand your question. Please try asking something else."



with st.expander('Talk to Chatbot'):
    st.write("Use keywords like laptop,phone,password,discounts,return,cancel order,support,brands,warranty")


    text2 = st.text_input('Enter Text:')

    response = get_response(text2)

    st.write("Chatbot:", response)
    st.write("To ask another question,clear input and press enter")