import pandas as pd
import tensorflow as tf 
import streamlit as st

import re

from tensorflow.keras.models import load_model 


def main():
    st.title("Sarcasm detection model")
    
    test_input = st.text_area("Enter Text here")
    
    
    st.button("Reset", type="primary")
    if st.button('Predict'):
        
        pre_processed_srting = preprocess_string(test_input)
        
        prediction = predict(pre_processed_srting)
        
        if prediction > 0.5:
        
            st.write('Sarcastic')
            st.write(f'Score: {prediction}')
        else:
            
            st.write(f'Not Sarcastic ')
            st.write(f'Score: {prediction}')  
    else:
        st.write("")
        
    
def preprocess_string(string):
    
    string = decontractions(string)
    
    test_df = pd.DataFrame({'comment': [string],
                        'label':[0]})
    
    test_dataset = df_to_dataset(test_df, shuffle=False)
    
    return test_dataset
    

def decontractions(phrase):
    """decontracted takes text and convert contractions into natural form.
     ref: https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python/47091490#47091490"""
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"won\’t", "will not", phrase)
    phrase = re.sub(r"can\’t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)

    phrase = re.sub(r"n\’t", " not", phrase)
    phrase = re.sub(r"\’re", " are", phrase)
    phrase = re.sub(r"\’s", " is", phrase)
    phrase = re.sub(r"\’d", " would", phrase)
    phrase = re.sub(r"\’ll", " will", phrase)
    phrase = re.sub(r"\’t", " not", phrase)
    phrase = re.sub(r"\’ve", " have", phrase)
    phrase = re.sub(r"\’m", " am", phrase)

    return phrase

    
def predict(df):
    
    encoder = tf.keras.layers.TextVectorization(max_tokens=2000) 
    encoder.adapt(df.map(lambda text,label:text))
    
    model = load_model('models\saracasm_class')
    predictions = model.predict(df)
    
    return predictions[0][0]

    
def df_to_dataset(dataframe, shuffle=True, batch_size=1024):
    df = dataframe.copy()
    labels = df.pop('label')
    df = df['comment']
    ds = tf.data.Dataset.from_tensor_slices((df, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds    
    
    
if __name__=='__main__':
    main() 

