import streamlit as st
from time import sleep
import json
import os
import pickle
from groq import Groq
import base64
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setting up Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def image_classifier_using_qroq(client, image):
    base64_image = base64.b64encode(image.read()).decode('utf-8')
    chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image? The first line should say, the image depicts a .... Followed by a detailed description of less than 4 lines on the next paragraph."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
        }
    ],
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    )
    return chat_completion.choices[0].message.content


# Load model and encoder
with open("student_score_model.pkl", "rb") as pickle_file:
    bundle = pickle.load(pickle_file)

model = bundle["model"]
encoder = bundle["encoder"]


# Title of the Application 
st.title(
    body="Image Classification (Pets)", 
    anchor=False, 
    help="This app tells you whether your image contains a pet in it or not", 
    width="stretch"
)

# Data Collection Section
st.header(
    body="Data Collection Section",
    anchor=False,
    help="Here, just fill this form with your details",
    divider="blue",
    width="stretch"
)

# File Upload

## Initialize form submitted session state
if "form_submitted" not in st.session_state:
    st.session_state.form_submitted = False

## Simple form data for identification
with st.form(key="id-form", enter_to_submit=False,border=True, width="stretch", height="content"):
    name=st.text_input(label="Enter your name: ", placeholder="Nana")
    age=st.slider(label="Select your age", min_value=15, max_value=30, step=1)
    stay=st.selectbox(label="Where do you stay?", options=["Ayeduase", "New Site", "Boadi", "Appiadu", "On-campus"])
    submitted = st.form_submit_button("Submit Data")

if submitted:
    st.session_state.form_submitted = True

    new_user_data = {
        "username": name,
        "userage": age,
        "stay": stay
    }

    # Save data to JSON
    if not os.path.exists("data.json") or os.path.getsize("data.json") == 0:
        initial_data = [new_user_data]
        with open("data.json", "w") as data_file:
            json.dump(initial_data, data_file, indent=4)
    else:
        try:
            with open("data.json", "r") as data_file:
                existing_data = json.load(data_file)
            
            if not isinstance(existing_data, list):
                print("Error: The existing file content is not a list. Cannot append")
            existing_data.append(new_user_data)
            
            with open("data.json", "w") as data_file:
                json.dump(existing_data, data_file, indent=4)

        except Exception as e:
            print(f"ERROR!: {e}")


    # Progress Bar
    progress_message = "Submitting Data. Please wait."
    progress_bar = st.progress(value=0, text=progress_message)
    for percent_complete in range(100):
        sleep(0.02)
        progress_bar.progress(percent_complete+1, text=progress_message)
    progress_bar.empty()
    st.balloons()
    st.write("Data submitted successfully")


# Upload Image now for prediction

if st.session_state.form_submitted:
    # Image Upload Section
    st.header(
        body="Image Upload Section",
        anchor=False,
        help="Here, just upload image and await prediction",
        divider="blue",
        width="stretch"
    )

    user_upload = st.file_uploader("Upload an image file (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

    if user_upload:
        st.image(user_upload, caption="Uploaded image")
        response = image_classifier_using_qroq(client=client, image=user_upload)
        st.subheader("Image description", divider="blue")
        st.write(response)
        st.balloons()

        # Prediction Section
        st.header(
        body="Prediction Section",
        anchor=False,
        help="Here, just fill form data and make prediction. Based on data entered, likelihood of performing well in exams will be predicted.",
        divider="blue",
        width="stretch"
        )

        with st.form(key="pred-form", enter_to_submit=False,border=True, width="stretch", height="content"):
            gender=st.radio(label="Enter your gender: ", options=["Male", "Female"])
            parent_education_level=st.selectbox(label="Select your Parent Education Level", options=["some high school", "high school (completed)", "some college", "associate's degree", "bachelor's degree", "master's degree"])
            test_preparation_course=st.selectbox(label="Did student complete a test prep course?", options=["Completed", "None"])
            submitted2 = st.form_submit_button("Make Prediction")

        if submitted2:
            user_input_pred = pd.DataFrame([
                {
                    "gender": gender.lower(),
                    "parental level of education": parent_education_level,
                    "test preparation course": test_preparation_course.lower()
                }
            ])

            # Encode user data
            user_encoded = encoder.transform(user_input_pred)

            # Predict 
            predicted_score = model.predict(user_encoded)[0]
            st.write(f"Predicted math score: {predicted_score:.2f}")