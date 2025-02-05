import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st
import plotly.express as px
import speech_recognition as sr
# from googletrans import Translator
# import spacy
# import openai

# Load the model and data
model = pk.load(open('model.pkl', 'rb'))
cars_data = pd.read_csv("Cardetails.csv")
# translator = Translator()
# nlp = spacy.load("en_core_web_sm")
# openai.api_key = "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"

# Extract brand names
def get_brand_name(car_name):
    car_name = car_name.split(' ')[0]
    return car_name.strip()

cars_data['name'] = cars_data['name'].apply(get_brand_name)

# Streamlit App Configuration
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar Navigation
with st.sidebar:
    st.title("CarPriceX")
    nav_option = st.radio(
        "Navigate to:",
        ["Home", "About", "Predict Price", "Insights & Trends", "Recommendations", 
         "Car Loan Calculator", "Insurance Estimation", "Community"]
    )

# Home Section
if nav_option == "Home":
    st.title("Welcome to CarPriceX 🚗")
    st.write("A tool to predict car prices, explore market trends, and calculate loans and insurance.")
    
# About Section
elif nav_option == "About":
    st.title("CarPriceX")
    st.markdown("""
    ## 📜 About This Model
    **CarPriceX** is an advanced tool to help you make informed decisions about buying or selling cars.  
    Key features include predictions, insights, recommendations, EMI calculators, and more.

    Welcome to **CarPriceX**, an advanced car price prediction model designed to help you make informed 
    decisions when buying or selling a used car. This tool leverages machine learning algorithms trained on 
    a comprehensive dataset of car listings to estimate the price of a car based on various key features.
        
    ### Key Features:
    - **Predict Price**: Get an estimated price for any car based on its specifications, including brand, model year, mileage, fuel type, and more.
    - **Insights & Trends**: Explore valuable data visualizations such as average prices by car brand, how mileage affects price, and other market insights to help you understand car pricing trends.
    - **Recommendations**: Based on your preferences and inputs, receive recommendations for similar cars, including options for specific years, brands, and other features.
    - **Car Loan & EMI Calculator**: Estimate your monthly EMI for a car loan based on loan amount, interest rate, and loan term, helping you better plan your car financing.
    - **Insurance Estimation**: Get an estimate for the car insurance premium based on the car’s details and your location, making it easier to calculate the full cost of ownership.
    """)


# Predict Price Section
elif nav_option == "Predict Price":
    st.markdown("## 🔮 Predict Car Price")
    with st.expander("Enter Car Details", expanded=True):
        name = st.selectbox("Car Brand", cars_data['name'].unique())
        year = st.slider("Manufacturing Year", 1994, 2024)
        km_driven = st.slider("Kilometers Driven", 11, 200000, step=1000)
        fuel = st.selectbox("Fuel Type", cars_data['fuel'].unique())
        seller_type = st.selectbox("Seller Type", cars_data['seller_type'].unique())
        transmission = st.selectbox("Transmission", cars_data['transmission'].unique())
        owner = st.selectbox("Owner Type", cars_data['owner'].unique())
        mileage = st.slider("Mileage (km/l)", 10, 40, step=1)
        engine = st.slider("Engine Capacity (CC)", 700, 5000, step=100)
        max_power = st.slider("Maximum Power (BHP)", 0, 2000, step=50)
        seats = st.slider("Seats", 5, 10)

    car_condition = st.selectbox("Car Condition", ["Excellent", "Good", "Fair", "Poor"])
    condition_multiplier = {"Excellent": 1.2, "Good": 1.1, "Fair": 1.0, "Poor": 0.8}

    if st.button("🚀 Predict"):
        input_data = pd.DataFrame([[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]],
                                  columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats'])
        encoding_dict = {
            'owner': {'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3, 'Fourth & Above Owner': 4, 'Test Drive Car': 5},
            'fuel': {'Diesel': 1, 'Petrol': 2, 'LPG': 3, 'CNG': 4},
            'seller_type': {'Individual': 1, 'Dealer': 2, 'Trustmark Dealer': 3},
            'transmission': {'Manual': 1, 'Automatic': 2},
            'name': dict(zip(cars_data['name'].unique(), range(1, len(cars_data['name'].unique()) + 1)))
        }
        for col, mapping in encoding_dict.items():
            input_data[col].replace(mapping, inplace=True)

        predicted_price = model.predict(input_data) * condition_multiplier[car_condition]
        st.markdown(f"### 💰 Predicted Price: **₹ {predicted_price[0]:,.2f}**")
        st.write("### Prediction Details:")
        st.json(input_data.to_dict(orient='records')[0])

# Add other sections like Insights, Recommendations, Community
# Code continues with additional feature integrations like financial features, voice assistant, etc.
# Insights & Trends Section
elif nav_option == "Insights & Trends":
    st.markdown("## 📊 Market Insights")
    st.write("Explore car price trends and attributes to make informed decisions.")
    brand_filter = st.multiselect("Filter by Brand:", options=cars_data['name'].unique(), default=cars_data['name'].unique())
    filtered_data = cars_data[cars_data['name'].isin(brand_filter)]

    st.markdown("### 📈 Average Price by Brand")
    avg_price = filtered_data.groupby('name')['selling_price'].mean().sort_values(ascending=False)
    st.bar_chart(avg_price)

    st.markdown("### 📉 Mileage vs Selling Price")
    scatter_fig = px.scatter(
        filtered_data,
        x="mileage",
        y="selling_price",
        color="fuel",
        title="Mileage vs Selling Price",
        labels={"selling_price": "Selling Price (₹)", "mileage": "Mileage (km/l)"}
    )
    st.plotly_chart(scatter_fig, use_container_width=True)

# Recommendations Section
elif nav_option == "Recommendations":
    st.markdown("## 💡 Personalized Recommendations")
    st.write("Here are some car models similar to your criteria:")
    recommendation_criteria = st.slider("Filter by Manufacturing Year:", 1994, 2024, (2010, 2020))
    recommendations = cars_data[(cars_data['year'] >= recommendation_criteria[0]) & (cars_data['year'] <= recommendation_criteria[1])].sample(5)
    st.write(recommendations)

# Car Loan EMI Calculator Section
elif nav_option == "Car Loan Calculator":
    st.markdown("### 💳 Car Loan EMI Calculator")
    loan_amount = st.number_input("Loan Amount (₹)", min_value=0)
    interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0)
    loan_term = st.number_input("Loan Term (Years)", min_value=1, max_value=10)

    if interest_rate == 0:
        emi = loan_amount / (loan_term * 12)
    else:
        emi = (loan_amount * interest_rate / 1200) / (1 - (1 + interest_rate / 1200) ** (-loan_term * 12))

    st.write(f"EMI: ₹ {emi:,.2f} per month")

# Car Insurance Estimation Section
elif nav_option == "Insurance Estimation":
    st.markdown("### 🛡️ Car Insurance Estimation")
    car_value = st.number_input("Car Value (₹)", min_value=10000)
    insurance_type = st.selectbox("Select Insurance Type", ["Comprehensive", "Third-Party", "Zero Depreciation"])

    insurance_rate = {"Comprehensive": 0.05, "Third-Party": 0.03, "Zero Depreciation": 0.07}
    estimated_insurance = car_value * insurance_rate[insurance_type]
    st.write(f"Estimated Insurance Cost: ₹ {estimated_insurance:,.2f}")

# # Voice Assistant Section
# elif nav_option == "Voice Assistant":
#     st.markdown("### 🎤 Voice Assistant for Car Details Entry")
#     recognizer = sr.Recognizer()

#     def listen():
#         with sr.Microphone() as source:
#             st.write("Listening...")
#             audio = recognizer.listen(source)
#             try:
#                 query = recognizer.recognize_google(audio)
#                 st.write("You said: " + query)
#                 # Process the query
#                 translated_query = translator.translate(query, src='en', dest='en').text
#                 st.write("Processed Query: ", translated_query)
#             except sr.UnknownValueError:
#                 st.write("Sorry, I didn't catch that.")
#             except sr.RequestError as e:
#                 st.write(f"Could not request results; {e}")

#     if st.button("Start Listening"):
#         listen()

# Community Features Section
elif nav_option == "Community":
    st.markdown("## 🌟 Community Features")
    st.markdown("### User Reviews & Ratings")
    car_model = st.selectbox("Select a Car Model to Review:", cars_data['name'].unique())
    user_review = st.text_area("Write your review here:")
    rating = st.slider("Rate this car (1-5):", 1, 5)

    if st.button("Submit Review"):
        st.write("Thank you for your feedback!")

#     st.markdown("### Live Chat with Support")
#     user_query = st.text_input("Type your query here:")
#     if st.button("Ask Support"):
#         response = openai.Completion.create(
#             engine="text-davinci-003",
#             prompt=f"Provide assistance for: {user_query}",
#             max_tokens=100
#         )
#         st.write("Support Response: ", response.choices[0].text.strip())

#     st.markdown("### ChatGPT Integration")
#     chat_query = st.text_input("Ask ChatGPT something about cars:")
#     if st.button("Ask ChatGPT"):
#         chat_response = openai.Completion.create(
#             engine="text-davinci-003",
#             prompt=chat_query,
#             max_tokens=150
#         )
#         st.write("ChatGPT says: ", chat_response.choices[0].text.strip())
# Financial Features Section: Total Ownership Cost
elif nav_option == "Financial Features":
    st.markdown("## 💰 Financial Features")
    st.markdown("### 🚗 Tax Calculator")
    car_price = st.number_input("Enter Car Price (₹):", min_value=10000)
    location = st.selectbox("Select Your Location:", ["Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata"])
    tax_rates = {"Delhi": 0.1, "Mumbai": 0.12, "Bangalore": 0.11, "Chennai": 0.09, "Kolkata": 0.08}
    estimated_tax = car_price * tax_rates[location]
    st.write(f"Estimated Tax: ₹ {estimated_tax:,.2f}")

    st.markdown("### 📊 Total Ownership Cost")
    insurance_cost = st.number_input("Insurance Cost (₹):", min_value=1000)
    emi_cost = st.number_input("Monthly EMI (₹):", min_value=1000)
    ownership_duration = st.slider("Ownership Duration (Years):", 1, 10)
    total_cost = estimated_tax + (insurance_cost * ownership_duration) + (emi_cost * ownership_duration * 12)
    st.write(f"Total Ownership Cost: ₹ {total_cost:,.2f}")

# # Voice Assistant: Intent Detection
# elif nav_option == "Voice Assistant (Advanced)":
#     st.markdown("### 🎤 Advanced Voice Assistant")
#     st.write("Use natural language commands for car searches.")

#     def advanced_listen():
#         with sr.Microphone() as source:
#             st.write("Listening...")
#             audio = recognizer.listen(source)
#             try:
#                 query = recognizer.recognize_google(audio)
#                 st.write("You said: " + query)
#                 # Basic intent detection
#                 if "SUV under" in query:
#                     budget = int(''.join(filter(str.isdigit, query)))
#                     filtered_cars = cars_data[(cars_data['body_type'] == "SUV") & (cars_data['selling_price'] <= budget)]
#                     st.write("Here are some SUVs under your budget:")
#                     st.write(filtered_cars)
#                 elif "manual transmission" in query:
#                     filtered_cars = cars_data[cars_data['transmission'] == "Manual"]
#                     st.write("Here are cars with manual transmission:")
#                     st.write(filtered_cars)
#                 else:
#                     st.write("Command not recognized. Try something like 'Find me a 2015 SUV under ₹5,00,000.'")
#             except sr.UnknownValueError:
#                 st.write("Sorry, I didn't catch that.")
#             except sr.RequestError as e:
#                 st.write(f"Could not request results; {e}")

#     if st.button("Start Listening (Advanced)"):
#         advanced_listen()

# # Multilingual Support
# elif nav_option == "Multilingual Support":
#     st.markdown("## 🌐 Multilingual Support")
#     st.write("Translate your queries and results into regional languages.")
#     available_languages = {"English": "en", "Hindi": "hi", "Tamil": "ta", "Telugu": "te", "Marathi": "mr"}
#     selected_language = st.selectbox("Choose a Language:", list(available_languages.keys()))
#     query = st.text_area("Type your query in English:")

#     if st.button("Translate Query"):
#         translated_query = translator.translate(query, src="en", dest=available_languages[selected_language]).text
#         st.write(f"Translated Query ({selected_language}): {translated_query}")

#     # Simulate query response translation
#     response = "Here are some options for you."  # Example response
#     translated_response = translator.translate(response, src="en", dest=available_languages[selected_language]).text
#     st.write(f"Response ({selected_language}): {translated_response}")

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Developed with ❤️ by Parul Yaduvanshi & Sambhav Maheshwari")


# updated code VII


# import pandas as pd
# import numpy as np
# import pickle as pk
# import streamlit as st
# import plotly.express as px
# import speech_recognition as sr

# # Load the model and data
# model = pk.load(open('model.pkl', 'rb'))
# cars_data = pd.read_csv("Cardetails.csv")

# # Extract brand names
# def get_brand_name(car_name):
#     car_name = car_name.split(' ')[0]
#     return car_name.strip()

# cars_data['name'] = cars_data['name'].apply(get_brand_name)

# # Streamlit App Configuration
# st.set_page_config(
#     page_title="Car Price Prediction",
#     page_icon="🚗",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Sidebar Navigation
# with st.sidebar:
#     st.title("CarPriceX")
#     nav_option = st.radio(
#         "Navigate to:",
#         ["Home", "About", "Predict Price", "Insights & Trends", "Recommendations", 
#          "Car Loan Calculator", "Insurance Estimation", "Voice Assistant"]
#     )

# # Home Section
# if nav_option == "Home":
#     st.title("Welcome to CarPriceX 🚗")
#     st.write("A tool to predict car prices, explore market trends, and calculate loans and insurance.")

# # About Section
# elif nav_option == "About":
#     st.title("CarPriceX")
#     st.markdown(
#         """
#         ## 📜 About This Model
        
#         Welcome to **CarPriceX**, an advanced car price prediction model designed to help you make informed decisions when buying or selling a used car. This tool leverages machine learning algorithms trained on a comprehensive dataset of car listings to estimate the price of a car based on various key features.
        
#         ### Key Features:
#         - **Predict Price**: Get an estimated price for any car based on its specifications, including brand, model year, mileage, fuel type, and more.
#         - **Insights & Trends**: Explore valuable data visualizations such as average prices by car brand, how mileage affects price, and other market insights to help you understand car pricing trends.
#         - **Recommendations**: Based on your preferences and inputs, receive recommendations for similar cars, including options for specific years, brands, and other features.
#         - **Car Loan EMI Calculator**: Estimate your monthly EMI for a car loan based on loan amount, interest rate, and loan term, helping you better plan your car financing.
#         - **Car Condition Rating**: Get an estimated condition rating of the car based on factors such as mileage, age, and other relevant details, giving you a clearer understanding of the car’s value.
#         - **Interactive Car Visualization**: Explore visual representations of car data through charts and graphs, helping you understand relationships between various factors such as mileage and price.
#         - **Car Insurance Estimation**: Get an estimate for the car insurance premium based on the car’s details and your location, making it easier to calculate the full cost of ownership.

        
#         ### How It Works:
#         **CarPriceX** uses advanced machine learning models trained on extensive data from the used car market. By simply entering a car's specifications (such as brand, year, mileage, fuel type, etc.), the app predicts its selling price. It also provides valuable insights into trends and market behavior to help you understand what factors impact car pricing.
        
#         Whether you're looking to buy, sell, or finance a car, **CarPriceX** empowers you with accurate predictions and in-depth market data, making the process smoother and more transparent.
#         """
#     )

# # Predict Price Section
# elif nav_option == "Predict Price":
#     st.markdown("## 🔮 Predict Car Price")
#     with st.expander("Enter Car Details", expanded=True):
#         name = st.selectbox("Car Brand", cars_data['name'].unique(), help="Choose the car's brand.")
#         year = st.slider("Manufacturing Year", 1994, 2024, help="Year the car was manufactured.")
#         km_driven = st.slider("Kilometers Driven", 11, 200000, step=1000, help="Total kilometers the car has been driven.")
#         fuel = st.selectbox("Fuel Type", cars_data['fuel'].unique(), help="Type of fuel used by the car.")
#         seller_type = st.selectbox("Seller Type", cars_data['seller_type'].unique(), help="Type of seller.")
#         transmission = st.selectbox("Transmission", cars_data['transmission'].unique(), help="Manual or Automatic.")
#         owner = st.selectbox("Owner Type", cars_data['owner'].unique(), help="Ownership history.")
#         mileage = st.slider("Mileage (km/l)", 10, 40, step=1, help="Fuel efficiency of the car.")
#         engine = st.slider("Engine Capacity (CC)", 700, 5000, step=100, help="Engine displacement in cubic centimeters.")
#         max_power = st.slider("Maximum Power (BHP)", 0, 2000, step=50, help="Car's maximum power output.")
#         seats = st.slider("Seats", 5, 10, help="Number of seats in the car.")

#     car_condition = st.selectbox(
#         "Car Condition",
#         ["Excellent", "Good", "Fair", "Poor"],
#         help="Select the condition of the car."
#     )
#     condition_multiplier = {"Excellent": 1.2, "Good": 1.1, "Fair": 1.0, "Poor": 0.8}

#     if st.button("🚀 Predict"):
#         # Prepare input data
#         input_data = pd.DataFrame([[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]],
#                                   columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats'])
#         encoding_dict = {
#             'owner': {'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3, 'Fourth & Above Owner': 4, 'Test Drive Car': 5},
#             'fuel': {'Diesel': 1, 'Petrol': 2, 'LPG': 3, 'CNG': 4},
#             'seller_type': {'Individual': 1, 'Dealer': 2, 'Trustmark Dealer': 3},
#             'transmission': {'Manual': 1, 'Automatic': 2},
#             'name': dict(zip(cars_data['name'].unique(), range(1, len(cars_data['name'].unique()) + 1)))
#         }
#         for col, mapping in encoding_dict.items():
#             input_data[col].replace(mapping, inplace=True)

#         # Predict
#         predicted_price = model.predict(input_data) * condition_multiplier[car_condition]
#         st.markdown(f"### 💰 Predicted Price (adjusted for condition): **₹ {predicted_price[0]:,.2f}**")
#         st.write("### Prediction Details:")
#         st.json(input_data.to_dict(orient='records')[0])

# # Insights & Trends Section
# elif nav_option == "Insights & Trends":
#     st.markdown("## 📊 Market Insights")
#     st.write("Explore car price trends and attributes to make informed decisions.")
#     brand_filter = st.multiselect("Filter by Brand:", options=cars_data['name'].unique(), default=cars_data['name'].unique())
#     filtered_data = cars_data[cars_data['name'].isin(brand_filter)]

#     st.markdown("### 📈 Average Price by Brand")
#     avg_price = filtered_data.groupby('name')['selling_price'].mean().sort_values(ascending=False)
#     st.bar_chart(avg_price)

#     st.markdown("### 📉 Mileage vs Selling Price")
#     scatter_fig = px.scatter(
#         filtered_data, 
#         x="mileage", 
#         y="selling_price", 
#         color="fuel", 
#         title="Mileage vs Selling Price",
#         labels={"selling_price": "Selling Price (₹)", "mileage": "Mileage (km/l)"}
#     )
#     st.plotly_chart(scatter_fig, use_container_width=True)

# # Recommendations Section
# elif nav_option == "Recommendations":
#     st.markdown("## 💡 Personalized Recommendations")
#     st.write("Here are some car models similar to your criteria:")
#     recommendation_criteria = st.slider("Filter by Manufacturing Year:", 1994, 2024, (2010, 2020))
#     recommendations = cars_data[(cars_data['year'] >= recommendation_criteria[0]) & (cars_data['year'] <= recommendation_criteria[1])].sample(5)
#     st.write(recommendations)

# # Car Loan EMI Calculator Section
# elif nav_option == "Car Loan Calculator":
#     st.markdown("### 💳 Car Loan EMI Calculator")
#     loan_amount = st.number_input("Loan Amount (₹)", min_value=0)
#     interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0)
#     loan_term = st.number_input("Loan Term (Years)", min_value=1, max_value=10)

#     if interest_rate == 0:
#         emi = loan_amount / (loan_term * 12)
#     else:
#         emi = (loan_amount * interest_rate / 1200) / (1 - (1 + interest_rate / 1200) ** (-loan_term * 12))

#     st.write(f"EMI: ₹ {emi:,.2f} per month")

# # Car Insurance Estimation Section
# elif nav_option == "Insurance Estimation":
#     st.markdown("### 🛡️ Car Insurance Estimation")
#     car_value = st.number_input("Car Value (₹)", min_value=10000)
#     insurance_type = st.selectbox("Select Insurance Type", ["Comprehensive", "Third-Party", "Zero Depreciation"])

#     insurance_rate = {"Comprehensive": 0.05, "Third-Party": 0.03, "Zero Depreciation": 0.07}
#     estimated_insurance = car_value * insurance_rate[insurance_type]
#     st.write(f"Estimated Insurance Cost: ₹ {estimated_insurance:,.2f}")

# # Voice Assistant Section
# elif nav_option == "Voice Assistant":
#     st.markdown("### 🎤 Voice Assistant for Car Details Entry")
#     recognizer = sr.Recognizer()

#     def listen():
#         with sr.Microphone() as source:
#             st.write("Listening...")
#             audio = recognizer.listen(source)
#             try:
#                 st.write("You said: " + recognizer.recognize_google(audio))
#             except sr.UnknownValueError:
#                 st.write("Sorry, I didn't catch that.")
#             except sr.RequestError as e:
#                 st.write(f"Could not request results; {e}")

#     if st.button("Start Listening"):
#         listen()
















# updated code VI
# import pandas as pd
# import numpy as np
# import pickle as pk
# import streamlit as st
# import plotly.express as px
# import speech_recognition as sr

# # Load the model and data
# model = pk.load(open('model.pkl', 'rb'))
# cars_data = pd.read_csv("Cardetails.csv")

# # Extract brand names
# def get_brand_name(car_name):
#     car_name = car_name.split(' ')[0]
#     return car_name.strip()

# cars_data['name'] = cars_data['name'].apply(get_brand_name)

# # Streamlit App Configuration
# st.set_page_config(
#     page_title="Car Price Prediction",
#     page_icon="🚗",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Top Navigation Bar using HTML and CSS
# st.markdown("""
#     <style>
#         .navbar {
#             background-color: #333;
#             overflow: hidden;
#         }
#         .navbar a {
#             float: left;
#             display: block;
#             color: white;
#             text-align: center;
#             padding: 14px 16px;
#             text-decoration: none;
#             font-size: 17px;
#         }
#         .navbar a:hover {
#             background-color: #ddd;
#             color: black;
#         }
#     </style>
#     <div class="navbar">
#         <a href="javascript:void(0)" onclick="window.location.href='/';">Home</a>
#         <a href="javascript:void(0)" onclick="window.location.href='#about';">About</a>
#         <a href="javascript:void(0)" onclick="window.location.href='#predict';">Predict Price</a>
#         <a href="javascript:void(0)" onclick="window.location.href='#insights';">Insights & Trends</a>
#         <a href="javascript:void(0)" onclick="window.location.href='#recommendations';">Recommendations</a>
#         <a href="javascript:void(0)" onclick="window.location.href='#calculator';">Car Loan Calculator</a>
#         <a href="javascript:void(0)" onclick="window.location.href='#insurance';">Insurance Estimation</a>
#         <a href="javascript:void(0)" onclick="window.location.href='#voice';">Voice Assistant</a>
#     </div>
# """, unsafe_allow_html=True)

# # Sidebar
# with st.sidebar:
#     st.title("CarPriceX")
    
#     st.markdown("### Features")
#     nav_option = st.radio(
#         "Navigate to:",
#         ["About", "Predict Price", "Insights & Trends", "Recommendations", "Comparison Tool", 
#          "Car Loan Calculator", "Insurance Estimation", "Voice Assistant"]
#     )

# # About Section
# if nav_option == "About":
#     st.title("CarPriceX")
#     st.markdown(
#         """
#         ## 📜 About This Model
        
#         Welcome to **CarPriceX**, an advanced car price prediction model designed to help you make informed decisions when buying or selling a used car. This tool leverages machine learning algorithms trained on a comprehensive dataset of car listings to estimate the price of a car based on various key features.
        
#         ### Key Features:
#         - **Predict Price**: Get an estimated price for any car based on its specifications, including brand, model year, mileage, fuel type, and more.
#         - **Insights & Trends**: Explore valuable data visualizations such as average prices by car brand, how mileage affects price, and other market insights to help you understand car pricing trends.
#         - **Recommendations**: Based on your preferences and inputs, receive recommendations for similar cars, including options for specific years, brands, and other features.
#         - **Car Loan & EMI Calculator**: Estimate your monthly EMI for a car loan based on loan amount, interest rate, and loan term, helping you better plan your car financing.
#         - **Insurance Estimation**: Get an estimate for the car insurance premium based on the car’s details and your location, making it easier to calculate the full cost of ownership.
        
#         ### How It Works:
#         **CarPriceX** uses advanced machine learning models trained on extensive data from the used car market. By simply entering a car's specifications (such as brand, year, mileage, fuel type, etc.), the app predicts its selling price. It also provides valuable insights into trends and market behavior to help you understand what factors impact car pricing.
        
#         Whether you're looking to buy, sell, or finance a car, **CarPriceX** empowers you with accurate predictions and in-depth market data, making the process smoother and more transparent.
#         """
#     )

# # Predict Price Section
# elif nav_option == "Predict Price":
#     st.markdown("## 🔮 Predict Car Price")
#     with st.expander("Enter Car Details", expanded=True):
#         name = st.selectbox("Car Brand", cars_data['name'].unique(), help="Choose the car's brand.")
#         year = st.slider("Manufacturing Year", 1994, 2024, help="Year the car was manufactured.")
#         km_driven = st.slider("Kilometers Driven", 11, 200000, step=1000, help="Total kilometers the car has been driven.")
#         fuel = st.selectbox("Fuel Type", cars_data['fuel'].unique(), help="Type of fuel used by the car.")
#         seller_type = st.selectbox("Seller Type", cars_data['seller_type'].unique(), help="Type of seller.")
#         transmission = st.selectbox("Transmission", cars_data['transmission'].unique(), help="Manual or Automatic.")
#         owner = st.selectbox("Owner Type", cars_data['owner'].unique(), help="Ownership history.")
#         mileage = st.slider("Mileage (km/l)", 10, 40, step=1, help="Fuel efficiency of the car.")
#         engine = st.slider("Engine Capacity (CC)", 700, 5000, step=100, help="Engine displacement in cubic centimeters.")
#         max_power = st.slider("Maximum Power (BHP)", 0, 2000, step=50, help="Car's maximum power output.")
#         seats = st.slider("Seats", 5, 10, help="Number of seats in the car.")

#     car_condition = st.selectbox(
#         "Car Condition",
#         ["Excellent", "Good", "Fair", "Poor"],
#         help="Select the condition of the car."
#     )
#     condition_multiplier = {"Excellent": 1.2, "Good": 1.1, "Fair": 1.0, "Poor": 0.8}

#     if st.button("🚀 Predict"):
#         # Prepare input data
#         input_data = pd.DataFrame([[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]],
#                                   columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats'])
#         encoding_dict = {
#             'owner': {'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3, 'Fourth & Above Owner': 4, 'Test Drive Car': 5},
#             'fuel': {'Diesel': 1, 'Petrol': 2, 'LPG': 3, 'CNG': 4},
#             'seller_type': {'Individual': 1, 'Dealer': 2, 'Trustmark Dealer': 3},
#             'transmission': {'Manual': 1, 'Automatic': 2},
#             'name': dict(zip(cars_data['name'].unique(), range(1, len(cars_data['name'].unique()) + 1)))
#         }
#         for col, mapping in encoding_dict.items():
#             input_data[col].replace(mapping, inplace=True)

#         # Predict
#         predicted_price = model.predict(input_data) * condition_multiplier[car_condition]
#         st.markdown(f"### 💰 Predicted Price (adjusted for condition): **₹ {predicted_price[0]:,.2f}**")
#         st.write("### Prediction Details:")
#         st.json(input_data.to_dict(orient='records')[0])

# # Insights & Trends Section
# elif nav_option == "Insights & Trends":
#     st.markdown("## 📊 Market Insights")
#     st.write("Explore car price trends and attributes to make informed decisions.")
#     brand_filter = st.multiselect("Filter by Brand:", options=cars_data['name'].unique(), default=cars_data['name'].unique())
#     filtered_data = cars_data[cars_data['name'].isin(brand_filter)]

#     st.markdown("### 📈 Average Price by Brand")
#     avg_price = filtered_data.groupby('name')['selling_price'].mean().sort_values(ascending=False)
#     st.bar_chart(avg_price)

#     st.markdown("### 📉 Mileage vs Selling Price")
#     scatter_fig = px.scatter(
#         filtered_data, 
#         x="mileage", 
#         y="selling_price", 
#         color="fuel", 
#         title="Mileage vs Selling Price",
#         labels={"selling_price": "Selling Price (₹)", "mileage": "Mileage (km/l)"}
#     )
#     st.plotly_chart(scatter_fig, use_container_width=True)

# # Recommendations Section
# elif nav_option == "Recommendations":
#     st.markdown("## 💡 Personalized Recommendations")
#     st.write("Here are some car models similar to your criteria:")
#     recommendation_criteria = st.slider("Filter by Manufacturing Year:", 1994, 2024, (2010, 2020))
#     recommendations = cars_data[(cars_data['year'] >= recommendation_criteria[0]) & (cars_data['year'] <= recommendation_criteria[1])].sample(5)
#     st.write(recommendations)

# # Car Loan EMI Calculator Section
# elif nav_option == "Car Loan Calculator":
#     st.markdown("### 💳 Car Loan EMI Calculator")
#     loan_amount = st.number_input("Loan Amount (₹)", min_value=0)
#     interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0)
#     loan_term = st.number_input("Loan Term (Years)", min_value=1, max_value=10)

#     if interest_rate == 0:
#         emi = loan_amount / (loan_term * 12)
#     else:
#         emi = (loan_amount * interest_rate / 1200) / (1 - (1 + interest_rate / 1200) ** (-loan_term * 12))

#     st.write(f"EMI: ₹ {emi:,.2f} per month")

# # Car Insurance Estimation Section
# elif nav_option == "Insurance Estimation":
#     st.markdown("### 🛡️ Car Insurance Estimation")
#     car_value = st.number_input("Car Value (₹)", min_value=10000)
#     insurance_type = st.selectbox("Select Insurance Type", ["Comprehensive", "Third-Party", "Zero Depreciation"])

#     insurance_rate = {"Comprehensive": 0.05, "Third-Party": 0.03, "Zero Depreciation": 0.07}
#     estimated_insurance = car_value * insurance_rate[insurance_type]
#     st.write(f"Estimated Insurance Cost: ₹ {estimated_insurance:,.2f}")

# # Voice Assistant Section
# elif nav_option == "Voice Assistant":
#     st.markdown("### 🎤 Voice Assistant for Car Details Entry")
#     recognizer = sr.Recognizer()

#     def listen():
#         with sr.Microphone() as source:
#             st.write("Listening...")
#             audio = recognizer.listen(source)
#             try:
#                 st.write("You said: " + recognizer.recognize_google(audio))
#             except sr.UnknownValueError:
#                 st.write("Sorry, I didn't catch that.")
#             except sr.RequestError as e:
#                 st.write(f"Could not request results; {e}")

#     if st.button("Start Listening"):
#         listen()

 # updated code V
# import pandas as pd
# import numpy as np
# import pickle as pk
# import streamlit as st
# import plotly.express as px
# import speech_recognition as sr

# # Load the model and data
# model = pk.load(open('model.pkl', 'rb'))
# cars_data = pd.read_csv("Cardetails.csv")

# # Extract brand names
# def get_brand_name(car_name):
#     car_name = car_name.split(' ')[0]
#     return car_name.strip()

# cars_data['name'] = cars_data['name'].apply(get_brand_name)

# # Streamlit App Configuration
# st.set_page_config(
#     page_title="Car Price Prediction",
#     page_icon="🚗",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )
# # Top Navigation Bar using HTML and CSS
# st.markdown("""
#     <style>
#         .navbar {
#             background-color: #333;
#             overflow: hidden;
#         }
#         .navbar a {
#             float: left;
#             display: block;
#             color: white;
#             text-align: center;
#             padding: 14px 16px;
#             text-decoration: none;
#             font-size: 17px;
#         }
#         .navbar a:hover {
#             background-color: #ddd;
#             color: black;
#         }
#     </style>
#     <div class="navbar">
#         <a href="javascript:void(0)" onclick="window.location.href='/';">Home</a>
#         <a href="javascript:void(0)" onclick="window.location.href='#about';">About</a>
#         <a href="javascript:void(0)" onclick="window.location.href='#predict';">Predict Price</a>
#         <a href="javascript:void(0)" onclick="window.location.href='#insights';">Insights & Trends</a>
#         <a href="javascript:void(0)" onclick="window.location.href='#recommendations';">Recommendations</a>
#         <a href="javascript:void(0)" onclick="window.location.href='#calculator';">Car Loan Calculator</a>
#         <a href="javascript:void(0)" onclick="window.location.href='#insurance';">Insurance Estimation</a>
#         <a href="javascript:void(0)" onclick="window.location.href='#voice';">Voice Assistant</a>
#     </div>
# """, unsafe_allow_html=True)
# # Sidebar
# with st.sidebar:
#     st.title(" CarPriceX")
    
#     st.markdown("### Features")
#     nav_option = st.radio(
#         "Navigate to:",
#         ["About", "Predict Price", "Insights & Trends", "Recommendations", "Comparison Tool", 
#          "Car Loan Calculator", "Insurance Estimation", "Voice Assistant"]
#     )


# # About Section
# if nav_option == "About":
#     st.title("                                       CarPriceX")
#     st.markdown(
#         """
#         ## 📜 About This Model
        
#         Welcome to **CarPriceX**, an advanced car price prediction model designed to help you make informed decisions when buying or selling a used car. This tool leverages machine learning algorithms trained on a comprehensive dataset of car listings to estimate the price of a car based on various key features.
        
#         ### Key Features:
#         - **Car Price Prediction**: Get an estimated price for any car based on its specifications, including brand, model year, mileage, fuel type, and more.
#         - **Insights & Trends**: Explore valuable data visualizations such as average prices by car brand, how mileage affects price, and other market insights to help you understand car pricing trends.
#         - **Personalized Recommendations**: Based on your preferences and inputs, receive recommendations for similar cars, including options for specific years, brands, and other features.
#         - **Car Loan EMI Calculator**: Estimate your monthly EMI for a car loan based on loan amount, interest rate, and loan term, helping you better plan your car financing.
#         - **Car Condition Rating**: Get an estimated condition rating of the car based on factors such as mileage, age, and other relevant details, giving you a clearer understanding of the car’s value.
#         - **Comparison Tool**: Compare multiple car models side by side based on key attributes to find the best options for your needs and budget.
#         - **Interactive Car Visualization**: Explore visual representations of car data through charts and graphs, helping you understand relationships between various factors such as mileage and price.
#         - **Real-Time Market Data**: Stay updated with the latest car pricing trends and market data, giving you the edge in a competitive used car market.
#         - **Car Insurance Estimation**: Get an estimate for the car insurance premium based on the car’s details and your location, making it easier to calculate the full cost of ownership.
#         - **Vehicle History Report Integration**: Check for vehicle history reports to ensure the car’s reliability and avoid purchasing problematic vehicles.
#         - **Voice Assistant for Car Details Entry**: Quickly input car details using voice recognition, streamlining the process for a faster, hands-free experience.
#         - **Export Predictions to CSV/Excel**: Export your car price predictions and insights to a CSV or Excel file for further analysis or sharing.
        
#         ### How It Works:
#         **CarPriceX** uses advanced machine learning models trained on extensive data from the used car market. By simply entering a car's specifications (such as brand, year, mileage, fuel type, etc.), the app predicts its selling price. It also provides valuable insights into trends and market behavior to help you understand what factors impact car pricing.
        
#         Whether you're looking to buy, sell, or finance a car, **CarPriceX** empowers you with accurate predictions and in-depth market data, making the process smoother and more transparent.
#         """
#     )
#     # st.write("This app uses machine learning models to predict car prices based on various features like brand, model, year, mileage, and more.")
#     # st.write("The model is trained using a dataset containing historical data of used cars and their prices.")

# # Predict Price Section
# elif nav_option == "Predict Price":
#     st.markdown("## 🔮 Predict Car Price")
#     with st.expander("Enter Car Details", expanded=True):
#         name = st.selectbox("Car Brand", cars_data['name'].unique(), help="Choose the car's brand.")
#         year = st.slider("Manufacturing Year", 1994, 2024, help="Year the car was manufactured.")
#         km_driven = st.slider("Kilometers Driven", 11, 200000, step=1000, help="Total kilometers the car has been driven.")
#         fuel = st.selectbox("Fuel Type", cars_data['fuel'].unique(), help="Type of fuel used by the car.")
#         seller_type = st.selectbox("Seller Type", cars_data['seller_type'].unique(), help="Type of seller.")
#         transmission = st.selectbox("Transmission", cars_data['transmission'].unique(), help="Manual or Automatic.")
#         owner = st.selectbox("Owner Type", cars_data['owner'].unique(), help="Ownership history.")
#         mileage = st.slider("Mileage (km/l)", 10, 40, step=1, help="Fuel efficiency of the car.")
#         engine = st.slider("Engine Capacity (CC)", 700, 5000, step=100, help="Engine displacement in cubic centimeters.")
#         max_power = st.slider("Maximum Power (BHP)", 0, 2000, step=50, help="Car's maximum power output.")
#         seats = st.slider("Seats", 5, 10, help="Number of seats in the car.")

#     car_condition = st.selectbox(
#         "Car Condition",
#         ["Excellent", "Good", "Fair", "Poor"],
#         help="Select the condition of the car."
#     )
#     condition_multiplier = {"Excellent": 1.2, "Good": 1.1, "Fair": 1.0, "Poor": 0.8}

#     if st.button("🚀 Predict"):
#         # Prepare input data
#         input_data = pd.DataFrame([[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]],
#                                   columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats'])
#         encoding_dict = {
#             'owner': {'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3, 'Fourth & Above Owner': 4, 'Test Drive Car': 5},
#             'fuel': {'Diesel': 1, 'Petrol': 2, 'LPG': 3, 'CNG': 4},
#             'seller_type': {'Individual': 1, 'Dealer': 2, 'Trustmark Dealer': 3},
#             'transmission': {'Manual': 1, 'Automatic': 2},
#             'name': dict(zip(cars_data['name'].unique(), range(1, len(cars_data['name'].unique()) + 1)))
#         }
#         for col, mapping in encoding_dict.items():
#             input_data[col].replace(mapping, inplace=True)

#         # Predict
#         predicted_price = model.predict(input_data) * condition_multiplier[car_condition]
#         st.markdown(f"### 💰 Predicted Price (adjusted for condition): **₹ {predicted_price[0]:,.2f}**")
#         st.write("### Prediction Details:")
#         st.json(input_data.to_dict(orient='records')[0])

# # Insights & Trends Section
# elif nav_option == "Insights & Trends":
#     st.markdown("## 📊 Market Insights")
#     st.write("Explore car price trends and attributes to make informed decisions.")
#     brand_filter = st.multiselect("Filter by Brand:", options=cars_data['name'].unique(), default=cars_data['name'].unique())
#     filtered_data = cars_data[cars_data['name'].isin(brand_filter)]

#     st.markdown("### 📈 Average Price by Brand")
#     avg_price = filtered_data.groupby('name')['selling_price'].mean().sort_values(ascending=False)
#     st.bar_chart(avg_price)

#     st.markdown("### 📉 Mileage vs Selling Price")
#     scatter_fig = px.scatter(
#         filtered_data, 
#         x="mileage", 
#         y="selling_price", 
#         color="fuel", 
#         title="Mileage vs Selling Price",
#         labels={"selling_price": "Selling Price (₹)", "mileage": "Mileage (km/l)"}
#     )
#     st.plotly_chart(scatter_fig, use_container_width=True)

# # Recommendations Section
# elif nav_option == "Recommendations":
#     st.markdown("## 💡 Personalized Recommendations")
#     st.write("Here are some car models similar to your criteria:")
#     recommendation_criteria = st.slider("Filter by Manufacturing Year:", 1994, 2024, (2010, 2020))
#     recommendations = cars_data[(cars_data['year'] >= recommendation_criteria[0]) & (cars_data['year'] <= recommendation_criteria[1])].sample(5)
#     st.write(recommendations)

# # Car Loan & EMI Calculator Section
# # elif nav_option == "Car Loan Calculator":
# #     st.markdown("### 💳 Car Loan EMI Calculator")
# #     loan_amount = st.number_input("Loan Amount (₹)", min_value=0)
# #     interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0)
# #     loan_term = st.number_input("Loan Term (Years)", min_value=1, max_value=10)

# #     emi = (loan_amount * interest_rate / 1200) / (1 - (1 + interest_rate / 1200) ** (-loan_term * 12))
# #     st.write(f"EMI: ₹ {emi:,.2f} per month")
# # Car Loan EMI Calculator Section
# elif nav_option == "Car Loan Calculator":
#     st.markdown("### 💳 Car Loan EMI Calculator")
#     loan_amount = st.number_input("Loan Amount (₹)", min_value=0)
#     interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0)
#     loan_term = st.number_input("Loan Term (Years)", min_value=1, max_value=10)

#     if interest_rate == 0:
#         # When interest rate is 0, the EMI is simply the loan amount divided by the term
#         emi = loan_amount / (loan_term * 12)
#     else:
#         # Formula for EMI calculation
#         emi = (loan_amount * interest_rate / 1200) / (1 - (1 + interest_rate / 1200) ** (-loan_term * 12))

#     st.write(f"EMI: ₹ {emi:,.2f} per month")


# # Car Insurance Estimation Section
# elif nav_option == "Insurance Estimation":
#     st.markdown("### 🛡️ Car Insurance Estimation")
#     car_value = st.number_input("Car Value (₹)", min_value=10000)
#     insurance_type = st.selectbox("Select Insurance Type", ["Comprehensive", "Third-Party", "Zero Depreciation"])

#     insurance_rate = {"Comprehensive": 0.05, "Third-Party": 0.03, "Zero Depreciation": 0.07}
#     estimated_insurance = car_value * insurance_rate[insurance_type]
#     st.write(f"Estimated Insurance Cost: ₹ {estimated_insurance:,.2f}")

# # Comparison Tool Section
# elif nav_option == "Comparison Tool":
#     car1 = st.selectbox("Select Car 1", cars_data['name'].unique())
#     car2 = st.selectbox("Select Car 2", cars_data['name'].unique())
#     car1_data = cars_data[cars_data['name'] == car1].iloc[0]
#     car2_data = cars_data[cars_data['name'] == car2].iloc[0]

#     st.write(f"**{car1}** details:")
#     st.json(car1_data.to_dict())

#     st.write(f"**{car2}** details:")
#     st.json(car2_data.to_dict())

# # Voice Assistant Section
# elif nav_option == "Voice Assistant":
#     st.markdown("### 🎤 Voice Assistant for Car Details Entry")
#     recognizer = sr.Recognizer()

#     def listen():
#         with sr.Microphone() as source:
#             st.write("Listening...")
#             audio = recognizer.listen(source)
#             try:
#                 st.write("You said: " + recognizer.recognize_google(audio))
#             except sr.UnknownValueError:
#                 st.write("Sorry, I didn't catch that.")
#             except sr.RequestError as e:
#                 st.write(f"Could not request results; {e}")

#     if st.button("Start Listening"):
#         listen()

# updated code IV

# import pandas as pd
# import numpy as np
# import pickle as pk
# import streamlit as st
# import plotly.express as px

# # Load the model and data
# model = pk.load(open('model.pkl', 'rb'))
# cars_data = pd.read_csv("Cardetails.csv")

# # Extract brand names
# def get_brand_name(car_name):
#     car_name = car_name.split(' ')[0]
#     return car_name.strip()

# cars_data['name'] = cars_data['name'].apply(get_brand_name)

# # Streamlit App Configuration
# st.set_page_config(
#     page_title="Car Price Prediction",
#     page_icon="🚗",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Sidebar: Navigation
# with st.sidebar:
#     st.title("🚗 Car Price Prediction")
#     st.markdown("---")
#     st.markdown("### Features")
#     nav_option = st.radio(
#         "Navigate to:",
#         ["Predict Price", "Insights & Trends", "Recommendations", "About"]
#     )

# # Header
# st.markdown(f"<h1 style='text-align: center; color: #0078d7;'>🚘 Car Price Prediction ML Model</h1>", unsafe_allow_html=True)
# st.write("     Welcome to the **Car Price Prediction** app! Estimate car prices, explore trends, and get personalized recommendations. 🌟")

# # Section: Predict Price
# if nav_option == "Predict Price":
#     st.markdown("## 🔮 Predict Car Price")
#     with st.expander("Enter Car Details", expanded=True):
#         name = st.selectbox("Car Brand", cars_data['name'].unique(), help="Choose the car's brand.")
#         year = st.slider("Manufacturing Year", 1994, 2024, help="Year the car was manufactured.")
#         km_driven = st.slider("Kilometers Driven", 11, 200000, step=1000, help="Total kilometers the car has been driven.")
#         fuel = st.selectbox("Fuel Type", cars_data['fuel'].unique(), help="Type of fuel used by the car.")
#         seller_type = st.selectbox("Seller Type", cars_data['seller_type'].unique(), help="Type of seller.")
#         transmission = st.selectbox("Transmission", cars_data['transmission'].unique(), help="Manual or Automatic.")
#         owner = st.selectbox("Owner Type", cars_data['owner'].unique(), help="Ownership history.")
#         mileage = st.slider("Mileage (km/l)", 10, 40, step=1, help="Fuel efficiency of the car.")
#         engine = st.slider("Engine Capacity (CC)", 700, 5000, step=100, help="Engine displacement in cubic centimeters.")
#         max_power = st.slider("Maximum Power (BHP)", 0, 2000, step=50, help="Car's maximum power output.")
#         seats = st.slider("Seats", 5, 10, help="Number of seats in the car.")

#     if st.button("🚀 Predict"):
#         # Prepare input data
#         input_data = pd.DataFrame([[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]], 
#                                   columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats'])
#         encoding_dict = {
#             'owner': {'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3, 'Fourth & Above Owner': 4, 'Test Drive Car': 5},
#             'fuel': {'Diesel': 1, 'Petrol': 2, 'LPG': 3, 'CNG': 4},
#             'seller_type': {'Individual': 1, 'Dealer': 2, 'Trustmark Dealer': 3},
#             'transmission': {'Manual': 1, 'Automatic': 2},
#             'name': dict(zip(cars_data['name'].unique(), range(1, len(cars_data['name'].unique()) + 1)))
#         }
#         for col, mapping in encoding_dict.items():
#             input_data[col].replace(mapping, inplace=True)

#         # Predict
#         predicted_price = model.predict(input_data)
#         st.markdown(f"### 💰 Predicted Price: **₹ {predicted_price[0]:,.2f}**")
#         st.write("### Prediction Details:")
#         st.json(input_data.to_dict(orient='records')[0])

# # Section: Insights & Trends
# elif nav_option == "Insights & Trends":
#     st.markdown("## 📊 Market Insights")
#     st.write("Explore car price trends and attributes to make informed decisions.")
#     brand_filter = st.multiselect("Filter by Brand:", options=cars_data['name'].unique(), default=cars_data['name'].unique())
#     filtered_data = cars_data[cars_data['name'].isin(brand_filter)]

#     st.markdown("### 📈 Average Price by Brand")
#     avg_price = filtered_data.groupby('name')['selling_price'].mean().sort_values(ascending=False)
#     st.bar_chart(avg_price)

#     st.markdown("### 📉 Mileage vs Selling Price")
#     scatter_fig = px.scatter(
#         filtered_data, 
#         x="mileage", 
#         y="selling_price", 
#         color="fuel", 
#         title="Mileage vs Selling Price",
#         labels={"selling_price": "Selling Price (₹)", "mileage": "Mileage (km/l)"}
#     )
#     st.plotly_chart(scatter_fig, use_container_width=True)

# # Section: Recommendations
# elif nav_option == "Recommendations":
#     st.markdown("## 💡 Personalized Recommendations")
#     st.write("Here are some car models similar to your criteria:")
#     recommendation_criteria = st.slider("Filter by Manufacturing Year:", 1994, 2024, (2010, 2020))
#     recommendations = cars_data[(cars_data['year'] >= recommendation_criteria[0]) & (cars_data['year'] <= recommendation_criteria[1])].sample(5)
#     st.write(recommendations)

# # Section: About
# elif nav_option == "About":
#     st.markdown("## 📜 About This App")
#     st.write("""
#         This app uses a machine learning model to predict the price of a car based on various attributes like brand, year, mileage, and more.
#     """)
    
#     st.markdown("### 🧠 About the Model")
#     st.write("""
#         The car price prediction model is built using a regression algorithm that has been trained on a dataset containing car details such as:
#         - Car Brand
#         - Manufacturing Year
#         - Kilometers Driven
#         - Fuel Type
#         - Seller Type
#         - Transmission Type
#         - Ownership History
#         - Mileage
#         - Engine Capacity
#         - Maximum Power
#         - Seats
        
#         The model has been trained to learn patterns in how these features relate to the selling price of a car. 
#         It uses a **Random Forest Regression** (or another suitable regression model) to make accurate predictions based on historical data.
        
#         ### Key Steps in the Model:
#         1. **Data Preprocessing**: The dataset is cleaned, categorical features are encoded, and any missing values are handled.
#         2. **Feature Engineering**: Relevant features such as car brand, year, fuel type, etc., are selected and processed to be fed into the model.
#         3. **Model Training**: The model is trained using a regression algorithm on the preprocessed dataset.
#         4. **Prediction**: Once trained, the model predicts the car price based on user input features, offering a price estimate for the car.
        
#         ### Model Evaluation:
#         The model's performance is evaluated using metrics such as Mean Absolute Error (MAE) and R-squared (R²). These metrics help determine how accurately the model predicts car prices based on the input features.
#     """)


# updated code III
# import pandas as pd
# import numpy as np
# import pickle as pk
# import streamlit as st
# import plotly.express as px

# # Load the model and data
# model = pk.load(open('model.pkl', 'rb'))
# cars_data = pd.read_csv("Cardetails.csv")

# # Extract brand names
# def get_brand_name(car_name):
#     car_name = car_name.split(' ')[0]
#     return car_name.strip()

# cars_data['name'] = cars_data['name'].apply(get_brand_name)

# # Streamlit App Configuration
# st.set_page_config(
#     page_title="Car Price Prediction",
#     page_icon="🚗",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Sidebar: Theme and Navigation
# with st.sidebar:
#     st.image("https://via.placeholder.com/150", caption="Your Logo Here", use_column_width=True)
#     st.title("🚗 Car Price Prediction")
#     theme = st.radio("Select Theme:", ["Light", "Dark"], index=0)
#     st.markdown("---")
#     st.markdown("### Features")
#     nav_option = st.radio(
#         "Navigate to:",
#         ["Predict Price", "Insights & Trends", "Recommendations"]
#     )

# # Theme-based styling
# if theme == "Dark":
#     st.markdown(
#         """
#         <style>
#         body { background-color: #1E1E1E; color: white; }
#         </style>
#         """,
#         unsafe_allow_html=True
#     )

# # Header
# st.markdown(f"<h1 style='text-align: center; color: #0078d7;'>🚘 Car Price Prediction ML Model</h1>", unsafe_allow_html=True)
# st.write("     Welcome to the **Car Price Prediction** app! Estimate car prices, explore trends, and get personalized recommendations. 🌟")

# # Section: Predict Price
# if nav_option == "Predict Price":
#     st.markdown("## 🔮 Predict Car Price")
#     with st.expander("Enter Car Details", expanded=True):
#         name = st.selectbox("Car Brand", cars_data['name'].unique(), help="Choose the car's brand.")
#         year = st.slider("Manufacturing Year", 1994, 2024, help="Year the car was manufactured.")
#         km_driven = st.slider("Kilometers Driven", 11, 200000, step=1000, help="Total kilometers the car has been driven.")
#         fuel = st.selectbox("Fuel Type", cars_data['fuel'].unique(), help="Type of fuel used by the car.")
#         seller_type = st.selectbox("Seller Type", cars_data['seller_type'].unique(), help="Type of seller.")
#         transmission = st.selectbox("Transmission", cars_data['transmission'].unique(), help="Manual or Automatic.")
#         owner = st.selectbox("Owner Type", cars_data['owner'].unique(), help="Ownership history.")
#         mileage = st.slider("Mileage (km/l)", 10, 40, step=1, help="Fuel efficiency of the car.")
#         engine = st.slider("Engine Capacity (CC)", 700, 5000, step=100, help="Engine displacement in cubic centimeters.")
#         max_power = st.slider("Maximum Power (BHP)", 0, 2000, step=50, help="Car's maximum power output.")
#         seats = st.slider("Seats", 5, 10, help="Number of seats in the car.")

#     if st.button("🚀 Predict"):
#         # Prepare input data
#         input_data = pd.DataFrame([[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]],
#                                   columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats'])
#         encoding_dict = {
#             'owner': {'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3, 'Fourth & Above Owner': 4, 'Test Drive Car': 5},
#             'fuel': {'Diesel': 1, 'Petrol': 2, 'LPG': 3, 'CNG': 4},
#             'seller_type': {'Individual': 1, 'Dealer': 2, 'Trustmark Dealer': 3},
#             'transmission': {'Manual': 1, 'Automatic': 2},
#             'name': dict(zip(cars_data['name'].unique(), range(1, len(cars_data['name'].unique()) + 1)))
#         }
#         for col, mapping in encoding_dict.items():
#             input_data[col].replace(mapping, inplace=True)

#         # Predict
#         predicted_price = model.predict(input_data)
#         st.markdown(f"### 💰 Predicted Price: **₹ {predicted_price[0]:,.2f}**")
#         st.write("### Prediction Details:")
#         st.json(input_data.to_dict(orient='records')[0])

# # Section: Insights & Trends
# elif nav_option == "Insights & Trends":
#     st.markdown("## 📊 Market Insights")
#     st.write("Explore car price trends and attributes to make informed decisions.")
#     brand_filter = st.multiselect("Filter by Brand:", options=cars_data['name'].unique(), default=cars_data['name'].unique())
#     filtered_data = cars_data[cars_data['name'].isin(brand_filter)]

#     st.markdown("### 📈 Average Price by Brand")
#     avg_price = filtered_data.groupby('name')['selling_price'].mean().sort_values(ascending=False)
#     st.bar_chart(avg_price)

#     st.markdown("### 📉 Mileage vs Selling Price")
#     scatter_fig = px.scatter(
#         filtered_data, 
#         x="mileage", 
#         y="selling_price", 
#         color="fuel", 
#         title="Mileage vs Selling Price",
#         labels={"selling_price": "Selling Price (₹)", "mileage": "Mileage (km/l)"}
#     )
#     st.plotly_chart(scatter_fig, use_container_width=True)

# # Section: Recommendations
# elif nav_option == "Recommendations":
#     st.markdown("## 💡 Personalized Recommendations")
#     st.write("Here are some car models similar to your criteria:")
#     recommendation_criteria = st.slider("Filter by Manufacturing Year:", 1994, 2024, (2010, 2020))
#     recommendations = cars_data[(cars_data['year'] >= recommendation_criteria[0]) & (cars_data['year'] <= recommendation_criteria[1])].sample(5)
#     st.write(recommendations)
# updates code II

# import pandas as pd
# import numpy as np
# import pickle as pk
# import streamlit as st
# import plotly.express as px

# # Load the model
# model = pk.load(open('model.pkl', 'rb'))

# # Load the dataset
# cars_data = pd.read_csv("Cardetails.csv")

# # Extract brand names
# def get_brand_name(car_name):
#     car_name = car_name.split(' ')[0]
#     return car_name.strip()

# cars_data['name'] = cars_data['name'].apply(get_brand_name)

# # Set up Streamlit app
# st.set_page_config(
#     page_title="Car Price Prediction",
#     page_icon="🚗",
#     layout="wide"
# )

# # Sidebar
# with st.sidebar:
#     st.image("https://via.placeholder.com/150", caption="Your Logo Here", use_column_width=True)
#     st.title("Navigation")
#     st.markdown("### Features:")
#     nav_option = st.radio(
#         "Select a Section",
#         ["🚀 Predict Price", "📊 Insights & Trends", "💡 Recommendations"]
#     )

# # Header
# st.markdown("<h1 style='text-align: center; color: #0078d7;'>🚘 Car Price Prediction ML Model</h1>", unsafe_allow_html=True)
# st.write("Welcome to the **Car Price Prediction** app! Use this tool to estimate the price of a car based on its attributes, analyze trends, and get recommendations. 🌟")

# if nav_option == "🚀 Predict Price":
#     # Form for input
#     st.markdown("### Enter Car Details")
#     with st.expander("Expand to Input Car Details", expanded=True):
#         name = st.selectbox("Select Car Brand", cars_data['name'].unique(), help="Choose the brand of the car.")
#         year = st.slider("Car Manufactured Year", 1994, 2024, help="Year the car was manufactured.")
#         km_driven = st.slider("No. of kms driven", 11, 200000, step=1000, help="Total kilometers the car has been driven.")
#         fuel = st.selectbox("Fuel Type", cars_data['fuel'].unique(), help="Select the type of fuel the car uses.")
#         seller_type = st.selectbox("Seller Type", cars_data['seller_type'].unique(), help="Who is selling the car.")
#         transmission = st.selectbox("Transmission Type", cars_data['transmission'].unique(), help="Manual or automatic transmission.")
#         owner = st.selectbox("Owner", cars_data['owner'].unique(), help="Ownership details.")
#         mileage = st.slider("Car Mileage (km/l)", 10, 40, step=1, help="Fuel efficiency in kilometers per liter.")
#         engine = st.slider("Engine Capacity (CC)", 700, 5000, step=100, help="Engine capacity in cubic centimeters.")
#         max_power = st.slider("Max Power (BHP)", 0, 2000, step=50, help="Maximum power output in brake horsepower.")
#         seats = st.slider("No. of seats", 5, 10, help="Number of seats in the car.")

#     # Prediction
#     if st.button("🚀 Predict Price"):
#         input_data_model = pd.DataFrame(
#             [[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]],
#             columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats']
#         )

#         # Encoding categorical variables
#         encoding_dict = {
#             'owner': {'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3, 'Fourth & Above Owner': 4, 'Test Drive Car': 5},
#             'fuel': {'Diesel': 1, 'Petrol': 2, 'LPG': 3, 'CNG': 4},
#             'seller_type': {'Individual': 1, 'Dealer': 2, 'Trustmark Dealer': 3},
#             'transmission': {'Manual': 1, 'Automatic': 2},
#             'name': dict(zip(cars_data['name'].unique(), range(1, len(cars_data['name'].unique()) + 1)))
#         }

#         for col, mapping in encoding_dict.items():
#             input_data_model[col].replace(mapping, inplace=True)

#         # Predict the price
#         car_price = model.predict(input_data_model)

#         st.markdown(f"### 💰 Predicted Car Price: **₹ {car_price[0]:,.2f}**")
#         st.markdown(
#             f"<p style='color: green;'>Your car's predicted price is based on the provided attributes.</p>", 
#             unsafe_allow_html=True
#         )
#         st.write("### Input Details:")
#         st.json(input_data_model.to_dict(orient='records')[0])

# elif nav_option == "📊 Insights & Trends":
#     st.markdown("### Explore Car Market Trends")
#     avg_price = cars_data.groupby('name')['selling_price'].mean().sort_values()
#     st.bar_chart(avg_price)

#     st.markdown("#### Mileage vs Selling Price")
#     scatter_fig = px.scatter(
#         cars_data, 
#         x="mileage", 
#         y="selling_price", 
#         color="fuel", 
#         title="Mileage vs Selling Price",
#         labels={"selling_price": "Selling Price (₹)", "mileage": "Mileage (km/l)"}
#     )
#     st.plotly_chart(scatter_fig, use_container_width=True)

# elif nav_option == "💡 Recommendations":
#     st.markdown("### Recommendations Based on Your Input")
#     st.markdown("Here are some car models that are similar to your input criteria:")
#     # Example logic for recommendations
#     st.write(cars_data.sample(5))


#updated code I
# import pandas as pd
# import numpy as np
# import pickle as pk
# import streamlit as st

# # Load the model
# model = pk.load(open('model.pkl', 'rb'))

# # Load the dataset
# cars_data = pd.read_csv("Cardetails.csv")

# # Extract brand names
# def get_brand_name(car_name):
#     car_name = car_name.split(' ')[0]
#     return car_name.strip()

# cars_data['name'] = cars_data['name'].apply(get_brand_name)

# # Set up Streamlit app
# st.set_page_config(
#     page_title="Car Price Prediction",
#     page_icon="🚗",
#     layout="wide"
# )

# # Sidebar
# with st.sidebar:
#     st.image("https://via.placeholder.com/150", caption="Your Logo Here", use_column_width=True)
#     st.title("Navigation")
#     st.markdown("### Select a Section:")
#     st.radio("Go To", ["Car Price Prediction", "Insights & Analysis"])

# # Header
# st.markdown("<h1 style='text-align: center; color: blue;'>🚘 Car Price Prediction ML Model</h1>", unsafe_allow_html=True)
# st.write("Welcome to the **Car Price Prediction** app! Enter the details of the car to get an accurate price estimate. 🌟")

# # Form for input
# st.markdown("### Enter Car Details")
# with st.expander("Expand to Input Car Details"):
#     name = st.selectbox("Select Car Brand", cars_data['name'].unique())
#     year = st.slider("Car Manufactured Year", 1994, 2024)
#     km_driven = st.slider("No. of kms driven", 11, 200000, step=1000)
#     fuel = st.selectbox("Fuel Type", cars_data['fuel'].unique())
#     seller_type = st.selectbox("Seller Type", cars_data['seller_type'].unique())
#     transmission = st.selectbox("Transmission Type", cars_data['transmission'].unique())
#     owner = st.selectbox("Owner", cars_data['owner'].unique())
#     mileage = st.slider("Car Mileage (km/l)", 10, 40, step=1)
#     engine = st.slider("Engine Capacity (CC)", 700, 5000, step=100)
#     max_power = st.slider("Max Power (BHP)", 0, 2000, step=50)
#     seats = st.slider("No. of seats", 5, 10)

# # Prediction
# if st.button("🚀 Predict Price"):
#     input_data_model = pd.DataFrame(
#         [[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]],
#         columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats']
#     )

#     # Encoding categorical variables
#     input_data_model['owner'].replace(['First Owner', 'Second Owner', 'Third Owner',
#                                        'Fourth & Above Owner', 'Test Drive Car'], [1, 2, 3, 4, 5], inplace=True)
#     input_data_model['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'], [1, 2, 3, 4], inplace=True)
#     input_data_model['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'], [1, 2, 3], inplace=True)
#     input_data_model['transmission'].replace(['Manual', 'Automatic'], [1, 2], inplace=True)
#     input_data_model['name'].replace(['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
#                                       'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
#                                       'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
#                                       'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
#                                       'Ambassador', 'Ashok', 'Isuzu', 'Opel'], 
#                                      list(range(1, 32)), inplace=True)

#     # Predict the price
#     car_price = model.predict(input_data_model)
#     st.success(f"💰 Predicted Car Price: ₹ {car_price[0]:,.2f}")

# # Insights Section
# st.markdown("### Insights & Analysis 📊")
# with st.expander("View Average Prices by Brand"):
#     avg_price = cars_data.groupby('name')['selling_price'].mean().sort_values()
#     st.bar_chart(avg_price)


#basic code

# import pandas as pd
# import numpy as np
# import pickle as pk
# import streamlit as st

# model=pk. load(open('model.pkl','rb'))

# st.title('  Car Price Prediction ML Model')
# # st.subheader('Enter the details of the car to predict the price')

# cars_data=pd.read_csv("Cardetails.csv")

# def get_brand_name(car_name):
#     car_name=car_name.split(' ')[0]
#     return car_name.strip()
# cars_data['name']=cars_data['name'].apply(get_brand_name)

# name=st.selectbox("Select Car Brand",cars_data['name'].unique())

# year=st.slider("Car Manufactured Year",1994,2024)

# km_driven=st.slider("No. of kms driven",11,200000)

# fuel=st.selectbox("Fuel Type",cars_data['fuel'].unique())

# seller_type=st.selectbox("Seller Type",cars_data['seller_type'].unique())

# transmission=st.selectbox("Transmission Type",cars_data['transmission'].unique())

# owner=st.selectbox("Owner",cars_data['owner'].unique())

# mileage=st.slider("Car Mileage",10,40)

# engine=st.slider("Engine CC",700,5000)

# max_power=st.slider("Max Power",0,2000)

# seats=st.slider("No. of seats",5,10)

# if st.button("Predict"):
#     input_data_model = pd.DataFrame(
#         [[name,year,km_driven,fuel,seller_type,transmission,owner,mileage,engine,max_power,seats]],
#         columns=['name','year','km_driven','fuel','seller_type','transmission','owner','mileage','engine','max_power','seats']
#     )

#     input_data_model['owner'].replace(['First Owner', 'Second Owner', 'Third Owner',
#        'Fourth & Above Owner', 'Test Drive Car'],[1,2,3,4,5],inplace=True)

#     input_data_model['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'],[1,2,3,4],inplace=True)

#     input_data_model['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'],[1,2,3],inplace=True)

#     input_data_model['transmission'].replace(['Manual', 'Automatic'],[1,2],inplace=True)

#     input_data_model['name'].replace(['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
#        'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
#        'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
#        'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
#        'Ambassador', 'Ashok', 'Isuzu', 'Opel'],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],inplace=True)

#     # st.write(input_data_model)

    
#     car_price=model.predict(input_data_model)

#     st.markdown("Car price is going to be "+str(car_price[0]))


# # after running the above code use command "streamlit run app.py" to deploy this code into a web application.
# # 
