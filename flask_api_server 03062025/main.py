from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from joblib import load
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics.pairwise import cosine_similarity


app=Flask(__name__)

                                                    # RANDOM FOREST CLASSIFIER

rfc_model=load('random_forest_model.joblib') # to load the model

# Mappings for the responses (for preprocessing)
allocation_behavior_map={
    "I will allocate my income into flexible budgets according to my needs and goals.": 8,
    "I will allocate my money to specific categories (e.g, rent, groceries) and stick to the set amounts.": 2,
    "I will focus on saving or investing first, then freely spend for the remaining amount for the rest of the month.":3
    }

finance_management_map={
    "Save First, Spend Later: I make saving my top priority and only allocate what's left afterwards":6,
    "The Flexible Allocator: I divide my income between savings, essentials, and some fun—whatever feels right!":8,
    "The Tracker: I keep tabs on every single expense—I need to know where my money is going!":2
    }

budgeting_preference_map={
    "Chill Budgeting: I don't focus on a detailed plan, but I aim to stay within a budget while maintaining some flexibility.":8,
    "Strict Budgeting: I prefer to follow a strict budget, setting specific limits for each categories and ensuring I don't exceed those amounts":2,
    "Savings First, Budget Later: I focus on saving first, setting aside money for my goals before spending.":6
    }

debt_management_preference_map={
    "I set aside a fixed amount each month for both savings and debt repayment.":2,
    "I divide my income proportionally between savings, debt, and other expenses.":7,
    "I save first and then allocate any remaining funds for debt repayment.":3
}


# Defining a function for recommending specific saving strategy based on the user's classification
def recommend_strategy(classification):
    if classification == 'Master Planner':
        return 'Envelope Method'
    elif classification == 'Chill Spender':
        return 'Proportional'
    else:
        return 'Pay Yourself First'


@app.route('/')

def hello_world():
    return 'FiliPinanZ'


# App route for classifying users
@app.route('/classify',methods=["POST","GET"])

# function to classify users
def classify():
    print("POST request received")

    try:
        # to get the input data from the POST request
        rfc_data=request.get_json()
        print(f"\n{rfc_data}")
        # To convert the data into a dataframe
        rfc_df=pd.DataFrame([rfc_data])

        # for checking
        print(f"\nDATAFRAME: \n{rfc_df} \nDataframe columns: \n {rfc_df.columns}")

        # for preprocessing the data
        rfc_df['Allocation_Behavior']=rfc_df['Allocation_Behavior'].map(allocation_behavior_map)
        rfc_df['Finance_Management']=rfc_df['Finance_Management'].map(finance_management_map)
        rfc_df['Budgeting_Preferences']=rfc_df['Budgeting_Preferences'].map(budgeting_preference_map)
        rfc_df['Debt_Management_Preferences']=rfc_df['Debt_Management_Preferences'].map(debt_management_preference_map)        
        
    
        # to run the prediction
        prediction_rfc=rfc_model.predict(rfc_df)
        print(f"prediction: \n{prediction_rfc}")

        # Storing the predicted classification in classification variable
        classification = {
            0: 'Master Planner',
            1: 'Goal Getter',
            2: 'Chill Spender'
        }.get(prediction_rfc[0], 'Unknown')

        print(f"\nclassification\n{classification}")
   

        # to get the recommended saving strategy
        saving_strategy=recommend_strategy(classification)
        print(f"\n Saving strat: \n{saving_strategy}")

        return jsonify({
            "Classification":classification,
            "Recommended Saving Strategy":saving_strategy
        })
    except Exception as e:
        return jsonify({"error":str(e)}),400


                                            # CONTENT-BASED FILTERING (Saving Percentage)

# Defining the matrix that will be used for recommending the saving percentage (ITEM-FEATURES):
def define_saving_perc_matrix():
    cbf_saving_percentage_data={
        'Employment_Status':['Working Student', 'Working Student','Working Student','Working Student',
                            'Part-time','Part-time','Part-time','Part-time',
                            'Full-time Employee','Full-time Employee','Full-time Employee','Full-time Employee',
                            'Full-time Student','Full-time Student','Full-time Student','Full-time Student',
                            'Unemployed','Unemployed','Unemployed','Unemployed',
                            'Self-Employed','Self-Employed','Self-Employed','Self-Employed'],
        'Average_Monthly_Cashflow': ['Less than ₱5,000', '₱5,000 - ₱9,999', '₱10,000 - ₱15,000', 'More than ₱15,000'] * 6,
        'Cashflow_Frequency': ['Low', 'Medium', 'High', 'Very High'] * 6,
        'Recommended_Savings_Percentage': [
            5, 10, 15, 20,  # Working Student 
            8, 12, 18, 25,  # Part-time 
            12, 18, 22, 30, # Full-time Employee → Highest stable income, should save more
            5, 7, 10, 15,   # Full-time Student 
            3, 5, 7, 10,    # Unemployed
            8, 15, 20, 28   # Self-Employed 
        ]
    }
    
    # Converting the defined Data into a DF:
    cbf_saving_percent_df=pd.DataFrame(cbf_saving_percentage_data)

    return cbf_saving_percent_df

                                                    # FUNCTIONS FOR MAPPING (SAVING PERCENTAGE)
# For mapping the features: 
def map_for_cbf_saving_perc(cbf_saving_perc_df):
    # Mapping for Cashflow Category 
    cbf_saving_perc_df['Average_Monthly_Cashflow']=cbf_saving_perc_df['Average_Monthly_Cashflow'].map({
         'Less than ₱5,000': 'Low',
            '₱5,000 - ₱9,999': 'Medium',
            '₱10,000 - ₱15,000': 'High',
            'More than ₱15,000': 'Very High'
        }).fillna('Unknown')  

    return cbf_saving_perc_df

# For mapping the user input features: 
def map_for_user_cbf_saving_perc(user_cbf_saving_perc_df):
    # Mapping for Cashflow Category 
    user_cbf_saving_perc_df['Average_Monthly_Cashflow']=user_cbf_saving_perc_df['Average_Monthly_Cashflow'].map({
         'Less than ₱5,000': 'Low',
            '₱5,000 - ₱9,999': 'Medium',
            '₱10,000 - ₱15,000': 'High',
            'More than ₱15,000': 'Very High'
        }).fillna('Unknown')  

    user_cbf_saving_perc_df['Cashflow_Frequency']=user_cbf_saving_perc_df['Cashflow_Frequency'].map({
         'Bi-Weekly (every two (2) weeks)': 'High',
            'Weekly': 'Very High',
            'Monthly': 'Medium',
            'Quarterly (every three (3) months)': 'Low'
        }).fillna('Unknown')  
        
    return user_cbf_saving_perc_df
                                                    # FUNCTIONS FOR PREPROCESSING (SAVING PERCENTAGE)

def preprocess_cbf_savings_perc(cbf_saving_perc):
    # For Ordinal features:
    ordinal_ranks=[['Low', 'Medium', 'High', 'Very High'], # First column (for 1st feature)
                   ['Low', 'Medium', 'High', 'Very High']] # Second column (for 2nd feature)
    
    ordinal_cbf_saving_perc_features=['Average_Monthly_Cashflow','Cashflow_Frequency'] # Columns to encode

    # Initialize Ordinal Encoder
    ordinal_encoder=OrdinalEncoder(categories=ordinal_ranks)
    cbf_saving_perc[ordinal_cbf_saving_perc_features]=ordinal_encoder.fit_transform(cbf_saving_perc[ordinal_cbf_saving_perc_features])

    # For OnehotEncoder (Defining Employee Categories):
    user_employment_status_categories = ['Full-time Employee', 'Full-time Student', 'Part-time', 'Self-Employed', 'Unemployed', 'Working Student']
    
    # Initialize OneHotEncoder: 
    oneh_encoder=OneHotEncoder(categories=[user_employment_status_categories], sparse_output=False, handle_unknown='ignore')

    # Encoding now the Employment Status: 
    encoded_employment_status=oneh_encoder.fit_transform(cbf_saving_perc[['Employment_Status']])
    
    # Create column names for the new DataFrame
    encoded_columns = oneh_encoder.get_feature_names_out(['Employment_Status'])

    # Convert to DataFrame with correct column names
    encoded_employment_df = pd.DataFrame(encoded_employment_status, columns=encoded_columns)
    cbf_saving_perc=pd.concat([cbf_saving_perc,encoded_employment_df], axis=1)
    cbf_saving_perc.drop(columns=['Employment_Status'], inplace=True)

    print(f"\n Preprocessed Saving Percent Reco DataFrame \n{cbf_saving_perc}")

    return cbf_saving_perc

                                              
# Functions for converting the user's monthly cashflow input:
def categorize_user_cashflow_input(value):
    value=float(value)

    if value < 5000 :
        return 'Less than ₱5,000'
    elif 5000 <= value < 10000:
        return '₱5,000 - ₱9,999'
    elif 10000 <= value < 15000:
        return '₱10,000 - ₱15,000'
    else:
        return 'More than ₱15,000'

# Function for categorizing/mapping the Goal Duration:
def map_user_goal_duration(ui_savings_percentage_df):
    # The values that I have set here are generated from the median of each values like for instance, the median of 1-2 months is 1.5
    # Formula: (1+2)/2= 1.5, also yung sa years naka convert siya into months, hence why 18 yung value ni 1-2 years
    ui_savings_percentage_df['Goal_Duration']=ui_savings_percentage_df['Goal_Duration'].map({
    '1-2 months':1.5,
    '3-5 months':4,
    '6-12 months':9,
    '1-2 years':18,
    '3-5 years':48,
    '6-10 years':96
})

# Function for Savings Percentage Based on Target EFund Amount and Goal Duration:

def get_required_monthly_savings(ui_savings_percentage_df):
    user_target_efund_amount=ui_savings_percentage_df['Target_Efund_Amount'].values[0]
    
    # Have the Goal_Duration mapped first
    ui_savings_percentage_df = map_user_goal_duration(ui_savings_percentage_df)

     # Then assign the value of the mapped Goal Duration
    user_goal_duration = ui_savings_percentage_df['Goal_Duration'].values[0]
    print(f"USER MAPPED GOAL DURATION:\n{user_goal_duration}") # For checking

    required_monthly_savings = user_target_efund_amount/user_goal_duration.round(2)

    return required_monthly_savings, user_goal_duration


# API for recommending the Saving Percentage (this will be the one that will generate the recommendations for the Saving percentage)

@app.route('/recommend_saving_percentage', methods=['POST', 'GET'])

def generate_saving_perc_recommendation(): 
    print("POST request received")

                                                        # ITEM- FEATURES

    cbf_saving_percentage_matrix_df=define_saving_perc_matrix()
    print(f"ITEM FEATURES FOR SAVING PERCENTAGE:\n{cbf_saving_percentage_matrix_df}\n") # For checking
    
    # Mapping the matrix :
    mapped_cbf_saving_perc_matrix=map_for_cbf_saving_perc(cbf_saving_percentage_matrix_df)
    print(f"\nMAPPED ITEM FEATURES FOR SAVING PERCENTAGE:\n{mapped_cbf_saving_perc_matrix}\n") # For checking

    # Preprocessing the  matrix:
    preprocessed_cbf_saving_perc_matrix=preprocess_cbf_savings_perc(mapped_cbf_saving_perc_matrix)
    print(f"PREPROCESSED ITEM FEATURES FOR SAVING PERCENTAGE:\n{preprocessed_cbf_saving_perc_matrix}\n")# For checking

    # Storing the preprocessed Saving Percentage Matrix to the variable "cbf_saving_percentage_matrix_df" Dataframe 
    cbf_saving_percentage_matrix_df=preprocessed_cbf_saving_perc_matrix
    print(f"FINAL ITEM FEATURES FOR SAVING PERCENTAGE:\n{cbf_saving_percentage_matrix_df}\n") # For checking

    # Dropping the Recommended Savings Percentage Column, since this will be the one that will be used as the Matrix to compute for the Cosine Similarity
    cbf_saving_percentage_matrix=cbf_saving_percentage_matrix_df.drop(columns=['Recommended_Savings_Percentage'])
    cbf_saving_percentage_matrix=cbf_saving_percentage_matrix.to_numpy() # Converting Dataframe to Array (para maging matrix siya)

    print(f"CBF SAVING PERCENTAGE MATRIX:\n{cbf_saving_percentage_matrix}") # For checking

                                                        # USER DATA (Input of User from the Android App)
    try:
        # Getting the user's input 
        ui_cbf_saving_perc_data = request.get_json()
        print(f"USER INPUT FOR SAVING PERCENTAGE:\n{ui_cbf_saving_perc_data}\n")  # For checking

        # Convert JSON data into a DataFrame
        ui_cbf_saving_percentage_df = pd.DataFrame([ui_cbf_saving_perc_data])

        # Getting the user's Target Efund Amount and Goal Duration:
        ui_efund_and_goal_dur_df = ui_cbf_saving_percentage_df[['Goal_Duration', 'Target_Efund_Amount']].copy()
        print(f"USER INPUT TARGET AMOUNT AND GOAL DURATION:\n{ui_efund_and_goal_dur_df}") # For checking

        # Dropping Goal_Duration','Target_Efund_Amount',and 'Average_Monthly_Expense'
        ui_cbf_saving_perc_df = ui_cbf_saving_percentage_df.drop(columns=['Goal_Duration','Target_Efund_Amount','Average_Monthly_Expense'], errors='ignore')
        print(f"USER INPUT DF FOR SAVING PERCENTAGE:\n{ui_cbf_saving_perc_df}\nINFO:\n {ui_cbf_saving_perc_df.info()}") # For checking

        # Convert Cashflow Categories: 
        ui_cbf_saving_perc_df['Average_Monthly_Cashflow']=ui_cbf_saving_perc_df['Average_Monthly_Cashflow'].apply(categorize_user_cashflow_input)
        print(f"USER'S CONVERTED CASHFLOW CATEGORY :\n{ui_cbf_saving_perc_df['Average_Monthly_Cashflow']}\n") # For checking

        # Map User's Data: 
        mapped_cbf_user_saving_perc_df=map_for_user_cbf_saving_perc(ui_cbf_saving_perc_df)
        print(f"USER'S MAPPED DF FOR SAVING PERCENTAGE :\n{mapped_cbf_user_saving_perc_df}\n") # For checking

        # Preprocess User's Data: 
        preprocessed_cbf_user_saving_perc_df=preprocess_cbf_savings_perc(mapped_cbf_user_saving_perc_df)
        print(f"USER'S PREPROCESSED DF FOR SAVING PERCENTAGE :\n{preprocessed_cbf_user_saving_perc_df}\n") # For checking

        # Convert the User's Input Dataframe into Array to turn it into a Matrix:
        user_cbf_saving_perc_matrix = preprocessed_cbf_user_saving_perc_df
        user_cbf_saving_perc_matrix=user_cbf_saving_perc_matrix.to_numpy()
        print(f"USER'S INPUT SAVING PERCENTAGE MATRIX :\n{user_cbf_saving_perc_matrix}\n") # For checking

        # Compute Cosine Similarity
        saving_percentage_similarities_matrix = cosine_similarity(user_cbf_saving_perc_matrix, cbf_saving_percentage_matrix)
        best_match_saving_percentage = saving_percentage_similarities_matrix.argmax()

        # Get recommendation (!! THIS IS THE RECOMMENDATION THAT IS GENERATED FROM THE SYSTEM BASED ON USER'S INPUTS: !! )
        saving_percentage_recommendation = cbf_saving_percentage_matrix_df.iloc[best_match_saving_percentage]['Recommended_Savings_Percentage']
        print(f"\n Recommended Saving Percent\n{saving_percentage_recommendation}")


        # Check the user's Recommended Saving's Percentage Based on their Target Emergency Fund Amount and Goal Duration: 
        # Formula: Target Efnf Amount / Goal Duration:
        user_required_monthly_savings, user_goal_duration=get_required_monthly_savings(ui_efund_and_goal_dur_df)
        print(f"USER'S REQUIRED MONTHLY SAVINGS:\n{user_required_monthly_savings}\n") # For checking
        print(f"USER'S MAPPED GOAL DURATION:\n{user_goal_duration}\n") # For checking



        # Return JSON response
        return jsonify({
            "CBF Recommended Savings Percentage":saving_percentage_recommendation,
            "Goal Duration": user_goal_duration
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

                                
if __name__=='__main__':
    app.run(debug=True,host='0.0.0.0')