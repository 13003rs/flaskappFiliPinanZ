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
    "I’ll divide my income into flexible budgets based on my needs and goals.": 8,
    "I will allocate my money into specific categories (e.g, rent, groceries) and stick to those amounts.": 2,
    "I will prioritize allocating it for saving or investments right away and then spend freely for the rest of the month.":3
    }

finance_management_map={
    "Save First, Spend Later: I make saving my top priority and only allocate what's left afterwards":6,
    "The Flexible Allocator: I break down my income into a mix of savings, essentials, and some fun—whatever feels right!":8,
    "The Tracker: I keep tabs on every single expense—gotta know where my money’s going!":2
    }

budgeting_preference_map={
    "Chill Budgeting: I don't stress over a detailed plan but aim to stay generally within a budget while keeping some flexibility":8,
    "Strict Budgeting:I prefer to stick to a strict budget where each category has a set limit and don't go over those amounts.":2,
    "Savings First, Budget Later: I make saving my top priority before I spend, setting aside money right away for my goals.":6
    }

debt_management_preference_map={
    "I set aside a specific amount for both savings and debt repayment each month.":2,
    "I divide my income proportionally between savings, debt, and other expenses.":7,
    "I save first and then allocate whatever's left for debt repayment.":3
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
    return ''


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
        'Recommended_Savings_Percentage': [5, 10, 15, 20,
                                        5, 10, 15, 20,
                                        8, 12, 18, 25,
                                        3, 5, 8, 10,
                                        2, 5, 8, 10,
                                        5, 10, 15, 25]
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
    '6-10 years':96,
    '11-20 years':186,
    '20 years and up':240
})
    
# Function for Savings Percentage Based on Target EFund Amount and Goal Duration:

def get_required_monthly_savings(ui_savings_percentage_df):
    user_target_efund_amount=ui_savings_percentage_df['Target_Efund_Amount'].values[0]
    
    # Have the Goal_Duration mapped first
    user_goal_duration=map_user_goal_duration(ui_savings_percentage_df)

    # Then assign the value of the mapped Goal Duration
    user_goal_duration = ui_savings_percentage_df['Goal_Duration'].values[0]
    print(f"USER MAPPED GOAL DURATION:\n{user_goal_duration}") # For checking


    required_monthly_savings = user_target_efund_amount/user_goal_duration.round(2)

    return required_monthly_savings, user_goal_duration

# Function for determining the Final Savings Percentage
def determine_final_savings_percentage(user_required_monthly_savings, recommended_savings_amount, 
                                       saving_percentage_recommendation, ui_cbf_saving_percentage_df, 
                                       user_average_monthly_cashflow):

    final_recommended_savings_percentage = 0
    recommendation_message = ""

    # Determine Final Recommended Savings Percentage
    if user_required_monthly_savings <= recommended_savings_amount:
        final_recommended_savings_percentage = saving_percentage_recommendation
        recommendation_message = "Great! Your savings goal fits well with your financial situation. You can proceed with the recommended savings plan."

    else:
        # Get User's Monthly Expense
        user_monthly_expense = ui_cbf_saving_percentage_df['Average_Monthly_Expense'].values[0]

        # Calculate User's Max Savings Capacity
        user_max_saving_capacity = user_average_monthly_cashflow - user_monthly_expense
        print(f"\nUSER'S MAX SAVINGS CAPACITY:\n{user_max_saving_capacity}\n") # For checking


        # Calculate Max Possible Savings Percentage
        user_max_possible_savings_percentage = round((user_max_saving_capacity / user_average_monthly_cashflow) * 100, 2)
        print(f"\nUSER'S MAX SAVINGS PERCENTAGE:\n{user_max_possible_savings_percentage}\n") # For checking

        if user_required_monthly_savings <= user_max_saving_capacity:
            if user_max_possible_savings_percentage <= 20:
                final_recommended_savings_percentage = user_max_possible_savings_percentage
                recommendation_message = "You’re on track! This savings percentage works well for your finances."
            else:
                final_recommended_savings_percentage = user_max_possible_savings_percentage
                recommendation_message = "You can save this amount, but it's higher than the ideal 20%. We suggest adjusting your budget or timeline to ensure it won’t strain your budget."
        else:
            recommendation_message = "It looks like your required monthly savings to reach your emergency fund goal within your set duration is higher than what you can comfortably set aside each month."
            final_recommended_savings_percentage = user_max_possible_savings_percentage  

    return final_recommended_savings_percentage, recommendation_message


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

                             # THIS IS FOR GETTING THE RECOMMENDED SAVINGS PERCENTAGE NA KASAMA NA YUNG TARGET EFUND AND GOAL DURATION: 

        # Check the user's Recommended Saving's Percentage Based on their Target Emergency Fund Amount and Goal Duration: 
        # Formula: Target Efnf Amount / Goal Duration:
        user_required_monthly_savings, user_goal_duration=get_required_monthly_savings(ui_efund_and_goal_dur_df)
        print(f"USER'S REQUIRED MONTHLY SAVINGS:\n{user_required_monthly_savings}\n") # For checking
        print(f"USER'S MAPPED GOAL DURATION:\n{user_goal_duration}\n") # For checking


        # To get the exact amount of the recommended Savings Amount using the System's Generated Recommended Savings Percentage: 
        # Formula is: recommended_savings_amount = saving_percentage_recommendation * user's monthly income

        user_average_monthly_cashflow = ui_cbf_saving_perc_data.get('Average_Monthly_Cashflow', 0) # To get the User's Monthly Cashflow
        print(f"\nUSER'S AVERAGE MONTHLY CASHFLOW:\n{user_average_monthly_cashflow}\n") # For checking

        recommended_savings_amount=(saving_percentage_recommendation/100)*user_average_monthly_cashflow
        print(f"\nUSER'S RECOMMENDED SAVINGS AMOUNT:\n{recommended_savings_amount}\n") # For checking

       # Call the function to determine the final recommended savings percentage
        final_recommended_savings_percentage, recommendation_message= determine_final_savings_percentage(
            user_required_monthly_savings, recommended_savings_amount, 
            saving_percentage_recommendation, ui_cbf_saving_percentage_df, 
            user_average_monthly_cashflow
        )

     
        print(f"\nFINAL RECOMMENDED SAVINGS PERCENTAGE:\n{final_recommended_savings_percentage}\n Message \n{recommendation_message}\n") # For checking

        # Return JSON response
        return jsonify({
            "Recommended Saving Percentage": final_recommended_savings_percentage,
            "Message": recommendation_message,
            "CBF Recommended Savings Percentage":saving_percentage_recommendation,
            "Goal Duration": user_goal_duration
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

                                             # For Further Tailored  Recommendations  (Content-Based Filtering):
        # Note: This recommendations are the ones that are based on the ff: Saving_Strategy, Goal_Duration, Income_Level, and Cashflow_Frequency

# Defining the matrix that will be used for the further tailored recommendations  (ITEM-FEATURES):
def define_recommendations_matrix():
    cbf_recommendation_data = {
        'Saving_Strategy': ['Proportional', 'Proportional', 'Proportional',  # For Proportional
                            'Envelope Method', 'Envelope Method', 'Envelope Method',  #For Envelope Method
                            'Pay Yourself First', 'Pay Yourself First', 'Pay Yourself First'],  #For PYF

        'Goal_Duration': ['Short Term', 'Medium Term', 'Long Term',
                        'Short Term', 'Medium Term', 'Long Term',
                        'Short Term', 'Medium Term', 'Long Term'],
        
        'Income_Level': ['Low', 'Medium', 'High',
                        'Low', 'Medium', 'High',
                        'Low', 'Medium', 'High'],
        'Cashflow_Frequency': ['Low', 'Medium', 'High',
                            'Low', 'Medium', 'High',
                            'Low', 'Medium', 'High'],
        'Recommendation': [
            "Start by saving a small, fixed proportion (e.g., 10-20%) of your income. Prioritize emergency savings first.",
            "Increase the proportion of your income dedicated to your emergency fund, aiming for at least 20% to grow it faster.",
            "Save a higher proportion of your income (e.g., 10 to 20%) for long-term financial security, with an emphasis on building your emergency fund first.",
            
            "Allocate small amounts to essential categories (e.g., groceries, utilities), while gradually increasing contributions to your emergency fund as your income grows.",
            "Use the envelope system to allocate more towards savings for your emergency fund while balancing essential needs. Automate your savings if possible.",
            "Fully automate the envelope system for both expenses and savings, prioritizing your emergency fund as the primary goal.",
            
            "Set aside a fixed amount first (even if small) to ensure you're building an emergency fund before spending on other expenses.",
            "Increase the amount you set aside for your emergency fund first, leaving the rest for expenses. Focus on reaching your emergency fund goal as soon as possible.",
            "Save a fixed, larger portion of your income (at least 20%) upfront for your emergency fund. Automate the process to ensure consistency in saving."
        ]
    }
    cbf_recommendation_df=pd.DataFrame(cbf_recommendation_data)

    return cbf_recommendation_df

# For Preprocessing the Item-Features Matrix: 
def preprocess_cbf_recommendations(cbf_recommendation_data_df):

    # For Ordinal Features
    ordinal_cbf_recommendations=['Cashflow_Frequency','Income_Level'] # Ordinal Columns
    ordinal_cbf_feature_ranks = [
        ['Low', 'Medium', 'High'],  # Cashflow_Frequency ranking
        ['Low', 'Medium', 'High']   # Income_Level ranking
    ]
    # Initliaze the Ordinal Encoder:
    ordinal_cbf_reco_encoder=OrdinalEncoder(categories=ordinal_cbf_feature_ranks)

    # Apply the Ordinal Encoder: 
    cbf_recommendation_data_df[ordinal_cbf_recommendations]=ordinal_cbf_reco_encoder.fit_transform( cbf_recommendation_data_df[ordinal_cbf_recommendations])

    # Check the Encoded Ordinal DF:
    print(f"ENCODED ORDINAL CBF RECOMMENDATIONS:\n{ cbf_recommendation_data_df[ordinal_cbf_recommendations]}\n") # For checking

    # For Nominal Features:
    nominal_cbf_recommendations=['Saving_Strategy','Goal_Duration']
    saving_strategy_categories=['Envelope Method','Proportional','Pay Yourself First']    
    goal_duration_categories = ['Short Term', 'Medium Term', 'Long Term']

     # Initialize OneHotEncoder with predefined categories
    nominal_cbf_reco_encoder = OneHotEncoder(categories=[saving_strategy_categories, goal_duration_categories], 
                                             sparse_output=False)

    # Fit and transform all nominal features at once
    encoded_nominal = nominal_cbf_reco_encoder.fit_transform(cbf_recommendation_data_df[nominal_cbf_recommendations])

    # Convert to DataFrame with proper column names
    encoded_nominal_df = pd.DataFrame(encoded_nominal, 
                                      columns=nominal_cbf_reco_encoder.get_feature_names_out(nominal_cbf_recommendations))

    # Merge with original dataset (dropping the original nominal columns)
    cbf_recommendation_data_df = pd.concat([cbf_recommendation_data_df.drop(columns=nominal_cbf_recommendations), 
                                            encoded_nominal_df], axis=1)

    print(f"\n Preprocessed Recommendations DataFrame \n{cbf_recommendation_data_df}")
    
    return cbf_recommendation_data_df


def map_for_user_cbf_recommendations(user_cbf_recommendations_df):

    user_cbf_recommendations_df['Income_Level']=user_cbf_recommendations_df['Income_Level'].map({
    'Less than ₱5,000': 'Low',
    '₱5,000 - ₱9,999': 'Medium',
    '₱10,000 - ₱15,000': 'Medium',
    'More than ₱15,000': 'High'
    })

    user_cbf_recommendations_df['Cashflow_Frequency']=user_cbf_recommendations_df['Cashflow_Frequency'].map({
    'Bi-Weekly (every two (2) weeks)':'High',
    'Weekly':'High',
    'Monthly':'Medium',
    'Quarterly (every three (3) months)':'Low'
})
    user_cbf_recommendations_df['Goal_Duration']=user_cbf_recommendations_df['Goal_Duration'].map({
    '1-2 months':'Short Term',
    '3-5 months':'Short Term',
    '6-12 months':'Short Term',
    '1-2 years':'Short Term',
    '3-5 years':'Medium Term',
    '6-10 years':'Medium Term',
    '11-20 years':'Long Term',
    '20 years and up':'Long Term'
    })

    return user_cbf_recommendations_df


# App route for recommending 
@app.route('/recommend',methods=['POST','GET'])

def recommendations ():
    print("POST request received")    

                                                        # ITEM- FEATURES    
    
    # Define/Generate the Recommendation Matrix:
    cbf_recommendations_df=define_recommendations_matrix()
    print(f"ITEM FEATURES FOR RECOMMENDATIONS:\n{cbf_recommendations_df}\n") # For checking

    # Preprocess the Matrix:
    preprocessed_cbf_recommendations_df=preprocess_cbf_recommendations(cbf_recommendations_df)
    print(f"PREPROCESSED ITEM FEATURES FOR RECOMMENDATIONS:\n{preprocessed_cbf_recommendations_df}\n") # For checking

    # Storing of the Preprocessed cbf_recommendation_df to cbf_recommendations_df:
    cbf_recommendations_df=preprocessed_cbf_recommendations_df
    print(f"FINAL ITEM FEATURES FOR RECOMMENDATIONS:\n{cbf_recommendations_df}\n") # For checking

    # Dropping the "Recommendations" column since this will be the one that will be used as the matrix to compute for the Cosine Similarity:
    cbf_recommendations_matrix = cbf_recommendations_df.drop(columns=['Recommendation'])

    # Converting it to array:
    cbf_recommendations_matrix=cbf_recommendations_matrix.to_numpy()
    print(f"ITEM FEATURES MATRIX RECOMMENDATIONS:\n{cbf_recommendations_matrix}\n") # For checking



                                     # USER DATA (User's input from the Android app (for the recommendations) ):
    try:
        ui_cbf_recommendations_data=request.get_json()
        print(f"USER INPUT FOR RECOMMENDATIONS:\n{ui_cbf_recommendations_data}\n") # For checking

        # Convert it into a Dataframe: 
        ui_cbf_recommendations_df=pd.DataFrame([ui_cbf_recommendations_data])
        print(f"USER INPUT DATAFRAME FOR RECOMMENDATIONS:\n{ui_cbf_recommendations_df}\n") # For checking

        # Categorize User's Average Monthly Cashflow Input: 
        ui_cbf_recommendations_df['Income_Level']= ui_cbf_recommendations_df['Income_Level'].apply(categorize_user_cashflow_input)
        print(f"USER INPUT CATEGORIZED CASHFLOW FOR RECOMMENDATIONS:\n{ui_cbf_recommendations_df}\n") # For checking


        # Map the user's input:
        mapped_ui_cbf_recommendations=map_for_user_cbf_recommendations(ui_cbf_recommendations_df)
        print(f"MAPPED USER INPUT FOR RECOMMENDATIONS:\n{mapped_ui_cbf_recommendations}\n") # For checking

        # Preprocess the user's input:
        preprocessed_user_cbf_recommendations_df=preprocess_cbf_recommendations(mapped_ui_cbf_recommendations)
        print(f"PREPROCESSED USER INPUT FOR RECOMMENDATIONS:\n{preprocessed_user_cbf_recommendations_df}\n") # For checking

        # Storing the preprocessed user df to ui_cbf_recommendations_df:
        ui_cbf_recommendations_df=preprocessed_user_cbf_recommendations_df
        print(f"FINAL USER INPUT FOR RECOMMENDATIONS:\n{ui_cbf_recommendations_df}\n") # For checking

        # Converting Datadframe to array:
        ui_cbf_recommendations_matrix=ui_cbf_recommendations_df
        ui_cbf_recommendations_matrix=ui_cbf_recommendations_matrix.to_numpy()

        print(f"USER INPUT MATRIX FOR RECOMMENDATIONS:\n{ui_cbf_recommendations_matrix}\n") # For checking

        cbf_recommendation_similarities_matrix=cosine_similarity(ui_cbf_recommendations_matrix,cbf_recommendations_matrix)

        cbf_recommendations_best_match=cbf_recommendation_similarities_matrix.argmax()
        cbf_recommendation=cbf_recommendations_df.iloc[cbf_recommendations_best_match]['Recommendation']

        print(f"GENERATED RECOMMENDATIONS: {cbf_recommendation}")

        return jsonify({
            "Recommendation":cbf_recommendation
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    


if __name__=='__main__':
    app.run(debug=True,host='0.0.0.0')