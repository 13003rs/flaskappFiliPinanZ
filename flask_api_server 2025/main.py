from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from joblib import load
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics.pairwise import cosine_similarity


app=Flask(__name__)

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
    return 'Hello'


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


# -- Function for mapping user data for recommendation 
def map_user_data(cbf_user_df):
    cbf_user_df['Average_Monthly_Cashflow']=cbf_user_df['Average_Monthly_Cashflow'].map({
    'Less than ₱5,000': 'Low',
    '₱5,000 - ₱10,000': 'Medium',
    '₱10,000 - ₱15,000': 'Medium',
    'More than ₱15,000': 'High'
})
    cbf_user_df['Cashflow_Frequency']=cbf_user_df['Cashflow_Frequency'].map({
    'Bi-Weekly (every two (2) weeks)':'High',
    'Weekly':'High',
    'Monthly':'Medium',
    'Quarterly (every three (3) months)':'Low'
})
    cbf_user_df['Goal_Duration']=cbf_user_df['Goal_Duration'].map({
    '1-2 months':'Short Term',
    '3-5 months':'Short Term',
    '6-12 months':'Short Term',
    '1-2 years':'Short Term',
    '3-5 years':'Medium Term',
    '6-10 years':'Medium Term',
    '11-20 years':'Long Term',
    '20 years and up':'Long Term'
})

    return cbf_user_df

# -- Function for preprocessing user data for recommendation 
def preprocess_user_df(mapped_user_df):
    # encoding ordinal features
    user_ordinal_ranks=[['Low','Medium','High']]
    user_cbf_ordinal_features=['Average_Monthly_Cashflow','Cashflow_Frequency']

    ordinal_encoder = OrdinalEncoder(categories=user_ordinal_ranks * len(user_cbf_ordinal_features)) 

    mapped_user_df[user_cbf_ordinal_features]=ordinal_encoder.fit_transform(mapped_user_df[user_cbf_ordinal_features])

# For saving strategy 
    saving_strategy_categories=['Envelope Method','Proportional','Pay Yourself First']
    onehot_saving_strat_encoder = OneHotEncoder(categories=[saving_strategy_categories], sparse_output=False, handle_unknown='ignore')

    mapped_user_df['Saving_Strategy'] = mapped_user_df['Saving_Strategy'].str.strip().str.title()

    # Perform one-hot encoding for 'Saving_Strategy'
    encoded_user_saving_strategy = onehot_saving_strat_encoder.fit_transform(mapped_user_df[['Saving_Strategy']])

    encoded_user_saving_strat_df = pd.DataFrame(encoded_user_saving_strategy, columns=onehot_saving_strat_encoder.get_feature_names_out(['Saving_Strategy']))

    # Concatenate the one-hot encoded DataFrame with the original DataFrame
    mapped_user_df = pd.concat([mapped_user_df, encoded_user_saving_strat_df], axis=1)
    mapped_user_df.drop(columns=['Saving_Strategy'], inplace=True)

# For Goal Duration
    goal_duration_categories = ['Short Term', 'Medium Term', 'Long Term']

    # Clean the Goal_Duration column to ensure consistent formatting
    mapped_user_df['Goal_Duration'] = mapped_user_df['Goal_Duration'].str.strip().str.title()

    # Initialize OneHotEncoder with predefined categories
    onehot_goal_duration_encoder = OneHotEncoder(categories=[goal_duration_categories], sparse_output=False, handle_unknown='ignore')

    # Perform one-hot encoding for 'Goal_Duration'
    encoded_user_goal_duration = onehot_goal_duration_encoder.fit_transform(mapped_user_df[['Goal_Duration']])

    encoded_user_goal_duration_df = pd.DataFrame(encoded_user_goal_duration, columns=onehot_goal_duration_encoder.get_feature_names_out(['Goal_Duration']))

    # Concatenate the one-hot encoded DataFrame with the original DataFrame
    mapped_user_df = pd.concat([mapped_user_df, encoded_user_goal_duration_df], axis=1)

    # Drop the original 'Goal_Duration' column after encoding
    mapped_user_df.drop(columns=['Goal_Duration'], inplace=True)



    return mapped_user_df

# App route for recommending 
@app.route('/recommend',methods=['POST','GET'])

def recommend():
   
    print("POST request received")    

# Defining the recommendation data in a structured format
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
            "Save a higher proportion of your income (e.g., 30% or more) for long-term financial security, with an emphasis on building your emergency fund first.",
            
            "Allocate small amounts to essential categories (e.g., groceries, utilities), while gradually increasing contributions to your emergency fund as your income grows.",
            "Use the envelope system to allocate more towards savings for your emergency fund while balancing essential needs. Automate your savings if possible.",
            "Fully automate the envelope system for both expenses and savings, prioritizing your emergency fund as the primary goal.",
            
            "Set aside a fixed amount first (even if small) to ensure you're building an emergency fund before spending on other expenses.",
            "Increase the amount you set aside for your emergency fund first, leaving the rest for expenses. Focus on reaching your emergency fund goal as soon as possible.",
            "Save a fixed, larger portion of your income (at least 20%) upfront for your emergency fund. Automate the process to ensure consistency in saving."
        ]
    }
#---------------- ITEM-FEATURES
    try:
        cbf_recommendation_df = pd.DataFrame(cbf_recommendation_data)
        print(f"Columns of ITEM FEATURE MATRIX{cbf_recommendation_df.columns}")

        # Define the features and their rankings
        ordinal_cbf_features = ['Cashflow_Frequency', 'Income_Level']
        ordinal_feature_ranks = [['Low', 'Medium', 'High']]  

        # Initialize the OrdinalEncoder
        ordinal_encoder = OrdinalEncoder(categories=ordinal_feature_ranks * len(ordinal_cbf_features))  # Extend ranks for each feature

        # Apply encoding to the DataFrame
        cbf_recommendation_df[ordinal_cbf_features] = ordinal_encoder.fit_transform(cbf_recommendation_df[ordinal_cbf_features])

        # Display the updated DataFrame
        print(cbf_recommendation_df)

        oneh_encoder=OneHotEncoder(sparse_output=False)

# Encoding nominal features (item features)
        nominal_features=['Saving_Strategy','Goal_Duration']

        for column in nominal_features:
            encoded_nominal=oneh_encoder.fit_transform(cbf_recommendation_df[[column]])
            encoded_nominal_df=pd.DataFrame(encoded_nominal, columns=oneh_encoder.get_feature_names_out([column]))
                                            
            cbf_recommendation_df=pd.concat([cbf_recommendation_df,encoded_nominal_df],axis=1)
            cbf_recommendation_df.drop(columns=[column], inplace=True)

        print(f"\n Encoded item features:\n{cbf_recommendation_df}\n")

        # Generate item-features matrix
        cbf_recommendation_matrix=cbf_recommendation_df.drop(columns=['Recommendation'])
        cbf_recommendation_matrix=cbf_recommendation_matrix.to_numpy()
        print(f"\n Item-features matrix:\n{cbf_recommendation_matrix}\n")


    # User input data (from android)
        cbf_user_data=request.get_json()   # to get the input data from the POST request
        print(f"\nUser data: \n{cbf_user_data}\n")
        # To convert the data into a dataframe
        cbf_user_df=pd.DataFrame([cbf_user_data])

        # for checking
        print(f"\nDATAFRAME of USER CBF: \n{cbf_user_df} \nDataframe CBF columns: \n {cbf_user_df.columns}\n")

        mapped_cbf_user_df=map_user_data(cbf_user_df)
        print(f"\n Mapped user data:\n{mapped_cbf_user_df}\n")

        preprocessed_user_df_reco=preprocess_user_df(mapped_cbf_user_df)
        print(f"\n Preprocessed user data:\n{preprocessed_user_df_reco}\n")

        cbf_user_rec_matrix=preprocessed_user_df_reco.to_numpy()
        print(f"\n User matrix:\n{cbf_user_rec_matrix}\n")


        similarities_matrix = cosine_similarity(cbf_user_rec_matrix,cbf_recommendation_matrix)

        best_match_idx = similarities_matrix.argmax()  
        recommendation = cbf_recommendation_df.iloc[best_match_idx]['Recommendation']

        print(f"\nRecommendation: {recommendation}")

        return jsonify({
            "Recommendation":recommendation
            })
    except Exception as e:
         return jsonify({"error":str(e)}),400

#------ FOR SAVING PERCENTAGE RECOMMENDATION -----

def map_user_cbf_saving_perc(user_cbf_savingperc_df):
    user_cbf_savingperc_df['Average_Monthly_Cashflow'] = user_cbf_savingperc_df['Average_Monthly_Cashflow'].map({
        'Less than ₱5,000': 'Low',
        '₱5,000 - ₱10,000': 'Medium',
        '₱10,000 - ₱15,000': 'High',
        'More than ₱15,000': 'Very High'
    })
    user_cbf_savingperc_df['Cashflow_Frequency'] = user_cbf_savingperc_df['Cashflow_Frequency'].map({
        'Bi-Weekly (every two (2) weeks)': 'High',
        'Weekly': 'Very High',
        'Monthly': 'Medium',
        'Quarterly (every three (3) months)': 'Low'
    })
    return user_cbf_savingperc_df

def preprocess_user_saving_perc_df(mapped_cbf_user_saving_df):
    avg_cashflow_ranks_user = [['Low', 'Medium', 'High', 'Very High']]
    ordinal_avg_cashflow_encoder_user = OrdinalEncoder(categories=avg_cashflow_ranks_user)
    mapped_cbf_user_saving_df['Average_Monthly_Cashflow'] = ordinal_avg_cashflow_encoder_user.fit_transform(mapped_cbf_user_saving_df[['Average_Monthly_Cashflow']])
    
    user_cashflow_freq_ranks = [['Low', 'Medium', 'High', 'Very High']]
    user_ordinal_cashflow_freq_encoder = OrdinalEncoder(categories=user_cashflow_freq_ranks)
    mapped_cbf_user_saving_df['Cashflow_Frequency'] = user_ordinal_cashflow_freq_encoder.fit_transform(mapped_cbf_user_saving_df[['Cashflow_Frequency']])
    
    user_employment_status_categories = ['Full-time Employee', 'Full-time Student', 'Part-time', 'Self-Employed', 'Unemployed', 'Working Student']
    user_onehot_employment_encoder = OneHotEncoder(categories=[user_employment_status_categories], sparse_output=False, handle_unknown='ignore')
    encoded_user_employment = user_onehot_employment_encoder.fit_transform(mapped_cbf_user_saving_df[['Employment_Status']])
    encoded_user_employment_df = pd.DataFrame(encoded_user_employment, columns=user_onehot_employment_encoder.get_feature_names_out(['Employment_Status']))
    mapped_cbf_user_saving_df = pd.concat([mapped_cbf_user_saving_df, encoded_user_employment_df], axis=1)
    mapped_cbf_user_saving_df.drop(columns=['Employment_Status'], inplace=True)
    
    return mapped_cbf_user_saving_df

@app.route('/recommend_saving_percentage',methods=['POST','GET'])
def recommend_saving_percentage():

#--------------------   FOR ITEM -FEATURES
    cbf_saving_percentage_data = {
        'Employment_Status': ['Working Student', 'Working Student', 'Working Student', 'Working Student',
                            'Part-time', 'Part-time', 'Part-time', 'Part-time',
                            'Full-time Employee', 'Full-time Employee', 'Full-time Employee', 'Full-time Employee',
                            'Full-time Student', 'Full-time Student', 'Full-time Student', 'Full-time Student',
                            'Unemployed', 'Unemployed', 'Unemployed', 'Unemployed',
                            'Self-Employed', 'Self-Employed', 'Self-Employed', 'Self-Employed'],
        'Average_Monthly_Cashflow':['Less than ₱5,000', '₱5,000 - ₱10,000', '₱10,000 - ₱15,000','More than ₱15,000',
                                'Less than ₱5,000', '₱5,000 - ₱10,000', '₱10,000 - ₱15,000', 'More than ₱15,000',
                                'Less than ₱5,000', '₱5,000 - ₱10,000', '₱10,000 - ₱15,000', 'More than ₱15,000',
                                'Less than ₱5,000', '₱5,000 - ₱10,000', '₱10,000 - ₱15,000', 'More than ₱15,000',
                                'Less than ₱5,000', '₱5,000 - ₱10,000', '₱10,000 - ₱15,000', 'More than ₱15,000',
                                'Less than ₱5,000', '₱5,000 - ₱10,000', '₱10,000 - ₱15,000', 'More than ₱15,000'],
        'Cashflow_Frequency': ['Low', 'Medium', 'High', 'Very High',
                            'Low', 'Medium', 'High', 'Very High',
                            'Low', 'Medium', 'High', 'Very High',
                            'Low', 'Medium', 'High', 'Very High',
                            'Low', 'Medium', 'High', 'Very High',
                            'Low', 'Medium', 'High', 'Very High'],
        'Recommended_Savings_Percentage':[5, 10, 15, 20,
                                        5, 10, 15, 20,
                                        8, 12, 18, 25,
                                        3, 5, 8, 10,
                                        2, 5, 8, 10,
                                        5, 10, 15,25]
    }

    # Create DataFrame to store the criteria dictionary
    cbf_saving_percent_df_reco=pd.DataFrame(cbf_saving_percentage_data)
    print(f"\ncbf_saving_percent_df_reco \n{cbf_saving_percent_df_reco}")

    cbf_saving_percent_df_reco['Average_Monthly_Cashflow'] = cbf_saving_percent_df_reco['Average_Monthly_Cashflow'].map({
        'Less than ₱5,000': 'Low',
        '₱5,000 - ₱10,000': 'Medium',
        '₱10,000 - ₱15,000': 'High',
        'More than ₱15,000': 'Very High'
    })
    print(f"\n mapped_cbf_savingstrat_df \n{cbf_saving_percent_df_reco}")

    avg_cashflow_ranks = [['Low', 'Medium', 'High', 'Very High']]
    ordinal_avg_cashflow_encoder = OrdinalEncoder(categories=avg_cashflow_ranks)
    cbf_saving_percent_df_reco['Average_Monthly_Cashflow'] = ordinal_avg_cashflow_encoder.fit_transform(cbf_saving_percent_df_reco[['Average_Monthly_Cashflow']])
    
    cashflow_freq_ranks = [['Low', 'Medium', 'High', 'Very High']]
    ordinal_cashflow_freq_encoder = OrdinalEncoder(categories=cashflow_freq_ranks)
    cbf_saving_percent_df_reco['Cashflow_Frequency'] = ordinal_cashflow_freq_encoder.fit_transform(cbf_saving_percent_df_reco[['Cashflow_Frequency']])
    
    onehot_encoder_employment = OneHotEncoder(sparse_output=False)
    encoded_employment_status = onehot_encoder_employment.fit_transform(cbf_saving_percent_df_reco[['Employment_Status']])
    encoded_employment_df = pd.DataFrame(encoded_employment_status, columns=onehot_encoder_employment.get_feature_names_out(['Employment_Status']))
    cbf_saving_percent_df_reco = pd.concat([cbf_saving_percent_df_reco, encoded_employment_df], axis=1)
    cbf_saving_percent_df_reco.drop(columns=['Employment_Status'], inplace=True)

    print(f"\n Preprocessed Saving Percent Reco DataFrame \n{cbf_saving_percent_df_reco}")

    cbf_saving_percent_reco_matrix=cbf_saving_percent_df_reco.drop(columns=['Recommended_Savings_Percentage'])
    cbf_saving_percent_reco_matrix=cbf_saving_percent_reco_matrix.to_numpy()

    print("POST request received")   

#--------------------- USER INPUT (from android)
    try:
      user_saving_perc_data = request.get_json()
      user_cbf_savingperc_df = pd.DataFrame([user_saving_perc_data])
      print(f"\n User Input  Saving Percent Reco DataFrame \n{user_cbf_savingperc_df}")

      # Mapped user input
      mapped_cbf_user_saving_df = map_user_cbf_saving_perc(user_cbf_savingperc_df)
      print(f"\n User Mapped Saving Percent Reco DataFrame \n{mapped_cbf_user_saving_df}")

    # Preprocessed user input 
      preprocessed_user_saving_perc_df = preprocess_user_saving_perc_df(mapped_cbf_user_saving_df)
      print(f"\n User Preprocessed Saving Percent Reco DataFrame \n{mapped_cbf_user_saving_df}")


      user_cbf_saving_perc_matrix = preprocessed_user_saving_perc_df
      user_cbf_saving_perc_matrix.to_numpy()
      print(f"\n User Input Saving Percent Reco Matrix \n{mapped_cbf_user_saving_df}")


      saving_percentage_similarities_matrix = cosine_similarity(user_cbf_saving_perc_matrix, cbf_saving_percent_reco_matrix)
      
      best_match_saving_percentage = saving_percentage_similarities_matrix.argmax()
      saving_percentage_recommendation = cbf_saving_percent_df_reco.iloc[best_match_saving_percentage]['Recommended_Savings_Percentage']
      print(f"\n Recommended Saving Percent\n{saving_percentage_recommendation}")

      return jsonify({"Recommended Saving Percentage": saving_percentage_recommendation})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400
 

if __name__=='__main__':
    app.run(debug=True,host='0.0.0.0')