import pandas as pd
import glob
import os
import re

def generate_ensemble():
    # 1. Setup: Automatically find files and determine weights
    print("Scanning for submission files...")
    
    submissions = []
    # Find all files matching the pattern "submission_*.csv"
    files = glob.glob('./old/submission_*.csv')
    
    if not files:
        print("No 'submission_*.csv' files found in the current directory.")
        return

    for filename in files:
        # Don't include the output file if it already exists to avoid recursion
        if filename == 'submission_ensemble_weighted.csv':
            continue

        # Extract the score from the filename using regex
        # Pattern: looks for 'submission_' followed by digits
        match = re.search(r'submission_(\d+)', filename)
        
        if match:
            score = int(match.group(1))
            weight = score / 100.0
            submissions.append({'file': filename, 'weight': weight})
            print(f"Found {filename} -> Weight: {weight}")
        else:
            print(f"Skipping {filename}: Could not extract score from filename.")

    if not submissions:
        print("No valid submission files found.")
        return

    print(f"\nLoaded {len(submissions)} files for ensemble.")
    
    # We will store the weighted votes in a dataframe
    # We load the first file just to get the 'id' column layout
    try:
        base_df = pd.read_csv(submissions[0]['file'])
        # Create a new DataFrame to hold the aggregate score
        # Assuming the ID column is the first column
        ensemble_df = pd.DataFrame()
        ensemble_df['id'] = base_df.iloc[:, 0]
        ensemble_df['weighted_score_1'] = 0.0 # Score for class 1 (Disease)
        ensemble_df['weighted_score_0'] = 0.0 # Score for class 0 (Healthy)
    except Exception as e:
        print(f"Error reading first file: {e}")
        return

    # 2. Iterate through files and vote
    for sub in submissions:
        filename = sub['file']
        weight = sub['weight']
        
        print(f"Processing {filename}...")
        try:
            df = pd.read_csv(filename)
            
            # Look for the prediction column (usually the last column or named 'target'/'prediction')
            # We assume it's the second column for standard submission files
            pred_col = df.columns[1] 
            
            # LOGIC:
            # If model predicts 1, we add 'weight' to score_1
            # If model predicts 0, we add 'weight' to score_0
            
            # Add weight where prediction is 1
            ensemble_df.loc[df[pred_col] == 1, 'weighted_score_1'] += weight
            
            # Add weight where prediction is 0
            ensemble_df.loc[df[pred_col] == 0, 'weighted_score_0'] += weight
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # 3. Final Decision
    print("\nCalculating consensus...")
    
    # Create final prediction column
    # If score_1 > score_0, predict 1. Otherwise predict 0.
    ensemble_df['final_prediction'] = (ensemble_df['weighted_score_1'] > ensemble_df['weighted_score_0']).astype(int)
    
    # 4. Save
    output_filename = 'submission_100.csv'
    
    # Prepare final submission format (ID + Prediction)
    final_submission = pd.DataFrame()
    final_submission[base_df.columns[0]] = ensemble_df['id']
    final_submission[base_df.columns[1]] = ensemble_df['final_prediction']
    
    final_submission.to_csv(output_filename, index=False)
    
    print(f"\nSuccess! Ensemble generated: {output_filename}")
    
    # Optional: Analysis
    # Let's see how many times the ensemble disagreed with the best model (assumed to be the one with highest weight)
    try:
        # Sort submissions by weight to find the best one
        best_sub = sorted(submissions, key=lambda x: x['weight'], reverse=True)[0]
        print(f"\nComparing against best single model: {best_sub['file']} ({best_sub['weight']})")
        
        best_model = pd.read_csv(best_sub['file'])
        best_col = best_model.columns[1]
        diff = (final_submission.iloc[:, 1] != best_model[best_col]).sum()
        print(f"Analysis: The ensemble changed {diff} predictions compared to the best single file.")
    except:
        pass

if __name__ == "__main__":
    generate_ensemble()