import os
import numpy as np
import pandas as pd

def generate_credit_dataset(num_samples=2500, random_state=42):
    np.random.seed(random_state)

    applicant_ids = [f"ACC-{10000 + i}" for i in range(num_samples)]
    genders = np.random.choice(['Female', 'Male'], size=num_samples, p=[0.55, 0.45])
    ages = np.random.randint(21, 66, size=num_samples)
    marital_statuses = np.random.choice(['Married', 'Single', 'Civil Union', 'Separated'], size=num_samples, p=[0.58, 0.28, 0.08, 0.06])
    housing_types = np.random.choice(['Own House', 'Rented Apartment', 'With Parents', 'Municipal Housing'], size=num_samples, p=[0.65, 0.20, 0.10, 0.05])
    education_levels = np.random.choice(['Secondary / Secondary Special', 'Higher Education', 'Incomplete Higher', 'Academic Degree'], size=num_samples, p=[0.60, 0.32, 0.06, 0.02])
    income_types = np.random.choice(['Working', 'Commercial Associate', 'State Servant', 'Pensioner'], size=num_samples, p=[0.52, 0.25, 0.13, 0.10])
    
    # Income based on income type and education
    base_incomes = []
    for inc_t, edu in zip(income_types, education_levels):
        inc = np.random.normal(65000, 22000)
        if edu == 'Higher Education':
            inc += 25000
        elif edu == 'Academic Degree':
            inc += 45000
        if inc_t == 'Commercial Associate':
            inc += 15000
        elif inc_t == 'Pensioner':
            inc -= 20000
        base_incomes.append(max(18000, round(inc, -2)))
    
    annual_incomes = np.array(base_incomes)
    
    # Employment duration based on age
    employment_durations = []
    for age in ages:
        max_emp = max(0.5, age - 20)
        emp = np.random.uniform(0.5, max_emp) if max_emp > 0.5 else 0.5
        employment_durations.append(round(emp, 1))
    employment_durations = np.array(employment_durations)
    
    family_members = np.random.choice([1, 2, 3, 4, 5], size=num_samples, p=[0.25, 0.40, 0.20, 0.10, 0.05])
    existing_loans = np.random.choice([0, 1, 2, 3, 4], size=num_samples, p=[0.40, 0.35, 0.15, 0.07, 0.03])
    
    # Credit Inquiries (0 to 8)
    credit_inquiries = np.random.choice([0, 1, 2, 3, 4, 5, 6], size=num_samples, p=[0.35, 0.30, 0.18, 0.09, 0.05, 0.02, 0.01])
    
    # Past Due Records (0: None, 1: 1-29 days, 2: 30-59 days, 3: 60-89 days, 4: >90 days)
    past_due_records = np.random.choice([0, 1, 2, 3, 4], size=num_samples, p=[0.72, 0.16, 0.07, 0.03, 0.02])

    # Rule-based credit score & approval decision probability
    score = (
        (annual_incomes / 10000) * 1.8 +
        (employment_durations) * 1.5 -
        (past_due_records * 12.0) -
        (credit_inquiries * 4.5) -
        (existing_loans * 2.0) +
        np.where(housing_types == 'Own House', 4.0, 0.0) +
        np.where(education_levels == 'Higher Education', 5.0, 0.0) +
        np.where(education_levels == 'Academic Degree', 8.0, 0.0) +
        np.random.normal(0, 5, size=num_samples)
    )

    # Sigmoid function for approval probability
    prob = 1 / (1 + np.exp(-(score - 10) / 6.0))
    approval_status = (prob >= 0.48).astype(int)

    df = pd.DataFrame({
        'Applicant_ID': applicant_ids,
        'Gender': genders,
        'Age': ages,
        'Marital_Status': marital_statuses,
        'Housing_Type': housing_types,
        'Education_Level': education_levels,
        'Income_Type': income_types,
        'Annual_Income': annual_incomes,
        'Employment_Duration': employment_durations,
        'Family_Members': family_members,
        'Existing_Loans': existing_loans,
        'Credit_Inquiries': credit_inquiries,
        'Past_Due_Records': past_due_records,
        'Approval_Status': approval_status
    })

    # Save to data folder
    os.makedirs('data', exist_ok=True)
    csv_path = os.path.join('data', 'credit_card_approval_dataset.csv')
    df.to_csv(csv_path, index=False)
    print(f"Dataset successfully created at {csv_path} with {len(df)} samples.")
    print(f"Approval Distribution: Approved (1) = {sum(df['Approval_Status'] == 1)}, Rejected (0) = {sum(df['Approval_Status'] == 0)}")
    return df

if __name__ == '__main__':
    generate_credit_dataset()
