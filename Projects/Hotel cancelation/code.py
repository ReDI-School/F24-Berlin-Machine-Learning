import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# Load the dataset
file_path = r'Projects/Hotel cancelation/data/hotel_booking.csv'
data = pd.read_csv(file_path)

# Add an ID column for tracking rows
data['id'] = range(1, len(data) + 1)

# Evil modifications to the data
np.random.seed(42)

# 1. Shuffle values within columns
for col in ['lead_time', 'adr', 'stays_in_week_nights']:
    data[col] = data[col].sample(frac=1).values

# 2. Perturb numeric columns with heavy random noise
for col in ['lead_time', 'adr', 'stays_in_weekend_nights', 'stays_in_week_nights', 'days_in_waiting_list']:
    noise = np.random.normal(0, data[col].std() * 0.5, size=len(data))
    data[col] = data[col] + noise

# 3. Introduce misclassifications in the label column
mislabel_indices = np.random.choice(data.index, size=int(0.1 * len(data)), replace=False)  # Mislabel 10%
data.loc[mislabel_indices, 'is_canceled'] = 1 - data.loc[mislabel_indices, 'is_canceled']

# 4. Randomly replace some categorical values with nonsensical but plausible ones
for col in ['meal', 'country', 'market_segment', 'distribution_channel']:
    mask = np.random.rand(len(data)) < 0.2  # Replace 20% of values
    random_values = ['UNAVAILABLE', 'UNKNOWN', 'MISSING']
    data.loc[mask, col] = np.random.choice(random_values, size=mask.sum())

# 5. Modify 50% of 'January' to 'Veganuary' in 'arrival_date_month'
january_mask = data['arrival_date_month'] == 'January'
veganuary_mask = np.random.rand(len(data)) < 0.5
data.loc[january_mask & veganuary_mask, 'arrival_date_month'] = 'Veganuary'

# 6. Swap data between 'arrival_date_year' and 'arrival_date_week_number'
data['arrival_date_year'], data['arrival_date_week_number'] = (
    data['arrival_date_week_number'],
    data['arrival_date_year'],
)

# 7. Introduce missing values strategically
missing_mask = np.random.rand(len(data)) < 0.05  # 5% of rows
data.loc[missing_mask, ['lead_time', 'adr', 'meal', 'country']] = np.nan

# 8. Add believable synthetic features
data['estimated_check_in_duration'] = data['lead_time'] / 2 + np.random.normal(0, 10, size=len(data))
data['booking_difficulty_score'] = (
    data['days_in_waiting_list'] * 0.1
    + data['previous_cancellations'] * 0.5
    + data['adr'] * 0.01
    + np.random.normal(0, 1, size=len(data))
)

# 9. Add a fun Easter egg
easter_egg_indices = np.random.choice(data.index, size=1, replace=False)
data.loc[easter_egg_indices, 'name'] = 'Easter Bunny'
data.loc[easter_egg_indices, 'email'] = 'bunny@eggmail.com'
data.loc[easter_egg_indices, 'phone-number'] = '123-456-7890'
data.loc[easter_egg_indices, 'credit_card'] = 'EGG-EGG-EGG-EGG'

# Split into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42, stratify=data['is_canceled'])

# Separate the labels from the test set
test_data_no_labels = test_data.drop(columns=['is_canceled'])

# correct submission
solution = test_data[['id','is_canceled']]


# Create a sample submission file
sample_submission = test_data_no_labels[['id']].copy()
sample_submission['is_canceled'] = 0  # Default probability

# Save datasets
solution.to_csv('Projects/Hotel cancelation/solution.csv', index=False)
train_data.to_csv('Projects/Hotel cancelation/train.csv', index=False)
test_data_no_labels.to_csv('Projects/Hotel cancelation/test.csv', index=False)
sample_submission.to_csv('Projects/Hotel cancelation/sample_submission.csv', index=False)

print("Evil and fun challenge datasets created successfully!")
