import numpy as np
import pandas as pd

# Number of samples
num_samples = 1500

# Parameter ranges (min, max)
ranges = {
    'Temperature': (5, 30),
    'pH': (6.5, 9.0),
    'Heavy Metals': (0.0001, 0.01),
    'Oil/Grease': (0.01, 10),
    'Sulfur/Sulfite': (0.1, 20),
    'Nitrate/Nitrogen': (0.01, 50),
    'Phenol': (0.0, 0.5),
    'Total Phosphate': (0.001, 0.1),
    'Cyanide': (0.0, 0.1),
    'Total Organic Carbon': (0.1, 20),
    'Dissolved Oxygen': (0, 14),
    'Fluoride': (0.1, 5),
    'Agricultural Pesticides': (0.0, 0.05)
}

# Generate random data
data = {}
for param, (low, high) in ranges.items():
    data[param] = np.random.uniform(low, high, num_samples)

df = pd.DataFrame(data)

# Calculate mutation effects

def calculate_mutation_effects(df):
    # Linear effects
    linear_effects = (
        df['Heavy Metals'] * 400 +  # Direct DNA damage from heavy metals
        df['Oil/Grease'] * 100 +   # Hydrocarbon contamination
        df['Nitrate/Nitrogen'] * 60 +  # Formation of mutagenic nitrosamines
        df['Phenol'] * 200          # DNA adduct formation
    )

    # Non-linear effects
    nonlinear_effects = (
        (df['Cyanide'] ** 2 + df['Agricultural Pesticides'] ** 2) * 70 +  # Toxic interactions
        np.exp(-df['Dissolved Oxygen'] / 4) * 100 +  # Hypoxia-induced stress
        np.log1p(df['Temperature']) * 8 +  # Thermal stress
        np.sqrt(df['Total Organic Carbon']) * 15   # Organic pollutant byproducts
    )

    # Noise for variability
    noise = np.random.uniform(-5, 5, len(df))

    mutation_score = linear_effects + nonlinear_effects + noise

    return mutation_score

# Compute mutation scores
mutation_score = calculate_mutation_effects(df)
mutation_percentage = (mutation_score - mutation_score.min()) / (mutation_score.max() - mutation_score.min()) * 100
df['Mutation (%)'] = mutation_percentage

# Save the output to CSV
output_file = 'synthetic_data.csv'
df.to_csv(output_file, index=False, float_format='%.6f')
