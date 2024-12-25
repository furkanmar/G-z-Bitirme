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
    # Direct effects
    direct_effects = (
        df['Heavy Metals'] * 1200 +  # Strong DNA repair inhibition targeting 16S hyper-variable regions
        df['Phenol'] * 900 +        # Direct oxidative stress on 16S rRNA regions
        df['Agricultural Pesticides'] * 1000  # Direct DNA alkylation and mutation targeting
    )

    # Indirect effects
    indirect_effects = (
        df['Oil/Grease'] * 300 +    # Hydrocarbon contamination affecting microbial adaptability
        df['Nitrate/Nitrogen'] * 250 +  # Formation of mutagenic nitrosamines in 16S regions
        df['Sulfur/Sulfite'] * 250 +    # Oxidation-reduction imbalance disrupting cellular functions
        df['Total Phosphate'] * 200 +    # Hypoxia-induced stress influencing 16S stability
        df['Cyanide'] * 350 +           # Cellular respiration disruption targeting RNA stability
        df['Total Organic Carbon'] * 150 +  # Organic pollutant byproducts enhancing mutagenicity
        df['Dissolved Oxygen'] * 100      # ROS production under hypoxia stressing microbial populations
    )

    # Non-linear effects
    nonlinear_effects = (
        (df['Heavy Metals'] ** 10 + df['Phenol'] ** 10 + df['Agricultural Pesticides'] ** 10) * 200 +  # Amplified toxicity in hyper-variable regions
        np.exp(-df['Dissolved Oxygen'] / 2) * 300 +  # Stronger exponential hypoxia effects
        np.log1p(df['Temperature']) * 30 +             # Enhanced thermal stress impact on genetic stability
        np.sqrt(df['Total Organic Carbon']) * 60       # Synergistic effects with organic pollution
    )

    # Noise for variability
    noise = np.random.uniform(-50, 50, len(df))  # Increased noise level to simulate real-world variability

    mutation_score = direct_effects + indirect_effects + nonlinear_effects + noise

    return mutation_score

# Compute mutation scores
mutation_score = calculate_mutation_effects(df)
mutation_percentage = (mutation_score - mutation_score.min()) / (mutation_score.max() - mutation_score.min()) * 100
df['Mutation (%)'] = mutation_percentage

# Save the output to CSV
output_file = 'synthetic_data_v2_updated.csv'
df.to_csv(output_file, index=False, float_format='%.6f')
