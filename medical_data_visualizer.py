import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the data
df = pd.read_csv("medical_examination.csv")

# 2. Add 'overweight' column (BMI > 25 is considered overweight)
df['overweight'] = np.where((df['weight'] / ((df['height'] / 100) ** 2)) > 25, 1, 0)

# 3. Normalize data: Cholesterol and glucose normalization
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# 4. Draw Categorical Plot
def draw_cat_plot():
    # 5. Create a DataFrame for the cat plot
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6. Group and reformat the data to split by 'cardio' and count each feature
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # 7. Create a categorical plot using seaborn's catplot
    fig = sns.catplot(x='variable', y='total', hue='value', col='cardio', kind='bar', data=df_cat)

    # 8. Return the figure for output
    fig.savefig('catplot.png')
    return fig

# 10. Draw the Heat Map
def draw_heat_map():
    # 11. Clean the data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) & 
        (df['height'] >= df['height'].quantile(0.025)) & 
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12. Calculate the correlation matrix
    corr = df_heat.corr()

    # 13. Generate a mask for the upper triangle of the correlation matrix
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14. Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # 15. Draw the heatmap using seaborn's heatmap function
    sns.heatmap(corr, annot=True, mask=mask, square=True, fmt='.1f', center=0, vmax=0.3, vmin=-0.1, cbar_kws={"shrink": 0.5}, ax=ax)

    # 16. Save the figure
    fig.savefig('heatmap.png')
    return fig