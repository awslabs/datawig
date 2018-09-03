from datawig import SimpleImputer
from datawig.utils import random_split
from sklearn.metrics import f1_score, classification_report
import pandas as pd
import os

'''
Text Data
'''
df = pd.read_csv('../finish_val_data.csv')
df_train, df_test = random_split(df, split_ratios=[0.8, 0.2])

#Fit a Model Without HPO
imputer_text = SimpleImputer(
    input_columns=['title', 'text'],
    output_column='finish',
    output_path='imputer_text_model',
    num_hash_buckets=2**15,
    tokens='chars'
    )

imputer_text.fit(
    train_df=df_train,
    learning_rate=1e-4,
    num_epochs=50,
    final_fc_hidden_units=[512])

#Fit a Model With HPO
imputer_text = SimpleImputer(
    input_columns=['title', 'text'],
    output_column='finish',
    output_path='imputer_model',
    )

imputer_text.fit_hpo(
    train_df=df_train,
    num_epochs=50,
    learning_rate_candidates=[1e-3, 1e-4],
    final_fc_hidden_units_candidates=[[100]],
    num_hash_bucket_candidates=[2**10, 2**15],
    tokens_candidates=['chars', 'words']
    )

#------------------------------------------------------------------------------------

'''
Numerical Data
'''
#Generate synthetic numerical data
numeric_data = np.random.uniform(-np.pi, np.pi, (n_samples,))
df = pd.DataFrame({
    'x': numeric_data,
    '*2': numeric_data * 2. + np.random.normal(0, .1, (n_samples,)),
    '**2': numeric_data ** 2 + np.random.normal(0, .1, (n_samples,)),
    feature_col: random_data[feature_col].values,
    label_col: random_data[label_col].values
    })
df_train, df_test = random_split(df, split_ratios=[0.8, 0.2])

#Fit a model without HPO
imputer_numeric = SimpleImputer(
    input_columns=['x', feature_col],
    output_column="*2",
    output_path='imputer_numeric_model',
    latent_dim = 512,
    hidden_layers = 1
    )

imputer_numeric.fit(
    train_df=df_train,
    learning_rate=1e-4,
    num_epochs=50
    )

#Fit a model with HPO
imputer_numeric = SimpleImputer(
    input_columns=['x', feature_col],
    output_column="*2",
    output_path=output_path,
    )

imputer_numeric.fit_hpo(
    train_df=df_train,
    num_epochs=50,
    learning_rate_candidates=[1e-3, 1e-4],
    latent_dim_candidates=[50, 100],
    hidden_layers_candidates=[0, 2],
    final_fc_hidden_units=[[100]]
    )

#------------------------------------------------------------------------------------
'''
Image Data 
'''
def create_test_image(filename, color='red'):
    if color == 'red':
        color_vector = (155, 0, 0)
    elif color == 'green':
        color_vector = (0, 155, 0)
    elif color == 'blue':
        color_vector = (0, 0, 155)
    file = BytesIO()
    image = Image.new('RGBA', size=(50, 50), color=color_vector)
    image.save(filename, 'png')
    file.name = filename + '.png'
    file.seek(0)
    return file

#Create synthetic image data
img_path = './example_test_images'
os.makedirs(img_path, exist_ok=True)
colors = ['red', 'green', 'blue']

for color in colors:
    create_test_image(os.path.join(img_path, color + ".png"), color)

n_samples = 32
color_labels = [random.choice(colors) for _ in range(n_samples)]

df = pd.DataFrame({"image_files": color_labels,
                   "label": color_labels})

df['image_files'] = img_path + "/" + df['image_files'] + ".png"
df_train, df_test = random_split(df, split_ratios=[0.8, 0.2])

#Fit a model with HPO
imputer_image = SimpleImputer(
    input_columns=['image_files'],
    output_column='label',
    output_path='imputer_image_model'
)

imputer_image.fit_hpo(
    train_df=df_train,
    num_epochs=50,
    learning_rate_candidates=[1e-3, 1e-4],
    layer_dim=[[256], [1024, 512]],
    final_fc_hidden_units=[[100]]
)
