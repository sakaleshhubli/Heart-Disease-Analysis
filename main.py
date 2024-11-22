import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns
print("Setup Complete")

filepath = ("HeartDisease\\heart.csv")
data = pd.read_csv(filepath)

print(data.head())

print(data.describe().T)  #.T maane tranpose whch make the reading easy

# plt.figure(figsize=(16,6))
# sns.lineplot(data=data)
# plt.title("Heart Disease Data Trends")
# plt.xlabel("Target")      # Replace with appropriate label
# plt.ylabel(" ")      # Replace with appropriate label
# plt.show()


#Index(['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'], dtype='object')
#data_file.columns

# Variable	Description
# age	Age of the patient in years
# sex	Gender of the patient (0 = male, 1 = female)
# cp	Chest pain type:
# 0: Typical angina
# 1: Atypical angina
# 2: Non-anginal pain
# 3: Asymptomatic
# trestbps	Resting blood pressure in mm Hg
# chol	Serum cholesterol in mg/dl
# fbs	Fasting blood sugar level, categorized as above 120 mg/dl (1 = true, 0 = false)
# restecg	Resting electrocardiographic results:
# 0: Normal
# 1: Having ST-T wave abnormality
# 2: Showing probable or definite left ventricular hypertrophy
# thalach	Maximum heart rate achieved during a stress test
# exang	Exercise-induced angina (1 = yes, 0 = no)
# oldpeak	ST depression induced by exercise relative to rest
# slope	Slope of the peak exercise ST segment:
# 0: Upsloping
# 1: Flat
# 2: Downsloping
# ca	Number of major vessels (0-4) colored by fluoroscopy
# thal	Thalium stress test result:
# 0: Normal
# 1: Fixed defect
# 2: Reversible defect
# 3: Not described
# target	Heart disease status (0 = no disease, 1 = presence of disease)