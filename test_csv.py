import pandas as pd

# Cargar el CSV
df = pd.read_csv("candidatos.csv")

# Mostrar los primeros registros para verificar
print(df.head())

# Revisar si hay valores nulos o problemas en los datos
print("\n📌 Información del dataset:")
print(df.info())

print("\n🔍 Revisión de valores nulos:")
print(df.isnull().sum())

print("\nEjemplo de un registro:")
print(df.iloc[0])
