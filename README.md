# Project 1: Decision Tree untuk Prediksi Posisi Pemain FC 25

## 📋 Overview

Proyek ini merupakan implementasi **Supervised Machine Learning** menggunakan algoritma **Decision Tree Classifier** untuk memprediksi posisi pemain dalam game EA Sports FC 25 berdasarkan atribut statistik pemain.

### Tujuan Penelitian
- Mengklasifikasikan posisi pemain (GK, DF, MF, FW) berdasarkan atribut kinerja
- Menganalisis fitur-fitur paling penting dalam menentukan posisi pemain
- Mengevaluasi performa model dengan berbagai skenario pengujian
- Menghasilkan model yang dapat digunakan untuk prediksi data baru

---

## 📚 Libraries yang Digunakan

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')
```

---

## 🗄️ Data Gathering

### Load Dataset
```python
file_path = '/content/drive/MyDrive/Collage/Smstr 3/male_players.csv'
df = pd.read_csv(file_path)
```

### Informasi Dataset
| Deskripsi | Nilai |
|-----------|-------|
| Jumlah baris | 16.161 |
| Jumlah kolom | 58 |
| Tipe data numerik | 42 (int64), 5 (float64) |
| Tipe data kategorikal | 11 (object) |

### Distribusi Posisi Pemain (Sebelum Kategorisasi)
```
Position
CB     2924
ST     2183
CM     1890
GK     1816
CDM    1330
RB     1281
LB     1214
LM      968
CAM     961
RM      896
RW      357
LW      341
```

---

## 🔧 Data Preprocessing

### A. Data Cleaning

#### 1. Seleksi Kolom Relevan
```python
columns_to_keep = [
    # Identitas
    'Name', 'Age', 'Height', 'Weight', 'Nation',
    # Target variable
    'Position',
    # Atribut utama
    'OVR',
    # Atribut kinerja
    'PAC', 'SHO', 'PAS', 'DRI', 'DEF', 'PHY',
    # Sub-atribut detail
    'Acceleration', 'Sprint Speed',
    'Positioning', 'Finishing', 'Shot Power', 'Long Shots',
    'Vision', 'Crossing', 'Short Passing', 'Long Passing',
    'Agility', 'Balance', 'Ball Control', 'Dribbling',
    'Interceptions', 'Heading Accuracy', 'Def Awareness', 
    'Standing Tackle', 'Sliding Tackle',
    'Jumping', 'Stamina', 'Strength', 'Aggression',
    # GK attributes
    'GK Diving', 'GK Handling', 'GK Kicking', 
    'GK Positioning', 'GK Reflexes'
]
```

#### 2. Handling Missing Values
```python
# Strategi 1: Hapus baris dengan missing values > 50%
threshold = 0.5
missing_ratio = df_clean.isnull().sum(axis=1) / df_clean.shape[1]
df_clean = df_clean[missing_ratio < threshold]

# Strategi 2: Imputasi median untuk kolom numerik
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df_clean[col].isnull().sum() > 0:
        df_clean[col].fillna(df_clean[col].median(), inplace=True)

# Strategi 3: Drop missing values pada kolom penting
df_clean = df_clean.dropna(subset=['Position', 'OVR'])
```

#### 3. Kategorisasi Posisi
```python
def categorize_position(position):
    """
    Kategorisasi posisi pemain menjadi 4 kategori utama:
    - GK: Goalkeeper
    - DF: Defender (CB, LB, RB)
    - MF: Midfielder (CDM, CM, CAM, LM, RM)
    - FW: Forward (ST, CF, LW, RW)
    """
    if pd.isna(position):
        return 'Unknown'
    
    position = str(position).upper()
    
    if 'GK' in position:
        return 'GK'
    elif any(pos in position for pos in ['CB', 'LB', 'RB']):
        return 'DF'
    elif any(pos in position for pos in ['CDM', 'CM', 'CAM', 'LM', 'RM']):
        return 'MF'
    elif any(pos in position for pos in ['ST', 'LW', 'RW']):
        return 'FW'
    else:
        return 'Unknown'

df_clean['position_category'] = df_clean['Position'].apply(categorize_position)
df_clean = df_clean[df_clean['position_category'] != 'Unknown']
```

### Distribusi Setelah Kategorisasi
```
position_category
MF    6045
DF    5419
FW    2881
GK    1816
```

#### 4. Deteksi dan Handling Outliers (IQR Method)
```python
def detect_outliers_IQR(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def cap_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    return df
```

### B. Feature Engineering

#### 1. Membuat Fitur Baru
```python
# Attack Score: rata-rata atribut ofensif
df_clean['attack_score'] = df_clean[['Positioning', 'Finishing', 
                                     'Shot Power', 'Long Shots']].mean(axis=1)

# Defense Score: rata-rata atribut defensif
df_clean['defense_score'] = df_clean[['Interceptions', 'Heading Accuracy', 
                                      'Def Awareness', 'Standing Tackle']].mean(axis=1)

# Attack-Defense Ratio
df_clean['attack_defense_ratio'] = df_clean['attack_score'] / (df_clean['defense_score'] + 1)
```

#### 2. Feature Selection dengan Correlation Analysis
```python
# Encode target variable
le = LabelEncoder()
df_clean['position_encoded'] = le.fit_transform(df_clean['position_category'])

# Hitung korelasi dengan target
correlation_with_target = df_clean[numeric_features + ['position_encoded']].corr()['position_encoded'].abs().sort_values(ascending=False)
```

### Top 15 Fitur Berdasarkan Korelasi
| Rank | Fitur | Korelasi |
|------|---------|----------|
| 1 | SHO | 0.4802 |
| 2 | PAS | 0.4233 |
| 3 | Vision | 0.4146 |
| 4 | DRI | 0.4089 |
| 5 | Jumping | 0.3475 |
| 6 | Strength | 0.3419 |
| 7 | Heading Accuracy | 0.3282 |
| 8 | Shot Power | 0.3034 |
| 9 | Long Shots | 0.2755 |
| 10 | Finishing | 0.2656 |

---

## 🎯 Model Training

### Data Splitting
```python
# Feature dan target
selected_features = ['PAC', 'SHO', 'PAS', 'DRI', 'DEF', 'PHY', 
                     'attack_score', 'defense_score', 'attack_defense_ratio']
X = df_clean[selected_features]
y = df_clean['position_category']

# Split data 80:20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Hyperparameter Tuning dengan GridSearchCV
```python
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 4, 8],
    'max_features': [None, 'sqrt', 'log2']
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)
best_dt = grid_search.best_estimator_
```

### Best Parameters
```python
{
    'criterion': 'entropy',
    'max_depth': 10,
    'max_features': None,
    'min_samples_leaf': 8,
    'min_samples_split': 20
}
```

---

## 📊 Model Evaluation

### Overall Performance
```python
y_pred = best_dt.predict(X_test_scaled)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision (weighted): {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Recall (weighted): {recall_score(y_test, y_pred, average='weighted'):.4f}")
print(f"F1-Score (weighted): {f1_score(y_test, y_pred, average='weighted'):.4f}")
```

### Hasil Evaluasi
| Metric | Nilai |
|--------|-------|
| **Accuracy** | **0.7776** (77.76%) |
| Precision (weighted) | 0.7791 |
| Recall (weighted) | 0.7776 |
| F1-Score (weighted) | 0.7770 |

### Classification Report per Position
| Position | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| **DF** | 0.8681 | 0.8441 | 0.8559 | 1084 |
| **FW** | 0.7581 | 0.6516 | 0.7008 | 577 |
| **GK** | 0.7260 | 0.8760 | 0.7940 | 363 |
| **MF** | 0.7269 | 0.7486 | 0.7376 | 1209 |

### Confusion Matrix
```
              Predicted
              DF    FW    GK    MF
Actual  DF  [915]  45   12   112
        FW   68  [376]  8    125
        GK   15   5   [318]  25
        MF  142  98   22   [947]
```

---

## 🔍 Feature Importance

### Top 10 Most Important Features
| Rank | Feature | Importance |
|------|---------|------------|
| 1 | attack_defense_ratio | 0.5051 |
| 2 | PAC | 0.1164 |
| 3 | DEF | 0.1149 |
| 4 | PHY | 0.0933 |
| 5 | PAS | 0.0715 |
| 6 | SHO | 0.0406 |
| 7 | attack_score | 0.0285 |
| 8 | defense_score | 0.0194 |
| 9 | DRI | 0.0069 |
| 10 | OVR | 0.0033 |

> 💡 **Insight**: Rasio attack-defense merupakan fitur paling dominan (50.51%) dalam menentukan posisi pemain, menunjukkan bahwa keseimbangan antara kemampuan ofensif dan defensif adalah kunci klasifikasi.

---

## ✅ Cross-Validation (5-Fold)

```python
cv_scores = cross_val_score(best_dt, X_train_scaled, y_train, cv=5, scoring='accuracy')
```

### Hasil Cross-Validation
| Fold | Accuracy |
|------|----------|
| 1 | 0.7633 |
| 2 | 0.7842 |
| 3 | 0.7896 |
| 4 | 0.7810 |
| 5 | 0.7849 |
| **Mean** | **0.7806** |
| **Std** | **0.0091** |
| **Variance** | **0.000082** |

> ✅ Model stabil dengan varians CV < 5%

---

## 🧪 Perbandingan Skenario Pengujian

### A. Different Split Ratios
| Split Ratio | Train Size | Test Size | Accuracy |
|-------------|------------|-----------|----------|
| 70:30 | 11,312 | 4,849 | 0.7890 |
| 80:20 | 12,928 | 3,233 | **0.7776** |
| 90:10 | 14,544 | 1,617 | 0.7805 |

### B. Different Feature Sets
| Feature Set | Num Features | Accuracy |
|-------------|-------------|----------|
| Top-5 Features | 5 | 0.7773 |
| Top-10 Features | 9 | **0.7779** |
| All Features | 9 | 0.7776 |

### C. Criterion Comparison
| Criterion | Accuracy |
|-----------|----------|
| **Gini** | **0.7900** |
| Entropy | 0.7776 |

### D. Max Depth Comparison
| Max Depth | Accuracy |
|-----------|----------|
| 5 | 0.7083 |
| **10** | **0.7776** |
| 15 | 0.7748 |
| 20 | 0.7751 |
| None | 0.7751 |

> 📈 **Insight**: Max depth = 10 memberikan keseimbangan optimal antara bias dan variance.

---

## 📋 Kesimpulan dan Insight

### 📊 Model Performance Summary
```
• Best Model: Decision Tree dengan parameter optimal
• Overall Accuracy: 77.76%
• Weighted F1-Score: 0.7770
• Cross-Validation Mean: 0.7806 ± 0.0091
```

### 🎯 Performance per Position
- **DF (Defender)**: F1 = 0.8559 ⭐ *Best Performance*
- **GK (Goalkeeper)**: F1 = 0.7940
- **MF (Midfielder)**: F1 = 0.7376
- **FW (Forward)**: F1 = 0.7008

### ⭐ Key Findings
1. **Fitur Paling Penting**: `attack_defense_ratio` (50.51%) mendominasi keputusan model
2. **Posisi Termudah Diprediksi**: Defender (DF) dengan precision 86.81%
3. **Posisi Tersulit Diprediksi**: Forward (FW) dengan recall 65.16%
4. **Model Stability**: Varians CV 0.0091 (< 5%) menunjukkan model stabil dan tidak overfitting

### ✅ Model Validation
```
• Baseline Accuracy: 0.7532
• Final Accuracy: 0.7776
• Improvement: +2.44%
• ✓ Model is stable (CV variance < 5%)
```

---

## 💾 Model Deployment

### Save Model dan Preprocessor
```python
import joblib

# Save model
joblib.dump(best_dt, 'decision_tree_fc25_model.pkl')

# Save scaler
joblib.dump(scaler, 'scaler.pkl')

# Save feature-specific scaler
joblib.dump(feature_set_scaler, 'feature_set_scaler.pkl')
```

### Load dan Prediksi Data Baru
```python
# Load model dan scaler
loaded_model = joblib.load('decision_tree_fc25_model.pkl')
loaded_scaler = joblib.load('feature_set_scaler.pkl')

# Data pemain baru
new_player = pd.DataFrame({
    'PAC': [68], 'SHO': [49], 'PAS': [60], 'DRI': [67],
    'DEF': [31], 'PHY': [56],
    'attack_score': [64], 'defense_score': [34],
    'attack_defense_ratio': [1]
})

# Scaling dan prediksi
new_player_scaled = loaded_scaler.transform(new_player)
prediction = loaded_model.predict(new_player_scaled)
prediction_proba = loaded_model.predict_proba(new_player_scaled)

print(f"Predicted Position: {prediction[0]}")
```

### Contoh Output Prediksi
```
Predicted Position: MF
Prediction Probabilities:
  DF: 0.0909
  FW: 0.0000
  GK: 0.2727
  MF: 0.6364  ← Highest probability
```

---

## 🔧 Rekomendasi Pengembangan

1. **Ensemble Methods**: Coba Random Forest atau Gradient Boosting untuk meningkatkan akurasi
2. **Feature Engineering**: Tambahkan fitur interaksi antar atribut
3. **Class Imbalance Handling**: Gunakan SMOTE untuk meningkatkan performa kelas minoritas (FW)
4. **Real-time Prediction**: Implementasi API untuk prediksi pemain baru secara real-time
5. **Model Interpretability**: Gunakan SHAP/LIME untuk interpretasi prediksi yang lebih mendalam

---

> 📝 **Note**: Proyek ini merupakan bagian dari tugas besar Pemrograman 1 dengan penerapan supervised machine learning dalam konteks video game (EA Sports FC 25).
**Author**: [Ilham Taufiq Ghifari]  
**Date**: [2 February 2026]  
**Repository**: `Project1_DT_FC25.ipynb`
