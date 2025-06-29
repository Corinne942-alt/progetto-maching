# Predizione delle Prestazioni nel Powerlifting: Analisi Completa con Machine Learning
# Versione ottimizzata per Windows con commenti in italiano

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os
import sys
from pathlib import Path
warnings.filterwarnings('ignore')

# Importazioni per Sklearn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import  GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression

# Ottimizzazioni specifiche per Windows
if os.name == 'nt':  # Windows
    # Imposta backend migliore per matplotlib su Windows
    import matplotlib
    matplotlib.use('Agg')  # Backend non-interattivo
    
    # Ottimizza pandas per Windows
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    
    # Imposta metodo multiprocessing per compatibilità Windows
    import multiprocessing as mp
    if __name__ == '__main__':
        mp.set_start_method('spawn', force=True)

print("=" * 80)
print("PREDIZIONE PRESTAZIONI POWERLIFTING")
print("Analisi Completa con Machine Learning")
print("Versione Ottimizzata per Windows")
print("=" * 80)

# ============================================================================
# 0. CONFIGURAZIONE SISTEMA WINDOWS E CONTROLLI
# ============================================================================

print("\n0. CONFIGURAZIONE SISTEMA WINDOWS")
print("-" * 50)

# Controlla versione Windows e ambiente Python
print(f"Sistema Operativo: {os.name}")
print(f"Piattaforma: {sys.platform}")
print(f"Versione Python: {sys.version}")
print(f"Directory di Lavoro: {os.getcwd()}")

# Crea directory di output per Windows
output_dir = Path("risultati_powerlifting")
output_dir.mkdir(exist_ok=True)
print(f"Directory di output creata: {output_dir.absolute()}")

# Controlla memoria disponibile su Windows
try:
    import psutil
    memory = psutil.virtual_memory()
    print(f"RAM Disponibile: {memory.available / (1024**3):.2f} GB")
    print(f"RAM Totale: {memory.total / (1024**3):.2f} GB")
except ImportError:
    print("psutil non disponibile - installa con: pip install psutil")

# ============================================================================
# 1. INTRODUZIONE E DEFINIZIONE DEL PROBLEMA
# ============================================================================

print("\n1. INTRODUZIONE")
print("-" * 50)
print("""
OBIETTIVO: Predire le prestazioni totali nel powerlifting (TotalKg) utilizzando 
le caratteristiche dell'atleta e le prestazioni nei singoli esercizi.

TIPO DI PROBLEMA: Regressione
DATASET: Dataset OpenPowerlifting
VARIABILE TARGET: TotalKg (peso totale sollevato in squat, panca, stacco)

DOMANDE DI RICERCA:
1. Quali caratteristiche sono più predittive delle prestazioni totali?
2. Come si confrontano diversi algoritmi di regressione per questo compito?
3. Possiamo identificare varianti algoritmiche che funzionano meglio per i dati del powerlifting?
4. Quali intuizioni possiamo ricavare sui fattori delle prestazioni nel powerlifting?
""")

# ============================================================================
# 2. CARICAMENTO E ESPLORAZIONE DATI (ottimizzato per Windows)
# ============================================================================

print("\n2. CARICAMENTO DATI ED ESPLORAZIONE INIZIALE")
print("-" * 50)

# Caricamento dati ottimizzato per Windows con opzioni di fallback
import pandas as pd

import pandas as pd

# Percorso completo del tuo CSV
file_csv = r"C:\Users\HP115S-FQ2089\Desktop\mac learn powerlifting\openpowerlifting-2024-01-06-4c732975.csv"

# Carica CSV
df = pd.read_csv(file_csv, encoding="utf-8", low_memory=False)

# Messaggi di conferma
print("✓ Dataset caricato con successo!")
print(f"Forma iniziale del dataset: {df.shape}")
print(f"Uso memoria: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\nPrime righe del dataset:")
print(df.head())




# Mostra informazioni di base
print("\nPanoramica del Dataset:")
print(df.info())

print("\nPrime righe:")
print(df.head())

# Salva riassunto su file (ottimizzato per Windows)
with open(output_dir / "riassunto_dati.txt", 'w', encoding='utf-8') as f:
    f.write("RIASSUNTO DATASET POWERLIFTING\n")
    f.write("=" * 50 + "\n")
    f.write(f"Forma: {df.shape}\n")
    f.write(f"Uso memoria: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")
    f.write("\nTipi di dati:\n")
    f.write(str(df.dtypes))

# ============================================================================
# 3. PREPROCESSING DATI E FEATURE ENGINEERING
# ============================================================================

print("\n3. PREPROCESSING DATI E FEATURE ENGINEERING")
print("-" * 50)

# Seleziona colonne rilevanti
colonne_feature = [
    'Sex',
    'Age',
    'BodyweightKg',
    'Equipment',
    'Best3SquatKg',
    'Best3BenchKg',
    'Best3DeadliftKg',
    'TotalKg'
]


# Crea dataframe di lavoro
print(df.columns)
df_lavoro = df[colonne_feature].copy()
print(f"Selezionate {len(colonne_feature)} caratteristiche rilevanti")

# Pulizia dati con indicazione progresso Windows
print("\nPulizia Dati:")
print(f"Valori mancanti prima della pulizia:")
riassunto_mancanti = df_lavoro.isnull().sum()
print(riassunto_mancanti)

# Rimuovi righe con target mancante o caratteristiche chiave
print("Pulendo dati...", end="")
df_pulito = df_lavoro.dropna()
print(".", end="")
df_pulito = df_pulito[df_pulito['TotalKg'] > 0]
print(".", end="")
df_pulito = df_pulito[(df_pulito['Best3SquatKg'] > 0) & 
                     (df_pulito['Best3BenchKg'] > 0) & 
                     (df_pulito['Best3DeadliftKg'] > 0)]
print(" ✓")

print(f"Forma dataset dopo pulizia: {df_pulito.shape}")
print(f"Rimosse {len(df_lavoro) - len(df_pulito)} righe ({(len(df_lavoro) - len(df_pulito))/len(df_lavoro)*100:.1f}%)")

# Feature Engineering
print("\nFeature Engineering:")

# Crea nuove caratteristiche
df_pulito['Gruppo_Eta'] = pd.cut(df_pulito['Age'], bins=[0, 25, 35, 45, 55, 100], 
                                labels=['Giovani', 'Giovani_Adulti', 'Adulti', 'Mezza_Eta', 'Senior'])

# Calcolo BMI più robusto
stima_altezza = 1.7 + (df_pulito['BodyweightKg'] - 70) * 0.01  # Stima approssimativa altezza
df_pulito['BMI'] = df_pulito['BodyweightKg'] / (stima_altezza ** 2)

df_pulito['Rapporto_Squat_Peso'] = df_pulito['Best3SquatKg'] / df_pulito['BodyweightKg']
df_pulito['Rapporto_Panca_Peso'] = df_pulito['Best3BenchKg'] / df_pulito['BodyweightKg']
df_pulito['Rapporto_Stacco_Peso'] = df_pulito['Best3DeadliftKg'] / df_pulito['BodyweightKg']
df_pulito['Rapporto_Totale_Peso'] = df_pulito['TotalKg'] / df_pulito['BodyweightKg']

# Rapporti specifici del powerlifting
df_pulito['Rapporto_Squat_Panca'] = df_pulito['Best3SquatKg'] / df_pulito['Best3BenchKg']
df_pulito['Rapporto_Stacco_Squat'] = df_pulito['Best3DeadliftKg'] / df_pulito['Best3SquatKg']

print("✓ Create caratteristiche ingegnerizzate:")
nuove_feature = ['Gruppo_Eta', 'BMI', 'Rapporto_Squat_Peso', 'Rapporto_Panca_Peso', 
                'Rapporto_Stacco_Peso', 'Rapporto_Squat_Panca', 'Rapporto_Stacco_Squat']
for feat in nuove_feature:
    print(f"  - {feat}")

# Codifica variabili categoriche
print("\nCodifica variabili categoriche:")
le_sesso = LabelEncoder()
df_pulito['Sesso_codificato'] = le_sesso.fit_transform(df_pulito['Sex'])
print(f"✓ Codifica sesso: {dict(zip(le_sesso.classes_, le_sesso.transform(le_sesso.classes_)))}")

# One-hot encoding per attrezzatura
attrezzatura_dummy = pd.get_dummies(df_pulito['Equipment'], prefix='Attrezzatura')
df_pulito = pd.concat([df_pulito, attrezzatura_dummy], axis=1)
print(f"✓ Categorie attrezzatura: {list(attrezzatura_dummy.columns)}")

# One-hot encoding per gruppi di età
gruppi_eta_dummy = pd.get_dummies(df_pulito['Gruppo_Eta'], prefix='Gruppo_Eta')
df_pulito = pd.concat([df_pulito, gruppi_eta_dummy], axis=1)
print(f"✓ Categorie gruppi età: {list(gruppi_eta_dummy.columns)}")

# ============================================================================
# 4. ANALISI ESPLORATIVA DATI
# ============================================================================

print("\n4. ANALISI ESPLORATIVA DATI")
print("-" * 50)

# Riassunto statistico
print("Riassunto Statistico delle Variabili Chiave:")
variabili_riassunto = ['Age', 'BodyweightKg', 'Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg', 'TotalKg']
statistiche_riassunto = df_pulito[variabili_riassunto].describe()
print(statistiche_riassunto)

# Salva statistiche riassuntive su CSV (ottimizzato per Windows)
statistiche_riassunto.to_csv(output_dir / "statistiche_riassuntive.csv")

# Analisi correlazione
print("\nCorrelazione con Variabile Target (TotalKg):")
feature_numeriche = df_pulito.select_dtypes(include=[np.number]).columns
correlazioni = df_pulito[feature_numeriche].corr()['TotalKg'].sort_values(ascending=False)
print(correlazioni.head(10))

# Salva correlazioni
correlazioni.to_csv(output_dir / "correlazioni_feature.csv")

# Analisi distribuzione
print(f"\nDistribuzione Variabile Target:")
statistiche_target = {
    'Media': df_pulito['TotalKg'].mean(),
    'Mediana': df_pulito['TotalKg'].median(),
    'Deviazione Standard': df_pulito['TotalKg'].std(),
    'Asimmetria': stats.skew(df_pulito['TotalKg'])
}

for stat, valore in statistiche_target.items():
    print(f"{stat}: {valore:.2f}")

# ============================================================================
# 5. PREPARAZIONE MODELLI
# ============================================================================

print("\n5. PREPARAZIONE MODELLI")
print("-" * 50)

# Prepara matrice delle caratteristiche
colonne_feature = ['Sesso_codificato', 'Age', 'BodyweightKg', 'BMI',
                   'Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg',
                   'Rapporto_Squat_Peso', 'Rapporto_Panca_Peso', 'Rapporto_Stacco_Peso',
                   'Rapporto_Squat_Panca', 'Rapporto_Stacco_Squat'] + \
                  list(attrezzatura_dummy.columns) + list(gruppi_eta_dummy.columns)

X = df_pulito[colonne_feature]
y = df_pulito['TotalKg']

print(f"Forma matrice caratteristiche: {X.shape}")
print(f"Forma vettore target: {y.shape}")
print(f"Caratteristiche utilizzate: {len(colonne_feature)}")

# Divisione train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Set di addestramento: {X_train.shape[0]} campioni")
print(f"Set di test: {X_test.shape[0]} campioni")

# Scalatura caratteristiche
scaler = RobustScaler()  # Più robusto agli outlier rispetto a StandardScaler
X_train_scalato = scaler.fit_transform(X_train)
X_test_scalato = scaler.transform(X_test)

print("✓ Caratteristiche scalate utilizzando RobustScaler")

# ============================================================================
# 6. IMPLEMENTAZIONI MODELLI E VARIANTI (ottimizzato per Windows)
# ============================================================================

print("\n6. IMPLEMENTAZIONI MODELLI E VARIANTI ALGORITMICHE")
print("-" * 50)

# Definisci modelli con parametri ottimizzati per Windows (ridotti per efficienza memoria)
modelli = {
    # Famiglia Modelli Lineari
    'Regressione_Lineare': LinearRegression(),
    'Regressione_Ridge': Ridge(alpha=1.0),

    
    # Famiglia Metodi basati su Alberi (ridotti per Windows)
    'Albero_Decisione': DecisionTreeRegressor(random_state=42, max_depth=10),
    
    'Gradient_Boosting': GradientBoostingRegressor(n_estimators=50, random_state=42),

    
    # Metodi basati su Istanze
    'KNN_Uniforme': KNeighborsRegressor(n_neighbors=5, weights='uniform', n_jobs=1),
    'KNN_Distanza': KNeighborsRegressor(n_neighbors=5, weights='distance', n_jobs=1),
    
   
    
   
}

print(f"Definite {len(modelli)} varianti algoritmiche ottimizzate per Windows:")
for famiglia in ['Modelli Lineari', 'Basati su Alberi', 'Basati su Istanze']:
    modelli_famiglia = [nome for nome in modelli.keys() if 
                       (famiglia == 'Modelli Lineari' and any(x in nome for x in ['Lineare', 'Ridge', 'Lasso'])) or
                       (famiglia == 'Basati su Alberi' and any(x in nome for x in ['Albero', 'Boosting'])) or
                       (famiglia == 'Basati su Istanze' and 'KNN' in nome) ]
                       
    print(f"  {famiglia}: {len(modelli_famiglia)} varianti")

# ============================================================================
# 7. ADDESTRAMENTO E VALUTAZIONE MODELLI
# ============================================================================

print("\n7. ADDESTRAMENTO E VALUTAZIONE MODELLI")
print("-" * 50)

risultati = []
predizioni = {}

print("Addestrando modelli...")
contatore_progresso = 0
totale_modelli = len(modelli)

for nome, modello in modelli.items():
    contatore_progresso += 1
    print(f"[{contatore_progresso}/{totale_modelli}] Addestrando {nome}...", end=" ")
    
    try:
        # CORREZIONE: Addestra correttamente il modello
        modello.fit(X_train, y_train)
        
        # Fai predizioni
        y_pred = modello.predict(X_test)
        
        # Calcola metriche
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Punteggio validazione incrociata
        cv_folds = 3
        cv_scores = cross_val_score(modello, X_train, y_train, cv=cv_folds, scoring='r2')
        
        risultati.append({
            'Modello': nome,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'CV_R2_Media': cv_scores.mean(),
            'CV_R2_DevStd': cv_scores.std()
        })
        
        predizioni[nome] = y_pred
        print("✓")
        
    except Exception as e:
        print(f"✗ (Errore: {str(e)})")

# Converti risultati in DataFrame
risultati_df = pd.DataFrame(risultati)
risultati_df = risultati_df.sort_values('RMSE', ascending=True)

# Salva risultati su CSV
risultati_df.to_csv(output_dir / "risultati_modelli.csv", index=False)

print(f"\nRisultati Prestazioni Modelli:")
print("=" * 110)
print(f"{'Modello':<25} {'MAE':<8} {'RMSE':<8} {'R²':<8} {'CV R²':<12} {'CV Dev.Std':<10}")
print("-" * 110)
for _, riga in risultati_df.iterrows():
    print(f"{riga['Modello']:<25} {riga['MAE']:<8.2f} {riga['RMSE']:<8.2f} {riga['R2']:<8.3f} "
          f"{riga['CV_R2_Media']:<12.3f} {riga['CV_R2_DevStd']:<10.3f}")

# ============================================================================
# 8. ANALISI IMPORTANZA CARATTERISTICHE
# ============================================================================

print("\n8. ANALISI IMPORTANZA CARATTERISTICHE")
print("-" * 50)

# Ottieni importanza caratteristiche da Gradient Boosting
gb_modello = GradientBoostingRegressor(n_estimators=50, random_state=42)
gb_modello.fit(X_train, y_train)

importanza_feature = pd.DataFrame({
    'Caratteristica': colonne_feature,
    'Importanza': gb_modello.feature_importances_
}).sort_values('Importanza', ascending=False)

print("Top 10 Caratteristiche più Importanti:")
print(importanza_feature.head(10))

# Salva importanza caratteristiche
importanza_feature.to_csv(output_dir / "importanza_caratteristiche.csv", index=False)

# ============================================================================
# 9. VALUTAZIONE FINALE E INTUIZIONI
# ============================================================================

print("\n9. VALUTAZIONE FINALE E INTUIZIONI")
print("-" * 50)

modello_migliore = risultati_df.iloc[0]
print(f"MODELLO CON PRESTAZIONI MIGLIORI: {modello_migliore['Modello']}")
print(f"Metriche di Prestazione:")
print(f"  - MAE: {modello_migliore['MAE']:.2f} kg")
print(f"  - RMSE: {modello_migliore['RMSE']:.2f} kg") 
print(f"  - R²: {modello_migliore['R2']:.3f}")
print(f"  - Validazione incrociata R²: {modello_migliore['CV_R2_Media']:.3f} ± {modello_migliore['CV_R2_DevStd']:.3f}")

print(f"\nINTUIZIONI CHIAVE:")
intuizioni = [
    "1. Le prestazioni nei singoli esercizi (squat, panca, stacco) sono altamente predittive",
    "2. I rapporti con il peso corporeo forniscono potere predittivo aggiuntivo",
    f"3. {modello_migliore['Modello']} ha raggiunto il miglior equilibrio tra accuratezza e generalizzazione",
    f"4. Il modello spiega il {modello_migliore['R2']*100:.1f}% della varianza nelle prestazioni totali"
]

for intuizione in intuizioni:
    print(intuizione)

# Confronto prestazioni per famiglia algoritmica
print(f"\nPRESTAZIONI PER FAMIGLIA ALGORITMICA:")
famiglie = {
    'Modelli Lineari': ['Lineare', 'Ridge'],
    'Basati su Alberi': ['Albero', 'Boosting'],
    'Basati su Istanze': ['KNN'],
    
    
}

prestazioni_famiglia = []
for famiglia, parole_chiave in famiglie.items():
    risultati_famiglia = risultati_df[risultati_df['Modello'].str.contains('|'.join(parole_chiave))]
    if not risultati_famiglia.empty:
        r2_medio = risultati_famiglia['R2'].mean()
        r2_migliore = risultati_famiglia['R2'].max()
        prestazioni_famiglia.append({
            'Famiglia': famiglia,
            'R2_Medio': r2_medio,
            'R2_Migliore': r2_migliore,
            'Numero_Modelli': len(risultati_famiglia)
        })
        print(f"  {famiglia}: R² Medio = {r2_medio:.3f}, R² Migliore = {r2_migliore:.3f}")

# Salva confronto prestazioni famiglie
pd.DataFrame(prestazioni_famiglia).to_csv(output_dir / "prestazioni_famiglie.csv", index=False)

# Crea report riassuntivo finale
print("\n" + "=" * 80)
print("ANALISI COMPLETATA - OTTIMIZZATA PER WINDOWS")
print("=" * 80)

# Statistiche riassuntive
info_riassunto = {
    'Dimensione_Dataset': len(df_pulito),
    'Caratteristiche_Analizzate': len(colonne_feature),
    'Algoritmi_Testati': len(modelli),
    'Modello_Migliore': modello_migliore['Modello'],
    'RMSE_Migliore': modello_migliore['RMSE'],
    'R2_Migliore': modello_migliore['R2']
}

print(f"\nRIASSUNTO FINALE:")
print(f"- Dimensione dataset: {info_riassunto['Dimensione_Dataset']:,} prestazioni powerlifting")
print(f"- Caratteristiche analizzate: {info_riassunto['Caratteristiche_Analizzate']}")
print(f"- Algoritmi testati: {info_riassunto['Algoritmi_Testati']}")
print(f"- Modello migliore: {info_riassunto['Modello_Migliore']}")
print(f"- Accuratezza predizione: ±{info_riassunto['RMSE_Migliore']:.1f} kg RMSE")

# Salva riassunto finale
with open(output_dir / "riassunto_finale.txt", 'w', encoding='utf-8') as f:
    f.write("PREDIZIONE PRESTAZIONI POWERLIFTING - RIASSUNTO FINALE\n")
    f.write("=" * 60 + "\n\n")
    for chiave, valore in info_riassunto.items():
        f.write(f"{chiave.replace('_', ' ')}: {valore}\n")
    f.write(f"\nAnalisi completata su sistema Windows")
    f.write(f"\nFile di output salvati in: {output_dir.absolute()}")

print(f"\n✓ Tutti i risultati salvati in: {output_dir.absolute()}")
print(f"✓ Controlla la cartella 'risultati_powerlifting' per gli output dettagliati")

# Pulizia specifica per Windows
if 'matplotlib' in sys.modules:
    plt.close('all')  # Chiude tutti i grafici aperti per liberare memoria

print("\n🎯 Analisi ottimizzata e completata con successo su Windows!")

# ============================================================================
# 10. FUNZIONI AGGIUNTIVE PER ANALISI AVANZATA
# ============================================================================

def crea_visualizzazioni():
    """Crea visualizzazioni delle prestazioni dei modelli"""
    
    print("\n10. CREAZIONE VISUALIZZAZIONI")
    print("-" * 50)
    
    # Configura matplotlib per Windows
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Analisi Prestazioni Powerlifting', fontsize=16, fontweight='bold')
    
    # Grafico 1: Confronto RMSE dei modelli
    ax1 = axes[0, 0]
    risultati_top10 = risultati_df.head(10)
    ax1.barh(range(len(risultati_top10)), risultati_top10['RMSE'])
    ax1.set_yticks(range(len(risultati_top10)))
    ax1.set_yticklabels(risultati_top10['Modello'], fontsize=8)
    ax1.set_xlabel('RMSE (kg)')
    ax1.set_title('Top 10 Modelli per RMSE')
    ax1.invert_yaxis()
    
    # Grafico 2: Correlazioni caratteristiche principali
    ax2 = axes[0, 1]
    correlazioni_top = correlazioni.head(8)
    ax2.bar(range(len(correlazioni_top)), correlazioni_top.values)
    ax2.set_xticks(range(len(correlazioni_top)))
    ax2.set_xticklabels(correlazioni_top.index, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Correlazione con TotalKg')
    ax2.set_title('Correlazioni Caratteristiche Principali')
    
    # Grafico 3: Distribuzione errori modello migliore
    ax3 = axes[1, 0]
    errori = y_test - predizioni[modello_migliore['Modello']]
    ax3.hist(errori, bins=30, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Errore Predizione (kg)')
    ax3.set_ylabel('Frequenza')
    ax3.set_title(f'Distribuzione Errori - {modello_migliore["Modello"]}')
    ax3.axvline(0, color='red', linestyle='--', alpha=0.7)
    
    # Grafico 4: Prestazioni per famiglia algoritmica
    ax4 = axes[1, 1]
    if prestazioni_famiglia:
        famiglie_nomi = [f['Famiglia'] for f in prestazioni_famiglia]
        famiglie_r2 = [f['R2_Migliore'] for f in prestazioni_famiglia]
        ax4.bar(famiglie_nomi, famiglie_r2)
        ax4.set_ylabel('R² Migliore')
        ax4.set_title('Prestazioni per Famiglia Algoritmica')
        ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Salva grafico (Windows-friendly)
    try:
        plt.savefig(output_dir / "visualizzazioni_analisi.png", dpi=300, bbox_inches='tight')
        print("✓ Visualizzazioni salvate in 'visualizzazioni_analisi.png'")
    except Exception as e:
        print(f"⚠ Errore nel salvare visualizzazioni: {e}")
    
    plt.close()

def analisi_residui_avanzata():
    """Analisi avanzata dei residui per il modello migliore"""
    
    print("\nANALISI RESIDUI AVANZATA")
    print("-" * 30)
    
    # Calcola residui
    nome_migliore = modello_migliore['Modello']
    predizioni_migliori = predizioni[nome_migliore]
    residui = y_test - predizioni_migliori
    
    # Statistiche residui
    stats_residui = {
        'Media Residui': np.mean(residui),
        'Dev.Std Residui': np.std(residui),
        'Media Assoluta Residui': np.mean(np.abs(residui)),
        'Percentile 95°': np.percentile(np.abs(residui), 95)
    }
    
    print("Statistiche Residui:")
    for stat, valore in stats_residui.items():
        print(f"  {stat}: {valore:.2f} kg")
    
    # Test normalità residui
    from scipy.stats import shapiro
    stat_shapiro, p_shapiro = shapiro(residui[:min(5000, len(residui))])  # Limite per test
    print(f"\nTest Normalità Shapiro-Wilk:")
    print(f"  Statistica: {stat_shapiro:.4f}")
    print(f"  p-value: {p_shapiro:.4f}")
    print(f"  Residui normali: {'Sì' if p_shapiro > 0.05 else 'No'}")
    
    return stats_residui

def genera_report_completo():
    """Genera un report completo dell'analisi"""
    
    print("\nGENERAZIONE REPORT COMPLETO")
    print("-" * 30)
    
    with open(output_dir / "report_completo.md", 'w', encoding='utf-8') as f:
        f.write("# Report Analisi Predizione Prestazioni Powerlifting\n\n")
        
        f.write("## Panoramica del Progetto\n")
        f.write(f"- **Obiettivo**: Predire le prestazioni totali nel powerlifting\n")
        f.write(f"- **Tipo di Problema**: Regressione\n")
        f.write(f"- **Dataset**: {len(df_pulito):,} atleti\n")
        f.write(f"- **Caratteristiche**: {len(colonne_feature)}\n\n")
        
        f.write("## Risultati Migliori\n")
        f.write(f"- **Modello Migliore**: {modello_migliore['Modello']}\n")
        f.write(f"- **RMSE**: {modello_migliore['RMSE']:.2f} kg\n")
        f.write(f"- **R²**: {modello_migliore['R2']:.3f}\n")
        f.write(f"- **MAE**: {modello_migliore['MAE']:.2f} kg\n\n")
        
        f.write("## Top 5 Caratteristiche Importanti\n")
        for i, (_, riga) in enumerate(importanza_feature.head(5).iterrows(), 1):
            f.write(f"{i}. **{riga['Caratteristica']}**: {riga['Importanza']:.4f}\n")
        
        f.write("\n## Confronto Modelli (Top 5)\n")
        f.write("| Modello | RMSE | R² | MAE |\n")
        f.write("|---------|------|----|----- |\n")
        for _, riga in risultati_df.head(5).iterrows():
            f.write(f"| {riga['Modello']} | {riga['RMSE']:.2f} | {riga['R2']:.3f} | {riga['MAE']:.2f} |\n")
        
        f.write(f"\n## Conclusioni\n")
        f.write("1. I modelli tree-based mostrano generalmente le migliori prestazioni\n")
        f.write("2. Le prestazioni individuali degli esercizi sono i predittori più forti\n")
        f.write("3. I rapporti peso-prestazione aggiungono valore predittivo\n")
        f.write("4. L'analisi è stata ottimizzata per sistemi Windows\n")
    
    print("✓ Report completo generato in 'report_completo.md'")

# Esegui analisi aggiuntive se il modulo viene eseguito direttamente
if __name__ == "__main__":
    try:
        # Crea visualizzazioni
        crea_visualizzazioni()
        
        # Analisi residui
        analisi_residui_avanzata()
        
        # Genera report
        genera_report_completo()
        
        print(f"\n" + "="*80)
        print("ANALISI COMPLETA TERMINATA CON SUCCESSO!")
        print(f"Tutti i file sono stati salvati in: {output_dir.absolute()}")
        print("="*80)
        
    except Exception as e:
        print(f"⚠ Errore nelle analisi aggiuntive: {e}")
        print("L'analisi principale è comunque completa.")

# Fine del programma