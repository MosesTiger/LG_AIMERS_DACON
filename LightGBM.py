import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import optuna

# 설정
ROOT_DIR = "."
RANDOM_STATE = 42
N_TRIALS = 50  # 최적화 시도 횟수를 50으로 설정

def load_and_explore_data(file_path):
    data = pd.read_csv(file_path)
    print(f"Data loading complete. Shape: {data.shape}")
    
    # 기본 통계 정보
    print(data.describe())
    
    # 결측치 정보
    missing_data = data.isnull().sum()
    print("Missing data:\n", missing_data[missing_data > 0])
    
    return data

def preprocess_data(data, is_train=True):
    # 'OK' 값을 np.nan으로 변환
    data = data.replace({'OK': np.nan})
    
    # 결측치가 80% 이상인 열 제거
    threshold = 0.8
    data = data.dropna(thresh=int((1-threshold) * len(data)), axis=1)
    
    # 공정별로 컬럼 분리
    dam_cols = [col for col in data.columns if '_Dam' in col]
    autoclave_cols = [col for col in data.columns if '_AutoClave' in col]
    fill1_cols = [col for col in data.columns if '_Fill1' in col]
    fill2_cols = [col for col in data.columns if '_Fill2' in col]

    # 숫자형 데이터만 선택하여 mean 계산
    data['dam_mean'] = pd.to_numeric(data[dam_cols].stack(), errors='coerce').groupby(level=0).mean()
    data['autoclave_mean'] = pd.to_numeric(data[autoclave_cols].stack(), errors='coerce').groupby(level=0).mean()
    data['fill1_mean'] = pd.to_numeric(data[fill1_cols].stack(), errors='coerce').groupby(level=0).mean()
    data['fill2_mean'] = pd.to_numeric(data[fill2_cols].stack(), errors='coerce').groupby(level=0).mean()

    # 수치형 컬럼과 범주형 컬럼을 분리
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    
    if is_train and 'target' in categorical_cols:
        categorical_cols.remove('target')
    
    # 전처리 파이프라인 설정
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    return preprocessor, data

def select_features(X, y):
    selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE), threshold='median')
    selector.fit(X, y)
    return selector

def define_model(trial):
    chosen_model = trial.suggest_categorical('model', ['rf', 'gb', 'xgb', 'lgbm'])
    
    if chosen_model == 'rf':
        return RandomForestClassifier(
            n_estimators=trial.suggest_int('n_estimators', 100, 1000),
            max_depth=trial.suggest_int('max_depth', 5, 30),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
            random_state=RANDOM_STATE
        )
    elif chosen_model == 'gb':
        return GradientBoostingClassifier(
            n_estimators=trial.suggest_int('n_estimators', 100, 1000),
            learning_rate=trial.suggest_float('learning_rate', 1e-3, 1.0, log=True),
            max_depth=trial.suggest_int('max_depth', 3, 10),
            random_state=RANDOM_STATE
        )
    elif chosen_model == 'xgb':
        return XGBClassifier(
            n_estimators=trial.suggest_int('n_estimators', 100, 1000),
            max_depth=trial.suggest_int('max_depth', 3, 10),
            learning_rate=trial.suggest_float('learning_rate', 1e-3, 1.0, log=True),
            subsample=trial.suggest_float('subsample', 0.6, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
            random_state=RANDOM_STATE
        )
    else:  # lgbm
        return LGBMClassifier(
            n_estimators=trial.suggest_int('n_estimators', 100, 1000),
            num_leaves=trial.suggest_int('num_leaves', 20, 100),
            learning_rate=trial.suggest_float('learning_rate', 1e-3, 1.0, log=True),
            subsample=trial.suggest_float('subsample', 0.6, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
            random_state=RANDOM_STATE
        )

def objective(trial):
    model = define_model(trial)
    
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('selector', selector),
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('classifier', model)
    ])
    
    scores = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]  # numpy 배열 인덱싱 사용
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        score = f1_score(y_val, y_pred, pos_label=1)  # pos_label을 숫자로 변경
        scores.append(score)
    
    return np.mean(scores)

if __name__ == "__main__":
    train_data = load_and_explore_data(os.path.join(ROOT_DIR, "train.csv"))
    test_data = load_and_explore_data(os.path.join(ROOT_DIR, "test.csv"))

    preprocessor, train_data_processed = preprocess_data(train_data)
    _, test_data_processed = preprocess_data(test_data, is_train=False)

    X = train_data_processed.drop('target', axis=1)
    y = train_data_processed['target']

    # target을 숫자로 인코딩
    le = LabelEncoder()
    y = le.fit_transform(y)

    selector = select_features(preprocessor.fit_transform(X), y)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=N_TRIALS)

    print(f'Best trial: score {study.best_value:.4f}, params {study.best_params}')

    best_model = define_model(study.best_trial)
    
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('selector', selector),
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('classifier', best_model)
    ])

    pipeline.fit(X, y)

    # 검증 데이터에 대한 성능 평가
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    pipeline.fit(X_train, y_train)
    val_pred = pipeline.predict(X_val)
    print("Validation Performance:")
    print(classification_report(y_val, val_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, val_pred))
