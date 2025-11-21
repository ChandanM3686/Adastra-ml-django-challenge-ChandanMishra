
import os
import re
import numpy as np
import pandas as pd
from django.shortcuts import render
from django.conf import settings
from django.http import HttpResponse, FileResponse, Http404
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from catboost import CatBoostRegressor
import joblib

def convert_km_to_num(s):
    
    if pd.isna(s):
        return np.nan
    s = str(s).strip()
    m = re.match(r'^([0-9,.]*\d)([kKmM])$', s)
    if m:
        num = m.group(1).replace(',', '')
        suf = m.group(2).lower()
        try:
            val = float(num)
            if suf == 'k':
                return val * 1_000
            if suf == 'm':
                return val * 1_000_000
        except:
            return np.nan
    return np.nan

def make_numeric_series(srs):
   
    cleaned = srs.astype(str).str.strip().replace(
        {'nan':'', 'none':'', 'na':'', 'n/a':'', '-':''}, regex=False
    )
    shorthand_converted = cleaned.apply(convert_km_to_num)
    mask_shorthand = shorthand_converted.notna()
    cleaned_numeric = cleaned.copy()
    cleaned_numeric.loc[mask_shorthand] = shorthand_converted.loc[mask_shorthand].astype(str)
    cleaned_numeric = cleaned_numeric.str.replace(r'[^\d\.\-]', '', regex=True)
    numeric = pd.to_numeric(cleaned_numeric, errors='coerce')
    return numeric

def parse_money_to_float(x):
    
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    neg = False
    if s.startswith('(') and s.endswith(')'):
        neg = True
        s = s[1:-1].strip()
    s = re.sub(r'[^\d\.\-kKmM,]', '', s)
    m = re.match(r'^([0-9,]*\.?[0-9]*)([kKmM])$', s)
    if m:
        num = m.group(1).replace(',', '')
        suf = m.group(2).lower()
        try:
            val = float(num)
            if suf == 'k':
                val *= 1_000
            else:
                val *= 1_000_000
            return -val if neg else val
        except:
            return np.nan
    try:
        val = float(s.replace(',', ''))
        return -val if neg else val
    except:
        return np.nan

def prepare_and_train(train_df, test_df, date_format='%d-%m-%Y'):
    
    train_df['Ad_Date'] = pd.to_datetime(train_df['Ad_Date'], format=date_format, errors='coerce')
    test_df['Ad_Date']  = pd.to_datetime(test_df['Ad_Date'],  format=date_format, errors='coerce')

    for col in ['Clicks', 'Impressions', 'Cost', 'Leads', 'Conversions']:
        if col in train_df.columns:
            train_df[col] = make_numeric_series(train_df[col])
        if col in test_df.columns:
            test_df[col] = make_numeric_series(test_df[col])

    numeric_cols = [c for c in ['Clicks','Impressions','Cost','Leads','Conversions'] if c in train_df.columns]
    for c in numeric_cols:
        med = train_df[c].median(skipna=True)
        if np.isnan(med):
            med = 0.0
        train_df[c].fillna(med, inplace=True)
        if c in test_df.columns:
            test_df[c].fillna(med, inplace=True)

    train_df['Conversion Rate'] = (train_df.get('Conversions',0) / train_df.get('Clicks',1)).replace([np.inf, -np.inf], np.nan).fillna(0)
    test_df['Conversion Rate']  = (test_df.get('Conversions',0)  / test_df.get('Clicks',1)).replace([np.inf, -np.inf], np.nan).fillna(0)

    train_df['ctr'] = (train_df['Clicks'] / train_df['Impressions']).replace([np.inf, -np.inf], np.nan).fillna(0)
    test_df['ctr']  = (test_df['Clicks']  / test_df['Impressions']).replace([np.inf, -np.inf], np.nan).fillna(0)

    train_df['cost_per_click'] = (train_df['Cost'] / train_df['Clicks']).replace([np.inf, -np.inf], np.nan).fillna(0)
    test_df['cost_per_click']  = (test_df['Cost']  / test_df['Clicks']).replace([np.inf, -np.inf], np.nan).fillna(0)

  
    for df in (train_df, test_df):
        df['ad_year'] = df['Ad_Date'].dt.year.fillna(0).astype(int)
        df['ad_month'] = df['Ad_Date'].dt.month.fillna(0).astype(int)
        df['ad_day'] = df['Ad_Date'].dt.day.fillna(0).astype(int)
        df['ad_weekday'] = df['Ad_Date'].dt.weekday.fillna(0).astype(int)

    if 'Sale_Amount' in train_df.columns:
        train_df['Sale_Amount_numeric'] = train_df['Sale_Amount'].apply(parse_money_to_float)
        train_df = train_df.dropna(subset=['Sale_Amount_numeric']).copy()
    else:
        raise ValueError("train_df must contain 'Sale_Amount' column")

    cat_features_names = ['Campaign_Name','Location','Device','Keyword']
    cat_present = [c for c in cat_features_names if c in train_df.columns]
    for c in cat_present:
        train_df[c] = train_df[c].astype(str).fillna('missing')
        if c in test_df.columns:
            test_df[c] = test_df[c].astype(str).fillna('missing')

    y = train_df['Sale_Amount_numeric'].astype(float)
    X = train_df.drop(columns=['Sale_Amount','Sale_Amount_numeric','Ad_ID','Ad_Date'], errors='ignore')
    X_test = test_df.drop(columns=['Ad_ID','Ad_Date'], errors='ignore')

 
    for col in X.columns:
        if col not in X_test.columns:
            X_test[col] = 0 if pd.api.types.is_numeric_dtype(X[col]) else "missing"
    X_test = X_test.reindex(columns=X.columns, fill_value=0)

    for c in cat_present:
        X[c] = X[c].astype(str)
        X_test[c] = X_test[c].astype(str)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

    model_cv = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.03,
        depth=6,
        loss_function='RMSE',
        verbose=100
    )

    model_cv.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        cat_features=cat_present if len(cat_present) > 0 else None,
        early_stopping_rounds=100,
        use_best_model=True
    )

    pred_val = model_cv.predict(X_val)
    
    val_rmse = float(mean_squared_error(y_val, pred_val)) ** 0.5
    mape = float(mean_absolute_percentage_error(y_val, pred_val)) if len(y_val)>0 else np.nan
    approx_acc = float(1 - mape) if not np.isnan(mape) else None

    best_it = getattr(model_cv, 'best_iteration_', None)
    final_iters = best_it if best_it is not None and best_it > 0 else 800
    final_model = CatBoostRegressor(
        iterations=final_iters,
        learning_rate=0.03,
        depth=6,
        loss_function='RMSE',
        verbose=100
    )
    final_model.fit(X, y, cat_features=cat_present if len(cat_present) > 0 else None)

    preds_test = final_model.predict(X_test)

 
    out_dir = os.path.join(settings.BASE_DIR, 'model_output')
    os.makedirs(out_dir, exist_ok=True)

    model_path = os.path.join(out_dir, 'final_model.joblib')
    submission_path = os.path.join(out_dir, 'submission.csv')

    joblib.dump(final_model, model_path)

    submission = pd.DataFrame({'Ad_ID': test_df.get('Ad_ID', np.arange(len(preds_test))),
                               'Sale_Amount': preds_test})
    submission.to_csv(submission_path, index=False)

    return {
        'val_rmse': val_rmse,
        'mape': mape,
        'approx_accuracy': approx_acc,
        'model_path': model_path,
        'submission_path': submission_path,
        'sample_preds': preds_test[:10].tolist()
    }

def train_view(request):
    context = {}
    try:
        if request.method == 'POST':
            train_file = request.FILES.get('train_file')
            test_file = request.FILES.get('test_file')

            if train_file:
                train_df = pd.read_csv(train_file)
            else:
                train_path = os.path.join(settings.BASE_DIR, 'train.csv')
                train_df = pd.read_csv(train_path)

            if test_file:
                test_df = pd.read_csv(test_file)
            else:
                test_path = os.path.join(settings.BASE_DIR, 'sample_test.csv')
                test_df = pd.read_csv(test_path)

            results = prepare_and_train(train_df, test_df, date_format='%d-%m-%Y')
            context.update(results)
            context['message'] = 'Training complete. Model and submission saved.'
        return render(request, 'upload.html', context)
    except Exception as e:
        context['error'] = str(e)
        return render(request, 'upload.html', context)

def download_submission(request):
    
    out_dir = os.path.join(settings.BASE_DIR, 'model_output')
    filepath = os.path.join(out_dir, 'submission.csv')

    if not os.path.exists(filepath):
        raise Http404("submission.csv not found. Run training first.")

    return FileResponse(open(filepath, 'rb'), as_attachment=True, filename='submission.csv')

def upload_predict(request):
    return train_view(request)

