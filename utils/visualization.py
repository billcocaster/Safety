"""
Görselleştirme modülü
Veri analizi ve model sonuçları için görselleştirme fonksiyonları
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import json
from utils.config import *

def plot_risk_distribution(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Risk skoru dağılımını görselleştirir
    
    Args:
        df: Veri seti
        save_path: Kayıt yolu (opsiyonel)
    """
    plt.figure(figsize=(15, 10))
    
    # Ana risk dağılımı
    plt.subplot(2, 3, 1)
    plt.hist(df['RiskScore'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(df['RiskScore'].mean(), color='red', linestyle='--', label=f'Ortalama: {df["RiskScore"].mean():.2f}')
    plt.axvline(df['RiskScore'].median(), color='green', linestyle='--', label=f'Medyan: {df["RiskScore"].median():.2f}')
    plt.xlabel('Risk Skoru')
    plt.ylabel('Frekans')
    plt.title('Risk Skoru Dağılımı')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Risk kategorileri
    plt.subplot(2, 3, 2)
    risk_categories = pd.cut(df['RiskScore'], 
                           bins=[0, RISK_THRESHOLDS['low'], RISK_THRESHOLDS['medium'], 100],
                           labels=['Düşük', 'Orta', 'Yüksek'])
    category_counts = risk_categories.value_counts()
    colors = ['green', 'orange', 'red']
    plt.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%', colors=colors)
    plt.title('Risk Kategorileri Dağılımı')
    
    # Saat bazlı risk
    plt.subplot(2, 3, 3)
    df['Hour'] = pd.to_datetime(df['CreatedAt']).dt.hour
    hourly_risk = df.groupby('Hour')['RiskScore'].mean()
    plt.plot(hourly_risk.index, hourly_risk.values, marker='o', linewidth=2, markersize=6)
    plt.xlabel('Saat')
    plt.ylabel('Ortalama Risk Skoru')
    plt.title('Saat Bazlı Ortalama Risk')
    plt.grid(True, alpha=0.3)
    plt.xticks(range(0, 24, 2))
    
    # Gün bazlı risk
    plt.subplot(2, 3, 4)
    df['DayOfWeek'] = pd.to_datetime(df['CreatedAt']).dt.dayofweek
    day_names = ['Pazartesi', 'Salı', 'Çarşamba', 'Perşembe', 'Cuma', 'Cumartesi', 'Pazar']
    daily_risk = df.groupby('DayOfWeek')['RiskScore'].mean()
    plt.bar(range(7), daily_risk.values, color=['blue']*5 + ['red']*2)
    plt.xlabel('Gün')
    plt.ylabel('Ortalama Risk Skoru')
    plt.title('Gün Bazlı Ortalama Risk')
    plt.xticks(range(7), day_names, rotation=45)
    plt.grid(True, alpha=0.3)
    
    # MFA yöntemi bazlı risk
    plt.subplot(2, 3, 5)
    mfa_risk = df.groupby('MFAMethod')['RiskScore'].mean().sort_values(ascending=False)
    plt.barh(mfa_risk.index, mfa_risk.values, color='lightcoral')
    plt.xlabel('Ortalama Risk Skoru')
    plt.title('MFA Yöntemi Bazlı Risk')
    plt.grid(True, alpha=0.3)
    
    # Uygulama bazlı risk
    plt.subplot(2, 3, 6)
    app_risk = df.groupby('Application')['RiskScore'].mean().sort_values(ascending=False)
    plt.barh(app_risk.index, app_risk.values, color='lightgreen')
    plt.xlabel('Ortalama Risk Skoru')
    plt.title('Uygulama Bazlı Risk')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Risk dağılım grafiği kaydedildi: {save_path}")
    
    plt.show()

def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None):
    """
    Model eğitim geçmişini görselleştirir
    
    Args:
        history: Eğitim geçmişi
        save_path: Kayıt yolu (opsiyonel)
    """
    plt.figure(figsize=(15, 5))
    
    # Loss grafiği
    plt.subplot(1, 3, 1)
    plt.plot(history['loss'], label='Eğitim Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # MAE grafiği
    plt.subplot(1, 3, 2)
    plt.plot(history['mae'], label='Eğitim MAE', linewidth=2)
    plt.plot(history['val_mae'], label='Validation MAE', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Model MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Son epoch değerleri
    plt.subplot(1, 3, 3)
    metrics = ['Final Train Loss', 'Final Val Loss', 'Final Train MAE', 'Final Val MAE']
    values = [history['loss'][-1], history['val_loss'][-1], 
              history['mae'][-1], history['val_mae'][-1]]
    colors = ['blue', 'orange', 'green', 'red']
    
    bars = plt.bar(metrics, values, color=colors, alpha=0.7)
    plt.ylabel('Değer')
    plt.title('Son Epoch Metrikleri')
    plt.xticks(rotation=45)
    
    # Değerleri çubukların üzerine yaz
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Eğitim geçmişi grafiği kaydedildi: {save_path}")
    
    plt.show()

def plot_prediction_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, 
                            save_path: Optional[str] = None):
    """
    Tahmin vs gerçek değer grafiği
    
    Args:
        y_true: Gerçek değerler
        y_pred: Tahmin edilen değerler
        save_path: Kayıt yolu (opsiyonel)
    """
    plt.figure(figsize=(15, 5))
    
    # Scatter plot
    plt.subplot(1, 3, 1)
    plt.scatter(y_true, y_pred, alpha=0.6, color='blue')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
             'r--', linewidth=2, label='Mükemmel Tahmin')
    plt.xlabel('Gerçek Risk Skoru')
    plt.ylabel('Tahmin Edilen Risk Skoru')
    plt.title('Tahmin vs Gerçek')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Hata dağılımı
    plt.subplot(1, 3, 2)
    errors = y_true - y_pred
    plt.hist(errors, bins=50, alpha=0.7, color='green', edgecolor='black')
    plt.axvline(0, color='red', linestyle='--', label='Hata = 0')
    plt.xlabel('Tahmin Hatası')
    plt.ylabel('Frekans')
    plt.title('Hata Dağılımı')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Residual plot
    plt.subplot(1, 3, 3)
    plt.scatter(y_pred, errors, alpha=0.6, color='orange')
    plt.axhline(0, color='red', linestyle='--', label='Hata = 0')
    plt.xlabel('Tahmin Edilen Risk Skoru')
    plt.ylabel('Tahmin Hatası')
    plt.title('Residual Plot')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Tahmin analizi grafiği kaydedildi: {save_path}")
    
    plt.show()

def plot_feature_importance(feature_importance: Dict[str, float], 
                          save_path: Optional[str] = None, top_n: int = 15):
    """
    Özellik önem analizi görselleştirmesi
    
    Args:
        feature_importance: Özellik önem skorları
        save_path: Kayıt yolu (opsiyonel)
        top_n: Gösterilecek en önemli özellik sayısı
    """
    # En önemli özellikleri al
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features = [item[0] for item in sorted_features]
    importance = [item[1] for item in sorted_features]
    
    plt.figure(figsize=(12, 8))
    
    # Yatay bar grafiği
    bars = plt.barh(range(len(features)), importance, color='skyblue', alpha=0.8)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Özellik Önem Skoru')
    plt.title(f'En Önemli {top_n} Özellik')
    plt.gca().invert_yaxis()
    
    # Değerleri çubukların üzerine yaz
    for i, (bar, value) in enumerate(zip(bars, importance)):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{value:.4f}', ha='left', va='center')
    
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Özellik önem grafiği kaydedildi: {save_path}")
    
    plt.show()

def plot_user_behavior_patterns(df: pd.DataFrame, user_id: str, 
                              save_path: Optional[str] = None):
    """
    Kullanıcı davranış kalıplarını görselleştirir
    
    Args:
        df: Veri seti
        user_id: Kullanıcı kimliği
        save_path: Kayıt yolu (opsiyonel)
    """
    user_data = df[df['UserId'] == user_id].copy()
    user_data['CreatedAt'] = pd.to_datetime(user_data['CreatedAt'])
    user_data['Hour'] = user_data['CreatedAt'].dt.hour
    user_data['DayOfWeek'] = user_data['CreatedAt'].dt.dayofweek
    
    plt.figure(figsize=(15, 10))
    
    # Giriş zamanları heatmap
    plt.subplot(2, 3, 1)
    time_matrix = np.zeros((7, 24))
    for _, row in user_data.iterrows():
        time_matrix[row['DayOfWeek'], row['Hour']] += 1
    
    day_names = ['Pzt', 'Sal', 'Çar', 'Per', 'Cum', 'Cmt', 'Paz']
    plt.imshow(time_matrix, cmap='YlOrRd', aspect='auto')
    plt.colorbar(label='Giriş Sayısı')
    plt.xlabel('Saat')
    plt.ylabel('Gün')
    plt.title(f'Kullanıcı {user_id} - Giriş Zamanları')
    plt.xticks(range(0, 24, 2))
    plt.yticks(range(7), day_names)
    
    # Risk skoru zaman serisi
    plt.subplot(2, 3, 2)
    plt.plot(user_data['CreatedAt'], user_data['RiskScore'], marker='o', linewidth=2)
    plt.xlabel('Tarih')
    plt.ylabel('Risk Skoru')
    plt.title('Risk Skoru Zaman Serisi')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # IP adresi kullanımı
    plt.subplot(2, 3, 3)
    ip_counts = user_data['ClientIP'].value_counts().head(10)
    plt.barh(range(len(ip_counts)), ip_counts.values, color='lightgreen')
    plt.yticks(range(len(ip_counts)), ip_counts.index)
    plt.xlabel('Kullanım Sayısı')
    plt.title('En Çok Kullanılan IP Adresleri')
    plt.gca().invert_yaxis()
    
    # MFA yöntemi kullanımı
    plt.subplot(2, 3, 4)
    mfa_counts = user_data['MFAMethod'].value_counts()
    plt.pie(mfa_counts.values, labels=mfa_counts.index, autopct='%1.1f%%')
    plt.title('MFA Yöntemi Kullanımı')
    
    # Uygulama kullanımı
    plt.subplot(2, 3, 5)
    app_counts = user_data['Application'].value_counts()
    plt.barh(range(len(app_counts)), app_counts.values, color='lightcoral')
    plt.yticks(range(len(app_counts)), app_counts.index)
    plt.xlabel('Kullanım Sayısı')
    plt.title('Uygulama Kullanımı')
    plt.gca().invert_yaxis()
    
    # Risk skoru dağılımı
    plt.subplot(2, 3, 6)
    plt.hist(user_data['RiskScore'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(user_data['RiskScore'].mean(), color='red', linestyle='--', 
                label=f'Ortalama: {user_data["RiskScore"].mean():.2f}')
    plt.xlabel('Risk Skoru')
    plt.ylabel('Frekans')
    plt.title('Risk Skoru Dağılımı')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Kullanıcı davranış kalıpları grafiği kaydedildi: {save_path}")
    
    plt.show()

def plot_model_comparison(comparison_results: Dict[str, Any], 
                         save_path: Optional[str] = None):
    """
    Model karşılaştırma grafiği
    
    Args:
        comparison_results: Model karşılaştırma sonuçları
        save_path: Kayıt yolu (opsiyonel)
    """
    models = list(comparison_results.keys())
    metrics = ['test_mae', 'test_rmse', 'test_r_squared']
    metric_names = ['Test MAE', 'Test RMSE', 'Test R²']
    
    plt.figure(figsize=(15, 5))
    
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        plt.subplot(1, 3, i+1)
        
        values = [comparison_results[model]['evaluation'][metric] for model in models]
        colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum']
        
        bars = plt.bar(models, values, color=colors[:len(models)], alpha=0.8)
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} Karşılaştırması')
        plt.xticks(rotation=45)
        
        # Değerleri çubukların üzerine yaz
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.4f}', ha='center', va='bottom')
        
        plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model karşılaştırma grafiği kaydedildi: {save_path}")
    
    plt.show()

def plot_risk_heatmap(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Risk skoru heatmap'i
    
    Args:
        df: Veri seti
        save_path: Kayıt yolu (opsiyonel)
    """
    # Saat ve gün bazlı ortalama risk skorları
    df['Hour'] = pd.to_datetime(df['CreatedAt']).dt.hour
    df['DayOfWeek'] = pd.to_datetime(df['CreatedAt']).dt.dayofweek
    
    risk_matrix = df.groupby(['DayOfWeek', 'Hour'])['RiskScore'].mean().unstack()
    
    plt.figure(figsize=(15, 8))
    
    # Ana heatmap
    plt.subplot(1, 2, 1)
    day_names = ['Pazartesi', 'Salı', 'Çarşamba', 'Perşembe', 'Cuma', 'Cumartesi', 'Pazar']
    sns.heatmap(risk_matrix, cmap='RdYlBu_r', annot=True, fmt='.1f', 
                xticklabels=range(0, 24, 2), yticklabels=day_names)
    plt.xlabel('Saat')
    plt.ylabel('Gün')
    plt.title('Gün ve Saat Bazlı Ortalama Risk Skoru')
    
    # Risk kategorileri heatmap
    plt.subplot(1, 2, 2)
    df['RiskCategory'] = pd.cut(df['RiskScore'], 
                               bins=[0, RISK_THRESHOLDS['low'], RISK_THRESHOLDS['medium'], 100],
                               labels=['Düşük', 'Orta', 'Yüksek'])
    
    category_matrix = pd.crosstab(df['DayOfWeek'], df['Hour'], values=df['RiskCategory'], 
                                 aggfunc=lambda x: (x == 'Yüksek').mean())
    
    sns.heatmap(category_matrix, cmap='Reds', annot=True, fmt='.2f',
                xticklabels=range(0, 24, 2), yticklabels=day_names)
    plt.xlabel('Saat')
    plt.ylabel('Gün')
    plt.title('Yüksek Risk Oranı (Gün/Saat)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Risk heatmap grafiği kaydedildi: {save_path}")
    
    plt.show()

def set_plot_style():
    """
    Matplotlib stil ayarları
    """
    plt.style.use('seaborn-v0_8')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16
    
    # Türkçe karakter desteği
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'sans-serif'] 