"""
Proje konfigürasyon ayarları
"""

import os
from datetime import datetime, timedelta

# Veri konfigürasyonu
DATA_CONFIG = {
    'num_users': 1000,  # Toplam kullanıcı sayısı
    'num_logs_per_user': 50,  # Kullanıcı başına log sayısı
    'date_range_days': 90,  # Veri aralığı (gün)
    'randomness_ratio': 0.08,  # Rastgelelik oranı (%8)
}

# Risk ağırlıkları (0-1 arası, toplam 1 olmalı)
RISK_WEIGHTS = {
    'time_based': 0.25,      # Zaman bazlı risk
    'ip_based': 0.20,        # IP adresi bazlı risk
    'mfa_based': 0.15,       # MFA yöntemi bazlı risk
    'application_based': 0.10, # Uygulama bazlı risk
    'browser_based': 0.10,   # Tarayıcı bazlı risk
    'os_based': 0.10,        # İşletim sistemi bazlı risk
    'user_profile_based': 0.10  # Kullanıcı profili bazlı risk
}

# Zaman bazlı risk kuralları
TIME_RISK_RULES = {
    'work_hours': {
        'start': '08:00',
        'end': '18:00',
        'risk_multiplier': 0.5  # İş saatlerinde düşük risk
    },
    'night_hours': {
        'start': '22:00',
        'end': '06:00',
        'risk_multiplier': 2.0  # Gece saatlerinde yüksek risk
    },
    'weekend_multiplier': 1.5  # Hafta sonu risk çarpanı
}

# MFA yöntemleri ve risk seviyeleri
MFA_METHODS = {
    'SMS': {'risk_level': 0.3, 'frequency': 0.4},
    'OTP': {'risk_level': 0.2, 'frequency': 0.3},
    'Mail': {'risk_level': 0.5, 'frequency': 0.2},
    'Biometric': {'risk_level': 0.1, 'frequency': 0.1}
}

# Uygulamalar ve risk seviyeleri
APPLICATIONS = {
    'CRM': {'risk_level': 0.3, 'frequency': 0.3},
    'HR System': {'risk_level': 0.4, 'frequency': 0.2},
    'Finance': {'risk_level': 0.6, 'frequency': 0.15},
    'Admin Panel': {'risk_level': 0.8, 'frequency': 0.1},
    'Email': {'risk_level': 0.2, 'frequency': 0.25}
}

# Tarayıcılar
BROWSERS = ['Chrome', 'Firefox', 'Safari', 'Edge', 'Opera']

# İşletim sistemleri
OPERATING_SYSTEMS = ['Windows 10', 'Windows 11', 'macOS', 'Linux', 'Ubuntu']

# Kullanıcı birimleri
UNITS = ['Bilgi İşlem', 'Satış', 'Pazarlama', 'İnsan Kaynakları', 'Finans', 'Yönetim']

# Kullanıcı unvanları
TITLES = ['Uzman', 'Takım Lideri', 'Müdür', 'Direktör', 'Genel Müdür', 'Stajyer']

# Model konfigürasyonu
MODEL_CONFIG = {
    'sequence_length': 10,  # LSTM için sequence uzunluğu
    'embedding_dim': 32,    # Embedding boyutu
    'lstm_units': 64,       # LSTM birim sayısı
    'dropout_rate': 0.3,    # Dropout oranı
    'learning_rate': 0.001, # Öğrenme oranı
    'batch_size': 32,       # Batch boyutu
    'epochs': 50,           # Epoch sayısı
    'validation_split': 0.2 # Validation oranı
}

# Dosya yolları
PATHS = {
    'data_dir': 'data',
    'models_dir': 'models',
    'notebooks_dir': 'notebooks',
    'tests_dir': 'tests'
}

# Risk skoru eşikleri
RISK_THRESHOLDS = {
    'low': 30,      # Düşük risk
    'medium': 60,   # Orta risk
    'high': 80      # Yüksek risk
} 