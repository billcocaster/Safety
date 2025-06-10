"""
Veri işleme ve ön hazırlama modülü
Model eğitimi için veri hazırlama işlemleri
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List, Any
import pickle
import os
from utils.config import *

class DataProcessor:
    """
    Model eğitimi için veri hazırlama sınıfı
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.sequence_data = {}
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        CSV dosyasından veri yükleme
        
        Args:
            filepath: Dosya yolu
            
        Returns:
            pd.DataFrame: Yüklenen veri
        """
        print(f"Veri yükleniyor: {filepath}")
        df = pd.read_csv(filepath)
        df['CreatedAt'] = pd.to_datetime(df['CreatedAt'])
        print(f"Veri yüklendi: {df.shape}")
        return df
    
    def create_sequences(self, df: pd.DataFrame, user_id: str, sequence_length: int = 10) -> List[np.ndarray]:
        """
        Kullanıcı için zaman serisi sekansları oluşturur
        
        Args:
            df: Veri seti
            user_id: Kullanıcı kimliği
            sequence_length: Sekans uzunluğu
            
        Returns:
            List[np.ndarray]: Sekanslar
        """
        user_data = df[df['UserId'] == user_id].sort_values('CreatedAt')
        
        if len(user_data) < sequence_length:
            return []
        
        sequences = []
        for i in range(len(user_data) - sequence_length + 1):
            sequence = user_data.iloc[i:i+sequence_length]
            sequences.append(sequence)
            
        return sequences
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Özellik çıkarımı yapar
        
        Args:
            df: Ham veri
            
        Returns:
            pd.DataFrame: Özellikler eklenmiş veri
        """
        print("Özellik çıkarımı yapılıyor...")
        
        df_features = df.copy()
        
        # Zaman bazlı özellikler
        df_features['hour'] = df_features['CreatedAt'].dt.hour
        df_features['day_of_week'] = df_features['CreatedAt'].dt.dayofweek
        df_features['month'] = df_features['CreatedAt'].dt.month
        df_features['is_weekend'] = df_features['day_of_week'].isin([5, 6]).astype(int)
        df_features['is_work_hours'] = ((df_features['hour'] >= 8) & (df_features['hour'] <= 18)).astype(int)
        df_features['is_night'] = ((df_features['hour'] >= 22) | (df_features['hour'] <= 6)).astype(int)
        
        # Kullanıcı bazlı özellikler
        user_login_counts = df_features.groupby('UserId').size().reset_index(name='user_login_count')
        df_features = df_features.merge(user_login_counts, on='UserId', how='left')
        
        # IP bazlı özellikler
        ip_login_counts = df_features.groupby('ClientIP').size().reset_index(name='ip_login_count')
        df_features = df_features.merge(ip_login_counts, on='ClientIP', how='left')
        
        # Kullanıcı-IP kombinasyonu
        df_features['user_ip_count'] = df_features.groupby(['UserId', 'ClientIP']).cumcount() + 1
        
        # Son giriş zamanından geçen süre (saat)
        df_features = df_features.sort_values(['UserId', 'CreatedAt'])
        df_features['hours_since_last_login'] = df_features.groupby('UserId')['CreatedAt'].diff().dt.total_seconds() / 3600
        df_features['hours_since_last_login'] = df_features['hours_since_last_login'].fillna(0)
        
        # Kategorik değişkenler için encoding
        categorical_columns = ['MFAMethod', 'Application', 'Browser', 'OS', 'Unit', 'Title']
        
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_features[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df_features[col])
            else:
                df_features[f'{col}_encoded'] = self.label_encoders[col].transform(df_features[col])
        
        # Özellik sütunlarını belirle
        self.feature_columns = [
            'hour', 'day_of_week', 'month', 'is_weekend', 'is_work_hours', 'is_night',
            'user_login_count', 'ip_login_count', 'user_ip_count', 'hours_since_last_login',
            'MFAMethod_encoded', 'Application_encoded', 'Browser_encoded', 'OS_encoded',
            'Unit_encoded', 'Title_encoded'
        ]
        
        print(f"Özellik çıkarımı tamamlandı. Toplam {len(self.feature_columns)} özellik")
        
        return df_features
    
    def prepare_sequence_data(self, df: pd.DataFrame, sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        LSTM modeli için sekans verisi hazırlar
        
        Args:
            df: Özellikler eklenmiş veri
            sequence_length: Sekans uzunluğu
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: X (özellikler), y (hedef)
        """
        print("Sekans verisi hazırlanıyor...")
        
        sequences = []
        targets = []
        
        for user_id in df['UserId'].unique():
            user_sequences = self.create_sequences(df, user_id, sequence_length)
            
            for seq in user_sequences:
                if len(seq) == sequence_length:
                    # Özellik vektörü
                    feature_vector = seq[self.feature_columns].values
                    sequences.append(feature_vector)
                    
                    # Hedef (son girişin risk skoru)
                    target = seq.iloc[-1]['RiskScore']
                    targets.append(target)
        
        X = np.array(sequences)
        y = np.array(targets)
        
        print(f"Sekans verisi hazırlandı: {X.shape}")
        
        return X, y
    
    def prepare_tabular_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tabular model için veri hazırlar
        
        Args:
            df: Özellikler eklenmiş veri
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: X (özellikler), y (hedef)
        """
        print("Tabular veri hazırlanıyor...")
        
        X = df[self.feature_columns].values
        y = df['RiskScore'].values
        
        # Özellik standardizasyonu
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"Tabular veri hazırlandı: {X_scaled.shape}")
        
        return X_scaled, y
    
    def split_data(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
                   val_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Veriyi eğitim, validation ve test setlerine ayırır
        
        Args:
            X: Özellikler
            y: Hedefler
            test_size: Test seti oranı
            val_size: Validation seti oranı
            random_state: Rastgelelik tohumu
            
        Returns:
            Tuple: Eğitim, validation ve test setleri
        """
        # Önce test setini ayır
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=False
        )
        
        # Kalan veriyi eğitim ve validation'a ayır
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, shuffle=False
        )
        
        print(f"Veri bölünmesi:")
        print(f"  Eğitim: {X_train.shape[0]} örnek")
        print(f"  Validation: {X_val.shape[0]} örnek")
        print(f"  Test: {X_test.shape[0]} örnek")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_preprocessors(self, filepath: str):
        """
        Ön işleme modellerini kaydeder
        
        Args:
            filepath: Kayıt dosya yolu
        """
        preprocessors = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessors, f)
        
        print(f"Ön işleme modelleri kaydedildi: {filepath}")
    
    def load_preprocessors(self, filepath: str):
        """
        Ön işleme modellerini yükler
        
        Args:
            filepath: Yükleme dosya yolu
        """
        with open(filepath, 'rb') as f:
            preprocessors = pickle.load(f)
        
        self.label_encoders = preprocessors['label_encoders']
        self.scaler = preprocessors['scaler']
        self.feature_columns = preprocessors['feature_columns']
        
        print(f"Ön işleme modelleri yüklendi: {filepath}")
    
    def get_feature_importance_analysis(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Özellik önem analizi yapar
        
        Args:
            df: Özellikler eklenmiş veri
            
        Returns:
            Dict[str, float]: Özellik önem skorları
        """
        print("Özellik önem analizi yapılıyor...")
        
        # Risk skoru ile korelasyon hesapla
        correlations = {}
        
        for col in self.feature_columns:
            if col in df.columns:
                correlation = df[col].corr(df['RiskScore'])
                correlations[col] = abs(correlation)
        
        # Önem skorlarını sırala
        sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        
        print("En önemli özellikler:")
        for feature, importance in sorted_features[:10]:
            print(f"  {feature}: {importance:.4f}")
        
        return dict(sorted_features)
    
    def create_user_embeddings(self, df: pd.DataFrame, embedding_dim: int = 32) -> Dict[str, np.ndarray]:
        """
        Kullanıcı embedding'leri oluşturur
        
        Args:
            df: Veri seti
            embedding_dim: Embedding boyutu
            
        Returns:
            Dict[str, np.ndarray]: Kullanıcı embedding'leri
        """
        print("Kullanıcı embedding'leri oluşturuluyor...")
        
        user_embeddings = {}
        
        for user_id in df['UserId'].unique():
            user_data = df[df['UserId'] == user_id]
            
            # Kullanıcı özelliklerini hesapla
            user_features = {
                'avg_risk_score': user_data['RiskScore'].mean(),
                'risk_std': user_data['RiskScore'].std(),
                'login_frequency': len(user_data) / user_data['CreatedAt'].dt.days.nunique(),
                'unique_apps': user_data['Application'].nunique(),
                'unique_ips': user_data['ClientIP'].nunique(),
                'preferred_hour': user_data['CreatedAt'].dt.hour.mode().iloc[0] if not user_data['CreatedAt'].dt.hour.mode().empty else 12,
                'weekend_ratio': (user_data['CreatedAt'].dt.dayofweek >= 5).mean(),
                'night_ratio': ((user_data['CreatedAt'].dt.hour >= 22) | (user_data['CreatedAt'].dt.hour <= 6)).mean()
            }
            
            # Özellik vektörünü oluştur
            feature_vector = np.array([
                user_features['avg_risk_score'],
                user_features['risk_std'],
                user_features['login_frequency'],
                user_features['unique_apps'],
                user_features['unique_ips'],
                user_features['preferred_hour'],
                user_features['weekend_ratio'],
                user_features['night_ratio']
            ])
            
            # Embedding boyutuna uyarla (basit PCA benzeri yaklaşım)
            if len(feature_vector) < embedding_dim:
                # Eksik boyutları sıfırla
                embedding = np.pad(feature_vector, (0, embedding_dim - len(feature_vector)))
            else:
                # Fazla boyutları kes
                embedding = feature_vector[:embedding_dim]
            
            user_embeddings[user_id] = embedding
        
        print(f"{len(user_embeddings)} kullanıcı için embedding oluşturuldu")
        
        return user_embeddings 