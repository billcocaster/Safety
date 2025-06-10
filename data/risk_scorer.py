"""
Risk skoru hesaplama modülü
Kullanıcı giriş verilerine dayalı risk puanlama sistemi
"""

import pandas as pd
import numpy as np
from datetime import datetime, time
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import ipaddress
from utils.config import *

class RiskScorer:
    """
    Kullanıcı giriş verilerine dayalı risk skoru hesaplayan sınıf
    """
    
    def __init__(self, custom_weights: Dict[str, float] = None):
        """
        Args:
            custom_weights: Özel risk ağırlıkları (opsiyonel)
        """
        self.weights = custom_weights if custom_weights else RISK_WEIGHTS.copy()
        self.user_history = defaultdict(list)
        self.user_patterns = {}
        
    def calculate_time_based_risk(self, timestamp: datetime) -> float:
        """
        Zaman bazlı risk hesaplama
        
        Args:
            timestamp: Giriş zamanı
            
        Returns:
            float: 0-1 arası risk skoru
        """
        hour = timestamp.hour
        weekday = timestamp.weekday()
        
        # İş saatleri kontrolü
        work_start = time(8, 0)
        work_end = time(18, 0)
        current_time = time(hour, timestamp.minute)
        
        # Gece saatleri kontrolü
        night_start = time(22, 0)
        night_end = time(6, 0)
        
        risk_score = 0.5  # Temel risk skoru
        
        # İş saati dışı girişler
        if not (work_start <= current_time <= work_end):
            risk_score += 0.3
            
        # Gece saatleri girişleri
        if night_start <= current_time or current_time <= night_end:
            risk_score += 0.4
            
        # Hafta sonu girişleri
        if weekday >= 5:  # Cumartesi (5) ve Pazar (6)
            risk_score += 0.2
            
        # Erken sabah girişleri (05:00-07:00)
        if 5 <= hour < 7:
            risk_score += 0.1
            
        # Geç gece girişleri (23:00-01:00)
        if hour >= 23 or hour <= 1:
            risk_score += 0.2
            
        return min(risk_score, 1.0)
    
    def calculate_ip_based_risk(self, user_id: str, client_ip: str) -> float:
        """
        IP adresi bazlı risk hesaplama
        
        Args:
            user_id: Kullanıcı kimliği
            client_ip: Giriş IP adresi
            
        Returns:
            float: 0-1 arası risk skoru
        """
        if user_id not in self.user_patterns:
            return 0.5  # Kullanıcı geçmişi yoksa orta risk
            
        user_pattern = self.user_patterns[user_id]
        preferred_ip_range = user_pattern.get('preferred_ip_range', None)
        
        if not preferred_ip_range:
            return 0.5
            
        try:
            # IP adresinin tercih edilen aralıkta olup olmadığını kontrol et
            ip_obj = ipaddress.IPv4Address(client_ip)
            network = ipaddress.IPv4Network(preferred_ip_range, strict=False)
            
            if ip_obj in network:
                return 0.1  # Tercih edilen IP aralığında - düşük risk
            else:
                return 0.8  # Farklı IP aralığında - yüksek risk
                
        except ValueError:
            return 0.5  # IP adresi geçersizse orta risk
    
    def calculate_mfa_based_risk(self, user_id: str, mfa_method: str) -> float:
        """
        MFA yöntemi bazlı risk hesaplama
        
        Args:
            user_id: Kullanıcı kimliği
            mfa_method: Kullanılan MFA yöntemi
            
        Returns:
            float: 0-1 arası risk skoru
        """
        if user_id not in self.user_patterns:
            return MFA_METHODS.get(mfa_method, {}).get('risk_level', 0.5)
            
        user_pattern = self.user_patterns[user_id]
        preferred_mfa = user_pattern.get('preferred_mfa', None)
        
        if preferred_mfa == mfa_method:
            return 0.1  # Tercih edilen MFA yöntemi - düşük risk
        else:
            # Farklı MFA yöntemi kullanımı
            base_risk = MFA_METHODS.get(mfa_method, {}).get('risk_level', 0.5)
            return min(base_risk + 0.3, 1.0)  # Ek risk ekle
    
    def calculate_application_based_risk(self, user_id: str, application: str) -> float:
        """
        Uygulama bazlı risk hesaplama
        
        Args:
            user_id: Kullanıcı kimliği
            application: Giriş yapılan uygulama
            
        Returns:
            float: 0-1 arası risk skoru
        """
        base_risk = APPLICATIONS.get(application, {}).get('risk_level', 0.5)
        
        if user_id not in self.user_patterns:
            return base_risk
            
        user_pattern = self.user_patterns[user_id]
        preferred_apps = user_pattern.get('preferred_apps', [])
        
        if application in preferred_apps:
            return base_risk * 0.5  # Tercih edilen uygulama - risk azalt
        else:
            return min(base_risk * 1.5, 1.0)  # Farklı uygulama - risk artır
    
    def calculate_browser_based_risk(self, user_id: str, browser: str) -> float:
        """
        Tarayıcı bazlı risk hesaplama
        
        Args:
            user_id: Kullanıcı kimliği
            browser: Kullanılan tarayıcı
            
        Returns:
            float: 0-1 arası risk skoru
        """
        if user_id not in self.user_patterns:
            return 0.3  # Temel tarayıcı riski
            
        user_pattern = self.user_patterns[user_id]
        preferred_browser = user_pattern.get('preferred_browser', None)
        
        if preferred_browser == browser:
            return 0.1  # Tercih edilen tarayıcı - düşük risk
        else:
            return 0.6  # Farklı tarayıcı - orta risk
    
    def calculate_os_based_risk(self, user_id: str, os: str) -> float:
        """
        İşletim sistemi bazlı risk hesaplama
        
        Args:
            user_id: Kullanıcı kimliği
            os: Kullanılan işletim sistemi
            
        Returns:
            float: 0-1 arası risk skoru
        """
        if user_id not in self.user_patterns:
            return 0.3  # Temel OS riski
            
        user_pattern = self.user_patterns[user_id]
        preferred_os = user_pattern.get('preferred_os', None)
        
        if preferred_os == os:
            return 0.1  # Tercih edilen OS - düşük risk
        else:
            return 0.6  # Farklı OS - orta risk
    
    def calculate_user_profile_based_risk(self, user_id: str, unit: str, title: str) -> float:
        """
        Kullanıcı profili bazlı risk hesaplama
        
        Args:
            user_id: Kullanıcı kimliği
            unit: Kullanıcının birimi
            title: Kullanıcının unvanı
            
        Returns:
            float: 0-1 arası risk skoru
        """
        if user_id not in self.user_patterns:
            return 0.3  # Temel profil riski
            
        user_pattern = self.user_patterns[user_id]
        expected_unit = user_pattern.get('unit', None)
        expected_title = user_pattern.get('title', None)
        
        risk_score = 0.0
        
        # Birim değişikliği kontrolü
        if expected_unit and expected_unit != unit:
            risk_score += 0.4
            
        # Unvan değişikliği kontrolü
        if expected_title and expected_title != title:
            risk_score += 0.3
            
        return min(risk_score, 1.0)
    
    def build_user_patterns(self, df: pd.DataFrame):
        """
        Kullanıcı davranış kalıplarını oluşturur
        
        Args:
            df: Giriş logları DataFrame'i
        """
        print("Kullanıcı davranış kalıpları oluşturuluyor...")
        
        for user_id in df['UserId'].unique():
            user_data = df[df['UserId'] == user_id]
            
            # En sık kullanılan değerleri bul
            patterns = {
                'preferred_mfa': user_data['MFAMethod'].mode().iloc[0] if not user_data['MFAMethod'].mode().empty else None,
                'preferred_apps': user_data['Application'].value_counts().head(2).index.tolist(),
                'preferred_browser': user_data['Browser'].mode().iloc[0] if not user_data['Browser'].mode().empty else None,
                'preferred_os': user_data['OS'].mode().iloc[0] if not user_data['OS'].mode().empty else None,
                'unit': user_data['Unit'].iloc[0],  # Birim değişmez
                'title': user_data['Title'].iloc[0],  # Unvan değişmez
                'login_times': user_data['CreatedAt'].dt.hour.tolist(),
                'ip_addresses': user_data['ClientIP'].tolist()
            }
            
            # Tercih edilen IP aralığını belirle
            ip_ranges = []
            for ip in patterns['ip_addresses']:
                try:
                    ip_obj = ipaddress.IPv4Address(ip)
                    ip_ranges.append(f"{ip_obj.network_address}/{ip_obj.netmask}")
                except:
                    continue
            
            if ip_ranges:
                patterns['preferred_ip_range'] = max(set(ip_ranges), key=ip_ranges.count)
            else:
                patterns['preferred_ip_range'] = None
                
            self.user_patterns[user_id] = patterns
            
        print(f"{len(self.user_patterns)} kullanıcı için kalıp oluşturuldu")
    
    def calculate_risk_score(self, row: pd.Series) -> float:
        """
        Tek bir giriş kaydı için risk skoru hesaplar
        
        Args:
            row: Giriş kaydı (pandas Series)
            
        Returns:
            float: 0-100 arası risk skoru
        """
        user_id = row['UserId']
        timestamp = pd.to_datetime(row['CreatedAt'])
        
        # Her bileşen için risk skorlarını hesapla
        time_risk = self.calculate_time_based_risk(timestamp)
        ip_risk = self.calculate_ip_based_risk(user_id, row['ClientIP'])
        mfa_risk = self.calculate_mfa_based_risk(user_id, row['MFAMethod'])
        app_risk = self.calculate_application_based_risk(user_id, row['Application'])
        browser_risk = self.calculate_browser_based_risk(user_id, row['Browser'])
        os_risk = self.calculate_os_based_risk(user_id, row['OS'])
        profile_risk = self.calculate_user_profile_based_risk(user_id, row['Unit'], row['Title'])
        
        # Ağırlıklı ortalama hesapla
        weighted_risk = (
            time_risk * self.weights['time_based'] +
            ip_risk * self.weights['ip_based'] +
            mfa_risk * self.weights['mfa_based'] +
            app_risk * self.weights['application_based'] +
            browser_risk * self.weights['browser_based'] +
            os_risk * self.weights['os_based'] +
            profile_risk * self.weights['user_profile_based']
        )
        
        # 0-100 arası skala
        final_risk_score = weighted_risk * 100
        
        return round(final_risk_score, 2)
    
    def add_risk_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        DataFrame'e risk skorlarını ekler
        
        Args:
            df: Giriş logları DataFrame'i
            
        Returns:
            pd.DataFrame: Risk skorları eklenmiş DataFrame
        """
        print("Risk skorları hesaplanıyor...")
        
        # Kullanıcı kalıplarını oluştur
        self.build_user_patterns(df)
        
        # Her satır için risk skoru hesapla
        risk_scores = []
        for idx, row in df.iterrows():
            risk_score = self.calculate_risk_score(row)
            risk_scores.append(risk_score)
            
            if idx % 1000 == 0:
                print(f"İşlenen kayıt: {idx}/{len(df)}")
        
        # Risk skorunu DataFrame'e ekle
        df_with_risk = df.copy()
        df_with_risk['RiskScore'] = risk_scores
        
        print("Risk skorları hesaplandı")
        
        return df_with_risk
    
    def get_risk_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Risk skorları için özet istatistikler
        
        Args:
            df: Risk skorları eklenmiş DataFrame
            
        Returns:
            Dict: Risk özeti
        """
        risk_scores = df['RiskScore']
        
        summary = {
            'total_logs': len(df),
            'mean_risk': risk_scores.mean(),
            'median_risk': risk_scores.median(),
            'std_risk': risk_scores.std(),
            'min_risk': risk_scores.min(),
            'max_risk': risk_scores.max(),
            'low_risk_count': len(risk_scores[risk_scores < RISK_THRESHOLDS['low']]),
            'medium_risk_count': len(risk_scores[(risk_scores >= RISK_THRESHOLDS['low']) & 
                                               (risk_scores < RISK_THRESHOLDS['medium'])]),
            'high_risk_count': len(risk_scores[risk_scores >= RISK_THRESHOLDS['high']]),
            'risk_distribution': {
                '0-20': len(risk_scores[risk_scores < 20]),
                '20-40': len(risk_scores[(risk_scores >= 20) & (risk_scores < 40)]),
                '40-60': len(risk_scores[(risk_scores >= 40) & (risk_scores < 60)]),
                '60-80': len(risk_scores[(risk_scores >= 60) & (risk_scores < 80)]),
                '80-100': len(risk_scores[risk_scores >= 80])
            }
        }
        
        return summary 