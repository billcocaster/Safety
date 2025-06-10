"""
Mock veri üretici modülü
Gerçekçi kullanıcı giriş logları oluşturur
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any
import ipaddress
from utils.config import *

class MockDataGenerator:
    """
    Gerçekçi kullanıcı giriş logları üreten sınıf
    """
    
    def __init__(self):
        self.users = []
        self.user_profiles = {}
        self.ip_ranges = self._generate_ip_ranges()
        
    def _generate_ip_ranges(self) -> List[str]:
        """Farklı lokasyonlar için IP aralıkları oluşturur"""
        ip_ranges = []
        
        # Türkiye IP aralıkları (örnek)
        for i in range(10):
            base_ip = f"192.168.{i}.0/24"
            ip_ranges.append(base_ip)
            
        # Farklı şehirler için IP aralıkları
        for i in range(5):
            base_ip = f"10.{i}.0.0/16"
            ip_ranges.append(base_ip)
            
        return ip_ranges
    
    def _generate_user_profiles(self) -> Dict[str, Dict]:
        """Kullanıcı profilleri oluşturur"""
        profiles = {}
        
        for i in range(DATA_CONFIG['num_users']):
            user_id = f"U{i+1:05d}"
            
            # Kullanıcı tercihleri
            preferred_mfa = np.random.choice(
                list(MFA_METHODS.keys()), 
                p=[MFA_METHODS[m]['frequency'] for m in MFA_METHODS.keys()]
            )
            
            preferred_apps = np.random.choice(
                list(APPLICATIONS.keys()), 
                size=2, 
                replace=False,
                p=[APPLICATIONS[a]['frequency'] for a in APPLICATIONS.keys()]
            )
            
            preferred_browser = np.random.choice(BROWSERS, p=[0.4, 0.2, 0.2, 0.15, 0.05])
            preferred_os = np.random.choice(OPERATING_SYSTEMS, p=[0.4, 0.2, 0.2, 0.1, 0.1])
            
            # Çalışma saatleri (kullanıcıya özel)
            work_start = random.randint(7, 9)  # 07:00 - 09:00 arası başlangıç
            work_end = random.randint(17, 19)  # 17:00 - 19:00 arası bitiş
            
            # Tercih edilen IP aralığı
            preferred_ip_range = random.choice(self.ip_ranges)
            
            profiles[user_id] = {
                'preferred_mfa': preferred_mfa,
                'preferred_apps': preferred_apps,
                'preferred_browser': preferred_browser,
                'preferred_os': preferred_os,
                'work_start_hour': work_start,
                'work_end_hour': work_end,
                'preferred_ip_range': preferred_ip_range,
                'unit': random.choice(UNITS),
                'title': random.choice(TITLES),
                'login_frequency': random.uniform(0.8, 1.2)  # Giriş sıklığı çarpanı
            }
            
        return profiles
    
    def _generate_timestamp(self, user_id: str, day: int) -> datetime:
        """Kullanıcıya özel giriş zamanı oluşturur"""
        profile = self.user_profiles[user_id]
        
        # Temel tarih
        base_date = datetime.now() - timedelta(days=DATA_CONFIG['date_range_days'] - day)
        
        # Hafta içi/sonu kontrolü
        if base_date.weekday() >= 5:  # Hafta sonu
            # Hafta sonu daha az giriş
            if random.random() > 0.3:
                return None
            hour = random.randint(10, 18)  # Hafta sonu saatleri
        else:  # Hafta içi
            # İş saatleri içinde daha fazla giriş
            if random.random() < 0.7:
                hour = random.randint(profile['work_start_hour'], profile['work_end_hour'])
            else:
                # İş saati dışı girişler
                if random.random() < 0.5:
                    hour = random.randint(6, profile['work_start_hour'])
                else:
                    hour = random.randint(profile['work_end_hour'], 23)
        
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        
        return base_date.replace(hour=hour, minute=minute, second=second)
    
    def _generate_ip_address(self, user_id: str) -> str:
        """Kullanıcıya özel IP adresi oluşturur"""
        profile = self.user_profiles[user_id]
        
        # %85 ihtimalle tercih edilen IP aralığından
        if random.random() < 0.85:
            ip_range = profile['preferred_ip_range']
        else:
            # %15 ihtimalle farklı IP aralığından (şüpheli durum)
            ip_range = random.choice([r for r in self.ip_ranges if r != profile['preferred_ip_range']])
        
        # IP aralığından rastgele IP seç
        network = ipaddress.IPv4Network(ip_range, strict=False)
        ip = str(random.choice(list(network.hosts())))
        
        return ip
    
    def _generate_login_data(self, user_id: str, timestamp: datetime) -> Dict[str, Any]:
        """Tek bir giriş kaydı oluşturur"""
        profile = self.user_profiles[user_id]
        
        # Rastgelelik kontrolü
        use_random = random.random() < DATA_CONFIG['randomness_ratio']
        
        # MFA yöntemi seçimi
        if use_random:
            mfa_method = random.choice(list(MFA_METHODS.keys()))
        else:
            mfa_method = profile['preferred_mfa']
        
        # Uygulama seçimi
        if use_random:
            application = random.choice(list(APPLICATIONS.keys()))
        else:
            application = random.choice(profile['preferred_apps'])
        
        # Tarayıcı seçimi
        if use_random:
            browser = random.choice(BROWSERS)
        else:
            browser = profile['preferred_browser']
        
        # İşletim sistemi seçimi
        if use_random:
            os = random.choice(OPERATING_SYSTEMS)
        else:
            os = profile['preferred_os']
        
        return {
            'UserId': user_id,
            'MFAMethod': mfa_method,
            'CreatedAt': timestamp,
            'ClientIP': self._generate_ip_address(user_id),
            'Application': application,
            'Browser': browser,
            'OS': os,
            'Unit': profile['unit'],
            'Title': profile['title']
        }
    
    def generate_dataset(self) -> pd.DataFrame:
        """Tam veri setini oluşturur"""
        print("Mock veri seti oluşturuluyor...")
        
        # Kullanıcı profillerini oluştur
        self.user_profiles = self._generate_user_profiles()
        
        all_logs = []
        
        for user_id in self.user_profiles.keys():
            user_logs = []
            
            # Kullanıcı başına log sayısı
            num_logs = int(DATA_CONFIG['num_logs_per_user'] * 
                          self.user_profiles[user_id]['login_frequency'])
            
            for day in range(DATA_CONFIG['date_range_days']):
                # Günlük giriş sayısı (1-3 arası)
                daily_logs = random.randint(1, 3)
                
                for _ in range(daily_logs):
                    timestamp = self._generate_timestamp(user_id, day)
                    
                    if timestamp is not None:
                        log_data = self._generate_login_data(user_id, timestamp)
                        user_logs.append(log_data)
            
            all_logs.extend(user_logs)
        
        # DataFrame oluştur
        df = pd.DataFrame(all_logs)
        
        # Tarih sıralaması
        df = df.sort_values('CreatedAt').reset_index(drop=True)
        
        print(f"Veri seti oluşturuldu: {len(df)} kayıt, {len(self.user_profiles)} kullanıcı")
        
        return df
    
    def save_dataset(self, filename: str = 'mock_login_data.csv'):
        """Veri setini dosyaya kaydeder"""
        df = self.generate_dataset()
        filepath = f"{PATHS['data_dir']}/{filename}"
        df.to_csv(filepath, index=False)
        print(f"Veri seti kaydedildi: {filepath}")
        return df

if __name__ == "__main__":
    # Test amaçlı veri üretimi
    generator = MockDataGenerator()
    df = generator.save_dataset()
    print(df.head())
    print(f"\nVeri seti boyutu: {df.shape}") 