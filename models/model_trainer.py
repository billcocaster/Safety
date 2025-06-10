"""
Model eğitimi ve değerlendirme modülü
Kapsamlı model eğitimi süreci yönetimi
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from data.mock_data_generator import MockDataGenerator
from data.risk_scorer import RiskScorer
from data.data_processor import DataProcessor
from models.deep_learning_model import RiskPredictionModel
from utils.config import *
from utils.visualization import plot_training_history, plot_risk_distribution

class ModelTrainer:
    """
    Model eğitimi ve değerlendirme sürecini yöneten sınıf
    """
    
    def __init__(self, output_dir: str = 'outputs'):
        """
        Args:
            output_dir: Çıktı dosyaları için dizin
        """
        self.output_dir = output_dir
        self.data_generator = MockDataGenerator()
        self.risk_scorer = RiskScorer()
        self.data_processor = DataProcessor()
        self.model = RiskPredictionModel()
        
        # Çıktı dizinlerini oluştur
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/models", exist_ok=True)
        os.makedirs(f"{output_dir}/plots", exist_ok=True)
        os.makedirs(f"{output_dir}/reports", exist_ok=True)
        
    def generate_and_prepare_data(self, force_regenerate: bool = False) -> pd.DataFrame:
        """
        Mock veri üretir ve hazırlar
        
        Args:
            force_regenerate: Veriyi yeniden üretmek için
            
        Returns:
            pd.DataFrame: Hazırlanmış veri seti
        """
        data_file = f"{PATHS['data_dir']}/mock_login_data.csv"
        
        if not force_regenerate and os.path.exists(data_file):
            print("Mevcut veri seti yükleniyor...")
            df = self.data_processor.load_data(data_file)
        else:
            print("Yeni mock veri seti oluşturuluyor...")
            df = self.data_generator.save_dataset()
        
        # Risk skorlarını hesapla
        if 'RiskScore' not in df.columns:
            print("Risk skorları hesaplanıyor...")
            df = self.risk_scorer.add_risk_scores(df)
            
            # Risk skorları eklenmiş veriyi kaydet
            df.to_csv(data_file, index=False)
        
        return df
    
    def prepare_training_data(self, df: pd.DataFrame, model_type: str = 'lstm') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Model eğitimi için veriyi hazırlar
        
        Args:
            df: Ham veri seti
            model_type: Model tipi ('lstm', 'hybrid', 'tabular')
            
        Returns:
            Tuple: Eğitim, validation ve test setleri
        """
        print(f"{model_type.upper()} modeli için veri hazırlanıyor...")
        
        # Özellik çıkarımı
        df_features = self.data_processor.extract_features(df)
        
        # Model tipine göre veri hazırlama
        if model_type in ['lstm', 'hybrid']:
            X, y = self.data_processor.prepare_sequence_data(
                df_features, 
                sequence_length=self.model.config['sequence_length']
            )
        else:  # tabular
            X, y = self.data_processor.prepare_tabular_data(df_features)
        
        # Veri bölünmesi
        X_train, X_val, X_test, y_train, y_val, y_test = self.data_processor.split_data(
            X, y, test_size=0.2, val_size=0.2
        )
        
        print(f"Veri hazırlama tamamlandı:")
        print(f"  Eğitim: {X_train.shape}")
        print(f"  Validation: {X_val.shape}")
        print(f"  Test: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   model_type: str = 'lstm',
                   user_embeddings: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
        """
        Model eğitimi gerçekleştirir
        
        Args:
            X_train: Eğitim özellikleri
            y_train: Eğitim hedefleri
            X_val: Validation özellikleri
            y_val: Validation hedefleri
            model_type: Model tipi
            user_embeddings: Kullanıcı embedding'leri
            
        Returns:
            Dict: Eğitim sonuçları
        """
        print(f"{model_type.upper()} modeli eğitimi başlıyor...")
        
        # Model kayıt yolları
        model_path = f"{self.output_dir}/models/{model_type}_risk_model.h5"
        preprocessor_path = f"{self.output_dir}/models/{model_type}_preprocessors.pkl"
        
        # Model eğitimi
        training_results = self.model.train(
            X_train, y_train, X_val, y_val,
            model_type=model_type,
            user_embeddings=user_embeddings,
            model_path=model_path
        )
        
        # Model ve ön işleme modellerini kaydet
        self.model.save_model(model_path, preprocessor_path)
        self.data_processor.save_preprocessors(preprocessor_path)
        
        # Eğitim geçmişini kaydet
        history_path = f"{self.output_dir}/models/{model_type}_training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.model.history.history, f, indent=2)
        
        print(f"Model eğitimi tamamlandı ve kaydedildi")
        
        return training_results
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray,
                      model_type: str = 'lstm',
                      user_embeddings: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
        """
        Model değerlendirmesi yapar
        
        Args:
            X_test: Test özellikleri
            y_test: Test hedefleri
            model_type: Model tipi
            user_embeddings: Kullanıcı embedding'leri
            
        Returns:
            Dict: Değerlendirme sonuçları
        """
        print(f"{model_type.upper()} modeli değerlendiriliyor...")
        
        # Model değerlendirmesi
        evaluation_results = self.model.evaluate(
            X_test, y_test, user_embeddings=user_embeddings
        )
        
        # Tahminler
        predictions = self.model.predict(X_test, user_embeddings)
        
        # Detaylı analiz
        detailed_analysis = self._perform_detailed_analysis(y_test, predictions)
        evaluation_results.update(detailed_analysis)
        
        # Sonuçları kaydet
        results_path = f"{self.output_dir}/reports/{model_type}_evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        print(f"Model değerlendirmesi tamamlandı")
        
        return evaluation_results
    
    def _perform_detailed_analysis(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Detaylı analiz yapar
        
        Args:
            y_true: Gerçek değerler
            y_pred: Tahmin edilen değerler
            
        Returns:
            Dict: Analiz sonuçları
        """
        # Hata analizi
        errors = y_true - y_pred
        abs_errors = np.abs(errors)
        
        # Risk kategorilerine göre analiz
        risk_categories = {
            'low': (y_true < RISK_THRESHOLDS['low']),
            'medium': ((y_true >= RISK_THRESHOLDS['low']) & (y_true < RISK_THRESHOLDS['medium'])),
            'high': (y_true >= RISK_THRESHOLDS['high'])
        }
        
        category_analysis = {}
        for category, mask in risk_categories.items():
            if np.sum(mask) > 0:
                category_analysis[f'{category}_mae'] = np.mean(abs_errors[mask])
                category_analysis[f'{category}_rmse'] = np.sqrt(np.mean(errors[mask]**2))
                category_analysis[f'{category}_count'] = int(np.sum(mask))
        
        # Tahmin doğruluğu analizi
        accuracy_analysis = {
            'within_5_points': np.mean(abs_errors <= 5),
            'within_10_points': np.mean(abs_errors <= 10),
            'within_20_points': np.mean(abs_errors <= 20),
            'large_errors_50_plus': np.mean(abs_errors > 50)
        }
        
        # İstatistiksel analiz
        statistical_analysis = {
            'error_mean': np.mean(errors),
            'error_std': np.std(errors),
            'error_median': np.median(abs_errors),
            'error_95th_percentile': np.percentile(abs_errors, 95),
            'correlation': np.corrcoef(y_true, y_pred)[0, 1]
        }
        
        return {
            'category_analysis': category_analysis,
            'accuracy_analysis': accuracy_analysis,
            'statistical_analysis': statistical_analysis
        }
    
    def compare_models(self, model_types: List[str] = ['lstm', 'tabular']) -> Dict[str, Any]:
        """
        Farklı model tiplerini karşılaştırır
        
        Args:
            model_types: Karşılaştırılacak model tipleri
            
        Returns:
            Dict: Karşılaştırma sonuçları
        """
        print("Model karşılaştırması başlıyor...")
        
        # Veri hazırlama
        df = self.generate_and_prepare_data()
        
        comparison_results = {}
        
        for model_type in model_types:
            print(f"\n{model_type.upper()} modeli işleniyor...")
            
            # Veri hazırlama
            X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_training_data(
                df, model_type
            )
            
            # Kullanıcı embedding'leri (hibrit model için)
            user_embeddings = None
            if model_type == 'hybrid':
                user_embeddings = self.data_processor.create_user_embeddings(df)
            
            # Model eğitimi
            training_results = self.train_model(
                X_train, y_train, X_val, y_val,
                model_type=model_type,
                user_embeddings=user_embeddings
            )
            
            # Model değerlendirmesi
            evaluation_results = self.evaluate_model(
                X_test, y_test,
                model_type=model_type,
                user_embeddings=user_embeddings
            )
            
            # Sonuçları birleştir
            comparison_results[model_type] = {
                'training': training_results,
                'evaluation': evaluation_results
            }
        
        # Karşılaştırma raporu oluştur
        comparison_report = self._create_comparison_report(comparison_results)
        
        # Raporu kaydet
        report_path = f"{self.output_dir}/reports/model_comparison_report.json"
        with open(report_path, 'w') as f:
            json.dump(comparison_report, f, indent=2)
        
        print("Model karşılaştırması tamamlandı")
        
        return comparison_results
    
    def _create_comparison_report(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Model karşılaştırma raporu oluşturur
        
        Args:
            comparison_results: Karşılaştırma sonuçları
            
        Returns:
            Dict: Karşılaştırma raporu
        """
        report = {
            'comparison_date': datetime.now().isoformat(),
            'models_compared': list(comparison_results.keys()),
            'summary': {},
            'detailed_results': comparison_results
        }
        
        # Özet karşılaştırma
        summary = {}
        for model_type, results in comparison_results.items():
            summary[model_type] = {
                'best_epoch': results['training']['best_epoch'],
                'final_val_loss': results['training']['final_val_loss'],
                'final_val_mae': results['training']['final_val_mae'],
                'test_mae': results['evaluation']['test_mae'],
                'test_rmse': results['evaluation']['test_rmse'],
                'test_r_squared': results['evaluation']['test_r_squared']
            }
        
        report['summary'] = summary
        
        # En iyi model belirleme
        best_model = min(summary.keys(), key=lambda x: summary[x]['test_mae'])
        report['best_model'] = {
            'model_type': best_model,
            'test_mae': summary[best_model]['test_mae'],
            'test_rmse': summary[best_model]['test_rmse']
        }
        
        return report
    
    def generate_visualizations(self, df: pd.DataFrame, model_type: str = 'lstm'):
        """
        Görselleştirmeler oluşturur
        
        Args:
            df: Veri seti
            model_type: Model tipi
        """
        print("Görselleştirmeler oluşturuluyor...")
        
        # Risk dağılımı
        plot_risk_distribution(df, save_path=f"{self.output_dir}/plots/risk_distribution.png")
        
        # Eğitim geçmişi
        history_path = f"{self.output_dir}/models/{model_type}_training_history.json"
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            plot_training_history(history, save_path=f"{self.output_dir}/plots/{model_type}_training_history.png")
        
        # Özellik önem analizi
        self._plot_feature_importance(df, model_type)
        
        print("Görselleştirmeler tamamlandı")
    
    def _plot_feature_importance(self, df: pd.DataFrame, model_type: str):
        """
        Özellik önem analizi görselleştirmesi
        
        Args:
            df: Veri seti
            model_type: Model tipi
        """
        # Özellik önem analizi
        feature_importance = self.data_processor.get_feature_importance_analysis(df)
        
        # En önemli 10 özelliği al
        top_features = dict(list(feature_importance.items())[:10])
        
        # Görselleştirme
        plt.figure(figsize=(12, 8))
        features = list(top_features.keys())
        importance = list(top_features.values())
        
        plt.barh(range(len(features)), importance)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Özellik Önem Skoru')
        plt.title('En Önemli 10 Özellik')
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/plots/{model_type}_feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_comprehensive_report(self, model_type: str = 'lstm') -> str:
        """
        Kapsamlı rapor oluşturur
        
        Args:
            model_type: Model tipi
            
        Returns:
            str: Rapor dosya yolu
        """
        print("Kapsamlı rapor oluşturuluyor...")
        
        # Veri yükleme
        df = self.generate_and_prepare_data()
        
        # Risk özeti
        risk_summary = self.risk_scorer.get_risk_summary(df)
        
        # Model sonuçları
        results_path = f"{self.output_dir}/reports/{model_type}_evaluation_results.json"
        model_results = {}
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                model_results = json.load(f)
        
        # Rapor oluşturma
        report_content = self._generate_report_content(
            df, risk_summary, model_results, model_type
        )
        
        # Raporu kaydet
        report_path = f"{self.output_dir}/reports/{model_type}_comprehensive_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"Kapsamlı rapor oluşturuldu: {report_path}")
        
        return report_path
    
    def _generate_report_content(self, df: pd.DataFrame, risk_summary: Dict[str, Any],
                               model_results: Dict[str, Any], model_type: str) -> str:
        """
        Rapor içeriği oluşturur
        
        Args:
            df: Veri seti
            risk_summary: Risk özeti
            model_results: Model sonuçları
            model_type: Model tipi
            
        Returns:
            str: Rapor içeriği
        """
        content = f"""# Kullanıcı Giriş Güvenlik Risk Değerlendirme Sistemi - Kapsamlı Rapor

## Proje Özeti

Bu proje, kullanıcıların sisteme giriş yapmaya çalıştığı anda elde edilen verilerle anlık risk değerlendirmesi yapan bir yapay zeka modeli geliştirmeyi amaçlamaktadır.

## Veri Seti Analizi

### Genel İstatistikler
- **Toplam Kayıt Sayısı**: {risk_summary['total_logs']:,}
- **Ortalama Risk Skoru**: {risk_summary['mean_risk']:.2f}
- **Medyan Risk Skoru**: {risk_summary['median_risk']:.2f}
- **Risk Skoru Standart Sapması**: {risk_summary['std_risk']:.2f}

### Risk Kategorileri
- **Düşük Risk (< {RISK_THRESHOLDS['low']})**: {risk_summary['low_risk_count']:,} kayıt
- **Orta Risk ({RISK_THRESHOLDS['low']}-{RISK_THRESHOLDS['medium']})**: {risk_summary['medium_risk_count']:,} kayıt
- **Yüksek Risk (≥ {RISK_THRESHOLDS['high']})**: {risk_summary['high_risk_count']:,} kayıt

### Risk Dağılımı
"""
        
        for range_name, count in risk_summary['risk_distribution'].items():
            percentage = (count / risk_summary['total_logs']) * 100
            content += f"- **{range_name}**: {count:,} kayıt (%{percentage:.1f})\n"
        
        content += f"""
## Model Performansı ({model_type.upper()})

### Eğitim Sonuçları
"""
        
        if model_results:
            content += f"""
- **Test MAE**: {model_results.get('test_mae', 'N/A'):.4f}
- **Test RMSE**: {model_results.get('test_rmse', 'N/A'):.4f}
- **Test R²**: {model_results.get('test_r_squared', 'N/A'):.4f}
"""
        
        content += f"""
## Etiketleme Yöntemi

### Seçilen Yöntem: Hibrit Yaklaşım

Bu projede **hibrit etiketleme yöntemi** kullanılmıştır. Bu yaklaşım aşağıdaki bileşenleri birleştirir:

### 1. Zaman Bazlı Kurallar
- **İş Saatleri**: 08:00-18:00 arası girişler düşük risk
- **Gece Saatleri**: 22:00-06:00 arası girişler yüksek risk
- **Hafta Sonu**: Cumartesi ve Pazar girişleri orta risk
- **Erken Sabah**: 05:00-07:00 arası girişler orta risk

### 2. Kullanıcı Davranış Kalıpları
- **Tercih Edilen MFA Yöntemi**: Kullanıcının en sık kullandığı doğrulama yöntemi
- **Tercih Edilen Uygulamalar**: Kullanıcının en sık giriş yaptığı uygulamalar
- **Tercih Edilen IP Aralığı**: Kullanıcının en sık kullandığı IP aralığı
- **Tercih Edilen Tarayıcı/OS**: Kullanıcının en sık kullandığı tarayıcı ve işletim sistemi

### 3. İstatistiksel Yaklaşım
- **Giriş Sıklığı**: Kullanıcının giriş yapma sıklığı
- **IP Çeşitliliği**: Kullanıcının kullandığı farklı IP sayısı
- **Uygulama Çeşitliliği**: Kullanıcının giriş yaptığı farklı uygulama sayısı

### Seçilme Nedenleri

1. **Gerçekçilik**: Gerçek dünya senaryolarını yansıtır
2. **Esneklik**: Farklı risk faktörlerini ağırlıklandırabilir
3. **Özelleştirilebilirlik**: Firma gereksinimlerine göre ayarlanabilir
4. **Ölçeklenebilirlik**: Büyük veri setlerinde etkili çalışır
5. **Yorumlanabilirlik**: Risk skorlarının nedenleri açıklanabilir

## Model Mimarisi

### LSTM Tabanlı Derin Öğrenme Modeli

**Seçilen Mimari**: LSTM (Long Short-Term Memory)

**Neden LSTM?**
- **Zaman Serisi Analizi**: Kullanıcı davranışlarının zaman içindeki değişimini yakalar
- **Uzun Vadeli Bağımlılıklar**: Geçmiş giriş davranışlarının gelecek riski etkilemesini modelleyebilir
- **Gradient Problem Çözümü**: Uzun sekanslarda gradient kaybolma problemini çözer
- **Kullanıcı Profili Öğrenme**: Her kullanıcının benzersiz davranış kalıplarını öğrenebilir

### Model Katmanları
1. **LSTM Katmanı 1**: 64 birim, sequence return
2. **Batch Normalization**: Eğitim stabilizasyonu
3. **LSTM Katmanı 2**: 32 birim, sequence return false
4. **Batch Normalization**: Eğitim stabilizasyonu
5. **Dense Katmanı 1**: 16 birim, ReLU aktivasyon
6. **Dropout**: Overfitting önleme
7. **Dense Katmanı 2**: 8 birim, ReLU aktivasyon
8. **Dropout**: Overfitting önleme
9. **Çıkış Katmanı**: 1 birim, linear aktivasyon

## Risk Ağırlık Sistemi

### Esnek Ağırlıklandırma
- **Zaman Bazlı Risk**: %25 ağırlık
- **IP Bazlı Risk**: %20 ağırlık
- **MFA Bazlı Risk**: %15 ağırlık
- **Uygulama Bazlı Risk**: %10 ağırlık
- **Tarayıcı Bazlı Risk**: %10 ağırlık
- **OS Bazlı Risk**: %10 ağırlık
- **Kullanıcı Profili Bazlı Risk**: %10 ağırlık

### Ağırlık Özelleştirme
Bu ağırlıklar firma gereksinimlerine göre kolayca değiştirilebilir. Örneğin:
- IP güvenliği kritikse IP ağırlığı artırılabilir
- Zaman bazlı güvenlik önemliyse zaman ağırlığı artırılabilir

## Sonuçlar ve Öneriler

### Başarılı Yönler
1. **Gerçekçi Mock Veri**: Gerçek dünya senaryolarını yansıtan veri seti
2. **Esnek Risk Hesaplama**: Özelleştirilebilir ağırlık sistemi
3. **Derin Öğrenme Modeli**: Karmaşık davranış kalıplarını öğrenebilen model
4. **Kapsamlı Analiz**: Detaylı raporlama ve görselleştirme

### Geliştirme Önerileri
1. **Gerçek Veri Entegrasyonu**: Mock veri yerine gerçek sistem logları kullanılması
2. **Model Optimizasyonu**: Hyperparameter tuning ile model performansının artırılması
3. **Gerçek Zamanlı Tahmin**: Canlı sistem entegrasyonu
4. **Anomali Tespiti**: Ek anomali tespit algoritmaları eklenmesi

## Teknik Detaylar

### Kullanılan Teknolojiler
- **Python**: Ana programlama dili
- **TensorFlow/Keras**: Derin öğrenme framework'ü
- **Pandas**: Veri manipülasyonu
- **NumPy**: Sayısal hesaplamalar
- **Scikit-learn**: Makine öğrenmesi araçları
- **Matplotlib/Seaborn**: Görselleştirme

### Performans Metrikleri
- **MAE (Mean Absolute Error)**: Ortalama mutlak hata
- **RMSE (Root Mean Square Error)**: Kök ortalama kare hata
- **R² (R-squared)**: Belirleme katsayısı

### Veri Seti Özellikleri
- **Kullanıcı Sayısı**: {DATA_CONFIG['num_users']:,}
- **Kayıt Sayısı**: {len(df):,}
- **Zaman Aralığı**: {DATA_CONFIG['date_range_days']} gün
- **Rastgelelik Oranı**: %{DATA_CONFIG['randomness_ratio']*100:.1f}

---
*Rapor Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return content 