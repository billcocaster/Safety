"""
Ana çalıştırma dosyası
Kullanıcı Giriş Güvenlik Risk Değerlendirme Sistemi
"""

import os
import sys
import argparse
import numpy as np
from datetime import datetime

# Proje modüllerini import et
from models.model_trainer import ModelTrainer
from utils.visualization import set_plot_style
from utils.config import *

def main():
    """
    Ana fonksiyon
    """
    parser = argparse.ArgumentParser(description='Kullanıcı Giriş Güvenlik Risk Değerlendirme Sistemi')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'compare', 'report', 'demo'], 
                       default='demo', help='Çalışma modu')
    parser.add_argument('--model_type', choices=['lstm', 'tabular', 'hybrid'], 
                       default='lstm', help='Model tipi')
    parser.add_argument('--output_dir', default='outputs', help='Çıktı dizini')
    parser.add_argument('--force_regenerate', action='store_true', 
                       help='Veriyi yeniden üret')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("KULLANICI GİRİŞ GÜVENLİK RİSK DEĞERLENDİRME SİSTEMİ")
    print("=" * 80)
    print(f"Çalışma Modu: {args.mode.upper()}")
    print(f"Model Tipi: {args.model_type.upper()}")
    print(f"Çıktı Dizini: {args.output_dir}")
    print(f"Başlangıç Zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Görselleştirme stilini ayarla
    set_plot_style()
    
    # Model trainer'ı başlat
    trainer = ModelTrainer(output_dir=args.output_dir)
    
    try:
        if args.mode == 'train':
            run_training_mode(trainer, args)
        elif args.mode == 'evaluate':
            run_evaluation_mode(trainer, args)
        elif args.mode == 'compare':
            run_comparison_mode(trainer, args)
        elif args.mode == 'report':
            run_report_mode(trainer, args)
        elif args.mode == 'demo':
            run_demo_mode(trainer, args)
        else:
            print(f"Bilinmeyen mod: {args.mode}")
            return 1
            
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("=" * 80)
    print(f"İşlem tamamlandı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    return 0

def run_training_mode(trainer: ModelTrainer, args):
    """
    Eğitim modu
    """
    print("\n🎯 EĞİTİM MODU BAŞLATILIYOR...")
    
    # Veri hazırlama
    print("\n📊 Veri hazırlanıyor...")
    df = trainer.generate_and_prepare_data(force_regenerate=args.force_regenerate)
    
    # Eğitim verisi hazırlama
    print(f"\n🔧 {args.model_type.upper()} modeli için veri hazırlanıyor...")
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_training_data(
        df, args.model_type
    )
    
    # Kullanıcı embedding'leri (hibrit model için)
    user_embeddings = None
    if args.model_type == 'hybrid':
        print("👤 Kullanıcı embedding'leri oluşturuluyor...")
        user_embeddings = trainer.data_processor.create_user_embeddings(df)
    
    # Model eğitimi
    print(f"\n🚀 {args.model_type.upper()} modeli eğitiliyor...")
    training_results = trainer.train_model(
        X_train, y_train, X_val, y_val,
        model_type=args.model_type,
        user_embeddings=user_embeddings
    )
    
    # Model değerlendirmesi
    print(f"\n📈 {args.model_type.upper()} modeli değerlendiriliyor...")
    evaluation_results = trainer.evaluate_model(
        X_test, y_test,
        model_type=args.model_type,
        user_embeddings=user_embeddings
    )
    
    # Görselleştirmeler
    print("\n📊 Görselleştirmeler oluşturuluyor...")
    trainer.generate_visualizations(df, args.model_type)
    
    # Sonuçları yazdır
    print("\n" + "="*50)
    print("EĞİTİM SONUÇLARI")
    print("="*50)
    print(f"Model Tipi: {training_results['model_type']}")
    print(f"En İyi Epoch: {training_results['best_epoch']}")
    print(f"Final Train Loss: {training_results['final_train_loss']:.4f}")
    print(f"Final Val Loss: {training_results['final_val_loss']:.4f}")
    print(f"Final Train MAE: {training_results['final_train_mae']:.4f}")
    print(f"Final Val MAE: {training_results['final_val_mae']:.4f}")
    
    print("\n" + "="*50)
    print("DEĞERLENDİRME SONUÇLARI")
    print("="*50)
    print(f"Test MAE: {evaluation_results['test_mae']:.4f}")
    print(f"Test RMSE: {evaluation_results['test_rmse']:.4f}")
    print(f"Test R²: {evaluation_results['test_r_squared']:.4f}")
    
    print(f"\n✅ Eğitim tamamlandı! Sonuçlar {args.output_dir} dizininde kaydedildi.")

def run_evaluation_mode(trainer: ModelTrainer, args):
    """
    Değerlendirme modu
    """
    print("\n📊 DEĞERLENDİRME MODU BAŞLATILIYOR...")
    
    # Veri hazırlama
    df = trainer.generate_and_prepare_data()
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_training_data(
        df, args.model_type
    )
    
    # Kullanıcı embedding'leri
    user_embeddings = None
    if args.model_type == 'hybrid':
        user_embeddings = trainer.data_processor.create_user_embeddings(df)
    
    # Model değerlendirmesi
    evaluation_results = trainer.evaluate_model(
        X_test, y_test,
        model_type=args.model_type,
        user_embeddings=user_embeddings
    )
    
    # Sonuçları yazdır
    print("\n" + "="*50)
    print("DEĞERLENDİRME SONUÇLARI")
    print("="*50)
    for key, value in evaluation_results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

def run_comparison_mode(trainer: ModelTrainer, args):
    """
    Model karşılaştırma modu
    """
    print("\n⚖️ MODEL KARŞILAŞTIRMA MODU BAŞLATILIYOR...")
    
    # Model tiplerini karşılaştır
    model_types = ['lstm', 'tabular']
    if args.model_type == 'hybrid':
        model_types.append('hybrid')
    
    comparison_results = trainer.compare_models(model_types)
    
    # Sonuçları yazdır
    print("\n" + "="*60)
    print("MODEL KARŞILAŞTIRMA SONUÇLARI")
    print("="*60)
    
    for model_type, results in comparison_results.items():
        print(f"\n📊 {model_type.upper()} MODELİ:")
        print(f"  Test MAE: {results['evaluation']['test_mae']:.4f}")
        print(f"  Test RMSE: {results['evaluation']['test_rmse']:.4f}")
        print(f"  Test R²: {results['evaluation']['test_r_squared']:.4f}")
        print(f"  En İyi Epoch: {results['training']['best_epoch']}")
    
    # En iyi modeli belirle
    best_model = min(comparison_results.keys(), 
                    key=lambda x: comparison_results[x]['evaluation']['test_mae'])
    
    print(f"\n🏆 EN İYİ MODEL: {best_model.upper()}")
    print(f"  Test MAE: {comparison_results[best_model]['evaluation']['test_mae']:.4f}")

def run_report_mode(trainer: ModelTrainer, args):
    """
    Rapor oluşturma modu
    """
    print("\n📋 RAPOR OLUŞTURMA MODU BAŞLATILIYOR...")
    
    # Kapsamlı rapor oluştur
    report_path = trainer.create_comprehensive_report(args.model_type)
    
    print(f"\n✅ Rapor oluşturuldu: {report_path}")
    print("\n📄 Rapor özeti:")
    print("  - Veri seti analizi")
    print("  - Model performansı")
    print("  - Etiketleme yöntemi açıklaması")
    print("  - Model mimarisi")
    print("  - Risk ağırlık sistemi")
    print("  - Sonuçlar ve öneriler")

def run_demo_mode(trainer: ModelTrainer, args):
    """
    Demo modu - Hızlı gösterim
    """
    print("\n🎬 DEMO MODU BAŞLATILIYOR...")
    
    # Küçük veri seti ile hızlı demo
    print("\n📊 Demo veri seti oluşturuluyor...")
    
    # Konfigürasyonu demo için ayarla
    original_config = DATA_CONFIG.copy()
    DATA_CONFIG['num_users'] = 100  # Demo için daha az kullanıcı
    DATA_CONFIG['num_logs_per_user'] = 20  # Demo için daha az log
    
    try:
        # Veri hazırlama
        df = trainer.generate_and_prepare_data(force_regenerate=True)
        
        # Risk özeti
        risk_summary = trainer.risk_scorer.get_risk_summary(df)
        
        print("\n" + "="*50)
        print("DEMO VERİ SETİ ÖZETİ")
        print("="*50)
        print(f"Toplam Kayıt: {risk_summary['total_logs']:,}")
        print(f"Ortalama Risk: {risk_summary['mean_risk']:.2f}")
        print(f"Düşük Risk: {risk_summary['low_risk_count']:,} kayıt")
        print(f"Orta Risk: {risk_summary['medium_risk_count']:,} kayıt")
        print(f"Yüksek Risk: {risk_summary['high_risk_count']:,} kayıt")
        
        # Hızlı model eğitimi (sadece birkaç epoch)
        print(f"\n🚀 {args.model_type.upper()} modeli hızlı eğitimi...")
        
        # Model konfigürasyonunu demo için ayarla
        demo_config = MODEL_CONFIG.copy()
        demo_config['epochs'] = 5  # Demo için daha az epoch
        
        trainer.model.config = demo_config
        
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_training_data(
            df, args.model_type
        )
        
        # Kullanıcı embedding'leri
        user_embeddings = None
        if args.model_type == 'hybrid':
            user_embeddings = trainer.data_processor.create_user_embeddings(df)
        
        # Model eğitimi
        training_results = trainer.train_model(
            X_train, y_train, X_val, y_val,
            model_type=args.model_type,
            user_embeddings=user_embeddings
        )
        
        # Hızlı değerlendirme
        evaluation_results = trainer.evaluate_model(
            X_test, y_test,
            model_type=args.model_type,
            user_embeddings=user_embeddings
        )
        
        print("\n" + "="*50)
        print("DEMO SONUÇLARI")
        print("="*50)
        print(f"Model Tipi: {args.model_type.upper()}")
        print(f"Test MAE: {evaluation_results['test_mae']:.4f}")
        print(f"Test RMSE: {evaluation_results['test_rmse']:.4f}")
        print(f"Test R²: {evaluation_results['test_r_squared']:.4f}")
        
        # Örnek tahmin
        print(f"\n🔮 Örnek Tahmin:")
        sample_indices = np.random.choice(len(X_test), min(5, len(X_test)), replace=False)
        sample_X = X_test[sample_indices]
        sample_y = y_test[sample_indices]
        
        predictions = trainer.model.predict(sample_X, user_embeddings)
        
        for i, (true_val, pred_val) in enumerate(zip(sample_y, predictions)):
            print(f"  Örnek {i+1}: Gerçek={true_val:.2f}, Tahmin={pred_val:.2f}, Hata={abs(true_val-pred_val):.2f}")
        
        print(f"\n✅ Demo tamamlandı! Sistem başarıyla çalışıyor.")
        
    finally:
        # Orijinal konfigürasyonu geri yükle
        DATA_CONFIG.update(original_config)

if __name__ == "__main__":
    sys.exit(main()) 